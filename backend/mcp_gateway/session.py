"""One long lived MCP connection to the hub.

Why it exists: the hub ties a customer's login to the single MCP connection
that made it. verify_otp_and_authenticate on connection A authenticates the
tools on A and nothing else, so a customer's tool calls must go over the same
connection that logged them in. The short lived connections of
mcp_client.client cannot do that.

The connection lives in one asyncio task on the hub loop thread (the same
loop mcp_client.tools uses), because the SDK's transports are anyio task
groups that must be entered and left by the same task. Calls are serialised
with a lock. Any transport failure marks the session dead; it is never
reconnected silently, because a new connection would have no login. The
caller sees SessionDead and asks the customer to sign in again.

The public methods are blocking and meant for worker threads.
"""

import asyncio
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Callable, Optional

import anyio
try:
    import httpx2
except ModuleNotFoundError:
    import httpx as httpx2  # type: ignore[no-redef]
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import MCPError

import config
from mcp_client.client import HubClient, _WrongTransport, get_hub
from mcp_client.errors import HubAuthError, HubBreakerOpen, HubToolError, HubUnavailable
from mcp_client.tools import _runner

_CLOSE_SECONDS = 3.0


class SessionDead(Exception):
    """The connection is closed or broke; the login on it is gone."""


@dataclass(frozen=True)
class CallOutput:
    text: str
    structured: Any = None
    is_error: bool = False


class HubSession:
    def __init__(self, hub: Optional[HubClient] = None, clock: Callable[[], float] = time.monotonic):
        self._hub = hub
        self._clock = clock
        self.last_used = clock()
        self._dead = False
        self._opened = False
        self._task: Optional[asyncio.Task] = None
        self._session: Optional[ClientSession] = None
        self._stop: Optional[asyncio.Event] = None
        self._lock: Optional[asyncio.Lock] = None

    @property
    def hub(self) -> HubClient:
        return self._hub or get_hub()

    @property
    def alive(self) -> bool:
        return self._opened and not self._dead and self._task is not None and not self._task.done()

    # Blocking API

    def open(self, timeout: Optional[float] = None) -> None:
        """Connects and initialises. Raises HubUnavailable, HubAuthError or
        HubBreakerOpen; the session is closed again when it fails."""
        timeout = float(timeout if timeout is not None else config.MCP_HUB_TIMEOUT_SECONDS)
        try:
            _runner.run(self._aopen(timeout), timeout=timeout + 2.0)
        except BaseException:
            self.close()
            raise

    def call(self, name: str, arguments: dict, timeout: Optional[float] = None) -> CallOutput:
        """One tool call on this connection. Raises SessionDead, TimeoutError
        or HubToolError (a protocol level refusal of this request; the
        connection is still fine)."""
        timeout = float(timeout if timeout is not None else config.MCP_HUB_TIMEOUT_SECONDS)
        try:
            return _runner.run(self._acall(name, arguments, timeout), timeout=timeout + 2.0)
        except TimeoutError:
            raise
        except (SessionDead, HubToolError):
            raise
        except BaseException as exc:
            self._dead = True
            raise SessionDead(type(exc).__name__) from None

    def close(self) -> None:
        """Ends the connection. Safe to call twice and on a session that never opened."""
        self._dead = True
        if self._task is None:
            return
        try:
            _runner.run(self._aclose(), timeout=_CLOSE_SECONDS + 3.0)
        except BaseException:  # noqa: BLE001
            pass

    # Async side, on the hub loop

    async def _aopen(self, timeout: float) -> None:
        hub = self.hub
        if not hub.configured:
            raise HubUnavailable("the hub is not configured", reason="not_configured")
        left = hub.breaker_seconds_left()
        if left > 0:
            raise HubBreakerOpen(left)
        loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        self._lock = asyncio.Lock()
        ready: asyncio.Future = loop.create_future()
        self._task = loop.create_task(self._run(ready))
        try:
            await asyncio.wait_for(asyncio.shield(ready), timeout)
        except asyncio.TimeoutError:
            error = HubUnavailable("timed out opening a hub connection", reason="timeout")
            hub._note_failure(error)
            raise error from None
        except HubAuthError as error:
            hub._note_failure(error)
            raise
        except HubUnavailable as error:
            hub._note_failure(error)
            raise
        self._opened = True
        self.last_used = self._clock()
        hub._note_success()

    async def _run(self, ready: asyncio.Future) -> None:
        stack = AsyncExitStack()
        try:
            await self._connect(stack)
            if not ready.done():
                ready.set_result(None)
            await self._stop.wait()
        except BaseException as exc:  # noqa: BLE001
            self._dead = True
            if not ready.done():
                ready.set_exception(exc if isinstance(exc, Exception) else HubUnavailable("cancelled", reason="cancelled"))
        finally:
            self._dead = True
            try:
                with anyio.move_on_after(_CLOSE_SECONDS, shield=True):
                    await stack.aclose()
            except BaseException:  # noqa: BLE001
                pass

    async def _connect(self, stack: AsyncExitStack) -> None:
        """Opens the first transport candidate that speaks MCP. The transport
        the shared client already found out is tried first."""
        hub = self.hub
        cached = hub._transport
        candidates = [cached] if cached is not None else hub._candidates()
        tried = []
        for candidate in candidates:
            attempt = AsyncExitStack()
            statuses: list = []
            try:
                self._session = await self._enter(attempt, candidate, statuses)
            except BaseException as exc:  # noqa: BLE001
                try:
                    with anyio.move_on_after(_CLOSE_SECONDS, shield=True):
                        await attempt.aclose()
                except BaseException:  # noqa: BLE001
                    pass
                if isinstance(exc, asyncio.CancelledError):
                    raise
                failure = hub._classify([exc], statuses)
                if isinstance(failure, _WrongTransport) and cached is None:
                    tried.append(candidate[0])
                    continue
                if isinstance(failure, _WrongTransport):
                    with hub._lock:
                        hub._transport = None
                    raise HubUnavailable("the cached transport stopped working", reason="no_transport") from None
                raise failure from None
            stack.push_async_callback(attempt.aclose)
            if cached is None:
                with hub._lock:
                    hub._transport = candidate
            return
        raise HubUnavailable("no MCP transport answered: " + ", ".join(tried), reason="no_transport")

    async def _enter(self, attempt: AsyncExitStack, candidate: tuple, statuses: list) -> ClientSession:
        hub = self.hub
        kind, endpoint = candidate

        async def record(response: httpx2.Response) -> None:
            if response.status_code >= 400:
                statuses.append(response.status_code)

        headers = {hub.auth_header: hub._header_value()}
        # An idle connection must outlive the gaps between a customer's questions.
        read_timeout = max(float(config.HUB_SESSION_IDLE_SECONDS), 60.0) + 60.0

        def client_factory(headers=None, timeout=None, auth=None) -> httpx2.AsyncClient:
            return httpx2.AsyncClient(
                headers=headers, timeout=timeout or httpx2.Timeout(hub.timeout), auth=auth,
                event_hooks={"response": [record]},
            )

        if kind == "streamable":
            http = await attempt.enter_async_context(
                httpx2.AsyncClient(
                    headers=headers, timeout=httpx2.Timeout(hub.timeout, read=read_timeout),
                    event_hooks={"response": [record]},
                )
            )
            streams = await attempt.enter_async_context(streamable_http_client(endpoint, http_client=http))
        else:
            streams = await attempt.enter_async_context(
                sse_client(
                    endpoint, headers=headers, timeout=hub.timeout, sse_read_timeout=read_timeout,
                    httpx_client_factory=client_factory,
                )
            )
        session = await attempt.enter_async_context(ClientSession(streams[0], streams[1]))
        await session.initialize()
        return session

    async def _acall(self, name: str, arguments: dict, timeout: float) -> CallOutput:
        if not self.alive or self._session is None or self._lock is None:
            raise SessionDead("closed")
        async with self._lock:
            if not self.alive:
                raise SessionDead("closed")
            try:
                result = await asyncio.wait_for(self._session.call_tool(name, arguments), timeout)
            except asyncio.TimeoutError:
                # The login on the connection is still good; only this answer is late.
                raise TimeoutError("hub call timed out") from None
            except MCPError as exc:
                raise HubToolError(f"{name}: {' '.join(str(exc.error.message).split())[:160]}") from None
            except BaseException as exc:  # noqa: BLE001
                self._dead = True
                self._stop.set()
                if isinstance(exc, asyncio.CancelledError):
                    raise
                raise SessionDead(type(exc).__name__) from None
        self.last_used = self._clock()
        text = "\n".join(
            block.text for block in (getattr(result, "content", None) or []) if getattr(block, "type", "") == "text"
        ).strip()
        structured = getattr(result, "structured_content", None)
        if not text and isinstance(structured, dict) and set(structured) == {"result"} and isinstance(structured["result"], str):
            text = structured["result"].strip()
        return CallOutput(text=text, structured=structured, is_error=bool(getattr(result, "is_error", False)))

    async def _aclose(self) -> None:
        self._dead = True
        if self._stop is not None:
            self._stop.set()
        task = self._task
        if task is None:
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), _CLOSE_SECONDS + 1.0)
        except BaseException:  # noqa: BLE001
            task.cancel()
