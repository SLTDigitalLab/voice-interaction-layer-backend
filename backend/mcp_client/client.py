"""MCP client for the SLT hub, built on the official SDK (plan R4, ADR 2.6).

What it does, in the order a call meets it:

1. Breaker. After a failed call the hub is skipped for
   MCP_HUB_BREAKER_SECONDS (default 300), the same idea as the 5 minute
   cloud LLM breaker in agent.py: a hub that is down must cost a voice
   turn nothing, not a timeout every time. While it is open every call
   raises HubBreakerOpen at once with no network traffic.
2. Retry. Network class failures (connection refused, DNS, timeout,
   HTTP 5xx/408/429) are retried MCP_HUB_RETRIES times with exponential
   backoff, inside one overall time budget. An authentication failure
   (401/403) is NEVER retried: the same key will be refused again.
3. Transport detection. Found out once and then cached: SSE at
   MCP_HUB_URL/sse first (the real hub is SSE only), then streamable HTTP
   at MCP_HUB_URL and at MCP_HUB_URL/mcp (a URL that already ends in /mcp
   or /sse is tried as given first). The wrong
   guess shows up as 404, 405 or an MCP level refusal and moves on to the
   next candidate; a 401 stops the search at once (the transport may be
   right, the key is not), and a network failure does too (every
   candidate would fail the same way).
4. One short lived session per operation (connect, initialize, call,
   close). It costs a few round trips but holds no socket between voice
   turns, survives a hub restart without any reconnect logic, and keeps
   the cached transport the only state that matters.

The SDK's streamable HTTP client turns every HTTP error into a generic
MCPError("Server returned an error response") with no status, so a 401 and
a 503 look the same. An httpx response hook therefore records the status
codes of each attempt, and classification uses those.

The API key is sent in the header MCP_HUB_AUTH_HEADER (a value for
"Authorization" without a space gets "Bearer " in front). It never
appears in a log line, a status() result or an error message.

Everything here is async; tools.py runs it on a background event loop
thread so the synchronous Orchestrator can call it. status() is plain
sync and thread safe, for /health.
"""

import asyncio
import json
import random
import re
import socket
import ssl
import threading
import time
from datetime import datetime, timezone
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, TypeVar

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
from mcp_client.errors import HubAuthError, HubBreakerOpen, HubError, HubToolError, HubUnavailable

T = TypeVar("T")

TOOLS_CACHE_SECONDS = 300.0
# Longest a finished session may spend closing before it is cut off.
_CLOSE_SECONDS = 3.0
_BACKOFF_BASE_SECONDS = 0.25
_BACKOFF_MAX_SECONDS = 2.0
# Retry waits use "equal jitter": half the backoff plus a random share of the
# other half, so many callers that failed together do not retry together.
_JITTER = random.random
# Statuses that mean "this URL does not speak MCP the way we tried": try
# the next transport candidate.
_WRONG_TRANSPORT_STATUSES = {400, 404, 405, 406, 410, 415, 501}
_UNAVAILABLE_STATUSES = {408, 425, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class HubTool:
    name: str
    description: str = ""
    input_schema: dict = field(default_factory=dict)

    def required_string_argument(self) -> Optional[str]:
        """The argument name when this tool takes exactly one required
        property and it is a string, otherwise None. This is the shape
        "tool(text)" that lets the assistant call a tool it knows nothing
        else about."""
        required = self.input_schema.get("required") or []
        properties = self.input_schema.get("properties") or {}
        if len(required) != 1:
            return None
        name = required[0]
        if (properties.get(name) or {}).get("type") != "string":
            return None
        return name

    def string_argument(self, preferred: tuple = ()) -> Optional[str]:
        """Looser than required_string_argument, for a tool the operator
        named explicitly: the single required string, else a preferred
        name, else the only string property."""
        properties = self.input_schema.get("properties") or {}
        strings = [n for n, spec in properties.items() if (spec or {}).get("type") == "string"]
        required = [n for n in (self.input_schema.get("required") or []) if n in strings]
        if len(required) == 1:
            return required[0]
        for name in preferred:
            if name in strings:
                return name
        if len(strings) == 1:
            return strings[0]
        return None


@dataclass(frozen=True)
class ToolOutput:
    """What a tool call returned: the text blocks joined, and the
    structured content when the hub sends it (either may be empty)."""

    text: str
    structured: Any = None


class _WrongTransport(Exception):
    """Internal: this candidate does not speak MCP, try the next."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _network_reason(leaf: BaseException) -> str:
    """dns, tls or connect, from the cause chain of a transport error."""
    seen = 0
    cursor: Optional[BaseException] = leaf
    while cursor is not None and seen < 8:
        if isinstance(cursor, socket.gaierror):
            return "dns"
        if isinstance(cursor, ssl.SSLError):
            return "tls"
        cursor = cursor.__cause__ or cursor.__context__
        seen += 1
    return "connect"


def _leaves(exc: BaseException) -> list[BaseException]:
    """The real exceptions inside nested ExceptionGroups. The SDK runs its
    transports in anyio task groups, which wrap whatever went wrong."""
    if isinstance(exc, BaseExceptionGroup):
        found: list[BaseException] = []
        for inner in exc.exceptions:
            found.extend(_leaves(inner))
        return found
    return [exc]


class HubClient:
    def __init__(
        self,
        url: str,
        auth_header: str,
        api_key: str,
        timeout: float,
        retries: int,
        breaker_seconds: float,
        *,
        enabled: bool = True,
        clock: Callable[[], float] = time.monotonic,
        tools_cache_seconds: float = TOOLS_CACHE_SECONDS,
    ):
        self.url = (url or "").strip().rstrip("/")
        self.auth_header = (auth_header or "X-API-Key").strip()
        self._api_key = (api_key or "").strip()
        self.timeout = max(0.2, float(timeout))
        self.retries = max(0, int(retries))
        self.breaker_seconds = max(0.0, float(breaker_seconds))
        self.enabled = bool(enabled)
        self._clock = clock  # injectable so tests can move the breaker window
        self._tools_cache_seconds = tools_cache_seconds

        self._lock = threading.Lock()
        self._transport: Optional[tuple[str, str]] = None  # ("streamable" | "sse", endpoint url)
        self._breaker_until = 0.0
        self._reachable: Optional[bool] = None
        self._last_error: Optional[str] = None
        self._last_reason: Optional[str] = None
        self._auth = "unknown"  # "accepted" | "rejected" | "unknown"
        self._checked_at: Optional[str] = None  # ISO UTC time of the last outcome
        self._tools: Optional[list[HubTool]] = None
        self._tools_at = 0.0
        self.attempts = 0  # network attempts made, for tests and diagnostics
        self._background: set = set()  # sessions still closing, kept referenced

    # State

    @property
    def configured(self) -> bool:
        return bool(self.enabled and self.url and self._api_key)

    def breaker_seconds_left(self) -> float:
        with self._lock:
            return max(0.0, self._breaker_until - self._clock())

    def status(self) -> dict:
        """For /health. Never raises, never contains the key."""
        with self._lock:
            left = max(0.0, self._breaker_until - self._clock())
            transport = None
            if self._transport is not None:
                transport = "streamable-http" if self._transport[0] == "streamable" else "sse"
            return {
                "configured": self.configured,
                "enabled": self.enabled,
                "transport": transport,
                "reachable": self._reachable,
                "breaker_open": left > 0,
                "breaker_seconds_left": int(left) if left > 0 else 0,
                "tools": None if self._tools is None else len(self._tools),
                "last_error": self._last_error,
                "last_error_reason": self._last_reason,
                "auth": self._auth,
                "checked_at": self._checked_at,
            }

    def reset_breaker(self) -> None:
        with self._lock:
            self._breaker_until = 0.0

    def _scrub(self, text: str) -> str:
        """Short single line, key and URL credentials removed."""
        if self._api_key:
            text = text.replace(self._api_key, "***")
        text = re.sub(r"(https?://)[^/@\s]+@", r"\1***@", text)
        return " ".join(text.split())[:160]

    def _note_failure(self, error: HubError) -> None:
        with self._lock:
            self._reachable = False
            self._last_error = f"{type(error).__name__}: {self._scrub(str(error))}"
            self._last_reason = getattr(error, "reason", None)
            self._checked_at = _now_iso()
            if isinstance(error, HubAuthError):
                self._auth = "rejected"
            if self.breaker_seconds > 0 and not isinstance(error, HubToolError):
                self._breaker_until = self._clock() + self.breaker_seconds

    def _note_success(self) -> None:
        with self._lock:
            self._reachable = True
            self._last_error = None
            self._last_reason = None
            self._auth = "accepted"
            self._checked_at = _now_iso()
            self._breaker_until = 0.0

    def _header_value(self) -> str:
        if self.auth_header.lower() == "authorization" and " " not in self._api_key:
            return f"Bearer {self._api_key}"
        return self._api_key

    # Transports

    def _candidates(self) -> list[tuple[str, str]]:
        """SSE first: the SLT hub is SSE only (probed 2026-10-05: streamable
        HTTP at / and at /mcp fail, /sse initialises). Streamable HTTP is
        kept as the second choice so a hub that moves to it keeps working."""
        base = self.url
        if base.endswith("/mcp"):
            return [("streamable", base), ("sse", base[: -len("/mcp")] + "/sse")]
        if base.endswith("/sse"):
            root = base[: -len("/sse")]
            return [("sse", base), ("streamable", root), ("streamable", root + "/mcp")]
        return [("sse", base + "/sse"), ("streamable", base), ("streamable", base + "/mcp")]

    async def _attempt(self, candidate: tuple[str, str], op: Callable[[ClientSession], Awaitable[T]]) -> T:
        """One connection to one candidate. Translates whatever goes wrong
        into HubAuthError, HubUnavailable, HubToolError or _WrongTransport.

        The session lives in its own task and the caller gets the result
        the moment op finishes, not when the connection has been closed.
        Closing is the slow part (measured: a few hundred ms to seconds
        under load, from the DELETE that ends the session and from
        unwinding the streams), and nothing the caller needs depends on
        it. Teardown is capped at _CLOSE_SECONDS and its errors are
        ignored once a result exists."""
        kind, endpoint = candidate
        statuses: list[int] = []

        async def record(response: httpx2.Response) -> None:
            if response.status_code >= 400:
                statuses.append(response.status_code)

        headers = {self.auth_header: self._header_value()}
        timeout = httpx2.Timeout(self.timeout)

        def client_factory(headers=None, timeout=None, auth=None) -> httpx2.AsyncClient:  # SSE path
            return httpx2.AsyncClient(
                headers=headers, timeout=timeout or httpx2.Timeout(self.timeout), auth=auth, event_hooks={"response": [record]}
            )

        loop = asyncio.get_running_loop()
        outcome: asyncio.Future = loop.create_future()

        async def lifecycle() -> None:
            stack = AsyncExitStack()
            errors: list[BaseException] = []
            try:
                if kind == "streamable":
                    http = await stack.enter_async_context(
                        httpx2.AsyncClient(headers=headers, timeout=timeout, event_hooks={"response": [record]})
                    )
                    streams = await stack.enter_async_context(streamable_http_client(endpoint, http_client=http))
                else:
                    streams = await stack.enter_async_context(
                        sse_client(
                            endpoint,
                            headers=headers,
                            timeout=self.timeout,
                            sse_read_timeout=self.timeout,
                            httpx_client_factory=client_factory,
                        )
                    )
                session = await stack.enter_async_context(ClientSession(streams[0], streams[1]))
                await session.initialize()
                value = await op(session)
                if not outcome.done():
                    outcome.set_result(value)
            except BaseException as exc:  # noqa: BLE001  (classified below)
                errors.append(exc)
            try:
                with anyio.move_on_after(_CLOSE_SECONDS):
                    await stack.aclose()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
            if not outcome.done():
                outcome.set_exception(self._classify(errors, statuses))

        self.attempts += 1
        task = loop.create_task(lifecycle())
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        try:
            return await outcome
        except asyncio.CancelledError:
            task.cancel()
            raise

    def _classify(self, errors: list[BaseException], statuses: list[int]) -> Exception:
        leaves = [leaf for exc in errors for leaf in _leaves(exc)]
        # A cancellation here is anyio unwinding after a child task failed;
        # the failure itself is the other leaf, so look at that.
        real = [leaf for leaf in leaves if not isinstance(leaf, asyncio.CancelledError)]
        leaves = real or leaves
        for leaf in leaves:
            if isinstance(leaf, HubError):
                return leaf
        for leaf in leaves:
            if isinstance(leaf, httpx2.HTTPStatusError):
                statuses = statuses + [leaf.response.status_code]
        if any(code in (401, 403) for code in statuses):
            code = [c for c in statuses if c in (401, 403)][0]
            return HubAuthError(f"the hub refused the credentials (HTTP {code})", status=code)
        for leaf in leaves:
            if isinstance(leaf, (httpx2.TimeoutException, TimeoutError, asyncio.TimeoutError)):
                return HubUnavailable("timed out talking to the hub", reason="timeout")
            if isinstance(leaf, httpx2.TransportError):
                return HubUnavailable(
                    f"could not connect to the hub ({type(leaf).__name__})", reason=_network_reason(leaf)
                )
        if any(code in _UNAVAILABLE_STATUSES for code in statuses):
            code = [c for c in statuses if c in _UNAVAILABLE_STATUSES][0]
            return HubUnavailable(f"the hub answered HTTP {code}", reason="http_status", status=code)
        if statuses and not any(code in _WRONG_TRANSPORT_STATUSES for code in statuses):
            return HubUnavailable(f"the hub answered HTTP {statuses[0]}", reason="http_status", status=statuses[0])
        # A 404/405 style refusal, an MCP level error, a body that is not
        # MCP: this candidate is the wrong transport or path.
        reason = f"HTTP {statuses[0]}" if statuses else type(leaves[0]).__name__
        return _WrongTransport(reason)

    async def _with_session(self, op: Callable[[ClientSession], Awaitable[T]]) -> T:
        """Runs op on the cached transport, or detects the transport."""
        cached = self._transport
        candidates = [cached] if cached is not None else self._candidates()
        tried: list[str] = []
        for candidate in candidates:
            try:
                result = await self._attempt(candidate, op)
            except _WrongTransport as wrong:
                tried.append(f"{candidate[0]} {candidate[1].replace(self.url, '') or '/'} ({wrong})")
                if cached is not None:
                    # The hub changed under us: forget it and detect again next time.
                    with self._lock:
                        self._transport = None
                    raise HubUnavailable(
                        "the cached transport stopped working; detecting again", reason="no_transport"
                    ) from None
                continue
            if cached is None:
                with self._lock:
                    self._transport = candidate
            return result
        raise HubUnavailable("no MCP transport answered: " + "; ".join(tried), reason="no_transport")

    # Public API

    async def _call(self, op: Callable[[ClientSession], Awaitable[T]], budget: Optional[float] = None) -> T:
        if not self.configured:
            raise HubUnavailable(
                "the hub is not configured (needs MCP_HUB_ENABLED, MCP_HUB_URL and MCP_HUB_API_KEY)",
                reason="not_configured",
            )
        left = self.breaker_seconds_left()
        if left > 0:
            raise HubBreakerOpen(left)
        deadline = time.monotonic() + (budget if budget is not None else self.timeout * (self.retries + 1))
        retry = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.05:
                error = HubUnavailable("the time budget for the hub call ran out", reason="budget")
                self._note_failure(error)
                raise error
            try:
                result = await asyncio.wait_for(self._with_session(op), timeout=min(self.timeout, remaining))
            except asyncio.TimeoutError:
                failure: HubError = HubUnavailable("timed out talking to the hub", reason="timeout")
            except HubAuthError as error:
                self._note_failure(error)  # never retried
                raise
            except HubToolError:
                self._note_success()  # the hub itself answered fine
                raise
            except HubUnavailable as error:
                failure = error
            else:
                self._note_success()
                return result
            retry += 1
            backoff = min(_BACKOFF_MAX_SECONDS, _BACKOFF_BASE_SECONDS * (2 ** (retry - 1)))
            if retry > self.retries or deadline - time.monotonic() <= backoff + 0.2:
                self._note_failure(failure)
                raise failure
            await asyncio.sleep(backoff / 2 + (backoff / 2) * _JITTER())

    async def list_tools(self, refresh: bool = False, budget: Optional[float] = None) -> list[HubTool]:
        with self._lock:
            fresh = self._tools is not None and self._clock() - self._tools_at < self._tools_cache_seconds
            if fresh and not refresh:
                return list(self._tools or [])

        async def op(session: ClientSession) -> list[HubTool]:
            tools: list[HubTool] = []
            cursor = None
            for _ in range(20):  # a page limit, so a hub that loops cannot hold us
                result = await session.list_tools(params=None if cursor is None else _page(cursor))
                tools.extend(
                    HubTool(name=t.name, description=t.description or "", input_schema=dict(t.input_schema or {}))
                    for t in result.tools
                )
                cursor = getattr(result, "next_cursor", None)
                if not cursor:
                    break
            return tools

        tools = await self._call(op, budget)
        with self._lock:
            self._tools = tools
            self._tools_at = self._clock()
        return list(tools)

    @staticmethod
    def _gate(name: str, arguments: dict, access) -> dict:
        """The server side access check for customer tools (backend/access).
        Runs before any network call and whatever the model asked for: a
        customer tool needs a verified session, and its telephone_no is
        replaced by the verified number, written the way the hub documents it
        (0112150026). A caller that passes no context is a guest. Raises
        access.AccessDenied. Public tools pass untouched."""
        from access import GUEST, authorize_tool, bind_arguments

        context = access or GUEST
        authorize_tool(name, arguments, context)
        bound = bind_arguments(name, arguments, context)
        number = bound.get("telephone_no")
        if isinstance(number, str) and number.startswith("+94"):
            bound["telephone_no"] = "0" + number[3:]
        return bound

    async def call_tool_output(
        self, name: str, arguments: dict, budget: Optional[float] = None, access=None, trusted: bool = False
    ) -> ToolOutput:
        """The tool's result with its structured content kept. Raises
        HubToolError when the hub reports the call as failed.

        access is the caller's access.AccessContext. trusted=True skips the
        gate and is only for this repository's own login flow (OTP_BACKEND=hub
        and the customer lookup), never for anything driven by a user or a
        model."""
        if not trusted and config.ACCESS_CONTROL_ENABLED:
            arguments = self._gate(name, arguments, access)

        async def op(session: ClientSession) -> ToolOutput:
            try:
                result = await session.call_tool(name, arguments)
            except MCPError as exc:
                # A protocol level refusal of THIS request (unknown tool,
                # invalid params): the hub is healthy, the request is not.
                raise HubToolError(f"{name}: {self._scrub(str(exc.error.message))}") from None
            text = "\n".join(
                block.text for block in (getattr(result, "content", None) or []) if getattr(block, "type", "") == "text"
            ).strip()
            if getattr(result, "is_error", False):
                raise HubToolError(f"{name}: {self._scrub(text) or 'the tool reported an error'}")
            return ToolOutput(text=text, structured=getattr(result, "structured_content", None))

        return await self._call(op, budget)

    async def call_tool(
        self, name: str, arguments: dict, budget: Optional[float] = None, access=None, trusted: bool = False
    ) -> str:
        output = await self.call_tool_output(name, arguments, budget, access, trusted)
        if output.text:
            return output.text
        structured = output.structured
        if isinstance(structured, dict) and set(structured) == {"result"}:
            # The SDK wraps a plain return value as {"result": value}.
            structured = structured["result"]
            if isinstance(structured, str):
                return structured.strip()
        if structured in (None, "", {}, []):
            return ""
        return json.dumps(structured, ensure_ascii=False)


def _page(cursor: str):
    try:
        import mcp_types as types
    except ModuleNotFoundError:
        from mcp import types  # type: ignore[no-redef]

    return types.PaginatedRequestParams(cursor=cursor)


_hub: Optional[HubClient] = None
_hub_lock = threading.Lock()


def get_hub() -> HubClient:
    """The process wide client, built from config on first use."""
    global _hub
    with _hub_lock:
        if _hub is None:
            _hub = HubClient(
                url=config.MCP_HUB_URL,
                auth_header=config.MCP_HUB_AUTH_HEADER,
                api_key=config.MCP_HUB_API_KEY,
                timeout=config.MCP_HUB_TIMEOUT_SECONDS,
                retries=config.MCP_HUB_RETRIES,
                breaker_seconds=config.MCP_HUB_BREAKER_SECONDS,
                enabled=config.MCP_HUB_ENABLED,
            )
        return _hub


def set_hub(hub: Optional[HubClient]) -> None:
    """Replaces (or with None, clears) the process wide client. For tests."""
    global _hub
    with _hub_lock:
        _hub = hub
