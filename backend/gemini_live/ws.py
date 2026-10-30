"""The /ws/live proxy: page <-> Gemini Live, with the tools run server side.

The lead mounts it in one line (docs/patches/A_gemini_live.md):

    result = await gemini_live.ws.handle(websocket, access, language, voice_style, **options)

handle() runs after the websocket was accepted and the start message was read
(language, voice_style and the access context come from the server, never from
the page). It returns a LiveResult; ok False with a reason_class means the
voice router should try the next tier (nothing was said to the page except a
live_failed message).

Protocol (page <-> proxy), the /ws/voice event shapes are reused:

  page -> proxy   binary frame: PCM16 little endian mono 16 kHz microphone audio
                  text frame {"type": "end_session"}
  proxy -> page   binary frame: PCM16 little endian mono 24 kHz model speech
                  {"type": "live_ready"}                                   connected, set up
                  {"type": "status", "state": "listening"|"thinking"|"speaking", "detail": "<tool line>"?}
                  {"type": "partial_transcript", "text": cumulative, "language": "si"}
                  {"type": "final_transcript", "text": ..., "language": ...}
                  {"type": "agent_response_chunk", "text": cumulative}
                  {"type": "agent_response", "text": ..., "mode": "gemini live"}
                  {"type": "interrupt"}                                    barge in: stop playback, drop queued audio
                  {"type": "speaking_done"}
                  {"type": "confirmation", "confirmation": {...}}          modal, same as /ws/voice
                  {"type": "live_failed", "reason": "<reason_class>"}      genuine failure, the stream ends
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from tool_agent import messages as tool_messages
from tool_agent import scrub as tool_scrub

from . import errors
from .config import DEFAULT_MODEL, LiveSessionConfig
from .instructions import build_instruction
from .session import (
    AudioOut, Failed, GeminiLiveSession, GoAway, InputText, Interrupted, OutputText, Resumed, ToolActivity, TurnComplete,
)
from .tools import ToolExecutor, build_tools

log = logging.getLogger("gemini_live")

MODE = "gemini live"


@dataclass
class LiveResult:
    ok: bool
    reason_class: str = ""  # why it ended: a genuine failure class, or user_closed / safety_block
    turns: int = 0
    seconds: float = 0.0
    handle: Optional[str] = None  # last resumption handle, for a hand over


async def _json(websocket, payload: dict) -> None:
    await websocket.send_text(json.dumps(payload, ensure_ascii=False))


def _handover_note(context: str) -> str:
    """When the call moved here from another voice tier, the last lines of the
    conversation so far (the page sends them, already trimmed)."""
    context = (context or "").strip()[:800]
    if not context:
        return ""
    return (
        "\n\nThis conversation was already going on and moved to you after a connection change. "
        "Do not greet again. What was said so far:\n" + context
    )


async def handle(
    websocket,
    access,
    language: Optional[str],
    voice_style: str = "female_natural",
    *,
    api_key: str = "",
    model: str = DEFAULT_MODEL,
    voice_name: str = "Kore",
    setup_timeout_seconds: float = 8.0,
    user_name: Optional[str] = None,
    greet: bool = False,
    state_changing: bool = False,
    resumption_handle: Optional[str] = None,
    client_factory: Optional[Callable[[Any], Any]] = None,
    context: str = "",
) -> LiveResult:
    started = time.monotonic()
    lang = tool_messages.short(language)
    tools = await asyncio.to_thread(build_tools, access, None, state_changing)

    async def show_confirmation(view: dict) -> None:
        await _json(websocket, {"type": "confirmation", "confirmation": view})

    executor = ToolExecutor.for_tools(access, tools, lang, on_confirmation=show_confirmation)
    cfg = LiveSessionConfig(
        api_key=api_key,
        model=model,
        language=lang,
        voice_style=voice_style,
        voice_name=voice_name,
        system_instruction=build_instruction(lang, access, user_name, state_changing, greet) + _handover_note(context),
        tools=tools,
        resumption_handle=resumption_handle,
        setup_timeout_seconds=setup_timeout_seconds,
    )
    session = GeminiLiveSession(executor, client_factory=client_factory)
    if not await session.open(cfg):
        failure = session.closed_reason or errors.OUTAGE
        await _json(websocket, {"type": "live_failed", "reason": failure})
        return LiveResult(False, failure, 0, time.monotonic() - started)

    await _json(websocket, {"type": "live_ready"})
    await _json(websocket, {"type": "status", "state": "listening"})
    turns = 0
    outcome = {"reason": errors.USER_CLOSED}

    async def upstream() -> None:
        while True:
            message = await websocket.receive()
            kind = message.get("type")
            if kind == "websocket.disconnect":
                return
            data = message.get("bytes")
            if data:
                await session.send_audio(data)
                continue
            text = message.get("text")
            if text:
                try:
                    payload = json.loads(text)
                except ValueError:
                    continue
                if isinstance(payload, dict) and payload.get("type") == "end_session":
                    return

    async def downstream() -> None:
        nonlocal turns
        speaking = False
        async for event in session.events:
            if isinstance(event, AudioOut):
                if not speaking:
                    speaking = True
                    await _json(websocket, {"type": "status", "state": "speaking"})
                await websocket.send_bytes(event.pcm24k)
            elif isinstance(event, InputText):
                kind = "final_transcript" if event.final else "partial_transcript"
                await _json(websocket, {"type": kind, "text": event.text, "language": lang})
                if event.final:
                    turns += 1
                    await _json(websocket, {"type": "status", "state": "thinking"})
                    if state_changing and getattr(access, "is_verified", False) and (event.text or "").strip():
                        try:
                            import confirmations

                            if confirmations.pending_for(access) is not None:
                                hit = await asyncio.to_thread(confirmations.intercept, access, event.text, lang)
                                if hit is not None and getattr(hit, "text", ""):
                                    await _json(websocket, {"type": "agent_response", "text": hit.text, "mode": MODE})
                        except Exception:
                            pass
            elif isinstance(event, OutputText):
                if event.final:
                    cleaned = event.text or ""
                    try:
                        cleaned, dropped = tool_scrub.clean_text(
                            cleaned, "", tool_messages._dial().values()
                        )
                        if dropped:
                            dial_line = tool_messages.dial_text("tool_agent.not_found", lang)
                            cleaned = f"{cleaned} {dial_line}".strip() if cleaned.strip() else dial_line
                    except Exception:
                        cleaned = event.text or ""
                    await _json(websocket, {"type": "agent_response", "text": cleaned, "mode": MODE})
                else:
                    await _json(websocket, {"type": "agent_response_chunk", "text": event.text})
            elif isinstance(event, ToolActivity):
                key = tool_messages.STATUS_BY_TOOL.get(event.name)
                detail = tool_messages.text(key, lang) if key else None
                payload = {"type": "status", "state": "thinking"}
                if detail:
                    payload["detail"] = detail
                await _json(websocket, payload)
            elif isinstance(event, Interrupted):
                speaking = False
                await _json(websocket, {"type": "interrupt"})
                await _json(websocket, {"type": "status", "state": "listening"})
            elif isinstance(event, TurnComplete):
                speaking = False
                await _json(websocket, {"type": "speaking_done"})
                await _json(websocket, {"type": "status", "state": "listening"})
            elif isinstance(event, Failed):
                outcome["reason"] = event.reason_class
                await _json(websocket, {"type": "live_failed", "reason": event.reason_class})
            elif isinstance(event, (GoAway, Resumed)):
                continue  # handled inside the session; nothing for the page to do

    up = asyncio.create_task(upstream())
    down = asyncio.create_task(downstream())
    try:
        await asyncio.wait({up, down}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in (up, down):
            if not task.done():
                task.cancel()
        for task in (up, down):
            try:
                await task
            except BaseException:  # noqa: BLE001  disconnects and cancellations end the proxy quietly
                pass
        await session.close()
    reason = outcome["reason"]
    if reason == errors.USER_CLOSED and session.closed_reason in (errors.SAFETY_BLOCK,):
        reason = errors.SAFETY_BLOCK
    seconds = time.monotonic() - started
    log.info("gemini_live proxy ended reason=%s turns=%d seconds=%d", reason, turns, int(seconds))
    return LiveResult(not errors.is_failure(reason), reason, turns, seconds, session.resumption_handle)
