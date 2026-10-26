"""FastAPI real time backend for the voice interaction service.

Ties stt.py, vad.py, agent.py, and tts.py into one live conversational loop
over a WebSocket: audio in, noise robust voice activity detection,
streaming transcription, an orchestrated agent response (cloud multi agent
if reachable, local single agent otherwise), streaming speech out. This is
what turns the four standalone modules into an actual real time product
rather than four scripts someone runs one after another.

Protocol (WebSocket at /ws/voice):

    Client -> server:
        - Binary frames: raw float32 mono PCM audio chunks at
          vad.SAMPLE_RATE (16000 Hz), CHUNK_SECONDS long. The frontend
          captures microphone audio, resamples/converts it to this format,
          and sends each chunk as it is captured.
        - Text frames: small JSON control messages,
          {"type": "start_session", "language_hint": "en" | "si" | "ta" | null,
           "user_name": "..." | null, "greet": true | false,
           "session_token": "..." | null}
          sent once at the start of a session, before any audio, and
          {"type": "end_session"} to close cleanly. user_name null means
          Guest; greet (default true) asks the server to speak its
          introduction first. session_token (optional) is the token from
          POST /auth/verify; with it, or with the user_session cookie on
          the handshake, the user is a verified existing user and is
          addressed by the name on their account, whatever user_name says.

    Server -> client (all JSON text frames):
        {"type": "status", "state": "calibrating" | "listening" | "thinking" | "speaking"}
            "speaking" is sent when the first sentence's audio actually
            starts, which can be while the rest of the reply is still
            being generated (see _speak_streamed_reply below).
        {"type": "speaking_sentence", "text": "..."}
            Sent right before each sentence's audio starts, so the avatar's
            mouth animation follows the real audio sentence by sentence.
        {"type": "user_name", "name": "..."}
            Sent when the user introduces themselves mid conversation
            ("my name is ..."); later replies address them by that name.
        {"type": "partial_transcript", "text": "...", "language": "en"}
        {"type": "final_transcript", "text": "...", "language": "en"}
        {"type": "agent_response_chunk", "text": "..."}
            Sent repeatedly while the agent's reply is being generated,
            "text" holding the accumulated reply so far (not just the new
            piece), the same cumulative shape partial_transcript already
            uses. Lets the frontend show the reply appearing in real time
            during the "thinking" status instead of only once it is fully
            generated. See agent.py's Orchestrator.respond_stream(): only
            the local fallback path is actually streamed chunk by chunk;
            the cloud multi agent path yields its finished reply as one
            chunk (documented there, and in KNOWN_LIMITATIONS.md).
        {"type": "agent_response", "text": "...", "mode": "cloud multi agent (gemini)"
                                                          | "local single agent (Llama via Ollama)"}
            Sent once, after the last agent_response_chunk, with the
            complete text and which mode produced it. Also used for the
            spoken introduction (mode "Introduction") and the canned
            closing lines (mode "closing").
        {"type": "speaking_done"}
        {"type": "end_chat"}
            Sent once, right after the closing goodbye has been spoken (see
            the closing detection below), when the conversation has reached
            its own natural end rather than being cut off. The server closes
            the connection immediately after sending this.
        {"type": "error", "message": "..."}

    Audio playback itself happens server side in this pass (see
    ARCHITECTURE.md for why: streaming synthesized audio back over the
    socket for in-browser playback is a reasonable next step, but out of
    scope here, since this runs on one local machine for a live demo). The
    frontend reflects status and transcripts from these events; the actual
    voice comes out of the server machine's speakers via tts.speak_text().

Why transcription runs on a background thread per utterance: transcribe_stream()
(stt.py) is a plain synchronous generator, built that way so it can be unit
tested and driven from the standalone microphone test without any asyncio
involved. Its whisper calls block. Calling it directly inside this async
handler would stall the event loop (and every other connection) for the
duration of each transcription pass. Running it on a thread and bridging its
yielded events back into the event loop via an asyncio.Queue keeps audio
receiving and event sending fully concurrent with the blocking work, which
is what actually lets a partial transcript reach the client while the user
is still talking, not just after they stop.

Every stage logs its own latency (stt.py, agent.py, tts.py each already
print their own timings); this module additionally logs a per turn total so
real time performance has evidence behind it, not just a claim.
"""

import asyncio
import functools
import ctypes
import json
import queue
import re
import sys
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from admin import service as admin_service
from admin.routes import router as admin_router
from agent import Orchestrator
from auth.admins import seed_first_admin_from_env
from log_safety import cap_sensitive_loggers
from auth.access_context import access_from_request
from auth.dependencies import verified_user_from_request
from auth.routes import router as auth_router
from db.migrate import database_status, upgrade_to_head
from knowledge.retriever import get_retriever
from mcp_client.tools import compose_tool_callbacks, hub_context_for
from mcp_client.tools import status as mcp_hub_status
from mcp_client.tools import warm_up as warm_up_mcp_hub
from mcp_server.mount import mount_mcp
from otp.routes import router as otp_router
from stt import transcribe_stream
from stt import warm_up as warm_up_stt
from tts import SentenceChunker, speak_stream, speak_text
from tts import precache as precache_speech
from tts import warm_up as warm_up_tts
from tts import status as speech_status
from vad import SAMPLE_RATE, BargeInDetector, VoiceActivityDetector
from voicekit.api import create_router as create_voicekit_router
from voicekit.settings import VoicekitSettings

# LiveKit is an optional transport. Importing it loads a native library, so a
# machine that cannot load it (see KNOWN_LIMITATIONS.md) keeps /ws/voice.
try:
    from livekit_transport import router as livekit_router
    from livekit_transport import set_agent_status_provider as set_livekit_agent_status
    from livekit_transport import start_worker as start_livekit_worker
except Exception as _livekit_import_error:  # pragma: no cover, depends on the machine
    livekit_router = None
    set_livekit_agent_status = None
    start_livekit_worker = None
    print(f"LiveKit transport unavailable ({type(_livekit_import_error).__name__}); /ws/voice still works.")

FRONTEND_DIR = config.PROJECT_ROOT / "frontend"

# The SLT Mobitel knowledge base is the Orchestrator's tool/retrieval step
# (its existing tool_callback hook), and an external chat agent made active
# in the admin portal, if any, answers before the built in paths.
retriever = get_retriever()
# The MCP hub's search tool adds to the knowledge base notes, never replaces
# them: retrieval runs first and a hub failure leaves its output untouched.
_tool_callback = compose_tool_callbacks(retriever.context_for, hub_context_for)
if config.ANSWER_SOURCE == "mcp_hub":
    # The MCP hub is the only source of SLT facts: the tool agent fetches them
    # before the model runs (agent.py), so the local knowledge base and the
    # hub notes callback are not used for the assistant's answers.
    _tool_callback = None
orchestrator = Orchestrator(
    tool_callback=_tool_callback,
    external_agent_provider=admin_service.get_active_agent,
)
# MCP's ask_assistant gets its own instance: last_mode is per instance, so
# sharing one with the voice and chat paths could report another request's
# mode when they overlap.
mcp_orchestrator = Orchestrator(tool_callback=_tool_callback)


def _warm_up():
    """Loads everything that is otherwise loaded lazily by the first
    request: the Whisper model and noise reduction library (stt.py), the
    local Llama model plus any cloud SDK (agent.py), and the Google Text to
    Speech connection (tts.py). Measured on this
    machine, that was well over 15 seconds landing on the very first
    question of the first session. Runs on a background thread so the
    server accepts connections straight away; a session that starts
    during warm up is simply no faster than before."""
    start = time.time()
    try:
        warm_up_stt()
    except Exception as exc:
        print(f"Warm up: STT failed ({exc}).")
    try:
        orchestrator.warm_up()
    except Exception as exc:
        print(f"Warm up: agent failed ({exc}).")
    if _hub_only():
        print("Warm up: the local knowledge base is not loaded, the MCP hub is the only source of SLT answers.")
    else:
        try:
            # Loads the index and the embedding model, so the first question's
            # retrieval is as fast as every later one.
            retriever.warm_up()
        except Exception as exc:
            print(f"Warm up: knowledge retriever failed ({exc}).")
    try:
        warm_up_mcp_hub()
    except Exception as exc:
        print(f"Warm up: MCP hub failed ({exc}).")
    try:
        warm_up_tts()
        # The introductions are fixed text, so the slow to synthesize
        # ones (Sinhala, see tts.precache) can be ready before anyone
        # connects. Only the Guest version is fully fixed; a named
        # greeting still reuses the cached sentences after the name.
        for language in _GREETINGS:
            greeting_text, greeting_language = _greeting(None, language)
            precache_speech(greeting_text, greeting_language)
    except Exception as exc:
        print(f"Warm up: TTS failed ({exc}).")
    print(f"Warm up finished in {time.time() - start:.2f}s")


def _opt_out_of_windows_power_throttling():
    """Tells Windows not to power throttle this process (EcoQoS). On hybrid
    Intel CPUs, Windows runs processes it considers to be in the background
    (which includes this server when start.ps1 launches it as a hidden
    window) on the efficiency cores at reduced clock speed. Measured on the
    development machine, that made every Whisper pass about 5x slower:
    about 2.4s versus 0.47s for the same 5 second clip, and every partial
    and final transcript pays it. Uses the documented SetProcessInformation
    ProcessPowerThrottling call; does nothing on other platforms, and a
    failure is only logged, since the server still works, just slower."""
    if sys.platform != "win32":
        return

    class _PowerThrottlingState(ctypes.Structure):
        _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]

    process_power_throttling = 4  # PROCESS_INFORMATION_CLASS value
    execution_speed = 0x1  # PROCESS_POWER_THROTTLING_EXECUTION_SPEED
    # ControlMask says which policy is being set, StateMask 0 says "off".
    state = _PowerThrottlingState(1, execution_speed, 0)
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetProcessInformation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
        if not kernel32.SetProcessInformation(
            kernel32.GetCurrentProcess(), process_power_throttling, ctypes.byref(state), ctypes.sizeof(state)
        ):
            print("Power throttling opt out was refused by Windows; transcription may be slower.")
    except Exception as exc:
        print(f"Power throttling opt out failed ({exc}); transcription may be slower.")


# The running LiveKit worker handle, set at startup when VOICE_TRANSPORT is livekit.
livekit_state: dict = {}


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """One startup sequence for the whole app: power throttling opt out,
    database migrations and the first admin account (before anything can
    touch the database), then the background warm up, and the MCP session
    manager for as long as the app runs."""
    _opt_out_of_windows_power_throttling()
    upgrade_to_head()
    seed_first_admin_from_env()
    if config.WARM_UP_ON_STARTUP:
        threading.Thread(target=_warm_up, daemon=True).start()
    if config.mcp_hub_configured() and _hub_only():
        import mcp_gateway

        # The first guest question finds open hub connections instead of opening one.
        threading.Thread(target=mcp_gateway.prewarm, daemon=True).start()
    if config.VOICE_ROUTER_ENABLED:
        from voice_router import get_router

        # Off the event loop: a slow Gemini call must not delay start up.
        asyncio.get_running_loop().run_in_executor(None, get_router().probe_all)
    if config.VOICE_TRANSPORT == "livekit" and start_livekit_worker is not None:
        # Never raises: an unreachable LiveKit server is retried in the
        # background and shown on /health, it does not stop the app.
        livekit_state["worker"] = start_livekit_worker(sys.modules[__name__])
        # The page asks GET /api/livekit/config whether the agent is running and
        # uses /ws/voice and the local pipeline when it is not.
        set_livekit_agent_status(lambda: livekit_state["worker"].status())
    try:
        async with mcp_handle.lifespan():
            yield
    finally:
        worker = livekit_state.get("worker")
        if worker is not None:
            await asyncio.to_thread(worker.stop, 5)


# Third party loggers must never record customer data (see log_safety.py).
cap_sensitive_loggers()

app = FastAPI(title="Voice Interaction Backend", lifespan=_lifespan)

# The frontend is a static page opened from a different origin/port during
# development (see frontend/), so CORS needs to be open for the demo. This
# is a deliberate scope boundary, not an oversight: locking this down is
# part of the authentication and production deployment work explicitly out
# of scope for this pass (see README.md).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Existing user verification and admin sign in (/auth), the admin portal's
# API (/admin/api), and the MCP server with its OAuth endpoints (/mcp,
# /oauth/token, /.well-known/...). The frontend and admin page themselves
# are served as static files at the very end of this module, after every
# route, so they never shadow one.
app.include_router(auth_router)
app.include_router(otp_router)
if config.VERIFY_ENABLED:
    from verification.routes import router as verification_router

    app.include_router(verification_router)
app.include_router(admin_router)
from confirmations.routes import router as confirm_router  # noqa: E402  (confirm modal endpoints, /confirm)

app.include_router(confirm_router)
from feedback import routes as feedback_routes  # noqa: E402  (five star rating, /feedback)

app.include_router(feedback_routes.router)

# Additive routers: the LiveKit token endpoint (/api/livekit) and voicekit,
# the reusable text to speech API (/api/tts/stream, /ws/tts), which refuses
# every request until VOICEKIT_API_KEYS lists at least one key.
if livekit_router is not None:
    app.include_router(livekit_router)
if config.VOICEKIT_ENABLED:
    app.include_router(
        create_voicekit_router(
            VoicekitSettings(
                google_credentials_path=config.GOOGLE_APPLICATION_CREDENTIALS,
                azure_key=config.AZURE_SPEECH_KEY,
                azure_region=config.AZURE_SPEECH_REGION,
                cloud_provider=config.CLOUD_SPEECH_PROVIDER,
                local_voice_dir=str(config.VOICEKIT_LOCAL_VOICE_DIR),
                max_text_chars=config.VOICEKIT_MAX_TEXT_CHARS,
            ),
            lambda: config.VOICEKIT_API_KEYS,
            config.VOICEKIT_RATE_LIMIT_PER_MINUTE,
        )
    )


def _hub_only() -> bool:
    """True when the MCP hub is the only source of SLT answers. The scraped
    local knowledge base is then neither loaded nor read."""
    return config.ANSWER_SOURCE == "mcp_hub"


def _mcp_search(query: str, top_k: int) -> list[dict]:
    if _hub_only():
        # Other programs that ask this server for knowledge get the hub's own
        # knowledge base, not the scraped copy.
        import json as _json

        import mcp_gateway

        result = mcp_gateway.call("search_knowledgebase_tool", {"query": query, "top_k": top_k}, None)
        chunks = (result.data or {}).get("chunks") if result.ok and isinstance(result.data, dict) else None
        return [
            {
                "text": str(chunk.get("content", "")),
                "score": float(chunk.get("score", 0.0) or 0.0),
                "source_url": None,
                "retrieved_at": None,
                "title": "SLT MCP hub knowledge base",
            }
            for chunk in (chunks or [])
            if isinstance(chunk, dict)
        ]
    return [
        {
            "text": chunk.text,
            "score": chunk.score,
            "source_url": chunk.source_url,
            "retrieved_at": chunk.retrieved_at,
            "title": chunk.title,
        }
        for chunk in retriever.search(query, top_k)
    ]


def _mcp_ask(question: str, language: Optional[str]) -> tuple[str, str]:
    reply = mcp_orchestrator.respond(question, None, language)
    return reply, mcp_orchestrator.last_mode


def _mcp_package_details(name: str) -> list:
    # Structured package records exist only in the scraped knowledge base, which
    # is not used when the hub is the only source; search_knowledge has the text.
    return [] if _hub_only() else retriever.get_package_details(name)


mcp_handle = mount_mcp(app, _mcp_search, _mcp_package_details, _mcp_ask)

# vad.py's calibrated noise floor plus WebRTC VAD's frame classification
# reject most non-speech audio before it ever reaches transcription, but
# neither is perfect: a short percussive sound (a tap, a chair creak, a
# door) can still pass the majority-of-frames check. When that happens,
# faster-whisper does not reliably return an empty string for it; it is a
# well documented failure mode of Whisper models to hallucinate a short,
# generic phrase from training data instead (most commonly "you" or
# "thank you", echoing captioned video transcripts). final_text is
# non-empty in that case, so the `if not final_text` check below lets it
# through as if it were a real question. This list is a narrow, second net
# for exactly that known behavior, not a general filter on short replies:
# genuine short utterances that are not in this list still reach the agent.
_LIKELY_TRANSCRIPTION_HALLUCINATIONS = {
    "you", "thank you", "thanks for watching", "thanks", "bye", "bye bye",
}


def _looks_like_transcription_hallucination(text: str) -> bool:
    return text.strip().lower().rstrip(".!?") in _LIKELY_TRANSCRIPTION_HALLUCINATIONS


# Ends the conversation on the user's own terms, rather than only via the
# cut button: once an utterance sounds like the user is satisfied, the
# assistant asks a canned closing question instead of routing that turn to
# the agent; if the very next utterance also reads as closing (or a plain
# "no"), the assistant says goodbye and the connection ends, but a real new
# question instead just drops the closing offer and is answered normally
# (see the awaiting_close_confirmation handling in voice_ws below). A simple
# keyword match, not an LLM classification: fast, fully deterministic, and
# consistent with _looks_like_transcription_hallucination's own approach
# above. It is not exhaustive, especially for Sinhala/Tamil phrasing beyond
# the couple of common words included here; see KNOWN_LIMITATIONS.md.
_CLOSING_SIGNAL_LATIN = re.compile(
    r"\b(thank you|thanks|that'?s all|that is all|i'?m satisfied|im satisfied|"
    r"nothing else|no more questions|that will be all|i'?m done|im done|"
    r"i'?m good|im good)\b",
    re.IGNORECASE,
)
_CLOSING_SIGNAL_OTHER = ("ස්තුතියි", "නන්ද", "நன்றி")

_NEGATIVE_CLOSE_LATIN = re.compile(
    r"\b(no|nope|nah|nothing|that'?s it|that is it|no thanks|no thank you|bye|goodbye)\b",
    re.IGNORECASE,
)
_NEGATIVE_CLOSE_OTHER = ("නැහැ", "இல்லை")

# The canned closing lines, in the language the rest of the conversation
# is in (the language picked on the first screen); English otherwise.
_CLOSING_QUESTION_TEXT = {
    "en": "You're welcome! Is there anything else I can help you with, or shall I end our conversation here?",
    "si": "ඔබව සාදරයෙන් පිළිගනිමු! මට ඔබට තවත් යමක් උදව් කළ හැකිද, නැතහොත් අපගේ සංවාදය මෙතැනින් අවසන් කරන්නද?",
    "ta": "மகிழ்ச்சி! நான் உங்களுக்கு வேறு ஏதாவது உதவ முடியுமா, அல்லது நமது உரையாடலை இங்கே முடித்துக்கொள்ளலாமா?",
}
_CLOSING_GOODBYE_TEXT = {
    "en": "Thank you for using the SLT Mobitel voice assistant. Goodbye, and have a great day!",
    "si": "SLT Mobitel හඬ සහායකයා භාවිතා කළාට ස්තුතියි. ආයුබෝවන්, ඔබට සුබ දවසක්!",
    "ta": "SLT Mobitel குரல் உதவியாளரைப் பயன்படுத்தியதற்கு நன்றி. வணக்கம், இனிய நாளாக அமையட்டும்!",
}


def _in_language(lines: dict, language: Optional[str]) -> str:
    return lines.get(language) or lines["en"]


# The two voice styles tts.py understands. The browser's male/female switch
# sends one of them (start_session.voice_style, or a set_voice message during
# a session); anything else, or nothing, keeps the original female voice.
_VOICE_STYLES = ("female_natural", "male")
DEFAULT_VOICE_STYLE = "female_natural"


def _valid_voice_style(value) -> str:
    return value if value in _VOICE_STYLES else DEFAULT_VOICE_STYLE


# Lets the user tell the assistant their name by voice, not only on the
# user screen: "my name is Nimal", "call me Nimal". "I'm ..." is left out
# on purpose, since "I'm good" or "I'm looking for a plan" would otherwise
# rename the user. A small list of words that commonly follow these
# phrases without being a name ("call me back") guards the rest.
_NAME_INTRODUCTION = re.compile(
    r"\b(?:my name is|my name's|call me|i am called|i'm called)\s+([A-Za-z][A-Za-z']{1,29})",
    re.IGNORECASE,
)
_NOT_A_NAME = {"back", "later", "again", "now", "tomorrow", "not", "just", "a", "the", "so", "really"}


def _extract_user_name(text: str) -> Optional[str]:
    match = _NAME_INTRODUCTION.search(text)
    if not match:
        return None
    name = match.group(1).strip("'")
    if name.lower() in _NOT_A_NAME:
        return None
    return name[0].upper() + name[1:]


# The spoken introduction, in the language picked on the frontend's first
# screen, so a Sinhala or Tamil speaker is greeted in their own language
# from the first word. When no name was given, "Guest" is said in that
# language too, rather than an English word dropped into a Sinhala or
# Tamil sentence. Anything other than si/ta (including no language picked)
# falls back to English.
_GREETINGS = {
    "en": (
        "Guest",
        "Welcome to SLT Mobitel, {name}! I'm your voice assistant, and I understand "
        "English, Sinhala, and Tamil, so just start speaking whenever you're ready.",
    ),
    "si": (
        "අමුත්තා",
        "SLT Mobitel වෙත සාදරයෙන් පිළිගනිමු, {name}! මම ඔබේ හඬ සහායකයා. මට ඉංග්‍රීසි, "
        "සිංහල සහ දෙමළ භාෂා තේරෙනවා, ඒ නිසා ඔබ සූදානම් වූ විට කතා කිරීම ආරම්භ කරන්න.",
    ),
    "ta": (
        "விருந்தினர்",
        "SLT Mobitel-க்கு வரவேற்கிறோம், {name}! நான் உங்கள் குரல் உதவியாளர். எனக்கு "
        "ஆங்கிலம், சிங்களம் மற்றும் தமிழ் புரியும், எனவே நீங்கள் தயாரானதும் பேசத் தொடங்குங்கள்.",
    ),
}


def _greeting(user_name: Optional[str], language: Optional[str]) -> tuple[str, str]:
    """Returns (text, language) for the spoken introduction."""
    language = language if language in _GREETINGS else "en"
    guest_word, template = _GREETINGS[language]
    return template.format(name=user_name or guest_word), language


def _sounds_like_closing_signal(text: str) -> bool:
    if _CLOSING_SIGNAL_LATIN.search(text):
        return True
    return any(marker in text for marker in _CLOSING_SIGNAL_OTHER)


def _sounds_negative(text: str) -> bool:
    if _NEGATIVE_CLOSE_LATIN.search(text):
        return True
    return any(marker in text for marker in _NEGATIVE_CLOSE_OTHER)


async def _send_json(websocket: WebSocket, payload: dict):
    await websocket.send_text(json.dumps(payload))


class _ChunkQueueSource:
    """A synchronous iterator backed by a thread safe queue. The async
    websocket handler pushes items into it as they arrive; a background
    thread pulls from it via plain blocking Queue.get(), which is exactly
    the iterator shape transcribe_stream() (audio chunks) and
    tts.speak_stream() (reply sentences) already expect."""

    _SENTINEL = object()

    def __init__(self):
        self._queue: "queue.Queue" = queue.Queue()

    def put(self, chunk):
        self._queue.put(chunk)

    def close(self):
        self._queue.put(self._SENTINEL)

    def __iter__(self):
        return self

    def __next__(self):
        item = self._queue.get()
        if item is self._SENTINEL:
            raise StopIteration
        return item


async def _run_one_utterance(
    websocket: WebSocket,
    language_hint: Optional[str],
    vad: VoiceActivityDetector,
    voice_holder: Optional[dict] = None,
    seed: Optional[np.ndarray] = None,
):
    """Streams audio chunks from the websocket into transcribe_stream()
    running on a background thread, forwarding each partial or final event
    to the client the moment it is produced. Returns
    (final_text, detected_language, session_ended)."""
    loop = asyncio.get_running_loop()
    chunk_source = _ChunkQueueSource()
    if seed is not None and seed.size:
        # Barge in: the audio that interrupted the reply (about the last 400 ms
        # plus the confirming speech) opens the utterance, cut into the usual
        # chunk size, so the first word is not lost.
        step = int(SAMPLE_RATE * 0.25)
        for start in range(0, seed.size, step):
            chunk_source.put(seed[start : start + step])
    event_queue: "asyncio.Queue" = asyncio.Queue()
    worker_done = threading.Event()

    def _worker():
        try:
            for event in transcribe_stream(chunk_source, language_hint=language_hint, vad=vad):
                loop.call_soon_threadsafe(event_queue.put_nowait, event)
                if event.is_final:
                    break
        except Exception as exc:
            loop.call_soon_threadsafe(event_queue.put_nowait, exc)
        finally:
            worker_done.set()
            loop.call_soon_threadsafe(event_queue.put_nowait, None)

    threading.Thread(target=_worker, daemon=True).start()

    session_ended = False

    async def _receive_loop():
        nonlocal session_ended
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    session_ended = True
                    chunk_source.close()
                    return
                if message.get("bytes") is not None:
                    chunk_source.put(np.frombuffer(message["bytes"], dtype=np.float32))
                elif message.get("text") is not None:
                    try:
                        payload = json.loads(message["text"])
                    except ValueError:
                        continue
                    if payload.get("type") == "end_session":
                        session_ended = True
                        chunk_source.close()
                        return
                    if payload.get("type") == "set_voice" and voice_holder is not None:
                        voice_holder["voice_style"] = _valid_voice_style(payload.get("voice_style"))
        except WebSocketDisconnect:
            session_ended = True
            chunk_source.close()

    receive_task = asyncio.create_task(_receive_loop())

    final_text, final_language = None, None
    try:
        while True:
            item = await event_queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                await _send_json(websocket, {"type": "error", "message": f"Transcription error: {item}"})
                break
            if item.is_final:
                final_text, final_language = item.text, item.language
                if item.text:
                    await _send_json(
                        websocket, {"type": "final_transcript", "text": item.text, "language": item.language}
                    )
                chunk_source.close()
                break
            else:
                await _send_json(
                    websocket, {"type": "partial_transcript", "text": item.text, "language": item.language}
                )
    finally:
        chunk_source.close()  # idempotent-enough for our purposes; safe if already closed
        receive_task.cancel()
        try:
            await receive_task
        except (asyncio.CancelledError, Exception):
            pass

    return final_text, final_language, session_ended


def _status_text(key: Optional[str], language: Optional[str]) -> Optional[str]:
    """The localized status line for a tool_agent status key, or None."""
    if not key:
        return None
    from i18n import t

    text = t(key, language)
    return None if text == key else text


async def _stream_agent_reply(
    text_input: str, user_name: Optional[str], language: Optional[str], access=None, websocket=None, cancel=None
):
    """Runs orchestrator.respond_stream() (agent.py) on a background
    thread and yields each text chunk as it arrives, bridged into the
    event loop via an asyncio.Queue the same way _run_one_utterance above
    bridges transcribe_stream()'s blocking whisper calls. respond_stream()
    is itself a blocking generator (its requests.post call streams the
    HTTP response, but each iteration still blocks waiting on the next
    chunk from Ollama), so running it directly in this async handler would
    stall every other connection's event loop turn for the whole reply,
    exactly as an unstreamed call would."""
    loop = asyncio.get_running_loop()
    chunk_queue: "asyncio.Queue" = asyncio.Queue()

    def _status(key: str) -> None:
        # Keeps the "thinking" state and says what is being checked.
        if websocket is None:
            return
        detail = _status_text(key, language)
        if detail:
            asyncio.run_coroutine_threadsafe(
                _send_json(websocket, {"type": "status", "state": "thinking", "detail": detail}), loop
            )

    def _put(item):
        try:
            if not loop.is_closed():
                loop.call_soon_threadsafe(chunk_queue.put_nowait, item)
        except RuntimeError:
            pass

    def _worker():
        generator = None
        try:
            generator = orchestrator.respond_stream(text_input, user_name, language, access=access, status_cb=_status)
            for chunk in generator:
                if cancel is not None and cancel.is_set():
                    break  # barge in: stop pulling, which closes the model stream below
                _put(chunk)
        except Exception as exc:
            _put(exc)
        finally:
            close = getattr(generator, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:  # noqa: BLE001
                    pass
            _put(None)

    threading.Thread(target=_worker, daemon=True).start()

    while True:
        item = await chunk_queue.get()
        if item is None:
            return
        if isinstance(item, Exception):
            raise item
        yield item


def _for_speech(sentence: str, language: Optional[str], announcer=None) -> str:
    """The wording sent to TTS: amounts and units spelled out the way a person
    says them (tool_agent.voice). The transcript keeps the written sentence."""
    if config.ANSWER_SOURCE != "mcp_hub":
        return sentence
    try:
        from tool_agent import for_speech

        spoken = for_speech(sentence, language or "en")
    except Exception:  # noqa: BLE001  speech wording must never cost the reply
        return sentence
    if announcer is not None and spoken != sentence:
        announcer.written[spoken] = sentence
    return spoken


def _pending_confirmation(access) -> Optional[dict]:
    """The browser safe view of the customer's pending action, or None."""
    if access is None or not getattr(access, "is_verified", False):
        return None
    try:
        import confirmations

        return confirmations.pending_for(access)
    except Exception:  # noqa: BLE001
        return None


class _SentenceAnnouncer:
    """Passed to tts.speak_stream() as on_sentence_start. It runs on the
    TTS playback thread, right before each sentence's audio starts, and
    tells the client: "speaking" status on the first sentence, then the
    sentence itself (see the protocol above). It waits for the send to
    finish before letting playback continue, so the status the client
    shows never lags behind the audio. Also records when audio first
    started, for the time to first audio figure in the per turn log."""

    def __init__(self, websocket: WebSocket, loop: asyncio.AbstractEventLoop):
        self._websocket = websocket
        self._loop = loop
        self.first_audio_at: Optional[float] = None
        # Spoken wording (tool_agent.for_speech) back to the written sentence,
        # so the transcript and captions keep the original text.
        self.written: dict = {}

    def __call__(self, sentence: str):
        sentence = self.written.pop(sentence, sentence)
        messages = []
        if self.first_audio_at is None:
            self.first_audio_at = time.time()
            messages.append({"type": "status", "state": "speaking"})
        messages.append({"type": "speaking_sentence", "text": sentence})

        async def _send_all():
            for message in messages:
                await _send_json(self._websocket, message)

        try:
            asyncio.run_coroutine_threadsafe(_send_all(), self._loop).result(timeout=2)
        except Exception:
            pass  # client gone or slow; the audio itself still plays


async def _speak(
    websocket: WebSocket, text: str, language: Optional[str], voice_style: str = DEFAULT_VOICE_STYLE
) -> float:
    """Plays an already complete text via TTS (the introduction and the
    canned closing lines), then sends "speaking_done". speak_text() blocks
    for the entire playback, so it runs on a thread via run_in_executor:
    calling it directly in this async handler would stall the whole event
    loop, and every other connection with it, for that whole time, the
    same reason _run_one_utterance and _stream_agent_reply run their own
    blocking work on a thread rather than inline. Returns the playback
    time in seconds for the per turn latency log."""
    loop = asyncio.get_running_loop()
    tts_start = time.time()
    try:
        await loop.run_in_executor(
            None, speak_text, text, language or "en", voice_style, _SentenceAnnouncer(websocket, loop)
        )
    except Exception as exc:
        await _send_json(websocket, {"type": "error", "message": f"TTS error: {exc}"})
    await _send_json(websocket, {"type": "speaking_done"})
    return time.time() - tts_start


async def _speak_canned_reply(
    websocket: WebSocket,
    text: str,
    language: Optional[str],
    mode: str,
    voice_style: str = DEFAULT_VOICE_STYLE,
) -> float:
    """Announces a canned line (the introduction, the closing question or
    the goodbye) the same way a real agent reply is announced, skipping
    the agent call and the chunk by chunk streaming entirely, since the
    text is already known up front. "thinking" goes first so the client
    stops sending microphone audio while the line is being synthesized."""
    await _send_json(websocket, {"type": "status", "state": "thinking"})
    await _send_json(websocket, {"type": "agent_response", "text": text, "mode": mode})
    return await _speak(websocket, text, language, voice_style)


async def _speak_streamed_reply(
    websocket: WebSocket,
    text_input: str,
    user_name: Optional[str],
    language: Optional[str],
    voice_style: str = DEFAULT_VOICE_STYLE,
    access=None,
    barge: Optional[dict] = None,
) -> tuple[str, float, Optional[float]]:
    """Generates the agent's reply and speaks it at the same time. Each
    sentence goes to TTS the moment the agent finishes writing it, so the
    first sentence is already playing while the second is still being
    generated. Before this, nothing was spoken until the whole reply was
    finished, so the wait before the assistant started talking grew with
    the length of its answer.

    Returns (reply_text, agent_seconds, first_audio_at), the last being
    the wall clock time audio started, or None if nothing was spoken.

    Barge in: pass a dict with the session's calibrated "vad" as `barge`. While
    the reply is spoken the microphone audio the page keeps sending is checked
    by a BargeInDetector (stricter than the normal end of speech detection, see
    vad.py). When the user really starts talking, playback stops at once and
    queued sentences are dropped, the model stream and the TTS are cancelled,
    {"type": "interrupt"} is sent (the page already handles it) and the audio
    that interrupted is stored in barge["seed"] for the next utterance. If the
    client ended the session meanwhile, barge["ended"] is set instead."""
    loop = asyncio.get_running_loop()
    announcer = _SentenceAnnouncer(websocket, loop)
    sentence_source = _ChunkQueueSource()
    chunker = SentenceChunker()

    cancel = None
    detector = None
    if barge is not None and _barge_in_available(barge.get("vad")):
        cancel = threading.Event()
        detector = BargeInDetector(barge["vad"])
    tts_kwargs = {"cancel": cancel} if cancel is not None else {}
    tts_task = loop.run_in_executor(
        None,
        functools.partial(speak_stream, sentence_source, language or "en", voice_style, announcer, **tts_kwargs),
    )
    monitor = None
    if detector is not None:
        monitor = asyncio.ensure_future(
            _watch_for_barge_in(websocket, detector, announcer, barge.get("voice_holder"))
        )

    agent_start = time.time()
    reply_text = ""

    async def _generate():
        nonlocal reply_text
        try:
            async for chunk in _stream_agent_reply(text_input, user_name, language, access, websocket, cancel):
                reply_text += chunk
                await _send_json(websocket, {"type": "agent_response_chunk", "text": reply_text})
                for sentence in chunker.feed(chunk):
                    sentence_source.put(_for_speech(sentence, language, announcer))
        except Exception as exc:
            await _send_json(websocket, {"type": "error", "message": f"Agent error: {exc}"})
        finally:
            for sentence in chunker.flush():
                sentence_source.put(_for_speech(sentence, language, announcer))
            sentence_source.close()

    async def _barge_in(outcome) -> None:
        """Stops everything for this turn: the model stream, the synthesis and
        the playback (flushed), then tells the page."""
        cancel.set()
        if not generation.done():
            generation.cancel()
            try:
                await generation
            except BaseException:  # noqa: BLE001  cancelled on purpose
                pass
        sentence_source.close()
        try:
            await tts_task  # returns within about 100 ms, cancel is set
        except Exception:  # noqa: BLE001
            pass
        kind, seed = outcome
        if kind == "barge":
            barge["seed"] = seed
            await _send_json(websocket, {"type": "interrupt"})
        else:
            barge["ended"] = True

    generation = asyncio.ensure_future(_generate())
    if monitor is not None:
        await asyncio.wait({generation, monitor}, return_when=asyncio.FIRST_COMPLETED)
        if monitor.done():
            await _barge_in(monitor.result())
            return reply_text, time.time() - agent_start, announcer.first_audio_at
    await generation
    agent_time = time.time() - agent_start

    if reply_text:
        message = {"type": "agent_response", "text": reply_text, "mode": orchestrator.last_mode}
        if getattr(orchestrator, "last_timings", None):
            message["timings"] = dict(orchestrator.last_timings)
        if getattr(orchestrator, "last_expired", False):
            message["expired"] = True
        await _send_json(websocket, message)
        pending = await asyncio.to_thread(_pending_confirmation, access)
        if pending:
            await _send_json(websocket, {"type": "confirmation", "confirmation": pending})

    try:
        if monitor is not None:
            await asyncio.wait({tts_task, monitor}, return_when=asyncio.FIRST_COMPLETED)
            if monitor.done():
                await _barge_in(monitor.result())
                return reply_text, agent_time, announcer.first_audio_at
            monitor.cancel()
            try:
                await monitor
            except BaseException:  # noqa: BLE001
                pass
        await tts_task
    except Exception as exc:
        await _send_json(websocket, {"type": "error", "message": f"TTS error: {exc}"})
    await _send_json(websocket, {"type": "speaking_done"})
    return reply_text, agent_time, announcer.first_audio_at


def _barge_in_available(vad) -> bool:
    """True for the real calibrated detector. Test doubles without the WebRTC
    classifier and the noise floor keep the old behaviour (no barge in)."""
    return vad is not None and hasattr(vad, "_vad") and hasattr(vad, "noise_floor_rms")


async def _watch_for_barge_in(websocket: WebSocket, detector, announcer, voice_holder: Optional[dict]):
    """Reads the microphone chunks the page keeps sending while the reply is
    spoken. Returns ("barge", seed_audio) when the user really started talking,
    or ("ended", None) when the client ended the session or the socket closed.
    Nothing is judged before the first audio has actually started."""
    armed = False
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                return "ended", None
            data = message.get("bytes")
            if data is not None:
                if announcer.first_audio_at is None:
                    continue
                if not armed:
                    detector.reset()
                    armed = True
                seed = detector.feed(np.frombuffer(data, dtype=np.float32))
                if seed is not None:
                    return "barge", seed
            elif message.get("text") is not None:
                try:
                    payload = json.loads(message["text"])
                except ValueError:
                    continue
                if payload.get("type") == "end_session":
                    return "ended", None
                if payload.get("type") == "set_voice" and voice_holder is not None:
                    voice_holder["voice_style"] = _valid_voice_style(payload.get("voice_style"))
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001  a closed socket raises on receive
        return "ended", None


@app.websocket("/ws/voice")
async def voice_ws(websocket: WebSocket):
    await websocket.accept()

    language_hint: Optional[str] = None
    user_name: Optional[str] = None
    greet = True
    handover = False
    session_token: Optional[str] = None
    voice_holder = {"voice_style": DEFAULT_VOICE_STYLE}
    try:
        start_message = await websocket.receive_text()
        payload = json.loads(start_message)
        if payload.get("type") == "start_session":
            language_hint = payload.get("language_hint") or None
            user_name = (payload.get("user_name") or "").strip()[:40] or None
            greet = payload.get("greet", True) is not False
            handover = payload.get("handover") is True
            session_token = payload.get("session_token") or None
            voice_holder["voice_style"] = _valid_voice_style(payload.get("voice_style"))
    except Exception as exc:
        await _send_json(websocket, {"type": "error", "message": f"Invalid start message: {exc}"})
        await websocket.close()
        return

    # A verified existing user (session_token from POST /auth/verify, or
    # the user_session cookie sent with the WebSocket handshake) is always
    # addressed by the name on their account; a client supplied name is
    # only used when there is no verified session, the Guest path, exactly
    # as before. The lookup touches the database, so it runs on a thread.
    verified_user = await asyncio.to_thread(verified_user_from_request, websocket, session_token)
    if verified_user is not None:
        user_name = verified_user.display_name
    # What this caller may be told (guest or OTP verified), decided here from
    # the session token once when the voice session starts (the wizard runs
    # before it, so a verified cookie is already present). /chat reads it on
    # every request.
    access_context = await asyncio.to_thread(access_from_request, websocket, session_token, language_hint)

    # Text to speech is on by default: the assistant introduces itself out
    # loud, in the selected language, by name if one was given, before
    # anything else happens. It is
    # spoken before calibration, not during it, since on this single
    # machine setup the microphone would otherwise hear the introduction
    # and take it as the room's background noise level. The client does
    # not send audio while the status is "speaking", so none leaks in.
    try:
        if handover:
            # The call moved here from another voice tier: a short notice, not a second introduction.
            from voice_router import make_notice

            notice_language = language_hint if language_hint in _GREETINGS else "en"
            await _speak_canned_reply(
                websocket, make_notice(notice_language), notice_language, "Introduction", voice_holder["voice_style"]
            )
        elif greet:
            greeting_text, greeting_language = _greeting(user_name, language_hint)
            await _speak_canned_reply(
                websocket, greeting_text, greeting_language, "Introduction", voice_holder["voice_style"]
            )
    except WebSocketDisconnect:
        return

    vad = VoiceActivityDetector()
    await _send_json(websocket, {"type": "status", "state": "calibrating"})
    # The client sends about 1s of ambient audio (nobody speaking) right
    # after the introduction, before the user starts talking, so the
    # noise floor reflects the actual room rather than a fixed guess.
    # 1s rather than the earlier 1.5s: it is enough samples for a stable
    # average, and it is dead time before the first question.
    calibration_samples = 0
    calibration_chunks = []
    try:
        while calibration_samples < int(1.0 * SAMPLE_RATE):
            message = await websocket.receive_bytes()
            chunk = np.frombuffer(message, dtype=np.float32)
            calibration_chunks.append(chunk)
            calibration_samples += chunk.size
        vad.calibrate_noise_floor(np.concatenate(calibration_chunks))
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await _send_json(websocket, {"type": "error", "message": f"Calibration failed: {exc}"})

    # Set once an utterance sounds like the user is satisfied/thanking the
    # assistant: the very next utterance then decides whether the
    # conversation actually ends (see _sounds_like_closing_signal/
    # _sounds_negative above) or was a false alarm, in which case it is
    # answered normally instead.
    awaiting_close_confirmation = False
    pending_seed = None  # the speech that interrupted the last reply (barge in)
    last_spoken_reply = ""

    try:
        while True:
            turn_start = time.time()
            await _send_json(websocket, {"type": "status", "state": "listening"})

            was_barge_in = pending_seed is not None
            final_text, detected_language, session_ended = await _run_one_utterance(
                websocket, language_hint, vad, voice_holder, seed=pending_seed
            )
            pending_seed = None

            if session_ended:
                return
            if not final_text:
                continue
            if _looks_like_transcription_hallucination(final_text):
                print(f"Turn skipped: transcript '{final_text}' looks like a Whisper hallucination, not a real utterance.")
                continue
            if was_barge_in and last_spoken_reply:
                spoken_words = {w for w in re.findall(r"\w+", last_spoken_reply.lower()) if len(w) > 2}
                heard_words = [w for w in re.findall(r"\w+", final_text.lower()) if len(w) > 2]
                if len(heard_words) >= 3 and spoken_words and sum(1 for w in heard_words if w in spoken_words) / len(heard_words) >= 0.75:
                    print(f"Turn skipped: barge-in transcript '{final_text}' matched speaker echo.")
                    continue
            reply_start = time.time()
            # Replies (spoken and written) are in the language picked on the
            # first screen, not whatever Whisper guessed from the audio, so
            # an English word dropped into a Sinhala question does not flip
            # the reply to English. With no language picked, the reply
            # follows the language Whisper detected, as before.
            reply_language = language_hint or detected_language

            # A verified user's name comes from their account, so "my name
            # is ..." only renames a Guest.
            introduced_name = None if verified_user else _extract_user_name(final_text)
            if introduced_name and introduced_name != user_name:
                user_name = introduced_name
                await _send_json(websocket, {"type": "user_name", "name": user_name})

            if awaiting_close_confirmation:
                if _sounds_negative(final_text) or _sounds_like_closing_signal(final_text):
                    await _speak_canned_reply(
                        websocket,
                        _in_language(_CLOSING_GOODBYE_TEXT, reply_language),
                        reply_language,
                        "closing",
                        voice_holder["voice_style"],
                    )
                    await _send_json(websocket, {"type": "end_chat"})
                    return
                # Not a closing confirmation after all, a real new question;
                # drop the offer and fall through to answer it normally.
                awaiting_close_confirmation = False
            elif _sounds_like_closing_signal(final_text):
                await _speak_canned_reply(
                    websocket,
                    _in_language(_CLOSING_QUESTION_TEXT, reply_language),
                    reply_language,
                    "closing",
                    voice_holder["voice_style"],
                )
                awaiting_close_confirmation = True
                continue

            await _send_json(websocket, {"type": "status", "state": "thinking"})
            barge = {"vad": vad, "voice_holder": voice_holder}
            reply_text, agent_time, first_audio_at = await _speak_streamed_reply(
                websocket, final_text, user_name, reply_language, voice_holder["voice_style"], access_context, barge
            )
            if reply_text:
                last_spoken_reply = reply_text
            if barge.get("ended"):
                return
            if barge.get("seed") is not None:
                pending_seed = barge["seed"]
                print("Barge in: the reply was cut off, listening with the interrupting speech.")
                continue
            if not reply_text:
                continue

            first_audio = f"{first_audio_at - reply_start:.2f}s" if first_audio_at else "none"
            print(
                f"Turn latency: first audio {first_audio} after the final transcript, agent {agent_time:.2f}s, "
                f"total {time.time() - turn_start:.2f}s, mode={orchestrator.last_mode}"
            )
    except WebSocketDisconnect:
        return


class ChatRequest(BaseModel):
    text: str
    language: Optional[str] = None
    user_name: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    mode: str
    # Additive fields: the customer's pending action for the confirmation modal
    # (no tokens of the hub, only the one time id), whether their hub login
    # ended, and the tool and model time of this turn in milliseconds.
    # They are left out of the JSON unless they have a value, so the earlier
    # {reply, mode} shape is unchanged for every ordinary turn.
    confirmation: Optional[dict] = None
    expired: Optional[bool] = None
    timings: Optional[dict] = None


@app.post("/chat", response_model=ChatResponse, response_model_exclude_none=True)
async def chat(request: ChatRequest, http_request: Request):
    """Text only alternative to the voice pipeline, for the frontend's
    Start Chat screen. Runs the same Orchestrator the voice flow uses (so
    the same cloud multi agent versus local fallback behavior applies),
    just without any STT or TTS involved. orchestrator.respond() is a
    blocking call (HTTP requests to Ollama or a cloud provider), so it
    runs on a thread via run_in_executor rather than directly in this
    async handler, for the same reason server.py's WebSocket transcription
    does: a blocking call here would stall every other connection's event
    loop turn until it finished.

    A verified user (Authorization: Bearer token or the user_session
    cookie) is addressed by the name on their account, not by the
    user_name in the body; without a verified session the body's name is
    used as before."""
    loop = asyncio.get_running_loop()
    verified_user = await asyncio.to_thread(verified_user_from_request, http_request)
    user_name = verified_user.display_name if verified_user else request.user_name
    access_context = await asyncio.to_thread(access_from_request, http_request, None, request.language)
    reply = await loop.run_in_executor(
        None,
        functools.partial(orchestrator.respond, request.text, user_name, request.language, access=access_context),
    )
    pending = await asyncio.to_thread(_pending_confirmation, access_context)
    return ChatResponse(
        reply=reply,
        mode=orchestrator.last_mode,
        confirmation=pending,
        expired=True if getattr(orchestrator, "last_expired", False) else None,
        timings=dict(getattr(orchestrator, "last_timings", None) or {}) or None,
    )


def _answers_status() -> dict:
    status = {"source": config.ANSWER_SOURCE, "state_changing_tools": config.STATE_CHANGING_TOOLS_ENABLED}
    try:
        import mcp_gateway
        import tool_agent

        status["gateway"] = mcp_gateway.status()
        status["warm_connections"] = mcp_gateway.warm_status()
        status["tool_agent"] = tool_agent.stats()
        status["tool_latency_classes"] = _latency_classes()
    except Exception as exc:  # noqa: BLE001  /health must always answer
        status["error"] = type(exc).__name__
    return status


_latency_cache: dict = {"at": 0.0, "value": None}


def _latency_classes() -> Optional[dict]:
    """The hub's get_function_latencies answer (a static classification, not a
    measurement), cached for 5 minutes so /health stays cheap."""
    if _latency_cache["at"] and time.monotonic() - _latency_cache["at"] < 300:
        return _latency_cache["value"]
    import mcp_gateway

    value = mcp_gateway.latencies() or None
    _latency_cache.update(at=time.monotonic(), value=value)  # a failure is cached too
    return value


def _chain_error(provider: str):
    """The last error of the provider's tier in the fallback chain (the existing reason mapping), so /health
    shows why Gemini is not answering even when the chain, not the single provider path, made the call."""
    try:
        tier = (orchestrator.chain.status() if config.LLM_CHAIN else {}).get("tiers", {}).get(provider)
        return tier.get("last_error") if tier else None
    except Exception:  # noqa: BLE001  /health must always answer
        return None


def _voice_router_status():
    if not config.VOICE_ROUTER_ENABLED:
        return None
    from voice_router import health_status

    return health_status()


@app.get("/health")
async def health():
    """Plain status endpoint for a demo or a smoke test, reporting which
    provider paths are actually configured right now rather than what the
    code merely supports, so a quick check tells the real story."""
    return {
        "status": "ok",
        "cloud_llm_configured": config.cloud_llm_configured(),
        "cloud_llm_provider": config.CLOUD_LLM_PROVIDER,
        "cloud_llm_model": {
            "livekit": config.LIVEKIT_LLM_MODEL,
            "gemini": config.GEMINI_MODEL,
        }.get(config.CLOUD_LLM_PROVIDER, config.OPENAI_MODEL),
        # Why the cloud LLM is currently not answering (None if it is, or
        # if it has not been tried yet), so a quick check tells the story.
        "cloud_llm_last_error": orchestrator.last_cloud_error or _chain_error(config.CLOUD_LLM_PROVIDER),
        # Per tier status of the fallback chain (Gemini, LiveKit Inference,
        # local Llama): order, the mode of the last answer, and each tier's
        # state and last error reason. None when LLM_CHAIN is empty.
        "llm_chain": orchestrator.chain.status() if config.LLM_CHAIN else None,
        "access_control": {"enabled": config.ACCESS_CONTROL_ENABLED},
        "verification": {"enabled": config.VERIFY_ENABLED, "otp_backend": config.VERIFY_OTP_BACKEND},
        # Account answers for verified customers (read only hub tools). Customer
        # data goes to the local model only unless data_to_cloud_llm is true.
        "customer_service": {
            "enabled": config.CUSTOMER_TOOLS_ENABLED,
            "data_to_cloud_llm": config.CUSTOMER_DATA_TO_CLOUD_LLM,
            "data_source": config.CUSTOMER_DATA_SOURCE,
        },
        # The hub as the only source of SLT answers: the gateway (open customer
        # connections, last error, access matrix loaded), the tool agent's
        # per turn tool time, the static latency classes the hub publishes, and
        # whether buying or fault logging is switched on.
        "answers": await asyncio.to_thread(_answers_status),
        "cloud_speech_configured": config.cloud_speech_configured(),
        "cloud_speech_provider": config.CLOUD_SPEECH_PROVIDER,
        # Which voice engine spoke last and why a better one was skipped.
        "speech_engine": speech_status(),
        "supported_languages": config.SUPPORTED_LANGUAGES,
        # The single app's other parts: database migration state, the
        # knowledge index (loaded, size, build date, last retrieval time),
        # which external chat agent is answering (None means the built in
        # assistant), and the MCP server.
        "database": await asyncio.to_thread(database_status),
        "knowledge": (
            {"enabled": False, "reason": "ANSWER_SOURCE=mcp_hub, the MCP hub is the only source"}
            if _hub_only()
            else retriever.status()
        ),
        "external_agent": admin_service.active_agent_summary(),
        "mcp": mcp_handle.status(),
        "mcp_hub": mcp_hub_status(),
        "livekit": {
            "transport": config.VOICE_TRANSPORT,
            "configured": config.livekit_configured(),
            "available": livekit_router is not None,
            "worker": livekit_state["worker"].status() if livekit_state.get("worker") else None,
        },
        "voice_router": _voice_router_status(),
        "avatar_renderer": config.AVATAR_RENDERER,
        "voicekit": {"enabled": config.VOICEKIT_ENABLED, "api_keys_configured": bool(config.VOICEKIT_API_KEYS)},
    }


# Voice tiers: Gemini Live first, then LiveKit, then the local pipeline (docs/GEMINI_LIVE_PLAN.md).
def _voice_name(language: str, voice_style: str) -> str:
    from voicekit import voice_config

    names = voice_config.lookup("gemini_live", language, voice_style)
    return names[0] if names else "Kore"


def _livekit_usable() -> Optional[bool]:
    """None when LiveKit is not in use (another transport is chosen or it has no keys), else
    whether its agent worker is running right now."""
    if config.VOICE_TRANSPORT != "livekit" or not config.livekit_configured():
        return None
    worker = livekit_state.get("worker")
    try:
        return bool(worker and worker.status().get("state") == "running")
    except Exception:  # noqa: BLE001
        return False


@app.get("/api/voice/tier")
def voice_tier(http_request: Request):
    """Which voice tier a new session should start on. No secrets: a tier name and states."""
    if not config.VOICE_ROUTER_ENABLED:
        return {"enabled": False, "tier": "livekit" if config.livekit_configured() else "local"}
    from voice_router import get_router

    router = get_router()
    tier = router.pick().value
    skip_live = not config.GEMINI_LIVE_ENABLED
    if not skip_live and tier == "gemini_live" and not config.CUSTOMER_DATA_TO_CLOUD_LLM:
        # A signed in customer (cookie) is not sent to Gemini; /ws/live checks the token again.
        try:
            skip_live = access_from_request(http_request, None, None).is_verified
        except Exception:  # noqa: BLE001
            skip_live = False
    if tier == "gemini_live" and skip_live:
        tier = "livekit" if config.livekit_configured() else "local"
    if tier == "livekit":
        usable = _livekit_usable()
        if usable is None:
            tier = "local"  # LiveKit is not the chosen transport here: not a failure
        elif not usable:
            # Configured and meant to run, but the agent worker is not running: a genuine failure.
            from voice_router import Tier

            router.report(Tier.LIVEKIT, "outage")
            tier = router.pick().value
    return {"enabled": True, "tier": tier, "model": config.GEMINI_LIVE_MODEL, "status": router.status()}


@app.websocket("/ws/live")
async def live_ws(websocket: WebSocket):
    """Gemini Live through this server: the key and every tool call stay here."""
    await websocket.accept()
    try:
        start = json.loads(await websocket.receive_text())
    except Exception:
        await websocket.close()
        return
    language = start.get("language_hint") or "en"
    voice_style = _valid_voice_style(start.get("voice_style"))
    access = await asyncio.to_thread(access_from_request, websocket, start.get("session_token") or None, language)
    from voice_router import Tier, get_router

    if access.is_verified and not config.CUSTOMER_DATA_TO_CLOUD_LLM:
        # Customer account data stays on this machine's model (the owner's decision, the same rule
        # /chat follows). Gemini Live would be sent the hub's account answers, so a verified customer
        # is served by the next tier. Not a failure: the router is not told.
        await _send_json(websocket, {"type": "live_failed", "reason": "customer_data_local_only"})
        await websocket.close()
        return

    if not (config.GEMINI_LIVE_ENABLED and config.GEMINI_API_KEY):
        get_router().report(Tier.GEMINI_LIVE, "missing_key")
        await _send_json(websocket, {"type": "live_failed", "reason": "missing_key"})
        await websocket.close()
        return
    from gemini_live import ws as gemini_live_ws

    user_name = (start.get("user_name") or "").strip()[:40] or None
    verified = await asyncio.to_thread(verified_user_from_request, websocket, start.get("session_token") or None)
    if verified is not None:
        user_name = verified.display_name
    result = await gemini_live_ws.handle(
        websocket, access, language, voice_style,
        api_key=config.GEMINI_API_KEY,
        model=config.GEMINI_LIVE_MODEL,
        voice_name=_voice_name(language, voice_style),
        setup_timeout_seconds=config.GEMINI_LIVE_SETUP_TIMEOUT_SECONDS,
        user_name=user_name,
        greet=start.get("greet", True) is not False,
        state_changing=config.STATE_CHANGING_TOOLS_ENABLED,
        context=str(start.get("context") or ""),
    )
    from voice_router import Outcome

    # A session that held a conversation proves the tier; a hang up or a safety block
    # changes nothing; only a genuine failure trips the breaker.
    get_router().report(Tier.GEMINI_LIVE, Outcome.success() if result.ok and result.turns else result.reason_class)
    try:
        await websocket.close()
    except Exception:
        pass


# Static files last, so every route above wins. The admin portal first,
# since "/" would otherwise also match /admin/...; html=True serves
# index.html for a directory and redirects /admin to /admin/.
class RevalidatingStaticFiles(StaticFiles):
    """Static files that browsers must revalidate on every load. They already
    carry an ETag, so an unchanged file costs a 304, but a changed app.js or
    index.html is picked up at once. Without this a browser can keep running
    an old app.js for a long time, which here meant the old PIN form instead of
    the verification wizard."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/admin", RevalidatingStaticFiles(directory=FRONTEND_DIR / "admin", html=True), name="admin")
app.mount("/", RevalidatingStaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    from app_port import APP_PORT, ensure_free

    ensure_free("0.0.0.0")
    uvicorn.run(app, host="0.0.0.0", port=APP_PORT)
