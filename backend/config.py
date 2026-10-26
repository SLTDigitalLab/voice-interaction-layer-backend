"""Central configuration for the voice interaction backend.

All environment dependent settings (API keys, provider choices, model
sizes) are read here once, so every other module imports from config
instead of touching os.environ directly. That keeps credential handling in
one place and makes it obvious which features are active versus falling
back, without digging through every module during a demo.

Values are read from a .env file (see .env.example) via python-dotenv, or
from real environment variables if set. Nothing here requires an API key
to import; the *_configured() helpers just report whether one is present so
callers can decide whether to attempt a cloud path or go straight to the
local fallback.
"""

import base64
import json
import os
import secrets
from pathlib import Path

from dotenv import load_dotenv

if (
    "PYTEST_CURRENT_TEST" not in os.environ
    and os.environ.get("JWT_SECRET") != "test-jwt-secret-not-for-production-0123456789abcdef"
):
    load_dotenv()

# Cloud LLM (multi agent orchestration when reachable)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
# "livekit" (LiveKit Inference, called with the LiveKit key and secret below),
# "gemini" or "gpt". The cloud provider answers first; when it is not set up or
# fails, the local Ollama model answers alone.
CLOUD_LLM_PROVIDER = os.environ.get("CLOUD_LLM_PROVIDER", "livekit").strip().lower()
# A "-latest" alias follows Google's current stable model; pinned names are retired (gemini-2.5-flash answers 404
# "no longer available to new users" on 2026-10-16). Checked against models.list for the AI Studio key.
_raw_gemini_model = os.environ.get("GEMINI_MODEL", "gemini-flash-latest").strip()
if _raw_gemini_model.startswith("gemini/"):
    _raw_gemini_model = _raw_gemini_model.split("/", 1)[1]
GEMINI_MODEL = _raw_gemini_model or "gemini-flash-latest"
# Gemini Live (native speech to speech, tier 1 of the voice router). This is a
# different model family from GEMINI_MODEL above: a Live model such as
# gemini-3.8-live cannot be used for text generation and the other way round.
GEMINI_LIVE_ENABLED = os.environ.get("GEMINI_LIVE_ENABLED", "true").strip().lower() in ("1", "true", "yes")
GEMINI_LIVE_MODEL = os.environ.get("GEMINI_LIVE_MODEL", "gemini-3.8-live").strip()
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
# Extra "reasoning" model call before the reply is composed (see
# agent.py's Orchestrator). Off by default: it has to finish before the
# reply can even start, so it roughly doubles the wait for the first word.
CLOUD_REASONING_STAGE = os.environ.get("CLOUD_REASONING_STAGE", "false").strip().lower() in ("1", "true", "yes")

# Local LLM fallback (always available, no key required)
LOCAL_LLM_MODEL = os.environ.get("LOCAL_LLM_MODEL", "llama3.2:latest")
# 127.0.0.1 rather than localhost: on Windows, "localhost" is tried as
# IPv6 (::1) first, Ollama only listens on IPv4, and every request then
# waited about 2 seconds for that attempt to time out before falling back
# (measured: 2.04s per request with localhost, 0.03s with 127.0.0.1).
LOCAL_LLM_HOST = os.environ.get("LOCAL_LLM_HOST", "http://127.0.0.1:11434")

# Cloud speech (STT and TTS)
CLOUD_SPEECH_PROVIDER = os.environ.get("CLOUD_SPEECH_PROVIDER", "google").strip().lower()  # "google" or "azure"
GOOGLE_APPLICATION_CREDENTIALS = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
AZURE_SPEECH_KEY = os.environ.get("AZURE_SPEECH_KEY", "").strip()
AZURE_SPEECH_REGION = os.environ.get("AZURE_SPEECH_REGION", "").strip()

# Local STT fallback
LOCAL_WHISPER_MODEL_SIZE = os.environ.get("LOCAL_WHISPER_MODEL_SIZE", "base")
# Words Whisper should expect (faster-whisper hotwords). Empty turns it off.
# The other local transcription settings (STT_DENOISE, STT_BEAM_SIZE,
# STT_ACCURATE_MODEL, STT_CONFIDENCE_FLOOR, STT_VAD_FILTER) are defined
# further down, after
# the helper functions they use.
STT_HOTWORDS = os.environ.get("STT_HOTWORDS", "SLT Mobitel").strip()

SUPPORTED_LANGUAGES = {"en": "English", "si": "Sinhala", "ta": "Tamil"}

# Repository root and the local, gitignored data directory (database,
# knowledge base cache and index, generated secrets). Paths below default
# to it so the app runs from a fresh clone with no extra setup.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", "").strip() or PROJECT_ROOT / "data")


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    return value in ("1", "true", "yes")


# On /ws/voice, TTS plays through Python sounddevice directly on the host's
# speakers rather than through the browser's audio engine, so the browser's
# WebRTC Acoustic Echo Cancellation has no reference signal. Keep speaker
# barge-in off by default on /ws/voice unless headphones are used
# (VOICE_WS_BARGE_IN=true).
VOICE_WS_BARGE_IN = _env_bool("VOICE_WS_BARGE_IN", False)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


GEMINI_LIVE_SETUP_TIMEOUT_SECONDS = float(os.environ.get("GEMINI_LIVE_SETUP_TIMEOUT_SECONDS", "").strip() or 8)
# Voice tier fallback: Gemini Live, then LiveKit, then the local pipeline. A tier
# is skipped only after a genuine failure (see voice_router/reasons.py).
VOICE_ROUTER_ENABLED = os.environ.get("VOICE_ROUTER_ENABLED", "true").strip().lower() in ("1", "true", "yes")
VOICE_TIER_ORDER = os.environ.get("VOICE_TIER_ORDER", "gemini_live,livekit,local").strip()
# Tier 2 speech from LiveKit Inference TTS (needs the LiveKit credentials; off until someone has
# listened to it, see docs/VOICE_EVAL.md). Speech to text on tier 2 stays the local Whisper model.
VOICE_TIER2_INFERENCE = os.environ.get("VOICE_TIER2_INFERENCE", "false").strip().lower() in ("1", "true", "yes")
VOICE_ROUTER_BREAKER_SECONDS = float(os.environ.get("VOICE_ROUTER_BREAKER_SECONDS", "").strip() or 60)


def _local_secret(name: str, factory) -> str:
    """Returns a secret from the environment, or, when it is not set,
    one generated once and kept in DATA_DIR/secrets.json (gitignored), so
    tokens and encrypted fields survive a restart on a development machine
    without anyone having to invent a key first. A real deployment sets
    the environment variable instead; see .env.example."""
    value = os.environ.get(name, "").strip()
    if value:
        return value
    secrets_file = DATA_DIR / "secrets.json"
    stored = {}
    if secrets_file.exists():
        try:
            stored = json.loads(secrets_file.read_text(encoding="utf-8"))
        except ValueError:
            stored = {}
    if not stored.get(name):
        stored[name] = factory()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        secrets_file.write_text(json.dumps(stored, indent=2), encoding="utf-8")
    return stored[name]


def _new_fernet_key() -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii")


# Server startup: load Whisper, Ollama, TTS and the knowledge index before
# the first request. Tests turn it off so they start in milliseconds.
WARM_UP_ON_STARTUP = _env_bool("WARM_UP_ON_STARTUP", True)
# The address this server is reached at, used in token issuer/audience
# values and in the MCP authorization metadata.
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "http://127.0.0.1:8000").strip().rstrip("/")

# Database: SQLite file by default; any SQLAlchemy URL (for example
# postgresql+psycopg://user:pass@host/db) switches it with no code change.
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip() or f"sqlite:///{(DATA_DIR / 'app.db').as_posix()}"

# Signing key for every token this server issues (user sessions, admin
# sessions, MCP access tokens), and the key that encrypts stored external
# agent credentials.
JWT_SECRET = _local_secret("JWT_SECRET", lambda: secrets.token_urlsafe(48))
FIELD_ENCRYPTION_KEY = _local_secret("FIELD_ENCRYPTION_KEY", _new_fernet_key)
# Session cookies are sent over plain http on the local demo; set true
# behind https.
COOKIE_SECURE = _env_bool("COOKIE_SECURE", False)
USER_SESSION_MINUTES = _env_int("USER_SESSION_MINUTES", 60)
ADMIN_SESSION_MINUTES = _env_int("ADMIN_SESSION_MINUTES", 60)
MCP_TOKEN_MINUTES = _env_int("MCP_TOKEN_MINUTES", 15)

# Existing user verification (phone number + PIN)
LOGIN_MAX_ATTEMPTS = _env_int("LOGIN_MAX_ATTEMPTS", 5)
LOGIN_LOCKOUT_MINUTES = _env_int("LOGIN_LOCKOUT_MINUTES", 15)
# Per client address, across all accounts, to slow down guessing.
LOGIN_RATE_LIMIT_PER_MINUTE = _env_int("LOGIN_RATE_LIMIT_PER_MINUTE", 10)

# First admin account, created at startup only when no admin exists yet.
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin").strip()
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "").strip()

# External chat agents registered in the admin portal
# Private, loopback and link local addresses are refused (SSRF protection)
# unless this development flag is on.
EXTERNAL_AGENT_ALLOW_PRIVATE = _env_bool("EXTERNAL_AGENT_ALLOW_PRIVATE", False)
EXTERNAL_AGENT_MAX_RESPONSE_BYTES = _env_int("EXTERNAL_AGENT_MAX_RESPONSE_BYTES", 1_000_000)
EXTERNAL_AGENT_DEFAULT_TIMEOUT = _env_float("EXTERNAL_AGENT_DEFAULT_TIMEOUT", 15.0)

# Knowledge base (retrieval augmented generation)
KNOWLEDGE_ENABLED = _env_bool("KNOWLEDGE_ENABLED", True)
KNOWLEDGE_DIR = Path(os.environ.get("KNOWLEDGE_DIR", "").strip() or DATA_DIR / "knowledge")
# Multilingual embedding model served by the same local Ollama as the
# fallback LLM, so Sinhala and Tamil questions find English content.
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "bge-m3").strip()
EMBEDDING_HOST = os.environ.get("EMBEDDING_HOST", "").strip() or LOCAL_LLM_HOST
RETRIEVAL_TOP_K = _env_int("RETRIEVAL_TOP_K", 4)
# Chunks scoring below this cosine similarity are not used as context.
# 0.53 measured (docs/RAG_EVAL.md): small talk tops out at 0.524 with
# bge-m3, real SLT Mobitel questions mostly score 0.55 or more.
RETRIEVAL_MIN_SCORE = _env_float("RETRIEVAL_MIN_SCORE", 0.53)
SCRAPER_USER_AGENT = os.environ.get(
    "SCRAPER_USER_AGENT",
    "SLTDigitalLab-KnowledgeBot/1.0 (voice assistant prototype; contact: SLT Digital Lab)",
).strip()
SCRAPER_DELAY_SECONDS = max(1.0, _env_float("SCRAPER_DELAY_SECONDS", 1.5))
SCRAPER_MAX_PAGES = _env_int("SCRAPER_MAX_PAGES", 400)

# Local speech to text accuracy (see stt.py). Measured on 144 spoken test
# clips (three voices, one with background noise): the noise reduction
# filter that used to run before every transcription made Whisper worse,
# 11.0% word errors against 3.3% without it, so it is off by default.
STT_DENOISE = _env_bool("STT_DENOISE", False)
# Beam search width: how many candidate transcripts Whisper compares.
STT_BEAM_SIZE = _env_int("STT_BEAM_SIZE", 5)
# Whisper's own speech filter, run before decoding. With the brand hotword
# set, Whisper writes words such as "SLT Mobitel" for audio that contains no
# speech (30 of 30 noise clips with the larger model, 7 of 30 with the base
# one); with this filter on, 0 of 30, at no cost in accuracy or speed.
STT_VAD_FILTER = _env_bool("STT_VAD_FILTER", True)
# A larger model for the final transcript of a turn when the main one is
# not confident (empty turns it off). The final transcript decides what is
# asked, so a second look at the uncertain ones is worth its extra second.
STT_ACCURATE_MODEL = os.environ.get("STT_ACCURATE_MODEL", "small").strip()
# Seconds of silence that end an utterance. It used to be a whole second plus up to half a
# second of chunk rounding; with 0.25 s chunks 0.75 s ends the turn about 0.5 s sooner.
# Sinhala speech to text. The stock Whisper models cannot read Sinhala (measured on test speech: a
# match of 0.16 for base, 0.13 for small, romanized or garbage output). A Whisper small model fine tuned
# for Sinhala (CTranslate2 format) matched 0.52 and writes real Sinhala script. It is kept outside git in
# STT_SINHALA_MODEL_DIR and fetched once at start up from STT_SINHALA_MODEL_REPO (about 490 MB) when
# STT_SINHALA_AUTO_DOWNLOAD is true; until it is there Sinhala behaves as before.
STT_SINHALA_MODEL_DIR = Path(os.environ.get("STT_SINHALA_MODEL_DIR", "").strip() or DATA_DIR / "models" / "whisper-sinhala")
STT_SINHALA_MODEL_REPO = os.environ.get("STT_SINHALA_MODEL_REPO", "janiduchamika/faster-whisper-small-sinhala-ct2-float16").strip()
STT_SINHALA_AUTO_DOWNLOAD = _env_bool("STT_SINHALA_AUTO_DOWNLOAD", True)
STT_SILENCE_SECONDS = _env_float("STT_SILENCE_SECONDS", 0.75)
# Languages whose FINAL transcript is made by STT_ACCURATE_MODEL directly: measured on synthesized
# Tamil speech the small model matched the sentence 0.75 against 0.43 for base in the same time.
# Sinhala is left out on purpose: neither Whisper model could read it (0.16 and 0.13).
STT_ACCURATE_LANGUAGES = tuple(x.strip() for x in os.environ.get("STT_ACCURATE_LANGUAGES", "ta").split(",") if x.strip())
# Average log probability of a transcript below which the larger model is
# asked to check it (closer to 0 means more confident).
STT_CONFIDENCE_FLOOR = _env_float("STT_CONFIDENCE_FLOOR", -0.40)

# MCP server
MCP_ENABLED = _env_bool("MCP_ENABLED", True)


# Voice transport: "livekit" is real time browser audio in and out through a
# LiveKit room, "websocket" is the original /ws/voice pipeline with server
# side playback. Both stay available; this only picks what the frontend uses.
# livekit without LIVEKIT_URL, key and secret behaves as websocket (the page
# asks GET /api/livekit/config and falls back), so a missing server never
# leaves the page without a voice.
VOICE_TRANSPORT = os.environ.get("VOICE_TRANSPORT", "livekit").strip().lower()
LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "").strip()
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "").strip()
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "").strip()
LIVEKIT_TOKEN_TTL_SECONDS = _env_int("LIVEKIT_TOKEN_TTL_SECONDS", 600)
LIVEKIT_AGENT_NAME = os.environ.get("LIVEKIT_AGENT_NAME", "voice-assistant").strip()
# LiveKit Inference: the language model that writes the replies when
# CLOUD_LLM_PROVIDER=livekit. Measured through the gateway from the development
# machine, first token took 1.3 to 2.2 s (gpt-4o-mini fastest of four tried),
# against 0.3 to 0.9 s for the local model; see ARCHITECTURE.md.
LIVEKIT_LLM_MODEL = os.environ.get("LIVEKIT_LLM_MODEL", "openai/gpt-4o-mini").strip()
LIVEKIT_INFERENCE_URL = os.environ.get("LIVEKIT_INFERENCE_URL", "https://agent-gateway.livekit.cloud/v1").strip().rstrip("/")
LIVEKIT_LLM_TIMEOUT_SECONDS = _env_float("LIVEKIT_LLM_TIMEOUT_SECONDS", 12.0)
# Same cap the local model uses (see agent.py), so a long answer is not read out.
LIVEKIT_LLM_MAX_TOKENS = _env_int("LIVEKIT_LLM_MAX_TOKENS", 220)

# Which avatar the page draws: "talkinghead" is the lip synced avatar in
# frontend/avatar (its model files come from frontend/avatar/fetch_assets.py;
# without them, or without WebGL, the page falls back), "legacy" is bot3d.js
# (Three.js r128) and the flat 2D face.
AVATAR_RENDERER = os.environ.get("AVATAR_RENDERER", "talkinghead").strip().lower()
# Which human models the 3D renderer shows: "south_asian" (medium brown skin, black SLT T shirt;
# frontend/avatar/assets/male_sl.glb and female_sl.glb, rebuilt with tools/avatar/build_avatar.py)
# or "original" (male.glb and mpfb.glb). A missing south_asian file falls back to the original.
AVATAR_VARIANT = os.environ.get("AVATAR_VARIANT", "original").strip().lower()

# voicekit: the reusable text to speech package and its HTTP/WebSocket API.
VOICEKIT_ENABLED = _env_bool("VOICEKIT_ENABLED", True)
# Comma separated API keys accepted by /api/tts/stream and /ws/tts. Empty
# means the API refuses every request (it is never open by default).
VOICEKIT_API_KEYS = [k.strip() for k in os.environ.get("VOICEKIT_API_KEYS", "").split(",") if k.strip()]
VOICEKIT_MAX_TEXT_CHARS = _env_int("VOICEKIT_MAX_TEXT_CHARS", 2000)
VOICEKIT_RATE_LIMIT_PER_MINUTE = _env_int("VOICEKIT_RATE_LIMIT_PER_MINUTE", 30)
# Folder holding local open voice models (Piper .onnx files), if any.
VOICEKIT_LOCAL_VOICE_DIR = Path(
    os.environ.get("VOICEKIT_LOCAL_VOICE_DIR", "").strip() or DATA_DIR / "voices"
)

# MCP hub (this app as an MCP client). The key only ever comes from the
# environment; nothing here is logged.
MCP_HUB_URL = os.environ.get("MCP_HUB_URL", "https://mcphub.sltdigitallab.lk").strip().rstrip("/")
MCP_HUB_AUTH_HEADER = os.environ.get("MCP_HUB_AUTH_HEADER", "X-API-Key").strip()
MCP_HUB_API_KEY = os.environ.get("MCP_HUB_API_KEY", "").strip()
MCP_HUB_ENABLED = _env_bool("MCP_HUB_ENABLED", True)
MCP_HUB_TIMEOUT_SECONDS = _env_float("MCP_HUB_TIMEOUT_SECONDS", 10.0)
MCP_HUB_RETRIES = _env_int("MCP_HUB_RETRIES", 2)
# The longest a spoken reply waits for the hub's notes. Measured against the
# real hub, a knowledge search takes 3 to 7 seconds, far over the first word
# budget, so the reply path gives up after this and answers from retrieval.
MCP_HUB_CONTEXT_BUDGET_SECONDS = _env_float("MCP_HUB_CONTEXT_BUDGET_SECONDS", 1.0)
# Whether the hub's search tool is also asked on every spoken reply. Off by
# default: it takes 3 to 7 seconds (measured 2026-10-05), which would delay the
# first word, and it overlaps the local knowledge base. The hub is still used
# for one time codes either way.
MCP_HUB_IN_REPLIES = _env_bool("MCP_HUB_IN_REPLIES", False)
# After a failed hub call, skip the hub for this long (same idea as the
# 5 minute cloud LLM breaker in agent.py).
# When the hub is the only source of answers one slow reply must not silence it for
# five minutes, so the pause after a failure is short in that mode.
MCP_HUB_BREAKER_SECONDS = _env_int(
    "MCP_HUB_BREAKER_SECONDS", 20 if os.environ.get("ANSWER_SOURCE", "mcp_hub").strip().lower() == "mcp_hub" else 300
)
# Comma separated names of hub tools offered to the Orchestrator. Empty
# means every tool that takes one required text argument.
MCP_HUB_TOOLS = [t.strip() for t in os.environ.get("MCP_HUB_TOOLS", "").split(",") if t.strip()]
# Hub tool that finds a customer by landline number (empty: use the local DB).
MCP_HUB_CUSTOMER_LOOKUP_TOOL = os.environ.get("MCP_HUB_CUSTOMER_LOOKUP_TOOL", "").strip()

# Existing user verification methods, comma separated: "pin" (landline plus
# PIN, as before) and/or "otp" (landline plus a one time code).
AUTH_METHODS = [m.strip().lower() for m in os.environ.get("AUTH_METHODS", "pin").split(",") if m.strip()]
OTP_LENGTH = _env_int("OTP_LENGTH", 6)
OTP_TTL_SECONDS = _env_int("OTP_TTL_SECONDS", 300)
OTP_MAX_ATTEMPTS = _env_int("OTP_MAX_ATTEMPTS", 3)
OTP_RESEND_COOLDOWN_SECONDS = _env_int("OTP_RESEND_COOLDOWN_SECONDS", 60)
OTP_RATE_LIMIT_PER_NUMBER_PER_HOUR = _env_int("OTP_RATE_LIMIT_PER_NUMBER_PER_HOUR", 5)
OTP_RATE_LIMIT_PER_IP_PER_HOUR = _env_int("OTP_RATE_LIMIT_PER_IP_PER_HOUR", 20)
# "console" (development only, prints the code), "email" or "sms".
OTP_SENDER = os.environ.get("OTP_SENDER", "console").strip().lower()
# Where codes come from: "hub" (the MCP hub's own customer lookup and OTP
# tools), "local" (generated and checked here, delivered by OTP_SENDER), or
# "auto" (the hub while it is reachable, local otherwise).
OTP_BACKEND = os.environ.get("OTP_BACKEND", "auto").strip().lower()
SMTP_HOST = os.environ.get("SMTP_HOST", "").strip()
SMTP_PORT = _env_int("SMTP_PORT", 587)
SMTP_USER = os.environ.get("SMTP_USER", "").strip()
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "").strip()
SMTP_FROM = os.environ.get("SMTP_FROM", "").strip()
SMS_GATEWAY_URL = os.environ.get("SMS_GATEWAY_URL", "").strip()
SMS_GATEWAY_API_KEY = os.environ.get("SMS_GATEWAY_API_KEY", "").strip()
SMS_GATEWAY_AUTH_HEADER = os.environ.get("SMS_GATEWAY_AUTH_HEADER", "Authorization").strip()
SMS_SENDER_ID = os.environ.get("SMS_SENDER_ID", "SLTMobitel").strip()


# --- LLM fallback chain, access control, verification wizard ----------------
# Order of the reply providers per turn. Names: gemini, livekit, local. An
# empty value keeps the older single provider behaviour (CLOUD_LLM_PROVIDER).
LLM_CHAIN = [t.strip().lower() for t in os.environ.get("LLM_CHAIN", "gemini,livekit,local").split(",") if t.strip()]
LLM_BREAKER_SECONDS = _env_int("LLM_BREAKER_SECONDS", 300)

# Guests may not receive customer service answers (see backend/access).
ACCESS_CONTROL_ENABLED = _env_bool("ACCESS_CONTROL_ENABLED", True)
ACCESS_POLICY_PATH = os.environ.get("ACCESS_POLICY_PATH", "").strip()  # empty means backend/access/policy.json

# Existing user verification wizard (backend/verification).
VERIFY_ENABLED = _env_bool("VERIFY_ENABLED", True)
VERIFY_OTP_TTL_SECONDS = _env_int("VERIFY_OTP_TTL_SECONDS", 120)
VERIFY_OTP_RESEND_COOLDOWN_SECONDS = _env_int("VERIFY_OTP_RESEND_COOLDOWN_SECONDS", 30)
VERIFY_OTP_MAX_ATTEMPTS = _env_int("VERIFY_OTP_MAX_ATTEMPTS", 3)
VERIFY_RATE_LIMIT_PER_IP_PER_HOUR = _env_int("VERIFY_RATE_LIMIT_PER_IP_PER_HOUR", 20)
VERIFY_RATE_LIMIT_PER_PHONE_PER_HOUR = _env_int("VERIFY_RATE_LIMIT_PER_PHONE_PER_HOUR", 6)
VERIFY_TICKET_MINUTES = _env_int("VERIFY_TICKET_MINUTES", 10)
NAME_MATCH_REQUIRED = _env_bool("NAME_MATCH_REQUIRED", True)
VERIFY_OTP_SENDER = os.environ.get("VERIFY_OTP_SENDER", "").strip().lower() or OTP_SENDER
# Where the verification wizard finds a customer: "local" (the customer_directory
# table) or "mcp_hub" (the hub tool named by MCP_HUB_CUSTOMER_LOOKUP_TOOL). The
# default is local because the hub has no lookup tool today (docs/MCP_HUB.md);
# choosing mcp_hub without that tool makes the wizard say "unavailable".
CUSTOMER_DATA_SOURCE = os.environ.get("CUSTOMER_DATA_SOURCE", "local").strip().lower()
if CUSTOMER_DATA_SOURCE not in ("local", "mcp_hub"):
    CUSTOMER_DATA_SOURCE = "local"

# Which one time code backend the wizard uses: "local" (our own code, HMAC,
# 120 s) or "hub" (the hub's validate_customer_and_send_otp and
# verify_otp_and_authenticate, which also authenticates the hub session that the
# customer tools need). Our own limits (120 s, 3 attempts, single use, resend
# wait) are applied on top in both modes. Default: hub when the hub is the only source of answers
# (ANSWER_SOURCE=mcp_hub, the default) and it is configured, so the code is sent by the hub to the
# customer's registered number and checked by the hub; local otherwise. Hub mode sends a real SMS
# or e-mail to a real customer on every run.
_OTP_DEFAULT = (
    "hub"
    if os.environ.get("ANSWER_SOURCE", "mcp_hub").strip().lower() == "mcp_hub" and os.environ.get("MCP_HUB_API_KEY", "").strip()
    else "local"
)
VERIFY_OTP_BACKEND = os.environ.get("VERIFY_OTP_BACKEND", "").strip().lower() or _OTP_DEFAULT
if VERIFY_OTP_BACKEND not in ("local", "hub"):
    VERIFY_OTP_BACKEND = "local"

# Account answers for verified customers through the hub's read only customer
# tools (backend/customer_service). Customer data is only given to the local
# model unless CUSTOMER_DATA_TO_CLOUD_LLM is true. Tools that change an account
# (purchases, subscriptions, logging a fault) are not offered at all.
CUSTOMER_TOOLS_ENABLED = _env_bool("CUSTOMER_TOOLS_ENABLED", True)
CUSTOMER_DATA_TO_CLOUD_LLM = _env_bool("CUSTOMER_DATA_TO_CLOUD_LLM", False)
CUSTOMER_TOOLS_BUDGET_SECONDS = _env_float("CUSTOMER_TOOLS_BUDGET_SECONDS", 8.0)
CUSTOMER_TOOLS_MAX_CHARS = _env_int("CUSTOMER_TOOLS_MAX_CHARS", 6000)

# MCP hub as the only source for SLT answers (docs/MCP_INTEGRATION_PLAN.md).
# ANSWER_SOURCE=mcp_hub: SLT questions are answered only from hub tool results
# (the old local knowledge base and the model's own knowledge are not used for
# SLT facts). Any other value keeps the earlier behaviour.
ANSWER_SOURCE = os.environ.get("ANSWER_SOURCE", "mcp_hub").strip().lower()
TOOL_ACCESS_PATH = os.environ.get("TOOL_ACCESS_PATH", "") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "tool_access.yaml"
)
# The hub's login belongs to one open connection, so each verified customer
# keeps one. Idle ones are closed after HUB_SESSION_IDLE_SECONDS.
HUB_SESSION_IDLE_SECONDS = _env_int("HUB_SESSION_IDLE_SECONDS", 900)
HUB_SESSION_MAX = _env_int("HUB_SESSION_MAX", 50)
EXPOSE_LATENCY_TOOL = _env_bool("EXPOSE_LATENCY_TOOL", False)
# Knowledge base chunks below this score are treated as "not found" (the hub's
# own scores run 0 to 1: measured on the live hub, relevant questions scored 0.42 to
# 0.74 and unrelated ones (cricket, weather) 0.36 to 0.37).
KB_MIN_SCORE = _env_float("KB_MIN_SCORE", 0.40)
KB_MAX_CHARS = _env_int("KB_MAX_CHARS", 4000)
# Warm shared hub connections for the public read only tools (mcp_gateway/warm.py): a guest's question
# then skips the connect, initialise and slow first call (measured 2 to 4 s on the live hub).
# After a hub sign in the customer's name is looked for in the read only customer tools' answers (the verify
# answer has none), for at most this many seconds; 0 turns the lookup off (mcp_gateway/profile.py).
HUB_NAME_LOOKUP_SECONDS = float(os.environ.get("HUB_NAME_LOOKUP_SECONDS", "").strip() or 6)
HUB_WARM_SHARED = os.environ.get("HUB_WARM_SHARED", "true").strip().lower() in ("1", "true", "yes")
HUB_WARM_POOL_SIZE = int(os.environ.get("HUB_WARM_POOL_SIZE", "").strip() or 3)
HUB_WARM_IDLE_SECONDS = float(os.environ.get("HUB_WARM_IDLE_SECONDS", "").strip() or 300)

# The five star rating asked when a chat or voice session ends (POST /feedback/rating).
RATING_ENABLED = _env_bool("RATING_ENABLED", True)
RATING_RATE_LIMIT_PER_HOUR = _env_int("RATING_RATE_LIMIT_PER_HOUR", 30)
# Buying, subscribing and logging a fault. Off until the owner has approved a
# live test; when on, each one still needs a confirmed pending action.
STATE_CHANGING_TOOLS_ENABLED = _env_bool("STATE_CHANGING_TOOLS_ENABLED", False)
CONFIRMATION_TTL_SECONDS = _env_int("CONFIRMATION_TTL_SECONDS", 60)


def livekit_configured() -> bool:
    """True when a LiveKit URL, key and secret are all present (no network call)."""
    return bool(LIVEKIT_URL and LIVEKIT_API_KEY and LIVEKIT_API_SECRET)


def mcp_hub_configured() -> bool:
    """True when the hub is enabled and an API key is present (no network call)."""
    return bool(MCP_HUB_ENABLED and MCP_HUB_URL and MCP_HUB_API_KEY)


def database_is_sqlite() -> bool:
    return DATABASE_URL.startswith("sqlite")


def cloud_llm_configured() -> bool:
    """True if the configured cloud LLM provider has a usable API key.
    Does not make a network call, so it cannot detect an invalid key or an
    outage; that is handled at call time by the agent's own fallback logic.
    """
    if CLOUD_LLM_PROVIDER == "livekit":
        return livekit_configured()
    if CLOUD_LLM_PROVIDER == "gemini":
        return bool(GEMINI_API_KEY)
    if CLOUD_LLM_PROVIDER == "gpt":
        return bool(OPENAI_API_KEY)
    return False


def cloud_speech_configured() -> bool:
    """True if the configured cloud speech provider has usable credentials."""
    if CLOUD_SPEECH_PROVIDER == "google":
        return bool(GOOGLE_APPLICATION_CREDENTIALS)
    if CLOUD_SPEECH_PROVIDER == "azure":
        return bool(AZURE_SPEECH_KEY and AZURE_SPEECH_REGION)
    return False
