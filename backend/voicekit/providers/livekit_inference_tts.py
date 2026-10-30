"""LiveKit Inference text to speech (tier 2): Inworld 2.0 for Sinhala, Gemini 3.8 Flash for the rest.

Uses livekit.agents.inference.TTS, which sends the text to the LiveKit Cloud gateway
(billed to the LiveKit project, not to Google). Needs LIVEKIT_URL, LIVEKIT_API_KEY and
LIVEKIT_API_SECRET in the environment; without them (or without the livekit-agents package)
available() is False and stream() raises ProviderUnavailable before any network call.

Same contract as the other providers: a function from one sentence to 16 bit mono PCM
AudioChunk events, ProviderUnavailable when it cannot speak, cancel checked between chunks.
No WordTiming events (the gateway does not return them), so clients lip sync from the audio.

Voices come from config/voices.yaml, tier livekit_inference, entries "<model>:<voice>", for
example "inworld/inworld-tts-2:Ashley". The model ids were read in the LiveKit docs on
2026-10-15 (docs, not run). STATUS: never run here, no Inference TTS call was made. The
voice ids and whether Ashley or Edward speak good Sinhala are unverified.
"""

import asyncio
import os
import queue
import threading
from typing import Iterator, List, Optional, Tuple

from ..types import AudioChunk, ProviderUnavailable
from ..voice_config import lookup

NAME = "livekit_inference"
_DONE = object()
_REQUIRED_ENV = ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")
MODELS = (
    "google/gemini-3.8-flash-lite-tts",
    "google/gemini-3.8-flash-tts",
    "inworld/inworld-tts-2",
    "inworld/inworld-tts-2-flash",
)


def _env(name: str, settings) -> str:
    return (getattr(settings, name.lower(), "") or os.environ.get(name, "") or "").strip()


def missing_credentials(settings=None) -> List[str]:
    return [n for n in _REQUIRED_ENV if not _env(n, settings)]


def available(settings=None) -> bool:
    if missing_credentials(settings):
        return False
    try:
        from livekit.agents import inference  # noqa: F401
    except Exception:
        return False
    return True


def split_entry(entry: str) -> Tuple[str, Optional[str]]:
    """'inworld/inworld-tts-2:Ashley' -> ('inworld/inworld-tts-2', 'Ashley')."""
    model, sep, voice = entry.partition(":")
    return model, (voice or None) if sep else None


def _make_tts(model: str, voice: Optional[str], language: str, settings):
    """Separate function so tests can replace it with a fake."""
    from livekit.agents import inference

    kwargs = {"language": language}
    if voice:
        kwargs["voice"] = voice
    return inference.TTS(
        model,
        api_key=_env("LIVEKIT_API_KEY", settings),
        api_secret=_env("LIVEKIT_API_SECRET", settings),
        **kwargs,
    )


_DEFAULT_MAKE_TTS = _make_tts


def _pump(tts_obj, text: str, out: "queue.Queue", cancel: threading.Event) -> None:
    async def run():
        stream = tts_obj.synthesize(text)
        try:
            async for ev in stream:
                if cancel.is_set():
                    return
                frame = ev.frame
                out.put((bytes(frame.data), int(frame.sample_rate), int(getattr(frame, "num_channels", 1) or 1)))
        finally:
            try:
                await stream.aclose()
            except Exception:
                pass

    try:
        asyncio.run(run())
    except Exception as exc:
        out.put(exc)
    finally:
        out.put(_DONE)


def _to_mono(pcm: bytes, channels: int) -> bytes:
    if channels <= 1:
        return pcm
    import numpy as np

    arr = np.frombuffer(pcm[: len(pcm) - len(pcm) % (2 * channels)], dtype="<i2").reshape(-1, channels)
    return arr.mean(axis=1).astype("<i2").tobytes()


def stream(text: str, language: str, voice: str, settings, cancel: threading.Event) -> Iterator[object]:
    missing = missing_credentials(settings)
    if missing:
        raise ProviderUnavailable("LiveKit Inference not configured: missing " + ", ".join(missing))
    if _make_tts is _DEFAULT_MAKE_TTS:
        try:
            import livekit.agents  # noqa: F401
        except Exception as exc:
            raise ProviderUnavailable("livekit-agents is not installed") from exc
    entries = [e for e in lookup("livekit_inference", language, voice) if e]
    if not entries:
        raise ProviderUnavailable(f"no LiveKit Inference voice configured for {language}")
    last = "no voice tried"
    for entry in entries:
        model, voice_id = split_entry(entry)
        try:
            tts_obj = _make_tts(model, voice_id, language, settings)
        except Exception as exc:
            last = f"setup failed: {type(exc).__name__}"
            continue
        q: "queue.Queue" = queue.Queue()
        threading.Thread(target=_pump, args=(tts_obj, text, q, cancel), daemon=True, name="voicekit-lk-tts").start()
        got_audio = False
        failed = None
        while True:
            item = q.get()
            if item is _DONE:
                break
            if isinstance(item, Exception):
                failed = item
                continue
            pcm, rate, channels = item
            pcm = _to_mono(pcm, channels)
            if pcm:
                got_audio = True
                yield AudioChunk(pcm, rate)
        if got_audio:
            return  # a failure after audio keeps what was played; the core moves on
        last = f"{type(failed).__name__}" if failed else "no audio"
    raise ProviderUnavailable(f"LiveKit Inference failed: {last}")
