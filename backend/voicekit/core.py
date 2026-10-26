"""voicekit core: text in, a stream of speech events out.

    for event in synthesize_stream("Hello there.", language="en"):
        ...   # SentenceStart, WordTiming, AudioChunk ..., SynthesisDone

Provider fallback order is cloud (Google or Azure), edge-tts, local model,
SAPI5, taken from settings.provider_order. A provider that fails before
producing audio is skipped for that sentence and, for 30 seconds, for that
language (so a provider that is out of credit or offline does not add a failed
request to every sentence). Nothing here raises for a provider failure: if
every provider fails the stream ends with SynthesisDone(provider="none").
The only error raised on purpose is InvalidRequest.

Synthesis runs on a worker thread, so a blocking input iterable (an LLM reply
still being written) and the provider network calls overlap with whatever the
caller does with the events. SynthStream.cancel() is thread safe and is what
the sinks call for barge in.

Run `python backend/voicekit/core.py` for a self test that prints latency.
"""

import os
import sys

# Allow `python backend/voicekit/core.py`: relative imports need the package
# on sys.path, and this directory must NOT stay on it, because our types.py
# would shadow the standard library module of the same name (so nothing but
# os and sys, already loaded, may be imported before this runs).
if __name__ == "__main__" and not __package__:
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _here]
    sys.path.insert(0, os.path.dirname(_here))
    from voicekit.core import run_selftest

    raise SystemExit(run_selftest())

import dataclasses
import queue
import re
import threading
import time
from typing import Iterable, Iterator, List, Union

from .providers import azure, edge, gemini_tts, google_cloud, livekit_inference_tts, local_model, piper_si, sapi
from .normalizer_adapter import prepare_for_tts
from .providers.base import clean_for_speech
from .settings import VoicekitSettings
from .types import (
    AudioChunk,
    InvalidRequest,
    ProviderUnavailable,
    SentenceStart,
    SynthesisDone,
    SynthEvent,
    WordTiming,
)
from .voices import EDGE_VOICES, GENDER, GOOGLE_VOICES, LANGUAGES, VOICE_STYLES

COOLDOWN_SECONDS = 30.0
_cooldown: dict = {}


class SentenceChunker:
    """Cuts a reply that arrives in small pieces into whole sentences as soon
    as each is complete, so the first can be spoken while the model is still
    writing the second. The very first piece may also be cut at a comma once
    it is 40 characters long, because a long opening sentence would otherwise
    delay the first word. Same rules as tts.SentenceChunker (reimplemented,
    voicekit does not import the host project)."""

    _BOUNDARY = re.compile(r"(?<=[.!?।])\s+|\n+")
    _CLAUSE_BOUNDARY = re.compile(r"(?<=[,;:])\s+")
    FIRST_CLAUSE_MIN_CHARS = 40

    def __init__(self):
        self._buffer = ""
        self._emitted_any = False

    def feed(self, text: str) -> List[str]:
        self._buffer += text
        parts = self._BOUNDARY.split(self._buffer)
        self._buffer = parts.pop()  # no boundary after it yet, may still grow
        sentences = [p.strip() for p in parts if p.strip()]
        if not sentences and not self._emitted_any and len(self._buffer) >= self.FIRST_CLAUSE_MIN_CHARS:
            clauses = self._CLAUSE_BOUNDARY.split(self._buffer)
            if len(clauses) > 1:
                self._buffer = clauses.pop()
                sentences = [" ".join(c.strip() for c in clauses)]
        if sentences:
            self._emitted_any = True
        return sentences

    def flush(self) -> List[str]:
        rest, self._buffer = self._buffer.strip(), ""
        return [rest] if rest else []


def _cloud_module(settings: VoicekitSettings):
    return azure if settings.cloud_provider == "azure" else google_cloud


def _provider_chain(settings: VoicekitSettings):
    modules = {
        "cloud": _cloud_module(settings), "edge": edge, "local_model": local_model, "sapi": sapi,
        # Opt in slots, only used when a host puts them in provider_order.
        "livekit_inference": livekit_inference_tts, "gemini_tts": gemini_tts, "piper_si": piper_si,
    }
    return [modules[slot] for slot in settings.provider_order if slot in modules]


def _validate(language: str, voice: str, settings: VoicekitSettings) -> None:
    if language not in LANGUAGES:
        raise InvalidRequest(f"unknown language {language!r}, expected one of {LANGUAGES}")
    if voice not in VOICE_STYLES:
        raise InvalidRequest(f"unknown voice {voice!r}, expected one of {VOICE_STYLES}")


class SynthStream:
    """Iterator of SynthEvent with a thread safe cancel(). Created by
    synthesize_stream(); iterate it like any generator. close() and cancel()
    are the same; after either, iteration simply stops."""

    _MAX_BUFFERED_EVENTS = 128

    def __init__(self, source, language: str, voice: str, settings: VoicekitSettings):
        self._source = source
        self._language = language
        self._voice = voice
        self._settings = settings
        self._queue: "queue.Queue" = queue.Queue(maxsize=self._MAX_BUFFERED_EVENTS)
        self._cancel = threading.Event()
        self._finished = False
        self._start = time.perf_counter()
        self._first_audio_ms = 0.0
        self._provider_used = "none"
        self._thread = threading.Thread(target=self._run, daemon=True, name="voicekit-synth")
        self._thread.start()

    # Consumer side

    def __iter__(self):
        return self

    def __next__(self) -> SynthEvent:
        while not self._finished:
            if self._cancel.is_set():
                self._finished = True
                break
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if isinstance(item, Exception):
                self._finished = True
                raise item
            if isinstance(item, SynthesisDone):
                self._finished = True
            return item
        raise StopIteration

    def cancel(self) -> None:
        self._cancel.set()

    close = cancel

    # Worker side

    def _put(self, item) -> bool:
        while not self._cancel.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _sentences(self) -> Iterator[str]:
        chunker = SentenceChunker()
        total = 0
        pieces = [self._source] if isinstance(self._source, str) else self._source
        for piece in pieces:
            if self._cancel.is_set():
                return
            total += len(piece or "")
            if total > self._settings.max_text_chars:
                raise InvalidRequest(f"text is longer than {self._settings.max_text_chars} characters")
            yield from chunker.feed(piece or "")
        yield from chunker.flush()

    def _speak_sentence(self, text: str, index: int) -> bool:
        """Tries each provider in order; returns True if any produced audio."""
        language = self._language
        text = prepare_for_tts(text, language)
        for module in _provider_chain(self._settings):
            if self._cancel.is_set():
                return False
            if time.monotonic() < _cooldown.get((module.NAME, language), 0.0):
                continue
            if not module.available(self._settings):
                continue  # not configured is normal, not worth a log line
            produced = False
            try:
                for event in module.stream(text, language, self._voice, self._settings, self._cancel):
                    event = dataclasses.replace(event, sentence_index=index)
                    if isinstance(event, AudioChunk) and not produced:
                        produced = True
                        if self._provider_used == "none":
                            self._provider_used = module.NAME
                            self._first_audio_ms = (time.perf_counter() - self._start) * 1000.0
                            print(f"voicekit first audio ({module.NAME}, {language}): {self._first_audio_ms:.0f} ms")
                    if not self._put(event):
                        return produced
                if produced:
                    return True
                reason = "no audio"
            except ProviderUnavailable as exc:
                reason = str(exc)
            except Exception as exc:  # a provider bug must not break the stream
                reason = f"{type(exc).__name__}"
            if self._cancel.is_set():
                return produced  # a cancelled request is not a provider failure
            if produced:
                return True  # failed part way: keep what was heard, do not repeat it
            if module.available(self._settings):
                _cooldown[(module.NAME, language)] = time.monotonic() + COOLDOWN_SECONDS
            print(f"voicekit: {module.NAME} unavailable ({reason}), falling back")
        return False

    def _run(self) -> None:
        try:
            index = 0
            for sentence in self._sentences():
                spoken = clean_for_speech(sentence)
                if not spoken:
                    continue
                if not self._put(SentenceStart(text=spoken, sentence_index=index)):
                    return
                self._speak_sentence(spoken, index)
                index += 1
            total_ms = (time.perf_counter() - self._start) * 1000.0
            self._put(SynthesisDone(self._provider_used, self._first_audio_ms, total_ms))
            print(f"voicekit total ({self._provider_used}, {self._language}): {total_ms:.0f} ms")
        except InvalidRequest as exc:
            self._put(exc)
        except Exception as exc:  # the source iterable broke
            print(f"voicekit: input stream failed ({type(exc).__name__})")
            total_ms = (time.perf_counter() - self._start) * 1000.0
            self._put(SynthesisDone(self._provider_used, self._first_audio_ms, total_ms))


def synthesize_stream(
    text_or_iterable: Union[str, Iterable[str]],
    language: str = "en",
    voice: str = "female_natural",
    settings: VoicekitSettings = None,
) -> SynthStream:
    """Starts synthesis and returns an iterator of SynthEvent (see types.py).
    Raises InvalidRequest at once for an unknown language or voice, or for a
    string that is empty or too long; an iterable that grows too long raises
    it from the iterator instead."""
    settings = settings or VoicekitSettings()
    _validate(language, voice, settings)
    if isinstance(text_or_iterable, str):
        if not text_or_iterable.strip():
            raise InvalidRequest("text is empty")
        if len(text_or_iterable) > settings.max_text_chars:
            raise InvalidRequest(f"text is longer than {settings.max_text_chars} characters")
    return SynthStream(text_or_iterable, language, voice, settings)


def list_voices(settings: VoicekitSettings = None) -> list:
    """Every voice voicekit knows, as {language, voice, provider, name,
    license, gender}. "voice" is the style to pass to synthesize_stream.
    Local model entries appear only when their files exist (settings given)."""
    out = []
    for (lang, style), name in EDGE_VOICES.items():
        out.append(dict(language=lang, voice=style, provider="edge", name=name, gender=GENDER[style],
                        license="Microsoft service terms via edge-tts; no key; unofficial endpoint, no SLA"))
    for (lang, style), (_, name) in GOOGLE_VOICES.items():
        out.append(dict(language=lang, voice=style, provider="google", name=name, gender=GENDER[style],
                        license="Google Cloud Text to Speech: paid API, terms of the Google Cloud account"))
    for style, name in (("female_natural", "Microsoft Zira"), ("male", "Microsoft David")):
        out.append(dict(language="en", voice=style, provider="sapi", name=name, gender=GENDER[style],
                        license="Operating system voice (Windows SAPI5), licensed with Windows"))
    if settings is not None:
        for lang in LANGUAGES:
            for style in VOICE_STYLES:
                path = local_model.model_path(settings, lang, style)
                if path is not None:
                    out.append(dict(language=lang, voice=style, provider="local_model", name=path.stem,
                                    gender=GENDER[style], license=local_model.licence_of(path)))
    return out


def run_selftest() -> int:
    """Speaks a sample in each language and prints latency and PCM sanity."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    samples = {
        "en": "This is a test of the voicekit speech stream. It speaks two sentences.",
        "si": "ආයුබෝවන්, මෙය පරීක්ෂණයකි.",
        "ta": "வணக்கம், இது ஒரு சோதனை.",
    }
    ok = True
    for lang, text in samples.items():
        chunks = words = 0
        audio_ms = 0.0
        done = None
        for ev in synthesize_stream(text, language=lang):
            if isinstance(ev, AudioChunk):
                assert len(ev.pcm) % 2 == 0 and ev.sample_rate > 0
                chunks += 1
                audio_ms += ev.duration_ms
            elif isinstance(ev, WordTiming):
                words += 1
            elif isinstance(ev, SynthesisDone):
                done = ev
        print(f"[{lang}] provider={done.provider} first_audio={done.first_audio_ms:.0f} ms "
              f"total={done.total_ms:.0f} ms chunks={chunks} audio={audio_ms:.0f} ms words={words}")
        ok = ok and done.provider != "none"
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(run_selftest())
