"""Multilingual, natural voice text to speech for the voice interaction backend.

Fallback order, from best to last resort:

    1. The configured cloud provider (Google Cloud TTS Neural2, or Azure
       Neural TTS), when real credentials are set. Highest quality, but
       needs an account, credentials, and is subject to that provider's
       free tier quota.
    2. edge-tts: an unofficial but very widely used library that calls
       Microsoft Edge's own text to speech backend (the same neural voice
       catalog Azure Cognitive Services uses) with no API key, account, or
       credentials at all, only internet access. This is what actually
       gets this project a natural sounding, non robotic voice, including
       a real male voice option, without anyone needing to set anything
       up; see edge_tts_voices() below for the specific voices used. It
       needs internet access, so it is not a true offline/local option,
       but it needs nothing to configure, unlike the cloud providers above.
    3. pyttsx3 (SAPI5), fully offline, for English only. This machine has
       exactly two installed SAPI5 voices, "David" (male) and "Zira"
       (female); genuinely the last resort for voice quality, but the only
       option with zero network dependency.
    4. For Sinhala or Tamil, if even edge-tts fails (no internet, or the
       service is unreachable), a short spoken English notice instead of
       mangling the text through an English voice or failing silently, per
       this project's rule against silently faking or skipping a blocked
       capability. This is now the genuinely rare case, not the common
       one: edge-tts alone gets Sinhala and Tamil a real, natural voice, no
       credentials required (verified live, see KNOWN_LIMITATIONS.md).

Streaming playback: speak_stream() takes sentences one at a time, even
while the agent is still generating the rest of its reply, and runs
synthesis and playback on two separate threads, so sentence N+1 is being
synthesized while sentence N is playing. Before this, each sentence was
synthesized only after the previous one had finished playing, which put a
full network round trip of silence between every sentence, and nothing was
spoken until the whole reply existed. speak_text() is the same thing for a
reply that is already complete.
"""

import queue
import re
import sys
import threading
import time
from typing import Callable, Iterable, Optional

import pyttsx3

import config

# edge-tts and Azure Cognitive Services draw from the same neural voice
# catalog (edge-tts calls Microsoft Edge's own backend for the same
# voices), so this one map serves both _synthesize_edge_tts() and
# _speak_cloud_azure() below rather than keeping two copies in sync.
from voicekit import voice_config as _voice_config  # noqa: E402

_NEURAL_VOICE_MAP = _voice_config.edge_map()

# The voices used before the newer ones above. If Microsoft's service refuses a
# newer voice name, the older one is tried before the offline fallback.
_NEURAL_FALLBACK_MAP = {
    ("en", "female_natural"): "en-US-JennyNeural",
    ("en", "male"): "en-US-GuyNeural",
    ("ta", "female_natural"): "ta-IN-PallaviNeural",
    ("ta", "male"): "ta-IN-ValluvarNeural",
}

# Which engine spoke the last sentence and why a better one was skipped, for
# /health. "edge-tts 403" shows up there when Microsoft's free service blocks an
# old edge-tts version (fixed by upgrading the package), instead of the voice
# silently becoming the Windows SAPI voice.
_speech_state = {"engine": None, "last_error": None}


def status() -> dict:
    return dict(_speech_state)

_LOCAL_FEMALE_VOICE_ID = None
_LOCAL_MALE_VOICE_ID = None
_voices_scanned = False


def _scan_local_voices():
    """Finds this machine's installed SAPI5 voice IDs once, so later calls
    do not re-enumerate voices on every utterance."""
    global _LOCAL_FEMALE_VOICE_ID, _LOCAL_MALE_VOICE_ID, _voices_scanned
    if _voices_scanned:
        return
    try:
        engine = pyttsx3.init()
        for voice in engine.getProperty("voices"):
            name = (voice.name or "").lower()
            gender = str(getattr(voice, "gender", "") or "").lower()
            if "female" in gender or "zira" in name:
                _LOCAL_FEMALE_VOICE_ID = voice.id
            elif "male" in gender or "david" in name:
                _LOCAL_MALE_VOICE_ID = voice.id
        engine.stop()
    except Exception as exc:
        print(f"TTS: could not enumerate local voices ({exc}).")
    _voices_scanned = True


def _split_sentences(text: str) -> list[str]:
    """Splits text into sentence sized chunks for streamed playback. A
    simple punctuation split is good enough here; this is not meant to be
    a linguistically precise sentence tokenizer, just a way to start
    speaking before the whole reply is ready."""
    parts = re.split(r"(?<=[.!?।])\s+", text.strip())
    return [p for p in parts if p.strip()]


def _speak_local(text_input: str, voice_style: str) -> bool:
    """Speaks text_input using the local SAPI5 engine. Returns True if
    playback happened, False if the local engine could not be used."""
    _scan_local_voices()
    try:
        engine = pyttsx3.init()
        voice_id = _LOCAL_FEMALE_VOICE_ID if voice_style == "female_natural" else _LOCAL_MALE_VOICE_ID
        if voice_id:
            engine.setProperty("voice", voice_id)
        cancel = _current_cancel()
        if cancel is not None:
            if cancel.is_set():
                return True

            def _stop_if_cancelled(*_args, **_kwargs):
                if cancel.is_set():
                    engine.stop()

            engine.connect("started-word", _stop_if_cancelled)
        engine.say(text_input)
        engine.runAndWait()
        engine.stop()
        return True
    except Exception as exc:
        print(f"TTS error: local playback failed ({exc}).")
        return False


# Explicit Chirp3-HD voice names rather than letting Google auto pick any
# voice that matches a requested gender: gender only selection can land on
# any tier in Google's catalog, including the much older, audibly synthetic
# "Standard" voices, not necessarily the natural sounding ones. Chirp3-HD
# is Google's newest, most natural voice generation (confirmed against the
# real account's own voice list, not assumed); the same character names
# exist across its supported languages, so one small map covers both.
# Sinhala has no entry at all: confirmed live against the real API that
# Google Cloud Text to Speech has zero si-LK voices in its catalog, not a
# transient error, so _speak_cloud_google() skips the network call for
# Sinhala entirely rather than making a request that can only fail. See
# KNOWN_LIMITATIONS.md.
_GOOGLE_CHIRP_VOICE_MAP = {
    ("en", "female_natural"): ("en-US", "en-US-Chirp3-HD-Kore"),
    ("en", "male"): ("en-US", "en-US-Chirp3-HD-Puck"),
    ("ta", "female_natural"): ("ta-IN", "ta-IN-Chirp3-HD-Kore"),
    ("ta", "male"): ("ta-IN", "ta-IN-Chirp3-HD-Puck"),
}


_google_tts_client = None


# Barge in. speak_stream() runs every play callable on its own thread and records
# the caller's cancel event here, so the low level players below (which take no
# arguments) can notice it. Setting the event stops the sound within a few
# milliseconds and makes speak_stream() drop everything still queued.
_playback = threading.local()
_CANCEL_POLL_SECONDS = 0.01


def _current_cancel() -> Optional[threading.Event]:
    return getattr(_playback, "cancel", None)


def _play(data, samplerate) -> None:
    import sounddevice as sd
    from vad import echo_reference

    cancel = _current_cancel()
    if cancel is not None and cancel.is_set():
        return
    echo_reference.start_clip(data, samplerate)
    sd.play(data, samplerate)
    if cancel is None:
        sd.wait()
        return
    try:
        duration = len(data) / float(samplerate)
    except TypeError:
        duration = 0.0
    deadline = time.monotonic() + duration + 0.25
    while time.monotonic() < deadline:
        if cancel.wait(_CANCEL_POLL_SECONDS):
            sd.stop()  # flushes what sounddevice still holds for this clip
            echo_reference.stop_clip()
            return
    sd.wait()


def _google_voice(language: str, voice_style: str):
    return _GOOGLE_CHIRP_VOICE_MAP.get((language, voice_style)) or _GOOGLE_CHIRP_VOICE_MAP.get(
        (language, "female_natural")
    )


def _get_google_client():
    global _google_tts_client
    if _google_tts_client is None:
        from google.cloud import texttospeech

        _google_tts_client = texttospeech.TextToSpeechClient()
    return _google_tts_client


_GOOGLE_STREAM_SAMPLE_RATE = 24000


def _play_pcm_stream(first_chunk: bytes, chunks: "queue.Queue", samplerate: int) -> None:
    """Plays 16 bit mono PCM as it arrives: first_chunk right away, then
    whatever the receiving thread puts on chunks, until it puts None (or
    an exception, if the stream broke part way). Leaving the with block
    waits for the buffered audio to finish playing."""
    import sounddevice as sd
    from vad import echo_reference

    leftover = b""
    cancel = _current_cancel()
    with sd.RawOutputStream(samplerate=samplerate, channels=1, dtype="int16") as out:
        item = first_chunk
        while item is not None:
            if cancel is not None and cancel.is_set():
                out.abort()  # drops the buffered audio instead of draining it
                echo_reference.stop_clip()
                break
            if isinstance(item, Exception):
                print(f"TTS: Google stream broke part way ({item}).")
                break
            data = leftover + item
            usable = len(data) - (len(data) % 2)  # whole 16 bit samples only
            if usable:
                # Small writes, so a cancel is noticed within about 100 ms of audio.
                step = 4800
                for start in range(0, usable, step):
                    if cancel is not None and cancel.is_set():
                        echo_reference.stop_clip()
                        break
                    piece = data[start : min(start + step, usable)]
                    echo_reference.feed_stream_pcm16(piece, samplerate)
                    out.write(piece)
            leftover = data[usable:]
            while True:
                try:
                    item = chunks.get(timeout=0.05)
                    break
                except queue.Empty:
                    if cancel is not None and cancel.is_set():
                        item = b""
                        break


def _stream_cloud_google(text_input: str, language: str, voice_style: str) -> Optional[Callable[[], None]]:
    """Starts synthesizing text_input with Google's streaming Text to
    Speech API and returns a callable that plays it, or None (never
    raises) if unavailable. Measured here, the streaming API delivers its
    first audio in about 0.3s, versus about 1.4s for the regular request
    below, which only answers once the whole sentence is synthesized; the
    voice is the same. Audio keeps arriving on a background thread, so
    this still works as a prefetch while the previous sentence plays.
    Waits only for the first chunk, so a failure is known here and the
    caller can fall back before anything has been played."""
    if not config.GOOGLE_APPLICATION_CREDENTIALS:
        return None
    voice_entry = _google_voice(language, voice_style)
    if voice_entry is None:
        return None
    lang_code, voice_name = voice_entry
    try:
        from google.cloud import texttospeech

        client = _get_google_client()
        streaming_config = texttospeech.StreamingSynthesizeConfig(
            voice=texttospeech.VoiceSelectionParams(language_code=lang_code, name=voice_name),
            streaming_audio_config=texttospeech.StreamingAudioConfig(
                audio_encoding=texttospeech.AudioEncoding.PCM,
                sample_rate_hertz=_GOOGLE_STREAM_SAMPLE_RATE,
            ),
        )
        stream_requests = [
            texttospeech.StreamingSynthesizeRequest(streaming_config=streaming_config),
            texttospeech.StreamingSynthesizeRequest(input=texttospeech.StreamingSynthesisInput(text=text_input)),
        ]
    except Exception as exc:
        print(f"TTS: Google streaming TTS unavailable ({exc}), falling back.")
        return None

    chunks: "queue.Queue" = queue.Queue()

    def _receive():
        try:
            for response in client.streaming_synthesize(iter(stream_requests)):
                if response.audio_content:
                    chunks.put(response.audio_content)
        except Exception as exc:
            chunks.put(exc)
        finally:
            chunks.put(None)

    threading.Thread(target=_receive, daemon=True).start()
    first_chunk = chunks.get()
    if first_chunk is None or isinstance(first_chunk, Exception):
        print(f"TTS: Google streaming TTS unavailable ({first_chunk}), falling back.")
        return None
    return lambda: _play_pcm_stream(first_chunk, chunks, _GOOGLE_STREAM_SAMPLE_RATE)


def _synthesize_cloud_google(text_input: str, language: str, voice_style: str):
    """Synthesizes text_input via Google Cloud Text to Speech's Chirp3-HD
    voices and returns (samples, samplerate) without playing it, so the
    caller can synthesize the next sentence while this one plays. Returns
    None (never raises) if unavailable, so the caller can fall back
    cleanly. The client is created once and reused: building a new one
    per sentence repeated the credential load and connection setup every
    time."""
    if not config.GOOGLE_APPLICATION_CREDENTIALS:
        return None
    voice_entry = _google_voice(language, voice_style)
    if voice_entry is None:
        return None
    lang_code, voice_name = voice_entry
    try:
        import io

        import soundfile as sf
        from google.cloud import texttospeech

        voice_params = texttospeech.VoiceSelectionParams(
            language_code=lang_code,
            name=voice_name,
        )
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)
        response = _get_google_client().synthesize_speech(
            input=texttospeech.SynthesisInput(text=text_input),
            voice=voice_params,
            audio_config=audio_config,
        )
        return sf.read(io.BytesIO(response.audio_content), dtype="float32")
    except Exception as exc:
        print(f"TTS: Google Cloud TTS unavailable ({exc}), falling back.")
        return None


def _speak_cloud_azure(text_input: str, language: str, voice_style: str) -> bool:
    """Synthesizes and plays text_input via Azure Neural TTS. Returns False
    (never raises) if unavailable, so the caller can fall back cleanly."""
    if not (config.AZURE_SPEECH_KEY and config.AZURE_SPEECH_REGION):
        return False
    try:
        import azure.cognitiveservices.speech as speechsdk

        voice_name = _NEURAL_VOICE_MAP.get((language, voice_style), "en-US-JennyNeural")

        speech_config = speechsdk.SpeechConfig(
            subscription=config.AZURE_SPEECH_KEY, region=config.AZURE_SPEECH_REGION
        )
        speech_config.speech_synthesis_voice_name = voice_name
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
        result = synthesizer.speak_text_async(text_input).get()
        return result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted
    except Exception as exc:
        print(f"TTS: Azure Neural TTS unavailable ({exc}), falling back.")
        return False


def _synthesize_edge_tts(text_input: str, language: str, voice_style: str):
    """Synthesizes text_input via edge-tts (see this module's docstring
    for what it is and why it sits here in the fallback order) and returns
    (samples, samplerate) without playing it. Returns None (never raises)
    on any failure, most commonly no internet access, so the caller falls
    back to the fully offline local voice.

    edge-tts's own API is async only (it streams audio over a websocket to
    Microsoft's service); asyncio.run() drives it synchronously here since
    this always runs on speak_stream()'s synthesis thread, which has no
    event loop of its own. soundfile can decode the MP3 bytes edge-tts
    returns directly (confirmed against a real sample; no separate MP3
    decoding library needed)."""
    voice_name = (
        _NEURAL_VOICE_MAP.get((language, voice_style))
        or _NEURAL_VOICE_MAP.get((language, "female_natural"))
        or "en-US-JennyNeural"
    )
    try:
        import asyncio
        import io

        import edge_tts
        import soundfile as sf

        async def _synthesize() -> bytes:
            communicate = edge_tts.Communicate(text_input, voice_name)
            audio_bytes = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_bytes.extend(chunk["data"])
            return bytes(audio_bytes)

        audio_bytes = b""
        for candidate in (voice_name, _NEURAL_FALLBACK_MAP.get((language, voice_style))):
            if not candidate:
                continue
            voice_name = candidate
            try:
                audio_bytes = asyncio.run(asyncio.wait_for(_synthesize(), timeout=8.0))
            except Exception as exc:  # noqa: BLE001  try the older voice, then give up
                _speech_state["last_error"] = type(exc).__name__ + ": " + str(exc)[:80]
                audio_bytes = b""
            if audio_bytes:
                break
        if not audio_bytes:
            print(f"TTS: edge-tts unavailable ({_speech_state['last_error']}), falling back.")
            _speech_state["engine"] = "offline (SAPI)"
            return None
        _speech_state.update(engine="edge-tts neural", last_error=None)
        return sf.read(io.BytesIO(audio_bytes), dtype="float32")
    except Exception as exc:
        _speech_state.update(engine="offline (SAPI)", last_error=type(exc).__name__ + ": " + str(exc)[:80])
        print(f"TTS: edge-tts unavailable ({exc}), falling back.")
        return None


def _speak_offline(sentence: str, language: str, voice_style: str) -> None:
    """Last resort once every network voice has failed: the local SAPI5
    voice for English, or, for Sinhala or Tamil, a short spoken English
    notice instead of mangling the text through an English voice. See
    KNOWN_LIMITATIONS.md. Genuinely rare now that edge-tts covers both
    languages with no credentials required."""
    if language == "en":
        if not _speak_local(sentence, voice_style):
            print("TTS error: local playback failed and no cloud or edge-tts path is available.")
        return
    notice = (
        f"Natural {config.SUPPORTED_LANGUAGES.get(language, language)} speech needs "
        "either a cloud provider or internet access for the free voice service, "
        "and neither is available right now."
    )
    print(f"TTS: {notice}")
    _speak_local(notice, voice_style)


# Markdown the LLM sometimes adds (bold markers, bullet stars, headings)
# means nothing when spoken, and some voices read the symbols out loud.
_UNSPOKEN_MARKUP = re.compile(r"[*#_`]+")


def _normalized_for_speech(sentence: str, language: str) -> str:
    """Numbers, rupee amounts, phone numbers and telecom words written out the way
    they are said (voice_text); the text on screen is not touched. Never raises."""
    try:
        from voicekit.normalizer_adapter import prepare_for_tts

        return prepare_for_tts(sentence, language) or sentence
    except Exception:  # noqa: BLE001
        return sentence


def _prepare_sentence(sentence: str, language: str, voice_style: str) -> Callable[[], None]:
    """Does the slow, network bound part (synthesis) for one sentence and
    returns a callable that plays it. Runs on speak_stream()'s synthesis
    thread, ahead of playback. Tries, in order: the configured cloud
    provider, then edge-tts, then the offline fallback. Azure's SDK
    synthesizes and plays in one call, so for Azure the whole thing
    happens at play time instead and gets no head start."""
    sentence = _UNSPOKEN_MARKUP.sub("", sentence).strip()
    if not sentence:
        return lambda: None
    sentence = _normalized_for_speech(sentence, language)
    start = time.perf_counter()

    if config.cloud_speech_configured():
        if config.CLOUD_SPEECH_PROVIDER == "azure":

            def _play_azure():
                if not _speak_cloud_azure(sentence, language, voice_style):
                    audio = _synthesize_edge_tts(sentence, language, voice_style)
                    if audio is not None:
                        _play(*audio)
                    else:
                        _speak_offline(sentence, language, voice_style)

            return _play_azure
        play = _stream_cloud_google(sentence, language, voice_style)
        if play is not None:
            print(f"TTS first audio (cloud streaming, {language}): {time.perf_counter() - start:.2f}s")
            return play
        audio = _synthesize_cloud_google(sentence, language, voice_style)
        if audio is not None:
            print(f"TTS synthesis (cloud, {language}): {time.perf_counter() - start:.2f}s")
            return lambda: _play(*audio)

    audio = _cached_edge_tts(sentence, language, voice_style)
    if audio is not None:
        print(f"TTS synthesis (edge-tts, {language}): {time.perf_counter() - start:.2f}s")
        return lambda: _play(*audio)

    return lambda: _speak_offline(sentence, language, voice_style)


# Synthesized edge-tts audio, keyed by (sentence, language, voice_style).
# edge-tts is what speaks Sinhala (Google has no Sinhala voice), and it took
# 2.5 to 4 seconds per sentence here, so fixed lines that come up again and
# again (the introduction, the closing lines) are worth keeping. Bounded so
# a long session of unique replies cannot grow it without limit; the
# oldest entry is dropped first.
_edge_tts_cache: dict = {}
_EDGE_TTS_CACHE_SIZE = 64


def _cached_edge_tts(sentence: str, language: str, voice_style: str):
    key = (sentence, language, voice_style)
    if key in _edge_tts_cache:
        return _edge_tts_cache[key]
    audio = _synthesize_edge_tts(sentence, language, voice_style)
    if audio is not None:
        if len(_edge_tts_cache) >= _EDGE_TTS_CACHE_SIZE:
            _edge_tts_cache.pop(next(iter(_edge_tts_cache)))
        _edge_tts_cache[key] = audio
    return audio


def precache(text_input: str, language: str, voice_style: str = "female_natural") -> None:
    """Synthesizes a known line ahead of time (server.py does this at
    startup for the Sinhala introduction), so when it is actually spoken
    the audio is already there. Only applies where edge-tts would be the
    voice used, i.e. a language Google Cloud Text to Speech cannot speak;
    Google's own streaming voice starts in about 0.3s and needs no cache."""
    if config.cloud_speech_configured() and (
        config.CLOUD_SPEECH_PROVIDER == "azure" or _google_voice(language, voice_style) is not None
    ):
        return
    start = time.perf_counter()
    for sentence in _split_sentences(text_input):
        _cached_edge_tts(_UNSPOKEN_MARKUP.sub("", sentence).strip(), language, voice_style)
    print(f"TTS precache ({language}): {time.perf_counter() - start:.2f}s")


class SentenceChunker:
    """Cuts a reply that arrives in small streamed pieces (a few tokens at
    a time from the LLM) into whole sentences as soon as each one is
    complete, so the first sentence can start being spoken while the agent
    is still writing the second. feed() returns any sentences completed by
    the new text; flush() returns whatever is left once the reply ends.

    The very first piece may also be cut at a comma or semicolon once it
    is FIRST_CLAUSE_MIN_CHARS long, because a long opening sentence would
    otherwise hold back the first spoken word until the model finishes
    writing all of it. Only the first: cutting every sentence at commas
    would make the voice pause unnaturally all the way through."""

    _BOUNDARY = re.compile(r"(?<=[.!?।])\s+|\n+")
    _CLAUSE_BOUNDARY = re.compile(r"(?<=[,;:])\s+")
    FIRST_CLAUSE_MIN_CHARS = 40

    def __init__(self):
        self._buffer = ""
        self._emitted_any = False

    def feed(self, text: str) -> list[str]:
        self._buffer += text
        parts = self._BOUNDARY.split(self._buffer)
        # The last part has no boundary after it yet, so it may still grow.
        self._buffer = parts.pop()
        sentences = [p.strip() for p in parts if p.strip()]
        if not sentences and not self._emitted_any and len(self._buffer) >= self.FIRST_CLAUSE_MIN_CHARS:
            clauses = self._CLAUSE_BOUNDARY.split(self._buffer)
            if len(clauses) > 1:
                self._buffer = clauses.pop()
                sentences = [" ".join(c.strip() for c in clauses)]
        if sentences:
            self._emitted_any = True
        return sentences

    def flush(self) -> list[str]:
        rest, self._buffer = self._buffer.strip(), ""
        return [rest] if rest else []


def speak_stream(
    sentences: Iterable[str],
    language: str = "en",
    voice_style: str = "female_natural",
    on_sentence_start: Optional[Callable[[str], None]] = None,
    cancel: Optional[threading.Event] = None,
) -> None:
    """Speaks sentences as they arrive from any iterable, including one
    that blocks while the agent is still generating (see server.py).
    A synthesis thread prepares up to two sentences ahead while this thread
    plays them in order, so playback of one sentence overlaps synthesis of
    the next instead of alternating with it. on_sentence_start, if given,
    is called with each sentence right before its audio starts, which is
    what lets the frontend flip to "speaking" and animate the avatar's
    mouth in step with the real audio rather than a guess.

    cancel (barge in): when the event is set the sentence that is playing is
    cut off at once, queued and not yet synthesized sentences are dropped, the
    source iterable is no longer consumed and the function returns."""
    prepared: "queue.Queue" = queue.Queue(maxsize=2)
    done = object()

    def _cancelled() -> bool:
        return cancel is not None and cancel.is_set()

    def _put(item) -> bool:
        while True:
            if _cancelled():
                return False
            try:
                prepared.put(item, timeout=0.05)
                return True
            except queue.Full:
                continue

    def _synthesis_worker():
        try:
            for sentence in sentences:
                if _cancelled():
                    break
                if not sentence or not sentence.strip():
                    continue
                try:
                    play = _prepare_sentence(sentence, language, voice_style)
                except Exception as exc:
                    print(f"TTS: could not prepare sentence ({exc}).")
                    continue
                if not _put((sentence, play)):
                    break
        finally:
            _put(done)  # waits for room; gives up only when cancelled (the player is leaving then)

    threading.Thread(target=_synthesis_worker, daemon=True).start()

    _playback.cancel = cancel
    try:
        while True:
            if _cancelled():
                return
            try:
                item = prepared.get(timeout=0.05)
            except queue.Empty:
                continue
            if item is done:
                return
            sentence, play = item
            if _cancelled():
                return
            if on_sentence_start is not None:
                on_sentence_start(sentence)
            start = time.perf_counter()
            play()
            print(f"TTS playback ({language}): {time.perf_counter() - start:.2f}s")
    finally:
        _playback.cancel = None


def warm_up() -> None:
    """Opens the Google Text to Speech connection with one tiny request at
    server startup (nothing is played), since the first request of a
    process also pays for credential loading and the connection handshake:
    about 1.5s versus 0.6s afterwards, measured here. No-op when Google
    speech is not configured; edge-tts has no persistent connection to
    warm."""
    if not (config.cloud_speech_configured() and config.CLOUD_SPEECH_PROVIDER == "google"):
        return
    start = time.perf_counter()
    if _stream_cloud_google("Hello.", "en", "female_natural") is not None:
        print(f"TTS warm up: {time.perf_counter() - start:.2f}s")


def speak_text(
    text_input: str,
    language: str = "en",
    voice_style: str = "female_natural",
    on_sentence_start: Optional[Callable[[str], None]] = None,
    cancel: Optional[threading.Event] = None,
) -> None:
    """Speaks an already complete text, sentence by sentence, through the
    same pipelined playback as speak_stream(). Tries, in order: the
    configured cloud provider (if real credentials are set), then edge-tts
    (natural, needs no credentials, only internet; see this module's
    docstring), then the local SAPI5 engine for English only, or a short
    spoken English notice for Sinhala or Tamil."""
    if text_input is None or not text_input.strip():
        return
    speak_stream(_split_sentences(text_input) or [text_input], language, voice_style, on_sentence_start, cancel)


def _run_standalone_test():
    """python tts.py: speaks a sample sentence in each supported language
    with the default female voice, then the same English sentence again
    with the male voice, so voice quality (and the cloud vs edge-tts vs
    local vs notice fallback behavior) can be compared and demoed
    directly, including hearing the male option this test specifically
    exercises."""
    # Windows consoles often default to a legacy codepage (cp1252) that
    # cannot encode Sinhala or Tamil script at all; printing the sample
    # text below would crash the whole test before it ever got to
    # speaking anything. Reconfiguring stdout to UTF-8 here, with
    # replacement for any character even UTF-8 output cannot render in
    # that terminal, keeps this test runnable from a plain terminal
    # rather than requiring the operator to change their system codepage
    # first. errors="replace" rather than "strict" because a print
    # statement failing is not worth treating as fatal here.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass  # older Python or a stream that does not support reconfigure(); best effort only

    samples = {
        "en": "This is a test of the natural voice text to speech system.",
        "si": "සිංහල බාෂාවේ පරීක්ෂණයකේ තක්සේත් ජාතිය සඩහශ චේතනය කරණයකින් වේ.",
        "ta": "இது இயற்கை குரலுக்கின் சோதனை ஓரு சோதனை வாக்கியம் ஆகும்.",
    }
    print(f"Cloud speech provider configured: {config.cloud_speech_configured()} ({config.CLOUD_SPEECH_PROVIDER})")
    for language, text in samples.items():
        print(f"\nSpeaking [{config.SUPPORTED_LANGUAGES[language]}, female]: {text}")
        speak_text(text, language=language, voice_style="female_natural")

    print("\nSpeaking [English, male]: This is the male voice option.")
    speak_text("This is the male voice option.", language="en", voice_style="male")


if __name__ == "__main__":
    _run_standalone_test()
