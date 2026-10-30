"""Streaming, multilingual speech to text for the real time voice pipeline.

Replaces the prototype's record-then-transcribe call with
transcribe_stream(), which consumes short audio chunks as they arrive (as
they would over a live WebSocket connection) and yields partial and final
transcript events, using VoiceActivityDetector (vad.py) to decide when
speech starts, when to refresh a partial guess, and when enough trailing
silence means the utterance is finished.

Honest engineering note on "streaming": faster-whisper (the local engine)
has no native partial decoding API; it transcribes whatever audio buffer
you give it, start to finish, each time you call it. So the "partial"
events here are produced by re-running transcription on the growing speech
buffer every few chunks, not by a model that natively emits words as it
hears them. This still cuts perceived latency a lot versus record-8-seconds
-then-transcribe (the prototype's behavior), because the user sees text
appear while they are still talking, but it is not the same as a true
streaming ASR API (which Google Cloud Speech to Text's streaming mode, for
example, actually provides). This tradeoff is documented in
KNOWN_LIMITATIONS.md rather than glossed over.

Language handling: English can always run locally via faster-whisper.
Sinhala and Tamil are attempted through the configured cloud speech
provider first (Google Cloud Speech to Text has explicit si-LK and ta-IN
support); if that is not configured or fails, they fall back to the same
local Whisper model, whose multilingual training does cover Sinhala and
Tamil but with materially lower accuracy than English, again documented
rather than hidden.
"""

import threading
import time
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
from faster_whisper import WhisperModel

import config
from vad import SAMPLE_RATE, VoiceActivityDetector, denoise_audio

_models: dict = {}


_sinhala_model = None
_sinhala_lock = threading.Lock()
_sinhala_fetching = False


def sinhala_model_ready() -> bool:
    return (config.STT_SINHALA_MODEL_DIR / "model.bin").exists()


def _get_sinhala_model() -> WhisperModel:
    global _sinhala_model
    with _sinhala_lock:
        if _sinhala_model is None:
            _sinhala_model = WhisperModel(str(config.STT_SINHALA_MODEL_DIR), device="cpu", compute_type="int8", cpu_threads=8)
        return _sinhala_model


def _fetch_sinhala_model() -> None:
    """Downloads the Sinhala model once, in the background. Never on a request path."""
    global _sinhala_fetching
    if sinhala_model_ready() or _sinhala_fetching or not config.STT_SINHALA_AUTO_DOWNLOAD or not config.STT_SINHALA_MODEL_REPO:
        return
    _sinhala_fetching = True

    def work() -> None:
        global _sinhala_fetching
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(config.STT_SINHALA_MODEL_REPO, local_dir=str(config.STT_SINHALA_MODEL_DIR))
            print("STT: the Sinhala speech model was downloaded.")
        except Exception as exc:  # noqa: BLE001
            print(f"STT: could not download the Sinhala speech model ({type(exc).__name__}); Sinhala keeps the default model.")
        finally:
            _sinhala_fetching = False

    threading.Thread(target=work, daemon=True).start()


def _run_sinhala(audio: np.ndarray) -> tuple[str, float]:
    """Greedy decoding with the voice filter and no temperature fallback: the default settings made the
    same model take 16 to 25 s on a 12 s sentence (retries on low confidence), these take about 3 s."""
    segments, _info = _get_sinhala_model().transcribe(
        audio, language="si", beam_size=1, temperature=0.0, condition_on_previous_text=False,
        without_timestamps=True, vad_filter=True,
    )
    segments = list(segments)
    text = "".join(segment.text for segment in segments).strip()
    confidence = float(np.mean([s.avg_logprob for s in segments])) if segments else -9.0
    return text, confidence


def _get_local_model(model_size: str) -> WhisperModel:
    """One loaded model per size, kept for the life of the process: the main
    one for every pass, and the larger one only if configured."""
    model = _models.get(model_size)
    if model is None:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        _models[model_size] = model
    return model


@dataclass
class TranscriptEvent:
    text: str
    is_final: bool
    language: str


def warm_up() -> None:
    """Loads the Whisper model and the noise reduction library and runs one
    throwaway pass of each, so the first real utterance of a session does
    not also pay for loading them (measured on this machine: about 6s for
    the Whisper model and about 7s for the first noisereduce/scipy import,
    versus about 0.4s and 0.03s per call once loaded). server.py calls this
    once at startup."""
    start = time.time()
    if config.STT_DENOISE:
        denoise_audio(np.zeros(SAMPLE_RATE, dtype=np.float32))
    silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
    for size in filter(None, [config.LOCAL_WHISPER_MODEL_SIZE, config.STT_ACCURATE_MODEL]):
        try:
            segments, _info = _get_local_model(size).transcribe(
                silence, language="en", beam_size=1, temperature=0.0, vad_filter=True
            )
            list(segments)
        except Exception as exc:
            # The larger model is optional (it may need a download that is
            # not possible right now); the main one failing is reported by
            # the first real transcription.
            print(f"STT warm up: model '{size}' not available ({exc}).")
    if sinhala_model_ready():
        try:
            _run_sinhala(silence)
        except Exception as exc:  # noqa: BLE001
            print(f"STT warm up: the Sinhala model is not usable ({exc}).")
    else:
        _fetch_sinhala_model()
    print(f"STT warm up: {time.time() - start:.2f}s")


def _run_whisper(
    model: WhisperModel, audio: np.ndarray, language_hint: Optional[str]
) -> tuple[str, str, Optional[float]]:
    """One Whisper pass: (text, language, confidence), where confidence is
    the average log probability of the segments (None when nothing was
    heard). Hotwords bias Whisper toward the brand name it otherwise hears
    as "Mobital" or "Mobilo", which then misses the knowledge base and
    confuses the agent: measured on spoken test questions, brand words
    recognised went from 46% to 85% with this set and the noise filter
    off."""
    segments, info = model.transcribe(
        audio,
        language=language_hint,
        beam_size=config.STT_BEAM_SIZE,
        hotwords=config.STT_HOTWORDS or None,
        vad_filter=config.STT_VAD_FILTER,
    )
    segments = list(segments)
    text = "".join(segment.text for segment in segments).strip()
    confidence = float(np.mean([segment.avg_logprob for segment in segments])) if segments else None
    return text, (language_hint or info.language), confidence


def _transcribe_local(audio: np.ndarray, language_hint: Optional[str], final: bool = False) -> tuple[str, str]:
    """The partial transcripts shown while the user is still talking use
    the fast main model only. The final transcript, which decides what is
    asked, gets a second look from the larger model when the main one was
    unsure of it (see STT_ACCURATE_MODEL and STT_CONFIDENCE_FLOOR): most
    turns are clear and pay nothing extra."""
    if language_hint == "si" and sinhala_model_ready():
        if not final:
            # A partial would cost about 3 s on this model and delay the end of the turn, and the stock
            # model only writes romanized noise for Sinhala: no partial text until the final one.
            return "", "si"
        try:
            text, _confidence = _run_sinhala(audio)
            if text:
                return text, "si"
        except Exception as exc:  # noqa: BLE001
            print(f"STT: the Sinhala model failed ({type(exc).__name__}); using the default model.")
    accurate = config.STT_ACCURATE_MODEL
    if final and accurate and language_hint in config.STT_ACCURATE_LANGUAGES:
        # The final transcript of these languages comes straight from the larger model: running
        # the small one first and the large one again would only cost more time.
        try:
            text, lang, _ = _run_whisper(_get_local_model(accurate), audio, language_hint)
            if text:
                return text, lang
        except Exception as exc:
            print(f"STT: '{accurate}' failed for {language_hint} ({exc}); using the main model.")
    text, lang, confidence = _run_whisper(_get_local_model(config.LOCAL_WHISPER_MODEL_SIZE), audio, language_hint)
    if (
        final
        and accurate
        and accurate != config.LOCAL_WHISPER_MODEL_SIZE
        and text
        and confidence is not None
        and confidence < config.STT_CONFIDENCE_FLOOR
    ):
        try:
            second_text, second_lang, _ = _run_whisper(_get_local_model(accurate), audio, language_hint)
            if second_text:
                print(f"STT: confidence {confidence:.2f} was low, rechecked with '{accurate}'.")
                return second_text, second_lang
        except Exception as exc:
            print(f"STT: recheck with '{accurate}' failed ({exc}); keeping the first transcript.")
    return text, lang


# After a cloud speech failure (for example the API not being enabled in the
# Google project, which answers 403 every time) the cloud is skipped for a
# while instead of being tried again for every transcript: partial
# transcripts are produced every 1.5 seconds of speech, so a failing call
# each time would add its network delay to the whole conversation. Same idea
# as the circuit breaker around the cloud language model in agent.py.
_CLOUD_RETRY_AFTER_SECONDS = 300
_cloud_retry_at = 0.0


def _transcribe_cloud_google(audio: np.ndarray, language_hint: str) -> Optional[tuple[str, str]]:
    global _cloud_retry_at
    if time.monotonic() < _cloud_retry_at:
        return None
    try:
        from google.cloud import speech

        client = speech.SpeechClient()
        lang_code = {"si": "si-LK", "ta": "ta-IN", "en": "en-US"}.get(language_hint, "en-US")
        audio_int16 = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16).tobytes()
        recognition_audio = speech.RecognitionAudio(content=audio_int16)
        recognition_config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=SAMPLE_RATE,
            language_code=lang_code,
        )
        response = client.recognize(config=recognition_config, audio=recognition_audio)
        text = " ".join(
            result.alternatives[0].transcript for result in response.results if result.alternatives
        ).strip()
        return (text, language_hint) if text else None
    except Exception as exc:
        _cloud_retry_at = time.monotonic() + _CLOUD_RETRY_AFTER_SECONDS
        reason = " ".join(str(exc).split())[:240]
        print(f"STT cloud error (google): {reason}; skipping the cloud for {_CLOUD_RETRY_AFTER_SECONDS}s, using local Whisper.")
        return None


def _transcribe_cloud(audio: np.ndarray, language_hint: str) -> Optional[tuple[str, str]]:
    """Attempts cloud speech recognition for the given language. Returns
    None (never raises) if cloud speech is not configured or the call
    fails, so the caller falls back to local Whisper without special
    casing at the call site."""
    if not config.cloud_speech_configured():
        return None
    if config.CLOUD_SPEECH_PROVIDER == "google":
        return _transcribe_cloud_google(audio, language_hint)
    # Azure Speech to Text is documented as a configured option but not
    # implemented for STT in this pass; see KNOWN_LIMITATIONS.md.
    return None


def _emit(audio: np.ndarray, language_hint: Optional[str], final: bool = False) -> tuple[str, str]:
    start = time.time()
    if config.STT_DENOISE:
        audio = denoise_audio(audio)

    if language_hint in ("si", "ta"):
        cloud_result = _transcribe_cloud(audio, language_hint)
        if cloud_result is not None:
            text, lang = cloud_result
            print(f"STT latency ({language_hint}, cloud): {time.time() - start:.2f}s")
            return text, lang
        print(
            f"STT: cloud speech not available for '{language_hint}', "
            "using local Whisper instead (lower accuracy expected for this language)."
        )

    text, lang = _transcribe_local(audio, language_hint, final)
    print(f"STT latency ({language_hint or 'auto'}, local): {time.time() - start:.2f}s")
    return text, lang


# Tuned for ~0.5s audio chunks, matching the chunk size main.py/server.py
# use when reading from the microphone or WebSocket.
CHUNK_SECONDS = 0.25
# Trailing silence ends an utterance (STT_SILENCE_SECONDS, 0.75 s by default; it was 1.5 s, then
# 1 s plus the rounding of 0.5 s chunks). A normal breath between phrases is shorter than that.
SILENCE_CHUNKS_TO_FINALIZE = max(1, round(config.STT_SILENCE_SECONDS / CHUNK_SECONDS))
REFRESH_CHUNKS = round(1.5 / CHUNK_SECONDS)  # emit a partial roughly every 1.5s of new speech


def transcribe_stream(
    audio_chunk_iterator: Iterator[np.ndarray],
    language_hint: Optional[str] = None,
    vad: Optional[VoiceActivityDetector] = None,
) -> Iterator[TranscriptEvent]:
    """Consumes an iterator of short float32 mono audio chunks (about
    CHUNK_SECONDS each, at SAMPLE_RATE) and yields TranscriptEvent objects:
    is_final=False partials as speech continues, then one is_final=True
    event once trailing silence (per the VoiceActivityDetector) indicates
    the utterance is complete.

    Pass an already calibrated VoiceActivityDetector (see vad.py) so the
    session's noise floor is respected; if none is given, a fresh one with
    default settings is created, which is fine for the standalone test
    below but not for a real session that should calibrate first.
    """
    vad = vad or VoiceActivityDetector()
    buffer: list[np.ndarray] = []
    trailing_silence_chunks = 0
    chunks_since_partial = 0
    speech_started = False

    for chunk in audio_chunk_iterator:
        is_speech = vad.is_speech(chunk)

        if is_speech:
            buffer.append(chunk)
            trailing_silence_chunks = 0
            speech_started = True
            chunks_since_partial += 1
        elif speech_started:
            buffer.append(chunk)  # keep a little trailing context for the final pass
            trailing_silence_chunks += 1

        if not speech_started:
            continue

        if chunks_since_partial >= REFRESH_CHUNKS:
            text, lang = _emit(np.concatenate(buffer), language_hint)
            if text:
                yield TranscriptEvent(text=text, is_final=False, language=lang)
            chunks_since_partial = 0

        if trailing_silence_chunks >= SILENCE_CHUNKS_TO_FINALIZE:
            text, lang = _emit(np.concatenate(buffer), language_hint, final=True)
            yield TranscriptEvent(text=text, is_final=True, language=lang)
            buffer = []
            speech_started = False
            trailing_silence_chunks = 0
            chunks_since_partial = 0

    # Stream ended (e.g. WebSocket closed) while mid utterance: flush what
    # was captured as a final event rather than dropping it silently.
    if speech_started and buffer:
        text, lang = _emit(np.concatenate(buffer), language_hint, final=True)
        if text:
            yield TranscriptEvent(text=text, is_final=True, language=lang)


def _microphone_chunk_iterator(seconds: float) -> Iterator[np.ndarray]:
    import sounddevice as sd

    block_frames = int(CHUNK_SECONDS * SAMPLE_RATE)
    num_blocks = max(1, int(round(seconds / CHUNK_SECONDS)))
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=block_frames) as stream:
        for _ in range(num_blocks):
            block, _overflow = stream.read(block_frames)
            yield block[:, 0]


def _run_standalone_test():
    """python stt.py: lets you pick a language, calibrates the noise floor
    briefly, then streams microphone audio through transcribe_stream() and
    prints partial and final events as they arrive, so streaming
    transcription can be verified in each supported language on its own."""
    import sounddevice as sd

    print("Supported languages:", ", ".join(f"{code} ({name})" for code, name in config.SUPPORTED_LANGUAGES.items()))
    language_hint = input("Language code to test (en/si/ta, blank for auto): ").strip() or None

    vad = VoiceActivityDetector()
    print("Calibrating noise floor: stay silent for 1.5 seconds...")
    calib = sd.rec(int(1.5 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    vad.calibrate_noise_floor(calib[:, 0])

    print("Speak for up to 10 seconds. A pause of about 1.5s ends the utterance.\n")
    for event in transcribe_stream(_microphone_chunk_iterator(10.0), language_hint=language_hint, vad=vad):
        label = "FINAL" if event.is_final else "partial"
        print(f"[{label}] ({event.language}) {event.text}")
        if event.is_final:
            break


if __name__ == "__main__":
    _run_standalone_test()
