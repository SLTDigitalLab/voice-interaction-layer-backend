"""One voice conversation over a LiveKit room.

This is the LiveKit counterpart of server.py's /ws/voice handler. It runs the
SAME pipeline stages, not copies of them: vad.VoiceActivityDetector,
stt.transcribe_stream and Orchestrator.respond_stream are called as they are,
each blocking one on a thread bridged into asyncio exactly the way server.py
bridges them. Only the two ends change: audio arrives as LiveKit frames
instead of WebSocket binary messages, and speech leaves as a published audio
track (voicekit synthesis) instead of server side playback.

The class uses only livekit.rtc primitives (Room events, AudioStream,
AudioSource, AudioFrame, AudioResampler, publish_data), so a fake room with
the same few methods is enough to test it. The worker (worker.py) connects the
room as the agent and hands it over; this class never opens a connection.

Wire format (kept identical to /ws/voice so the page has one handler):

    Agent to browser, data topic "va-events", JSON:
        status (calibrating | listening | thinking | speaking),
        partial_transcript, final_transcript, agent_response_chunk,
        agent_response, speaking_sentence, speaking_done, user_name,
        end_chat, error, plus {"type": "interrupt"} (barge in).
    Browser to agent, data topic "va-control", JSON:
        {"type": "set_voice", "voice_style": "female_natural" | "male"}
        {"type": "mute", "muted": true | false}
    Audio: a track named "agent-voice", 24000 Hz mono.

Audio flow. One ingest task turns every LiveKit frame into 16 kHz float32 mono
and cuts it into 100 ms blocks. A router task hands each block to whichever
stage currently wants audio (the "mode"):
    drop          nothing wants it (the greeting plays, the user is muted)
    calibrate     1 s of ambient audio sets the VAD noise floor
    listen        blocks are regrouped into stt.CHUNK_SECONDS chunks and fed to
                  transcribe_stream, like server.py's chunk source
    interruptible the agent is thinking or speaking: blocks go to a barge in
                  detector only
Barge in: while interruptible, VAD keeps running on the user's microphone. A
short run of speech blocks (about 200 ms, so a cough or a click does not cut
the assistant off) cancels reply generation and synthesis, clears the audio
source queue so the buffered speech stops at once, sends {"type": "interrupt"}
and switches to listen, seeding the next utterance with the last few blocks so
the first syllable that caused the interruption is not lost. Echo from the
assistant's own voice is removed by the browser's echo cancellation; this
class does not try to.
"""

import asyncio
import collections
import json
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
from livekit import rtc

from i18n import t
from vad import SAMPLE_RATE

from .errors import SessionError

EVENTS_TOPIC = "va-events"
CONTROL_TOPIC = "va-control"
AGENT_TRACK_NAME = "agent-voice"
AGENT_SAMPLE_RATE = 24000
VOICE_STYLES = ("female_natural", "male")

BLOCK_SECONDS = 0.1
BLOCK_SAMPLES = int(SAMPLE_RATE * BLOCK_SECONDS)
PREROLL_BLOCKS = 4  # 400 ms kept so the word that interrupted is not lost
OUTPUT_FRAME_MS = 20
# Reliable data packets are limited to about 15 KB; a cumulative reply longer
# than this is cut for the live display only (the final agent_response is cut
# the same way, the full text was still spoken).
MAX_EVENT_TEXT_CHARS = 12000


def _with_project_voice_settings(synthesize_stream):
    """voicekit takes its provider credentials as a settings object and does
    not read the project's config itself (it must stay liftable). Without
    them the session would only ever reach edge-tts, never the configured
    cloud voice that /ws/voice and the /api/tts API use, so the same chain is
    built here from config once."""
    try:
        import functools

        import config
        from voicekit.settings import VoicekitSettings

        settings = VoicekitSettings(
            google_credentials_path=config.GOOGLE_APPLICATION_CREDENTIALS,
            azure_key=config.AZURE_SPEECH_KEY,
            azure_region=config.AZURE_SPEECH_REGION,
            cloud_provider=config.CLOUD_SPEECH_PROVIDER,
            local_voice_dir=str(config.VOICEKIT_LOCAL_VOICE_DIR),
            max_text_chars=config.VOICEKIT_MAX_TEXT_CHARS,
        )
        if config.VOICE_TIER2_INFERENCE:
            # Tier 2 speaks with LiveKit Inference TTS first (Sinhala only exists on Inworld);
            # the rest of the chain stays behind it.
            settings.provider_order = ("livekit_inference",) + tuple(settings.provider_order)
        return functools.partial(synthesize_stream, settings=settings)
    except Exception:
        return synthesize_stream


@dataclass
class SessionSettings:
    """What the token said about this conversation. Empty language means
    auto detect, like language_hint null on /ws/voice."""

    language: str = ""
    voice_style: str = "female_natural"
    user_name: str = ""
    verified: bool = False
    # OTP verified (customer service allowed), set by the token endpoint.
    access_verified: bool = False
    greet: bool = True
    # The call moved here from another voice tier (voice_router): say the notice, not the introduction.
    handover: bool = False
    participant_identity: Optional[str] = None
    user_id: Optional[int] = None
    phone_e164: Optional[str] = None


@dataclass
class Bindings:
    """The pieces of server.py the session reuses without importing it (an
    import would be circular when server.py includes this package's router,
    and would build a second app). The lead builds one with
    Bindings.from_server(server_module); tests pass stubs."""

    orchestrator: Any  # respond_stream(text, user_name, language) and last_mode
    greeting: Callable[[Optional[str], Optional[str]], tuple]
    looks_like_hallucination: Callable[[str], bool]
    sounds_like_closing_signal: Callable[[str], bool]
    sounds_negative: Callable[[str], bool]
    closing_question: Callable[[Optional[str]], str]
    closing_goodbye: Callable[[Optional[str]], str]
    extract_user_name: Callable[[str], Optional[str]]

    @classmethod
    def from_server(cls, server_module) -> "Bindings":
        """Wires the existing helpers of server.py (the same functions the
        WebSocket path uses, so both transports greet, detect closings and
        filter hallucinations identically)."""
        m = server_module
        return cls(
            orchestrator=m.orchestrator,
            greeting=m._greeting,
            looks_like_hallucination=m._looks_like_transcription_hallucination,
            sounds_like_closing_signal=m._sounds_like_closing_signal,
            sounds_negative=m._sounds_negative,
            closing_question=lambda lang: m._in_language(m._CLOSING_QUESTION_TEXT, lang),
            closing_goodbye=lambda lang: m._in_language(m._CLOSING_GOODBYE_TEXT, lang),
            extract_user_name=m._extract_user_name,
        )


class _ChunkQueueSource:
    """A synchronous iterator over a thread safe queue: the event loop puts
    items in, a worker thread pulls them with a blocking get. Same shape as
    server.py's _ChunkQueueSource, which is what transcribe_stream (audio
    chunks) and the synthesizer (reply text pieces) expect."""

    _SENTINEL = object()

    def __init__(self):
        self._queue: "queue.Queue" = queue.Queue()

    def put(self, item):
        self._queue.put(item)

    def close(self):
        self._queue.put(self._SENTINEL)

    def __iter__(self):
        return self

    def __next__(self):
        item = self._queue.get()
        if item is self._SENTINEL:
            raise StopIteration
        return item


@dataclass
class _TurnState:
    first_audio_at: Optional[float] = None


def _is_sentence_start(event) -> bool:
    return hasattr(event, "text") and hasattr(event, "sentence_index") and not hasattr(event, "start_ms")


def _is_audio(event) -> bool:
    return hasattr(event, "pcm") and hasattr(event, "sample_rate")


class LiveKitVoiceSession:
    """One conversation. Construct with a connected room, call run() and
    await it; it returns when the conversation ends (closing goodbye, the user
    leaving, the room closing, aclose())."""

    def __init__(
        self,
        room,
        settings: SessionSettings,
        bindings: Bindings,
        *,
        audio_stream_factory: Optional[Callable[[Any], Any]] = None,
        audio_output_factory: Optional[Callable[[], Any]] = None,
        vad_factory: Optional[Callable[[], Any]] = None,
        transcribe_fn: Optional[Callable[..., Any]] = None,
        synth_fn: Optional[Callable[..., Any]] = None,
        chunk_seconds: Optional[float] = None,
        barge_in_debounce_ms: int = 200,
        calibration_seconds: float = 1.0,
        mic_wait_seconds: float = 60.0,
    ):
        self._room = room
        self.settings = settings
        self._b = bindings
        self._audio_stream_factory = audio_stream_factory
        self._audio_output_factory = audio_output_factory
        self._vad_factory = vad_factory
        self._transcribe_fn = transcribe_fn
        self._synth_fn = synth_fn
        self._chunk_seconds = chunk_seconds
        self._barge_blocks_needed = max(1, round(barge_in_debounce_ms / (BLOCK_SECONDS * 1000)))
        self._calibration_seconds = calibration_seconds
        self._mic_wait_seconds = mic_wait_seconds

        self._voice_style = settings.voice_style if settings.voice_style in VOICE_STYLES else VOICE_STYLES[0]
        self._muted = False
        self._vad = None
        self._source = None
        self._target_identity: Optional[str] = settings.participant_identity
        self._ingest_task: Optional[asyncio.Task] = None
        self._router_task: Optional[asyncio.Task] = None
        self._resamplers: dict = {}
        self._pending = np.zeros(0, dtype=np.float32)

        self._blocks: "asyncio.Queue" = asyncio.Queue()
        self._mode = "drop"
        self._calibration: list = []
        self._calibrated = asyncio.Event()
        self._listen_buf: list = []
        self._listen_samples = 0
        self._chunk_source: Optional[_ChunkQueueSource] = None
        self._preroll: "collections.deque" = collections.deque(maxlen=PREROLL_BLOCKS)
        self._barge = asyncio.Event()
        self._barge_run = 0
        self._closed = asyncio.Event()
        self._handlers: list = []
        self._send_failed_logged = False
        self.last_turn_latency: Optional[dict] = None

    # ------------------------------------------------------------------ setup

    def _resolve_defaults(self):
        """Imports the heavy defaults only now, so importing this module (for
        the token tests, say) does not load Whisper."""
        if self._vad_factory is None:
            from vad import VoiceActivityDetector

            self._vad_factory = VoiceActivityDetector
        if self._transcribe_fn is None:
            from stt import transcribe_stream

            self._transcribe_fn = transcribe_stream
        if self._chunk_seconds is None:
            try:
                from stt import CHUNK_SECONDS

                self._chunk_seconds = CHUNK_SECONDS
            except Exception:
                self._chunk_seconds = 0.5
        if self._synth_fn is None:
            try:
                from voicekit.core import synthesize_stream
            except Exception as exc:
                raise SessionError(f"voicekit is not available, so the assistant cannot speak ({exc})") from exc
            self._synth_fn = _with_project_voice_settings(synthesize_stream)
        if self._audio_stream_factory is None:
            self._audio_stream_factory = lambda track: rtc.AudioStream.from_track(
                track=track, sample_rate=SAMPLE_RATE, num_channels=1
            )
        if self._audio_output_factory is None:
            self._audio_output_factory = self._publish_real_track

    async def _publish_real_track(self):
        """Creates the 24 kHz mono source, wraps it in the "agent-voice" track
        and publishes it. A smallish queue (400 ms) keeps spoken sentence
        events close to the audio and makes barge in cut quickly."""
        source = rtc.AudioSource(AGENT_SAMPLE_RATE, 1, queue_size_ms=400)
        track = rtc.LocalAudioTrack.create_audio_track(AGENT_TRACK_NAME, source)
        await self._room.local_participant.publish_track(
            track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        )
        return source

    def _attach_room_events(self):
        def on(event, callback):
            self._room.on(event, callback)
            self._handlers.append((event, callback))

        on("track_subscribed", self._on_track_subscribed)
        on("data_received", self._on_data_received)
        on("participant_disconnected", self._on_participant_disconnected)
        on("disconnected", lambda *args: self._closed.set())
        # The user may have published before this class was attached.
        for participant in list(getattr(self._room, "remote_participants", {}).values()):
            for publication in list(getattr(participant, "track_publications", {}).values()):
                track = getattr(publication, "track", None)
                if track is not None:
                    self._on_track_subscribed(track, publication, participant)

    def _on_track_subscribed(self, track, publication, participant):
        if getattr(track, "kind", None) != rtc.TrackKind.KIND_AUDIO:
            return
        if self._target_identity is None:
            self._target_identity = participant.identity
        if participant.identity != self._target_identity or self._ingest_task is not None:
            return
        self._ingest_task = asyncio.ensure_future(self._ingest(track))

    def _on_participant_disconnected(self, participant):
        if participant.identity == self._target_identity:
            self._closed.set()

    def _on_data_received(self, packet):
        if getattr(packet, "topic", "") != CONTROL_TOPIC:
            return
        sender = getattr(packet, "participant", None)
        if self._target_identity and sender is not None and sender.identity != self._target_identity:
            return
        try:
            payload = json.loads(bytes(packet.data).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return
        kind = payload.get("type")
        if kind == "set_voice" and payload.get("voice_style") in VOICE_STYLES:
            self._voice_style = payload["voice_style"]
        elif kind == "mute":
            self._muted = payload.get("muted") is True
            self._barge_run = 0

    # ----------------------------------------------------------------- events

    async def _send(self, payload: dict):
        """Publishes one event on va-events. A failed publish (room closing)
        is logged once and never stops the conversation."""
        text = payload.get("text")
        if isinstance(text, str) and len(text) > MAX_EVENT_TEXT_CHARS:
            payload = {**payload, "text": text[-MAX_EVENT_TEXT_CHARS:]}
        try:
            await self._room.local_participant.publish_data(
                json.dumps(payload, ensure_ascii=False), reliable=True, topic=EVENTS_TOPIC
            )
        except Exception as exc:
            if not self._send_failed_logged:
                self._send_failed_logged = True
                print(f"LiveKit: could not publish an event ({exc.__class__.__name__}); further failures are not logged.")

    async def _status(self, state: str):
        await self._send({"type": "status", "state": state})

    # ------------------------------------------------------------------ audio

    def _to_16k_mono(self, frame) -> np.ndarray:
        """LiveKit frame to float32 mono at SAMPLE_RATE. The real AudioStream
        is asked for 16 kHz mono already, so this is a plain int16 to float
        conversion; the resampling branch only runs for a source that ignores
        that request (and for fakes)."""
        pcm = np.frombuffer(frame.data, dtype=np.int16)
        channels = getattr(frame, "num_channels", 1) or 1
        if channels > 1:
            pcm = pcm[: (pcm.size // channels) * channels].reshape(-1, channels).mean(axis=1).astype(np.int16)
        rate = getattr(frame, "sample_rate", SAMPLE_RATE)
        if rate != SAMPLE_RATE:
            pcm = self._resample_int16(pcm, rate, SAMPLE_RATE)
        return pcm.astype(np.float32) / 32768.0

    def _resample_int16(self, pcm: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
        key = (source_rate, target_rate)
        resampler = self._resamplers.get(key)
        if resampler is None:
            resampler = rtc.AudioResampler(source_rate, target_rate, num_channels=1)
            self._resamplers[key] = resampler
        frame = rtc.AudioFrame(pcm.tobytes(), source_rate, 1, pcm.size)
        out = resampler.push(frame)
        if not out:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate([np.frombuffer(f.data, dtype=np.int16) for f in out])

    async def _ingest(self, track):
        """Reads the user's microphone track and slices it into 100 ms blocks
        for the router."""
        stream = self._audio_stream_factory(track)
        try:
            async for event in stream:
                samples = self._to_16k_mono(event.frame)
                if samples.size == 0:
                    continue
                self._pending = np.concatenate([self._pending, samples])
                while self._pending.size >= BLOCK_SAMPLES:
                    self._blocks.put_nowait(self._pending[:BLOCK_SAMPLES].copy())
                    self._pending = self._pending[BLOCK_SAMPLES:]
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"LiveKit: microphone stream ended ({exc.__class__.__name__}: {exc}).")
        finally:
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:
                    pass

    async def _route_audio(self):
        while True:
            block = await self._blocks.get()
            if self._muted:
                self._barge_run = 0
                continue
            mode = self._mode
            if mode == "calibrate":
                self._calibration.append(block)
                if sum(b.size for b in self._calibration) >= int(self._calibration_seconds * SAMPLE_RATE):
                    try:
                        self._vad.calibrate_noise_floor(np.concatenate(self._calibration))
                    except Exception as exc:
                        print(f"LiveKit: calibration failed ({exc}); using the default noise floor.")
                    self._calibration = []
                    self._calibrated.set()
            elif mode == "listen":
                self._listen_buf.append(block)
                self._listen_samples += block.size
                if self._listen_samples >= int(self._chunk_seconds * SAMPLE_RATE):
                    self._chunk_source.put(np.concatenate(self._listen_buf))
                    self._listen_buf, self._listen_samples = [], 0
            elif mode == "interruptible":
                self._preroll.append(block)
                if self._vad.is_speech(block):
                    self._barge_run += 1
                    if self._barge_run >= self._barge_blocks_needed:
                        self._barge_run = 0
                        # Switch to listening right here (single threaded, so
                        # nothing slips between) and keep what was just heard.
                        self._enter_listen(seed=list(self._preroll))
                        self._barge.set()
                else:
                    self._barge_run = 0

    def _enter_listen(self, seed: Optional[list] = None):
        self._chunk_source = _ChunkQueueSource()
        self._listen_buf = list(seed or [])
        self._listen_samples = sum(b.size for b in self._listen_buf)
        self._preroll.clear()
        self._barge_run = 0
        self._mode = "listen"

    # ------------------------------------------------------------ STT and LLM

    async def _run_one_utterance(self, language_hint: Optional[str]):
        """transcribe_stream on a thread, events bridged into asyncio, each
        partial and the final sent to the page the moment it exists (the
        structure of server.py's _run_one_utterance, with the audio coming
        from the router instead of a socket). Returns (text, language)."""
        loop = asyncio.get_running_loop()
        source = self._chunk_source
        events: "asyncio.Queue" = asyncio.Queue()

        def _worker():
            try:
                for event in self._transcribe_fn(source, language_hint=language_hint, vad=self._vad):
                    loop.call_soon_threadsafe(events.put_nowait, event)
                    if event.is_final:
                        break
            except Exception as exc:
                loop.call_soon_threadsafe(events.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(events.put_nowait, None)

        threading.Thread(target=_worker, daemon=True, name="livekit-stt").start()

        final_text, final_language = None, None
        try:
            while True:
                getter = asyncio.ensure_future(events.get())
                closer = asyncio.ensure_future(self._closed.wait())
                try:
                    await asyncio.wait({getter, closer}, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    closer.cancel()
                if not getter.done():
                    getter.cancel()
                    break  # the session ended (user left); the source is closed below
                item = getter.result()
                if item is None:
                    break
                if isinstance(item, Exception):
                    await self._send({"type": "error", "message": f"Transcription error: {item}"})
                    break
                if item.is_final:
                    final_text, final_language = item.text, item.language
                    if item.text:
                        await self._send({"type": "final_transcript", "text": item.text, "language": item.language})
                    break
                await self._send({"type": "partial_transcript", "text": item.text, "language": item.language})
        finally:
            source.close()
        return final_text, final_language

    def _access(self):
        """Guest unless the signed token said this caller finished OTP.
        Resolves the verified user's phone_e164 from their user-<id> identity
        in the local database without exposing the phone number in the JWT."""
        from access import GUEST, AccessContext

        if self.settings.access_verified:
            uid = self.settings.user_id
            phone = self.settings.phone_e164
            ident = self.settings.participant_identity or self._target_identity or ""
            if uid is None and ident.startswith("user-") and ident[5:].isdigit():
                uid = int(ident[5:])
            if not phone and uid is not None:
                try:
                    from auth.users import get_user

                    user = get_user(uid)
                    if user is not None and getattr(user, "phone_e164", None):
                        phone = user.phone_e164
                except Exception:
                    pass
            return AccessContext.verified(uid, self.settings.user_name or None, phone, self.settings.language or None)
        return GUEST

    async def _stream_agent_reply(self, text: str, user_name: Optional[str], language: Optional[str], cancel):
        """Orchestrator.respond_stream on a thread, chunks bridged into the
        loop (server.py's _stream_agent_reply, plus a cancel flag so barge in
        stops pulling from the generator)."""
        loop = asyncio.get_running_loop()
        chunks: "asyncio.Queue" = asyncio.Queue()

        def _put(item):
            try:
                if not loop.is_closed():
                    loop.call_soon_threadsafe(chunks.put_nowait, item)
            except RuntimeError:
                pass

        def _worker():
            generator = None
            try:
                generator = self._b.orchestrator.respond_stream(text, user_name, language, access=self._access())
                for chunk in generator:
                    if cancel.is_set():
                        break
                    _put(chunk)
            except Exception as exc:
                _put(exc)
            finally:
                close = getattr(generator, "close", None)
                if close is not None:
                    try:
                        close()
                    except Exception:
                        pass
                _put(None)

        threading.Thread(target=_worker, daemon=True, name="livekit-agent").start()
        while True:
            item = await chunks.get()
            if item is None:
                return
            if isinstance(item, Exception):
                raise item
            yield item

    # ----------------------------------------------------------------- speech

    def _begin_speech(self, language: Optional[str], cancel):
        """Starts synthesis on a thread. Returns (text_in, publisher_task,
        state): feed reply text into text_in, close it when the text is
        complete, and await the publisher to know the audio has played."""
        loop = asyncio.get_running_loop()
        text_in = _ChunkQueueSource()
        synth_events: "asyncio.Queue" = asyncio.Queue()
        voice = self._voice_style
        synth_fn = self._synth_fn
        # voicekit knows en, si and ta; a language Whisper guessed outside them
        # would be refused, so speak it as English like server.py's fallback.
        language = language if language in ("en", "si", "ta") else "en"

        def _put_synth(item):
            try:
                if not loop.is_closed():
                    loop.call_soon_threadsafe(synth_events.put_nowait, item)
            except RuntimeError:
                pass

        def _worker():
            generator = None
            try:
                generator = synth_fn(text_in, language=language, voice=voice)
                for event in generator:
                    if cancel.is_set():
                        break
                    _put_synth(event)
            except Exception as exc:
                _put_synth(exc)
            finally:
                close = getattr(generator, "close", None)
                if close is not None:
                    try:
                        close()
                    except Exception:
                        pass
                _put_synth(None)

        threading.Thread(target=_worker, daemon=True, name="livekit-tts").start()
        state = _TurnState()
        publisher = asyncio.ensure_future(self._publish_synthesis(synth_events, state))
        return text_in, publisher, state

    def _resample_for_output(self, pcm: np.ndarray, rate: int) -> np.ndarray:
        if rate == AGENT_SAMPLE_RATE:
            return pcm
        return self._resample_int16(pcm, rate, AGENT_SAMPLE_RATE)

    async def _publish_synthesis(self, synth_events: "asyncio.Queue", state: _TurnState):
        """Turns voicekit events into published audio and UI events. A sentence
        is announced (speaking_sentence) right before its first audio chunk is
        handed to the source, and "speaking" goes out with the first one, the
        order /ws/voice uses. Finally waits until the source has played
        everything out, so speaking_done is true."""
        sentences: dict = {}
        announced: set = set()
        carry = b""
        leftover = np.zeros(0, dtype=np.int16)
        frame_samples = AGENT_SAMPLE_RATE * OUTPUT_FRAME_MS // 1000
        try:
            while True:
                event = await synth_events.get()
                if event is None:
                    break
                if isinstance(event, Exception):
                    await self._send({"type": "error", "message": f"TTS error: {event}"})
                    break
                if _is_sentence_start(event):
                    sentences[event.sentence_index] = event.text
                    continue
                if not _is_audio(event):
                    continue  # word timings and the done marker are not used here
                data = carry + bytes(event.pcm)
                usable = len(data) - (len(data) % 2)
                carry = data[usable:]
                if usable == 0:
                    continue
                pcm = self._resample_for_output(np.frombuffer(data[:usable], dtype=np.int16), event.sample_rate)
                if event.sentence_index not in announced:
                    announced.add(event.sentence_index)
                    if state.first_audio_at is None:
                        state.first_audio_at = time.time()
                        await self._status("speaking")
                    if event.sentence_index in sentences:
                        await self._send({"type": "speaking_sentence", "text": sentences[event.sentence_index]})
                leftover = np.concatenate([leftover, pcm])
                while leftover.size >= frame_samples:
                    await self._capture(leftover[:frame_samples])
                    leftover = leftover[frame_samples:]
            if leftover.size:
                await self._capture(leftover)
            await self._source.wait_for_playout()
        except asyncio.CancelledError:
            raise

    async def _capture(self, samples: np.ndarray):
        frame = rtc.AudioFrame(samples.astype(np.int16).tobytes(), AGENT_SAMPLE_RATE, 1, samples.size)
        await self._source.capture_frame(frame)

    async def _speak_streamed_reply(self, text: str, user_name, language, cancel):
        """Generates the reply and speaks it at once: each chunk the agent
        yields goes straight to the synthesizer, which cuts sentences and
        starts audio while later sentences are still being written (the
        point of server.py's _speak_streamed_reply, with voicekit in place of
        tts.speak_stream). Returns (reply_text, agent_seconds, first_audio_at)."""
        text_in, publisher, state = self._begin_speech(language, cancel)
        agent_start = time.time()
        reply_text = ""
        try:
            try:
                async for chunk in self._stream_agent_reply(text, user_name, language, cancel):
                    reply_text += chunk
                    await self._send({"type": "agent_response_chunk", "text": reply_text})
                    text_in.put(chunk)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._send({"type": "error", "message": f"Agent error: {exc}"})
            finally:
                text_in.close()
            agent_time = time.time() - agent_start
            if reply_text:
                await self._send(
                    {
                        "type": "agent_response",
                        "text": reply_text,
                        "mode": getattr(self._b.orchestrator, "last_mode", ""),
                    }
                )
            await publisher
        except BaseException:
            publisher.cancel()
            text_in.close()
            raise
        await self._send({"type": "speaking_done"})
        return reply_text, agent_time, state.first_audio_at

    async def _speak_canned(self, text: str, language: Optional[str], mode: str, cancel):
        """The introduction and the closing lines: text known up front, so no
        agent call (server.py's _speak_canned_reply)."""
        await self._status("thinking")
        await self._send({"type": "agent_response", "text": text, "mode": mode})
        text_in, publisher, _state = self._begin_speech(language, cancel)
        try:
            text_in.put(text)
            text_in.close()
            await publisher
        except BaseException:
            publisher.cancel()
            text_in.close()
            raise
        await self._send({"type": "speaking_done"})

    async def _run_interruptible(self, make_coro, cancel, interruptible: bool):
        """Runs a speaking coroutine and watches for barge in and for the
        session ending. Returns (result, interrupted). On interruption the
        generation and synthesis are cancelled, the audio queue is cleared
        (so buffered speech stops now) and "interrupt" is sent."""
        self._barge.clear()
        self._mode = "interruptible" if interruptible else "drop"
        self._barge_run = 0
        self._preroll.clear()
        task = asyncio.ensure_future(make_coro())
        barge_task = asyncio.ensure_future(self._barge.wait())
        closed_task = asyncio.ensure_future(self._closed.wait())
        try:
            done, _ = await asyncio.wait({task, barge_task, closed_task}, return_when=asyncio.FIRST_COMPLETED)
            if task in done:
                return task.result(), False
            cancel.set()
            task.cancel()
            try:
                await task
            except BaseException:
                pass
            try:
                self._source.clear_queue()
            except Exception:
                pass
            if barge_task in done:
                await self._send({"type": "interrupt"})
                print("LiveKit: barge in, the assistant stopped speaking.")
            return None, True
        finally:
            barge_task.cancel()
            closed_task.cancel()

    # ------------------------------------------------------------------- main

    async def run(self):
        self._resolve_defaults()
        self._attach_room_events()
        self._source = await self._maybe_await(self._audio_output_factory())
        self._router_task = asyncio.ensure_future(self._route_audio())
        try:
            await self._conversation()
        finally:
            await self.aclose()

    @staticmethod
    async def _maybe_await(value):
        return await value if asyncio.iscoroutine(value) or isinstance(value, asyncio.Future) else value

    async def _conversation(self):
        b, s = self._b, self.settings
        language_hint = s.language or None
        user_name = s.user_name or None
        verified = s.verified

        # The introduction first, with the mic ignored, then calibration, in
        # the order and for the reason server.py documents (the microphone
        # would otherwise hear the introduction as room noise).
        if s.greet:
            text, language = b.greeting(user_name, language_hint)
            if s.handover:
                from voice_router import make_notice

                language = language if language in ("en", "si", "ta") else "en"
                text = make_notice(language)
            _, _ = await self._run_interruptible(
                lambda: self._speak_canned(text, language, "Introduction", threading.Event()),
                threading.Event(),
                interruptible=False,
            )
            if self._closed.is_set():
                return

        self._vad = self._vad_factory()
        self._calibration = []
        self._calibrated.clear()
        self._mode = "calibrate"
        await self._status("calibrating")
        calibrated = asyncio.ensure_future(self._calibrated.wait())
        closed = asyncio.ensure_future(self._closed.wait())
        try:
            await asyncio.wait({calibrated, closed}, timeout=self._mic_wait_seconds, return_when=asyncio.FIRST_COMPLETED)
        finally:
            calibrated.cancel()
            closed.cancel()
        if self._closed.is_set():
            return
        if not self._calibrated.is_set():
            await self._send({"type": "error", "message": t("voice.no_mic_audio", self.settings.language or None)})
            raise SessionError("no microphone audio within the wait limit")

        awaiting_close_confirmation = False
        while not self._closed.is_set():
            turn_start = time.time()
            if self._mode != "listen":
                self._enter_listen()
            await self._status("listening")

            final_text, detected_language = await self._run_one_utterance(language_hint)
            # The utterance's chunk source is closed now; stop feeding it.
            self._mode = "drop"
            if self._closed.is_set():
                return
            if not final_text:
                if self._mode != "listen":
                    self._enter_listen()
                continue
            if b.looks_like_hallucination(final_text):
                print(f"Turn skipped: transcript '{final_text}' looks like a Whisper hallucination, not a real utterance.")
                self._enter_listen()
                continue
            reply_start = time.time()
            reply_language = language_hint or detected_language

            introduced = None if verified else b.extract_user_name(final_text)
            if introduced and introduced != user_name:
                user_name = introduced
                await self._send({"type": "user_name", "name": user_name})

            if awaiting_close_confirmation:
                if b.sounds_negative(final_text) or b.sounds_like_closing_signal(final_text):
                    text = b.closing_goodbye(reply_language)
                    await self._run_interruptible(
                        lambda: self._speak_canned(text, reply_language, "closing", threading.Event()),
                        threading.Event(),
                        interruptible=False,
                    )
                    await self._send({"type": "end_chat"})
                    return
                awaiting_close_confirmation = False
            elif b.sounds_like_closing_signal(final_text):
                text = b.closing_question(reply_language)
                cancel = threading.Event()
                _, interrupted = await self._run_interruptible(
                    lambda: self._speak_canned(text, reply_language, "closing", cancel), cancel, interruptible=True
                )
                awaiting_close_confirmation = True
                if not interrupted:
                    self._enter_listen()
                continue

            await self._status("thinking")
            cancel = threading.Event()
            result, interrupted = await self._run_interruptible(
                lambda: self._speak_streamed_reply(final_text, user_name, reply_language, cancel),
                cancel,
                interruptible=True,
            )
            if interrupted:
                continue  # already listening, seeded with the interrupting speech
            reply_text, agent_time, first_audio_at = result
            self._enter_listen()
            if not reply_text:
                continue
            first_audio = f"{first_audio_at - reply_start:.2f}s" if first_audio_at else "none"
            self.last_turn_latency = {
                "first_audio_s": (first_audio_at - reply_start) if first_audio_at else None,
                "agent_s": agent_time,
                "total_s": time.time() - turn_start,
            }
            print(
                f"LiveKit turn latency: first audio {first_audio} after the final transcript, "
                f"agent {agent_time:.2f}s, total {time.time() - turn_start:.2f}s, "
                f"mode={getattr(self._b.orchestrator, 'last_mode', '')}"
            )

    async def aclose(self):
        """Stops every task and releases the audio source. Safe to call twice."""
        self._closed.set()
        if self._chunk_source is not None:
            self._chunk_source.close()
        for task in (self._ingest_task, self._router_task):
            if task is not None and not task.done():
                task.cancel()
        for task in (self._ingest_task, self._router_task):
            if task is not None:
                try:
                    await task
                except BaseException:
                    pass
        for event, callback in self._handlers:
            try:
                self._room.off(event, callback)
            except Exception:
                pass
        self._handlers = []
        aclose = getattr(self._source, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:
                pass


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        from ._selftest import main

        raise SystemExit(main())
    print("python -m backend.livekit_transport.session --selftest runs the session against a fake room.")
