"""Hard barge in on the /ws/voice path, with synthetic audio only.

What is exercised:
  * vad.BargeInDetector: stricter speech condition (energy margin over the
    calibrated floor plus 300 ms of continuous speech), pre roll for the seed.
  * server._speak_streamed_reply with a scripted websocket (the microphone
    chunks the page now sends while the status is "speaking"), a fake model
    stream and a fake TTS that records cancellation.
  * tts.speak_stream / tts._play with a fake sounddevice: playback is stopped
    and the queued sentences are flushed.

Times are AUDIO times (chunks of 250 ms counted by the test), not wall clock,
except where a line says "real".
"""

import asyncio
import json
import sys
import threading
import time
import types

import numpy as np
import pytest

SR = 16000
CHUNK = 4000  # 250 ms, what the page sends
AMBIENT = 0.005  # calibrated room noise (RMS)
SPEECH_LEVEL = 0.12
ECHO_LEVEL = 0.02  # the assistant's own voice as the microphone hears it, speech like but quiet


# --------------------------------------------------------------- synthetic audio


def voiced(seconds, level=SPEECH_LEVEL, f0=140.0):
    """Harmonic signal with two formants and a syllable rate wobble. The WebRTC
    classifier labels it as speech (checked), unlike plain tones."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    sig = np.zeros(n)
    for k in range(1, 30):
        f = f0 * k
        if f > 4000:
            break
        amp = 1.0 / k * (1 + 3 * np.exp(-(((f - 700) / 200) ** 2)) + 2 * np.exp(-(((f - 1200) / 250) ** 2)))
        sig += amp * np.sin(2 * np.pi * f * t)
    sig *= 1 + 0.4 * np.sin(2 * np.pi * 4 * t)
    return (sig / np.sqrt(np.mean(sig**2)) * level).astype(np.float32)


def noise(seconds, level=AMBIENT, seed=3):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(seconds * SR)) * level).astype(np.float32)


def calibrated_vad():
    from vad import VoiceActivityDetector

    vad = VoiceActivityDetector()
    vad.calibrate_noise_floor(noise(1.0))
    return vad


# ------------------------------------------------------------ the detector itself


def feed_in_chunks(detector, signal, chunk=CHUNK):
    """Returns (audio_seconds_until_trigger, seed) or (None, None)."""
    for i in range(0, signal.size, chunk):
        seed = detector.feed(signal[i : i + chunk])
        if seed is not None:
            return (min(i + chunk, signal.size)) / SR, seed
    return None, None


def test_speech_triggers_within_two_chunks_and_seed_has_the_first_word():
    from vad import BargeInDetector

    detector = BargeInDetector(calibrated_vad())
    lead = noise(0.5)
    signal = np.concatenate([lead, voiced(2.0)])
    at, seed = feed_in_chunks(detector, signal)
    assert at is not None
    latency = at - lead.size / SR
    print(f"detector only: speech onset to trigger {latency * 1000:.0f} ms of audio (250 ms chunks)")
    assert latency <= 0.6
    # The seed starts about 150 ms before the first speech frame and runs to the end of the chunk.
    assert seed.size >= int(0.4 * SR)
    onset_in_seed = seed.size - int(latency * SR)
    assert onset_in_seed <= int(0.2 * SR), "the seed must reach back to the first word, not start after it"
    loud = float(np.sqrt(np.mean(seed[int(0.2 * SR) :] ** 2)))
    assert loud > 0.05


def test_noise_alone_does_not_trigger():
    from vad import BargeInDetector

    detector = BargeInDetector(calibrated_vad())
    # The room's own level, then twice as loud (a fan speeding up), for 6 s.
    for level in (AMBIENT, 2 * AMBIENT, 0.015):
        at, _ = feed_in_chunks(detector, noise(6.0, level))
        assert at is None, f"noise at {level} must not interrupt"


def test_the_assistants_own_echo_level_does_not_trigger():
    from vad import BargeInDetector

    detector = BargeInDetector(calibrated_vad())
    at, _ = feed_in_chunks(detector, voiced(8.0, ECHO_LEVEL) + noise(8.0))
    assert at is None, "speech shaped audio at echo level must not cut the reply off"


def test_a_short_cough_does_not_trigger():
    from vad import BargeInDetector

    detector = BargeInDetector(calibrated_vad())
    signal = np.concatenate([noise(0.5), voiced(0.15), noise(1.5)])
    at, _ = feed_in_chunks(detector, signal)
    assert at is None, "150 ms of speech is below the 300 ms confirmation"


def test_a_pause_inside_a_word_does_not_restart_the_count():
    from vad import BargeInDetector

    detector = BargeInDetector(calibrated_vad())
    signal = np.concatenate([noise(0.2), voiced(0.18), noise(0.06), voiced(0.4)])
    at, _ = feed_in_chunks(detector, signal, chunk=480)
    assert at is not None


# ------------------------------------------------- the /ws/voice reply with barge in


class ScriptedSocket:
    """Stands in for the WebSocket: receive() hands out one 250 ms microphone
    chunk per call (as fast as the loop allows, audio time is counted, not
    waited), send_text() records the events the page would get."""

    def __init__(self, mic):
        self.mic = mic
        self.index = 0
        self.events = []
        self.last_chunk_at = None

    async def send_text(self, text):
        self.events.append(json.loads(text))

    async def receive(self):
        await asyncio.sleep(0.004)
        chunk = self.mic(self.index)
        self.index += 1
        self.last_chunk_at = time.monotonic()
        return {"type": "websocket.receive", "bytes": chunk.astype(np.float32).tobytes()}

    def types(self):
        return [e["type"] for e in self.events]


class FakeTts:
    """Replaces tts.speak_stream: every sentence "plays" for sentence_seconds of
    real time and notices cancel like the real player does."""

    def __init__(self, sentence_seconds=0.4):
        self.sentence_seconds = sentence_seconds
        self.started = []
        self.finished = []
        self.stopped_at = None
        self.cancel_seen = False
        self.returned = threading.Event()

    def speak_stream(self, sentences, language="en", voice_style="female_natural", on_sentence_start=None, cancel=None):
        try:
            for sentence in sentences:
                if cancel is not None and cancel.is_set():
                    self.cancel_seen = True
                    self.stopped_at = time.monotonic()
                    return
                self.started.append(sentence)
                if on_sentence_start is not None:
                    on_sentence_start(sentence)
                if cancel is not None and cancel.wait(self.sentence_seconds):
                    self.cancel_seen = True
                    self.stopped_at = time.monotonic()
                    return  # cut mid sentence, the rest of the queue is dropped
                if cancel is None:
                    time.sleep(self.sentence_seconds)
                self.finished.append(sentence)
        finally:
            self.returned.set()


class FakeLlm:
    """Replaces orchestrator.respond_stream: a slow model that writes sentences
    and records whether its stream was closed before it finished."""

    def __init__(self, sentences=8, gap_seconds=0.25):
        self.total = sentences
        self.gap = gap_seconds
        self.yielded = 0
        self.closed_early = False
        self.completed = False

    def respond_stream(self, text, user_name=None, language=None, access=None, status_cb=None):
        try:
            for i in range(self.total):
                time.sleep(self.gap)
                self.yielded += 1
                yield f"Sentence number {i + 1} of the answer. "
            self.completed = True
        except GeneratorExit:
            self.closed_early = True
            raise


def run_reply(server_module, monkeypatch, mic_builder, llm, tts_fake):
    """Runs one streamed reply. mic_builder(state) returns the mic function; state
    exposes tts_fake so the microphone can react to what is being said."""
    monkeypatch.setattr(server_module, "speak_stream", tts_fake.speak_stream)
    monkeypatch.setattr(server_module.orchestrator, "respond_stream", llm.respond_stream)
    barge = {"vad": calibrated_vad(), "voice_holder": {"voice_style": "female_natural"}}
    socket = ScriptedSocket(mic_builder())

    async def go():
        return await server_module._speak_streamed_reply(
            socket, "tell me about the plans", "Guest", "en", "female_natural", None, barge
        )

    result = asyncio.run(asyncio.wait_for(go(), timeout=30))
    return result, socket, barge


def speaking_when(tts_fake, count, then):
    """Microphone: ambient noise until the TTS has started `count` sentences,
    then the user's speech. Records which chunk index the speech began on."""
    state = {"onset": None}
    speech = voiced(5.0)

    def mic(index):
        if state["onset"] is None and len(tts_fake.started) >= count:
            state["onset"] = index
        if state["onset"] is None:
            return then(index)
        offset = (index - state["onset"]) * CHUNK
        return speech[offset : offset + CHUNK] if offset + CHUNK <= speech.size else noise(0.25)

    return mic, state


@pytest.mark.parametrize(
    "where, start_after, sentences",
    [
        ("first word of the reply", 1, 8),
        ("mid reply", 4, 8),
        ("near the end of the reply", 7, 8),
    ],
)
def test_user_speech_cuts_the_reply_off(server_module, monkeypatch, where, start_after, sentences):
    tts_fake = FakeTts(sentence_seconds=0.4)
    llm = FakeLlm(sentences=sentences, gap_seconds=0.12 if start_after > 1 else 0.5)
    mic, state = speaking_when(tts_fake, start_after, lambda i: noise(0.25, AMBIENT, seed=i))

    (reply_text, _agent_time, first_audio), socket, barge = run_reply(
        server_module, monkeypatch, lambda: mic, llm, tts_fake
    )

    chunks_to_stop = socket.index - state["onset"]
    audio_latency = chunks_to_stop * 0.25
    real_stop = tts_fake.stopped_at - socket.last_chunk_at
    print(
        f"barge in {where}: speech onset to playback stop {audio_latency * 1000:.0f} ms of audio "
        f"({chunks_to_stop} chunks of 250 ms), real time from the confirming chunk to the player "
        f"stopping {real_stop * 1000:.0f} ms"
    )
    assert audio_latency < 0.6
    assert real_stop < 0.25, "the player must stop at once, not after the sentence"

    # Playback: stopped mid sentence, the queued sentences were flushed.
    assert tts_fake.cancel_seen
    assert len(tts_fake.started) <= start_after + 1
    assert tts_fake.returned.is_set()
    # Generation: the model stream was closed before it finished (or it had already finished).
    if start_after == 1:
        # The worker thread notices at the model's next piece (here at most 0.5 s), then closes the stream.
        deadline = time.monotonic() + 2
        while not llm.closed_early and time.monotonic() < deadline:
            time.sleep(0.01)
        assert llm.closed_early and not llm.completed
        assert llm.yielded < llm.total
    # The page is told with the existing event, and the speech seeds the next utterance.
    assert "interrupt" in socket.types()
    assert "speaking_done" not in socket.types()
    assert barge.get("seed") is not None and barge["seed"].size >= int(0.4 * SR)
    assert first_audio is not None


def test_noise_and_echo_leave_the_reply_alone(server_module, monkeypatch):
    for level_name, signal in (
        ("room noise", lambda i: noise(0.25, AMBIENT, seed=i)),
        ("louder noise", lambda i: noise(0.25, 3 * AMBIENT, seed=i)),
        ("own echo", None),
    ):
        tts_fake = FakeTts(sentence_seconds=0.15)
        llm = FakeLlm(sentences=4, gap_seconds=0.05)
        echo = voiced(30.0, ECHO_LEVEL)

        def mic(index, signal=signal, echo=echo):
            if signal is not None:
                return signal(index)
            offset = (index * CHUNK) % (echo.size - CHUNK)
            return echo[offset : offset + CHUNK] + noise(0.25, AMBIENT, seed=index)

        (reply_text, _t, _fa), socket, barge = run_reply(server_module, monkeypatch, lambda: mic, llm, tts_fake)
        assert "interrupt" not in socket.types(), level_name
        assert socket.types()[-1] == "speaking_done", level_name
        assert barge.get("seed") is None and not barge.get("ended")
        assert len(tts_fake.finished) == 4, level_name
        assert llm.completed and not llm.closed_early


def test_nothing_is_judged_before_the_first_audio(server_module, monkeypatch):
    """Speech arriving while the model is still thinking (no audio yet) is not a barge in."""
    tts_fake = FakeTts(sentence_seconds=0.1)
    llm = FakeLlm(sentences=2, gap_seconds=0.4)
    speech = voiced(2.0)
    mic = lambda index: speech[: CHUNK] if len(tts_fake.started) == 0 else noise(0.25, AMBIENT, seed=index)  # noqa: E731
    (reply_text, _t, _fa), socket, barge = run_reply(server_module, monkeypatch, lambda: mic, llm, tts_fake)
    assert "interrupt" not in socket.types()
    assert socket.types()[-1] == "speaking_done"


def test_the_double_never_used_path_keeps_working_without_barge_in(server_module, monkeypatch):
    """Without a calibrated detector (the existing tests' fake VAD) nothing changes."""
    tts_fake = FakeTts(sentence_seconds=0.05)
    llm = FakeLlm(sentences=2, gap_seconds=0.02)
    monkeypatch.setattr(server_module, "speak_stream", tts_fake.speak_stream)
    monkeypatch.setattr(server_module.orchestrator, "respond_stream", llm.respond_stream)
    socket = ScriptedSocket(lambda i: noise(0.25))

    async def go():
        return await server_module._speak_streamed_reply(socket, "hello", "Guest", "en", "female_natural", None)

    reply_text, _t, _fa = asyncio.run(asyncio.wait_for(go(), timeout=10))
    assert reply_text.startswith("Sentence number 1")
    assert socket.index == 0, "the socket is not read when barge in is off"
    assert socket.types()[-1] == "speaking_done"


# ------------------------------------------------------------ the real TTS player


class FakeSoundDevice(types.ModuleType):
    """Records what the player did. play() starts a clip, stop() flushes it."""

    def __init__(self):
        super().__init__("sounddevice")
        self.clips = []
        self.stop_calls = 0
        self.active_until = 0.0

    def play(self, data, samplerate):
        self.clips.append(len(data) / float(samplerate))
        self.active_until = time.monotonic() + len(data) / float(samplerate)

    def stop(self):
        self.stop_calls += 1
        self.active_until = 0.0

    def wait(self):
        while time.monotonic() < self.active_until:
            time.sleep(0.005)


def test_real_player_stops_at_once_and_flushes_the_queue(monkeypatch):
    pytest.importorskip("pyttsx3")
    import tts

    fake_sd = FakeSoundDevice()
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    prepared = []

    def fake_prepare(sentence, language, voice_style):
        prepared.append(sentence)
        clip = np.zeros(24000, dtype=np.int16)  # 1 s of audio
        return lambda: tts._play(clip, 24000)

    monkeypatch.setattr(tts, "_prepare_sentence", fake_prepare)

    started = []
    cancel = threading.Event()
    finished = threading.Event()

    def run():
        tts.speak_stream(
            ["one.", "two.", "three.", "four."], "en", "female_natural", started.append, cancel=cancel
        )
        finished.set()

    thread = threading.Thread(target=run)
    thread.start()
    deadline = time.monotonic() + 3
    while not started and time.monotonic() < deadline:
        time.sleep(0.005)
    time.sleep(0.2)  # 200 ms into the first second long clip
    cancelled_at = time.monotonic()
    cancel.set()
    assert finished.wait(2)
    stopped_after = time.monotonic() - cancelled_at
    thread.join(2)
    print(f"real player: cancel to speak_stream returned {stopped_after * 1000:.0f} ms, sd.stop called {fake_sd.stop_calls}x")
    assert stopped_after < 0.15
    assert fake_sd.stop_calls >= 1, "the audio device buffer must be flushed"
    assert fake_sd.clips == [1.0], "only the first sentence was ever handed to the device"
    assert started == ["one."], "queued sentences are dropped, not played"
    assert tts._current_cancel() is None, "the cancel context does not leak to later playback"


def test_real_player_without_cancel_behaves_as_before(monkeypatch):
    pytest.importorskip("pyttsx3")
    import tts

    fake_sd = FakeSoundDevice()
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    clip = np.zeros(2400, dtype=np.int16)
    monkeypatch.setattr(tts, "_prepare_sentence", lambda s, l, v: (lambda: tts._play(clip, 24000)))
    tts.speak_stream(["a.", "b."], "en", "female_natural")
    assert fake_sd.clips == [0.1, 0.1]
    assert fake_sd.stop_calls == 0


# ----------------------------------------------- AEC & Double-Talk Mode Tests


def test_aec_headphones_mode_instant_interruption():
    """Headphones: TTS reference is actively playing in the headphones, but no
    speaker audio leaks into the mic. The user's speech must trigger immediately."""
    from vad import BargeInDetector, EchoReference

    echo_ref = EchoReference()
    echo_ref.start_clip(voiced(6.0, level=0.20, f0=185.0), SR)

    detector = BargeInDetector(calibrated_vad(), echo_ref=echo_ref)
    lead = noise(0.5, AMBIENT)
    user_speech = voiced(2.0, level=0.12, f0=140.0)
    signal = np.concatenate([lead, user_speech])

    at, seed = feed_in_chunks(detector, signal)
    assert at is not None, "Human speech in headphone mode must trigger barge-in"
    latency = at - lead.size / SR
    print(f"Headphones mode trigger latency: {latency * 1000:.0f} ms")
    assert latency <= 0.6
    assert seed is not None and seed.size >= int(0.4 * SR)
    seed_rms = float(np.sqrt(np.mean(seed[int(0.2 * SR) :] ** 2)))
    assert seed_rms > 0.06, "Human voice in seed must remain clean and audible"


def test_aec_open_laptop_speakers_echo_alone_never_triggers():
    """Open Laptop Speakers: TTS is playing out loud. Mic picks up delayed,
    room-reflected, distorted speaker audio at loud levels (RMS ~0.13, 26x noise floor).
    AEC must cancel it so it never self-triggers."""
    from vad import BargeInDetector, EchoReference

    echo_ref = EchoReference()
    ref = voiced(10.0, level=0.22, f0=185.0)
    echo_ref.start_clip(ref, SR)

    # Simulate open laptop speaker acoustics: 180 ms delay + 64-tap room filter + nonlinearity
    delay = int(0.18 * SR)
    rir = np.zeros(64, dtype=np.float32)
    rir[0], rir[3], rir[11], rir[25] = 0.65, -0.32, 0.22, -0.15
    echo_conv = np.convolve(np.tanh(1.3 * ref) / 1.3, rir, mode="full")[: ref.size]
    echo_conv = (echo_conv / np.sqrt(np.mean(echo_conv**2)) * 0.13).astype(np.float32)

    mic_echo = np.zeros_like(ref)
    mic_echo[delay:] = echo_conv[:-delay]

    detector = BargeInDetector(calibrated_vad(), echo_ref=echo_ref)
    at, _ = feed_in_chunks(detector, mic_echo[: int(8.0 * SR)])
    assert at is None, "Loud open laptop speaker echo alone must never trigger barge-in"


def test_aec_open_laptop_speakers_double_talk_cuts_off_and_returns_clean_seed():
    """Open Laptop Speakers (Double-Talk): TTS is playing loudly. Speaker echo runs
    alone for 1.0s without triggering, then the human talks over the speaker.
    AEC must immediately detect double-talk, trigger barge-in, and return clean seed audio."""
    from vad import BargeInDetector, EchoReference

    echo_ref = EchoReference()
    ref = voiced(10.0, level=0.22, f0=185.0)
    echo_ref.start_clip(ref, SR)

    delay = int(0.18 * SR)
    rir = np.zeros(64, dtype=np.float32)
    rir[0], rir[3], rir[11], rir[25] = 0.65, -0.32, 0.22, -0.15
    echo_conv = np.convolve(np.tanh(1.3 * ref) / 1.3, rir, mode="full")[: ref.size]
    echo_conv = (echo_conv / np.sqrt(np.mean(echo_conv**2)) * 0.12).astype(np.float32)

    mic_echo = np.zeros_like(ref)
    mic_echo[delay:] = echo_conv[:-delay]

    # Human starts speaking at t = 1.0s over the ongoing loud speaker echo
    lead_s = 1.0
    human_s = 3.0
    human = voiced(human_s, level=0.12, f0=140.0)

    mic_signal = mic_echo[: int((lead_s + human_s) * SR)].copy()
    lead_samples = int(lead_s * SR)
    mic_signal[lead_samples : lead_samples + human.size] += human

    detector = BargeInDetector(calibrated_vad(), echo_ref=echo_ref)
    at, seed = feed_in_chunks(detector, mic_signal)

    assert at is not None, "Human talking over open laptop speakers must trigger barge-in"
    latency = at - lead_s
    print(f"Open speakers double-talk trigger latency: {latency * 1000:.0f} ms")
    assert 0.0 <= latency <= 0.6, "Double-talk must trigger promptly"
    assert seed is not None and seed.size >= int(0.4 * SR)
    seed_rms = float(np.sqrt(np.mean(seed[int(0.2 * SR) :] ** 2)))
    assert seed_rms > 0.05, "Seed audio must preserve the human voice"

