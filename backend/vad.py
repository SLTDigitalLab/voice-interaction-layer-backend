"""Noise robust voice activity detection for the real time voice pipeline.

Replaces the prototype's fixed RMS amplitude threshold with two layers that
work together:

1. WebRTC VAD (the `webrtcvad-wheels` package, a prebuilt Windows wheel of
   Google's WebRTC voice activity classifier): a small, fast, pretrained
   model that looks at the spectral shape of each short audio frame, not
   just its loudness. This makes it far more robust to steady background
   noise (fans, hum, traffic) than a single energy threshold, because
   noise and speech can be equally loud but look different spectrally.
2. An ambient noise calibration step, run once per session before the user
   starts talking, which measures the room's baseline energy and sets an
   adaptive noise floor from it. WebRTC VAD's classification is combined
   with this floor, so quiet speech in a quiet room is not swallowed by a
   floor tuned for a noisy one, and a loud non speech burst is less likely
   to pass in a noisy one.

A third piece, `denoise_audio()`, applies spectral gating noise reduction
(via the `noisereduce` library) to the audio actually sent to transcription.
This is a separate concern from `is_speech()`: VAD decides *when* speech is
happening, denoising cleans up *what* gets recorded, so background noise
baked into a real speech segment does not degrade transcription accuracy.
"""

from typing import Optional

import numpy as np
import webrtcvad

SAMPLE_RATE = 16000
FRAME_MS = 30  # webrtcvad only accepts 10, 20, or 30 ms frames
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)


class VoiceActivityDetector:
    """Stateful VAD: create one instance per session, calibrate it once,
    then call is_speech() per audio chunk."""

    def __init__(self, aggressiveness: int = 2):
        # aggressiveness 0-3, higher values filter out more non speech at
        # the cost of being more willing to clip quiet speech. 2 is a
        # reasonable middle ground for a conversational assistant.
        self._vad = webrtcvad.Vad(aggressiveness)
        self.noise_floor_rms = 0.01  # sane default until calibrate_noise_floor() runs
        self.calibrated = False

    def calibrate_noise_floor(self, sample_audio: np.ndarray) -> float:
        """Measures ambient noise energy from a short sample recorded at
        session start with nobody speaking, and sets the adaptive energy
        floor used alongside WebRTC VAD's per frame decision. Returns the
        floor that was set so the caller can log or display it."""
        if sample_audio is None or sample_audio.size == 0:
            self.calibrated = False
            return self.noise_floor_rms

        rms = float(np.sqrt(np.mean(np.square(sample_audio))))
        # Speech must clear ambient noise by a margin, not just cross zero,
        # otherwise a noisy room re-triggers on its own background level.
        # Capped at 0.035 so speaking during the 1 s calibration window cannot
        # set an unreachable threshold for the rest of the call.
        self.noise_floor_rms = min(max(rms * 2.5, 0.003), 0.035)
        self.calibrated = True
        return self.noise_floor_rms

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """audio_chunk: float32 mono numpy array sampled at SAMPLE_RATE.
        Internally split into WebRTC VAD's fixed size frames; the chunk
        counts as speech only if both the calibrated energy floor is
        cleared AND a majority of its frames are classified as speech by
        WebRTC VAD, so a loud non speech sound and a quiet steady hum are
        both rejected on their own terms."""
        if audio_chunk is None or audio_chunk.size == 0:
            return False

        rms = float(np.sqrt(np.mean(np.square(audio_chunk))))
        if rms < self.noise_floor_rms:
            return False

        pcm16 = np.clip(audio_chunk * 32768.0, -32768, 32767).astype(np.int16)
        num_frames = len(pcm16) // FRAME_SAMPLES
        if num_frames == 0:
            # Chunk shorter than one VAD frame; the energy gate above is
            # all we can apply.
            return True

        speech_frames = 0
        for i in range(num_frames):
            frame = pcm16[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]
            if self._vad.is_speech(frame.tobytes(), SAMPLE_RATE):
                speech_frames += 1

        return speech_frames / num_frames >= 0.5


# Barge in while the server itself is playing the reply (/ws/voice). The sound
# comes out of the machine's own speakers, so the browser's echo cancellation has
# no reference signal and cannot remove it. The guard is therefore stricter than
# the normal end of speech detection: a frame must be well above the calibrated
# floor (the floor is already 2.5x the room's own level) and the speech must run
# for a while before anything is cut off.
BARGE_ENERGY_MARGIN = 3.0  # frame RMS must be this many times the calibrated floor
BARGE_MIN_RMS = 0.055  # and never below this absolute level (prevents laptop speaker echo self-barge-in)
BARGE_CONFIRM_MS = 300  # continuous speech needed before the reply is cut off
BARGE_GAP_FRAMES = 3  # up to 90 ms of quiet inside a word does not restart the count
BARGE_PREROLL_MS = 150  # audio kept from before the first speech frame
BARGE_HISTORY_MS = 2000


class BargeInDetector:
    """Decides whether the user really started talking over the assistant.

    feed() takes the microphone chunks (any length, float32 mono, SAMPLE_RATE)
    and returns None until the speech has run for confirm_ms, then the audio to
    start the next utterance with: from preroll_ms before the first speech
    frame up to the end of the chunk that confirmed it, so the first word is
    not lost. Only frames that clear BOTH the stricter energy margin and the
    WebRTC classifier count, so the assistant's own voice coming back through
    the microphone and steady noise do not trigger it."""

    def __init__(
        self,
        vad: "VoiceActivityDetector",
        confirm_ms: int = BARGE_CONFIRM_MS,
        energy_margin: float = BARGE_ENERGY_MARGIN,
        min_rms: float = BARGE_MIN_RMS,
        gap_frames: int = BARGE_GAP_FRAMES,
        preroll_ms: int = BARGE_PREROLL_MS,
    ):
        self._vad = vad
        self._need_frames = max(1, round(confirm_ms / FRAME_MS))
        self._margin = energy_margin
        self._min_rms = min_rms
        self._gap_frames = gap_frames
        self._preroll = int(SAMPLE_RATE * preroll_ms / 1000)
        self._max_history = int(SAMPLE_RATE * BARGE_HISTORY_MS / 1000)
        self.reset()

    def reset(self) -> None:
        self._history = np.zeros(0, dtype=np.float32)
        self._carry = np.zeros(0, dtype=np.float32)
        self._frames_total = 0  # frames examined since reset (for the position of the run start)
        self._run_frames = 0
        self._gap = 0
        self._run_start_frame: Optional[int] = None
        self._dropped = 0  # samples trimmed from the front of the history

    @property
    def threshold_rms(self) -> float:
        return max(self._vad.noise_floor_rms * self._margin, self._min_rms)

    def _frame_is_speech(self, frame: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(np.square(frame))))
        if rms < self.threshold_rms:
            return False
        pcm16 = np.clip(frame * 32768.0, -32768, 32767).astype(np.int16)
        return bool(self._vad._vad.is_speech(pcm16.tobytes(), SAMPLE_RATE))

    def feed(self, chunk: np.ndarray) -> Optional[np.ndarray]:
        if chunk is None or chunk.size == 0:
            return None
        self._history = np.concatenate([self._history, chunk])
        data = np.concatenate([self._carry, chunk])
        usable = (data.size // FRAME_SAMPLES) * FRAME_SAMPLES
        self._carry = data[usable:]
        triggered = False
        for i in range(0, usable, FRAME_SAMPLES):
            frame = data[i : i + FRAME_SAMPLES]
            if self._frame_is_speech(frame):
                if self._run_start_frame is None:
                    self._run_start_frame = self._frames_total
                self._run_frames += 1
                self._gap = 0
            elif self._run_start_frame is not None:
                self._gap += 1
                if self._gap > self._gap_frames:
                    self._run_frames, self._gap, self._run_start_frame = 0, 0, None
            self._frames_total += 1
            if self._run_frames >= self._need_frames:
                triggered = True
                break
        if triggered:
            start = self._run_start_frame * FRAME_SAMPLES - self._preroll - self._dropped
            seed = self._history[max(0, start) :].copy()
            self.reset()
            return seed
        if self._history.size > self._max_history:
            cut = self._history.size - self._max_history
            self._history = self._history[cut:]
            self._dropped += cut
        return None


_denoise_import_failed = False


def denoise_audio(audio: np.ndarray, noise_sample: Optional[np.ndarray] = None) -> np.ndarray:
    """Applies spectral gating noise reduction to captured audio before it
    is sent to transcription. Falls back to returning the audio unchanged
    if noisereduce fails for any reason (e.g. a clip too short to analyze),
    since a transcription attempt on the raw audio is better than none.
    If the library cannot even be imported, that is remembered, so every
    later partial and final transcript does not retry the failing import."""
    global _denoise_import_failed
    if _denoise_import_failed:
        return audio
    try:
        import noisereduce as nr
    except Exception as exc:
        # Known blocker on this machine: noisereduce depends on scipy, and
        # scipy's compiled linalg extension is refused by a Windows
        # Application Control policy here (ImportError, not a bug in this
        # code). Documented in KNOWN_LIMITATIONS.md. Falling back to raw
        # audio keeps the pipeline working; VAD based speech gating still
        # applies, only the extra spectral cleanup step is skipped.
        _denoise_import_failed = True
        print(f"VAD: denoise_audio unavailable ({exc}), using raw audio from now on.")
        return audio
    try:
        if noise_sample is not None and noise_sample.size > 0:
            return nr.reduce_noise(y=audio, sr=SAMPLE_RATE, y_noise=noise_sample, stationary=True)
        return nr.reduce_noise(y=audio, sr=SAMPLE_RATE, stationary=True)
    except Exception as exc:
        print(f"VAD: denoise_audio failed on this clip ({exc}), using raw audio instead.")
        return audio


def _run_standalone_test():
    """python vad.py: records a short silent calibration sample, then two
    prompted test clips (quiet, then noisy or spoken), and reports how many
    analysis blocks the new VAD flags as speech versus the old prototype's
    fixed threshold, so the improvement can be shown as evidence."""
    import sounddevice as sd

    vad = VoiceActivityDetector(aggressiveness=2)

    print("Calibrating noise floor: stay silent for 1.5 seconds...")
    calib = sd.rec(int(1.5 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    floor = vad.calibrate_noise_floor(calib[:, 0])
    print(f"Noise floor set to RMS {floor:.5f}")

    OLD_FIXED_THRESHOLD = 0.02  # matches the prototype's SPEECH_RMS_THRESHOLD

    def _test_clip(label: str, seconds: float = 4.0):
        input(f"\nPress Enter, then {label} for about {seconds:.0f} seconds...")
        clip = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()
        audio = clip[:, 0]
        block = FRAME_SAMPLES * 10  # roughly 300ms analysis blocks
        blocks = [audio[i : i + block] for i in range(0, len(audio) - block, block)]
        if not blocks:
            print(f"{label}: clip too short to analyze.")
            return
        new_speech_blocks = sum(1 for b in blocks if vad.is_speech(b))
        old_speech_blocks = sum(
            1 for b in blocks if float(np.sqrt(np.mean(np.square(b)))) >= OLD_FIXED_THRESHOLD
        )
        print(
            f"{label}: {new_speech_blocks}/{len(blocks)} blocks flagged as speech by the "
            f"new calibrated VAD, vs {old_speech_blocks}/{len(blocks)} by the old fixed "
            f"threshold ({OLD_FIXED_THRESHOLD})."
        )

    _test_clip("stay quiet, or make background noise without speaking")
    _test_clip("speak normally")

    print(
        "\nA good result: the quiet/noise clip shows fewer speech blocks under the new "
        "VAD than the old threshold (fewer false triggers), while the speech clip still "
        "registers as speech under both (real speech is not lost)."
    )


if __name__ == "__main__":
    _run_standalone_test()
