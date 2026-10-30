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
import threading
import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import webrtcvad

SAMPLE_RATE = 16000
FRAME_MS = 30  # webrtcvad only accepts 10, 20, or 30 ms frames
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)


def _to_mono_float32_16k(data, samplerate: int) -> np.ndarray:
    """Converts any TTS playback buffer (1D/2D, int16/float32, any sample rate)
    into a 1D float32 mono waveform at SAMPLE_RATE (16 kHz) for AEC reference."""
    if data is None:
        return np.zeros(0, dtype=np.float32)
    arr = np.asarray(data)
    if arr.size == 0:
        return np.zeros(0, dtype=np.float32)
    if arr.ndim > 1:
        arr = np.mean(arr, axis=1)
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        scale = float(max(abs(info.min), abs(info.max)))
        arr = arr.astype(np.float32) / (scale if scale > 0 else 32768.0)
    else:
        arr = arr.astype(np.float32, copy=False)
    if samplerate <= 0 or samplerate == SAMPLE_RATE:
        return np.ascontiguousarray(arr, dtype=np.float32)
    out_len = int(round(arr.size * float(SAMPLE_RATE) / float(samplerate)))
    if out_len <= 0:
        return np.zeros(0, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, num=arr.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=out_len, endpoint=False)
    return np.interp(x_new, x_old, arr).astype(np.float32)


class EchoReference:
    """Thread-safe timeline of the assistant's own TTS speaker output at 16 kHz.

    On /ws/voice, TTS audio is played on the host via Python `sounddevice` while
    the user's microphone is captured in the browser and streamed over WebSocket.
    Recording the exact reference waveform here gives AcousticEchoCanceller the
    reference signal needed to subtract speaker echo while preserving human
    double-talk (barge-in)."""

    def __init__(self, max_history_seconds: float = 6.0):
        self._lock = threading.Lock()
        self._clips: list[dict] = []
        self._audio_clock: float = 0.0
        self._max_history = max_history_seconds

    def clear(self) -> None:
        with self._lock:
            self._clips.clear()
            self._audio_clock = 0.0

    def start_clip(self, data, samplerate: int, now: Optional[float] = None) -> None:
        samples = _to_mono_float32_16k(data, samplerate)
        if samples.size == 0 or float(np.max(np.abs(samples))) < 1e-5:
            return
        t0 = time.monotonic() if now is None else float(now)
        dur = samples.size / float(SAMPLE_RATE)
        with self._lock:
            self._truncate_active_locked(t0, self._audio_clock)
            self._prune_locked(t0)
            self._clips.append(
                {
                    "start_mono": t0,
                    "end_mono": t0 + dur,
                    "start_audio": self._audio_clock,
                    "end_audio": self._audio_clock + dur,
                    "samples": samples,
                }
            )

    def feed_stream_pcm16(self, pcm_bytes: bytes, samplerate: int, now: Optional[float] = None) -> None:
        if not pcm_bytes:
            return
        pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
        samples = _to_mono_float32_16k(pcm, samplerate)
        if samples.size == 0 or float(np.max(np.abs(samples))) < 1e-5:
            return
        t0 = time.monotonic() if now is None else float(now)
        dur = samples.size / float(SAMPLE_RATE)
        with self._lock:
            if self._clips and abs(t0 - self._clips[-1]["end_mono"]) < 0.25:
                last = self._clips[-1]
                last["samples"] = np.concatenate([last["samples"], samples])
                last["end_mono"] += dur
                last["end_audio"] += dur
            else:
                self._prune_locked(t0)
                self._clips.append(
                    {
                        "start_mono": t0,
                        "end_mono": t0 + dur,
                        "start_audio": self._audio_clock,
                        "end_audio": self._audio_clock + dur,
                        "samples": samples,
                    }
                )

    def stop_clip(self, now: Optional[float] = None) -> None:
        t_stop = time.monotonic() if now is None else float(now)
        with self._lock:
            self._truncate_active_locked(t_stop, self._audio_clock)

    def _truncate_active_locked(self, t_mono: float, t_audio: float) -> None:
        kept = []
        for clip in self._clips:
            wall_elapsed = t_mono - clip["start_mono"]
            audio_elapsed = t_audio - clip["start_audio"]
            use_audio = wall_elapsed < 0.35 * audio_elapsed and audio_elapsed > 0.0
            if use_audio:
                if clip["end_audio"] > t_audio:
                    keep_s = max(0.0, t_audio - clip["start_audio"])
                    keep_n = min(clip["samples"].size, int(keep_s * SAMPLE_RATE))
                    if keep_n <= 0:
                        continue
                    clip["samples"] = clip["samples"][:keep_n]
                    clip["end_audio"] = clip["start_audio"] + keep_n / float(SAMPLE_RATE)
                    clip["end_mono"] = clip["start_mono"] + keep_n / float(SAMPLE_RATE)
            else:
                if clip["end_mono"] > t_mono:
                    keep_s = max(0.0, t_mono - clip["start_mono"])
                    keep_n = min(clip["samples"].size, int(keep_s * SAMPLE_RATE))
                    if keep_n <= 0:
                        continue
                    clip["samples"] = clip["samples"][:keep_n]
                    clip["end_mono"] = clip["start_mono"] + keep_n / float(SAMPLE_RATE)
                    clip["end_audio"] = clip["start_audio"] + keep_n / float(SAMPLE_RATE)
            kept.append(clip)
        self._clips = kept

    def _prune_locked(self, t_mono: float) -> None:
        cutoff = t_mono - self._max_history
        self._clips = [c for c in self._clips if c["end_mono"] >= cutoff or c["end_audio"] >= self._audio_clock - self._max_history]

    def get_window(
        self,
        chunk_seconds: float = 0.25,
        lookback_s: float = 1.15,
        lookahead_s: float = 0.05,
        now: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """Returns the 16 kHz reference waveform over [t - lookback_s, t + lookahead_s],
        or None if no reference audio was active recently."""
        t_mono = time.monotonic() if now is None else float(now)
        total_samples = int(round((lookback_s + lookahead_s) * SAMPLE_RATE))
        with self._lock:
            if not self._clips:
                return None
            self._audio_clock += max(0.0, float(chunk_seconds))
            t_audio = self._audio_clock
            latest = self._clips[-1]
            wall_elapsed = t_mono - latest["start_mono"]
            audio_elapsed = t_audio - latest["start_audio"]
            use_audio = wall_elapsed < 0.35 * audio_elapsed and audio_elapsed > 0.0

            win_start = (t_audio if use_audio else t_mono) - lookback_s
            win_end = (t_audio if use_audio else t_mono) + lookahead_s
            start_key = "start_audio" if use_audio else "start_mono"
            end_key = "end_audio" if use_audio else "end_mono"

            # Check if any clip overlaps [win_start, win_end]
            has_overlap = any(c[end_key] >= win_start and c[start_key] <= win_end for c in self._clips)
            if not has_overlap:
                return None

            out = np.zeros(total_samples, dtype=np.float32)
            for clip in self._clips:
                c_start = clip[start_key]
                c_end = clip[end_key]
                if c_end <= win_start or c_start >= win_end:
                    continue
                ov_start = max(win_start, c_start)
                ov_end = min(win_end, c_end)
                if ov_end <= ov_start:
                    continue
                dst_i0 = int(round((ov_start - win_start) * SAMPLE_RATE))
                src_i0 = int(round((ov_start - c_start) * SAMPLE_RATE))
                n_copy = int(round((ov_end - ov_start) * SAMPLE_RATE))
                dst_i0 = max(0, min(total_samples, dst_i0))
                src_i0 = max(0, min(clip["samples"].size, src_i0))
                n_copy = max(0, min(total_samples - dst_i0, clip["samples"].size - src_i0, n_copy))
                if n_copy > 0:
                    out[dst_i0 : dst_i0 + n_copy] = clip["samples"][src_i0 : src_i0 + n_copy]

            if float(np.max(np.abs(out))) < 1e-4:
                return None
            return out


echo_reference = EchoReference()


def _stft_512(y: np.ndarray, n_fft: int = 512, hop: int = 160) -> np.ndarray:
    win = np.hanning(n_fft).astype(np.float32)
    pad = np.pad(y, (n_fft // 2, n_fft // 2), mode="reflect")
    frames = np.ascontiguousarray(sliding_window_view(pad, n_fft)[::hop]) * win
    return np.fft.rfft(frames, axis=1)


def _istft_512(spec: np.ndarray, length: int, n_fft: int = 512, hop: int = 160) -> np.ndarray:
    win = np.hanning(n_fft).astype(np.float32)
    frames = np.fft.irfft(spec, n=n_fft, axis=1).astype(np.float32) * win
    out_len = (frames.shape[0] - 1) * hop + n_fft
    out = np.zeros(out_len, dtype=np.float32)
    wsum = np.zeros(out_len, dtype=np.float32)
    win_sq = win * win
    for i in range(frames.shape[0]):
        s = i * hop
        out[s : s + n_fft] += frames[i]
        wsum[s : s + n_fft] += win_sq
    out /= np.maximum(wsum, 1e-6)
    start = n_fft // 2
    return out[start : start + length]


class AcousticEchoCanceller:
    """Two-stage reference-based Acoustic Echo Canceller (AEC) with Double-Talk
    Detection (DTD) for both Headphone and Open Laptop Speaker modes:

    1. Headphones Mode:
       When the microphone picks up no speaker echo while reference audio is
       playing (low cross-correlation and near-zero speaker coupling gain),
       microphone audio is passed through untouched so any user utterance
       triggers barge-in immediately.
    2. Open Laptop Speakers Mode (Single-Talk Echo):
       FFT cross-correlation finds the exact acoustic + WebSocket transport delay,
       a 96-tap multipath FIR filter subtracts the linear speaker-to-room impulse
       response, and a coherence-weighted STFT post-filter suppresses nonlinear
       speaker distortion and late reverberation so the assistant never triggers
       barge-in from its own voice.
    3. Open Laptop Speakers Mode (Human Double-Talk):
       When the user speaks over the open speaker, the FIR filter removes only the
       speaker echo component while the Double-Talk Detector freezes adaptation and
       preserves uncorrelated human speech energy so barge-in triggers immediately."""

    FIR_TAPS = 96
    PRE_TAPS = 8
    ANALYSIS_SAMPLES = 4000  # 250 ms context window at 16 kHz

    def __init__(self, echo_ref: Optional[EchoReference] = None):
        self._echo_ref = echo_reference if echo_ref is None else echo_ref
        self.reset()

    def reset(self) -> None:
        self._mic_ring = np.zeros(0, dtype=np.float32)
        self._locked_lag: Optional[int] = None
        self._tracked_h: Optional[np.ndarray] = None
        self._tracked_rms_gain: float = 0.0
        self._seen_headphones_silence: bool = False
        self.last_mode: str = "idle"

    def process(self, chunk: np.ndarray, now: Optional[float] = None) -> np.ndarray:
        if chunk is None or chunk.size == 0 or self._echo_ref is None:
            return chunk

        chunk_f32 = np.ascontiguousarray(chunk, dtype=np.float32)
        ref_win = self._echo_ref.get_window(
            chunk_seconds=chunk_f32.size / float(SAMPLE_RATE),
            lookback_s=1.15,
            lookahead_s=0.05,
            now=now,
        )
        if ref_win is None:
            self.last_mode = "idle"
            return chunk_f32

        # Maintain at least 250 ms of microphone context for delay & FIR estimation
        if chunk_f32.size >= self.ANALYSIS_SAMPLES:
            d = chunk_f32
            tail_only = False
        else:
            self._mic_ring = np.concatenate([self._mic_ring, chunk_f32])[-self.ANALYSIS_SAMPLES :]
            if self._mic_ring.size < 1600:
                return chunk_f32
            d = self._mic_ring
            tail_only = True

        L = d.size
        W = ref_win.size
        M = self.FIR_TAPS
        if W < L + M + 16:
            return chunk_f32

        mic_rms = float(np.sqrt(np.mean(d * d)))
        ref_active = ref_win[np.abs(ref_win) > 1e-3]
        ref_rms = float(np.sqrt(np.mean(ref_active * ref_active))) if ref_active.size > 0 else 0.0

        # If reference is actively playing loudly while mic is near room noise floor,
        # record that the user is on headphones (no speaker-to-mic acoustic coupling).
        if ref_rms > 0.04 and mic_rms < 0.018 and self._tracked_rms_gain < 0.05:
            self._seen_headphones_silence = True
            self.last_mode = "headphones"
            return chunk_f32

        if mic_rms < 1e-4 or ref_rms < 1e-3:
            return chunk_f32

        # Pre-emphasized FFT cross-correlation to find sample-accurate echo delay
        d_hp = np.empty_like(d)
        d_hp[0] = d[0]
        d_hp[1:] = d[1:] - 0.95 * d[:-1]
        x_hp = np.empty_like(ref_win)
        x_hp[0] = ref_win[0]
        x_hp[1:] = ref_win[1:] - 0.95 * ref_win[:-1]

        ed = float(np.dot(d_hp, d_hp))
        if ed < 1e-9:
            return chunk_f32

        max_lag = W - L - M
        n_fft = 1 << (W + L).bit_length()
        X_fft = np.fft.rfft(x_hp, n=n_fft)
        D_fft = np.fft.rfft(d_hp, n=n_fft)
        corr = np.fft.irfft(X_fft * np.conj(D_fft), n=n_fft)[: max_lag + 1]

        csum = np.concatenate([[0.0], np.cumsum(x_hp.astype(np.float64) ** 2)])
        ex = csum[L : L + max_lag + 1] - csum[: max_lag + 1]
        norm_corr = np.zeros_like(corr)
        valid = ex > 1e-6
        norm_corr[valid] = np.abs(corr[valid]) / np.sqrt(ex[valid] * ed + 1e-12)

        best_lag = int(np.argmax(norm_corr))
        best_rho = float(norm_corr[best_lag])

        if self._locked_lag is not None and 0 <= self._locked_lag <= max_lag:
            lo = max(0, self._locked_lag - 320)
            hi = min(max_lag + 1, self._locked_lag + 321)
            local_lag = lo + int(np.argmax(norm_corr[lo:hi]))
            local_rho = float(norm_corr[local_lag])
            if local_rho >= 0.18 or best_rho < local_rho * 1.35:
                best_lag = local_lag
                best_rho = local_rho

        start = int(np.clip(best_lag - self.PRE_TAPS, 0, W - L - M))
        x_aligned = ref_win[ best_lag : best_lag + L ]
        x_aligned_rms = float(np.sqrt(np.mean(x_aligned * x_aligned)))
        if x_aligned_rms < 1e-3:
            return chunk_f32

        # Stage 1: Multipath 96-tap FIR filter
        X_mat = np.ascontiguousarray(sliding_window_view(ref_win[start : start + L + M - 1], M))
        r_xx = X_mat.T @ X_mat
        r_xd = X_mat.T @ d
        reg = 1e-3 * (float(np.trace(r_xx)) / M + 1e-6)
        h_inst = np.linalg.solve(r_xx + reg * np.eye(M, dtype=np.float32), r_xd)
        echo_inst = X_mat @ h_inst
        res_inst = d - echo_inst
        r2_fir = 1.0 - float(np.dot(res_inst, res_inst)) / (float(np.dot(d, d)) + 1e-10)

        # Headphones detection: if prior silent chunks showed zero speaker coupling
        # OR both correlation and FIR explained variance are negligible, do not attenuate.
        if (self._seen_headphones_silence and self._tracked_rms_gain < 0.05) or (
            r2_fir < 0.14 and best_rho < 0.22 and self._tracked_rms_gain < 0.05
        ):
            self.last_mode = "headphones"
            return chunk_f32

        curr_gain = mic_rms / (x_aligned_rms + 1e-6)

        # Check if we already locked onto a speaker echo filter and the user just
        # started speaking over the speaker (causing an energy jump or new voice).
        is_double_talk = False
        res1 = res_inst
        if self._tracked_h is not None and self._tracked_rms_gain > 0.05:
            echo_tracked = X_mat @ self._tracked_h
            res_tracked = d - echo_tracked
            res_tracked_rms = float(np.sqrt(np.mean(res_tracked * res_tracked)))
            res_inst_rms = float(np.sqrt(np.mean(res_inst * res_inst)))
            if curr_gain > self._tracked_rms_gain * 1.25 and res_tracked_rms > max(0.035, res_inst_rms * 1.15):
                is_double_talk = True
                res1 = res_tracked
            elif r2_fir < 0.62 and res_inst_rms > 0.035:
                is_double_talk = True
                res1 = res_inst
        elif r2_fir < 0.62 and float(np.sqrt(np.mean(res_inst * res_inst))) > 0.035:
            is_double_talk = True

        # Update tracked speaker echo path only during clean Single-Talk Echo
        if not is_double_talk and r2_fir >= 0.65:
            self._locked_lag = best_lag
            self._seen_headphones_silence = False
            if self._tracked_h is None:
                self._tracked_h = h_inst.copy()
                self._tracked_rms_gain = curr_gain
            else:
                self._tracked_h = 0.7 * self._tracked_h + 0.3 * h_inst
                self._tracked_rms_gain = 0.7 * self._tracked_rms_gain + 0.3 * curr_gain

        # Stage 2: Coherence-weighted STFT fine-bin residual echo post-filter
        R_spec = _stft_512(res1)
        X_spec = _stft_512(x_aligned)
        mag_R = np.abs(R_spec)
        mag_X = np.abs(X_spec)
        mag_X_sm = mag_X.copy()
        if mag_X.shape[0] >= 3:
            mag_X_sm[1:-1] = 0.25 * mag_X[:-2] + 0.50 * mag_X[1:-1] + 0.25 * mag_X[2:]

        g_bin = np.sum(mag_R * mag_X_sm, axis=0) / (np.sum(mag_X_sm * mag_X_sm, axis=0) + 1e-8)

        if not is_double_talk and r2_fir >= 0.65:
            self.last_mode = "speakers_echo_only"
            est_echo_mag = 1.85 * g_bin[None, :] * mag_X_sm
            gain_mask = np.clip(1.0 - est_echo_mag / (mag_R + 1e-8), 0.03, 1.0)
            res2 = _istft_512(R_spec * gain_mask, L)
        else:
            self.last_mode = "speakers_double_talk"
            r_zm = mag_R - np.mean(mag_R, axis=0, keepdims=True)
            x_zm = mag_X_sm - np.mean(mag_X_sm, axis=0, keepdims=True)
            coh_bin = np.maximum(
                0.0,
                np.sum(r_zm * x_zm, axis=0)
                / np.sqrt(np.sum(r_zm * r_zm, axis=0) * np.sum(x_zm * x_zm, axis=0) + 1e-10),
            )
            est_echo_mag = 1.15 * (coh_bin * coh_bin)[None, :] * g_bin[None, :] * mag_X_sm
            gain_mask = np.clip(1.0 - est_echo_mag / (mag_R + 1e-8), 0.35, 1.0)
            res2 = _istft_512(R_spec * gain_mask, L)

        return res2[-chunk_f32.size :] if tail_only else res2


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
BARGE_ENERGY_MARGIN = 2.5  # frame RMS must be this many times the calibrated floor
BARGE_MIN_RMS = 0.03  # and never below this absolute level
BARGE_CONFIRM_MS = 300  # continuous speech needed before the reply is cut off
BARGE_GAP_FRAMES = 3  # up to 90 ms of quiet inside a word does not restart the count
BARGE_PREROLL_MS = 150  # audio kept from before the first speech frame
BARGE_HISTORY_MS = 2000


class BargeInDetector:
    """Decides whether the user really started talking over the assistant.

    feed() takes the microphone chunks (any length, float32 mono, SAMPLE_RATE),
    passes them through AcousticEchoCanceller (which removes active speaker echo
    while preserving headphone speech and double-talk over open speakers),
    and returns None until the speech has run for confirm_ms, then the audio to
    start the next utterance with: from preroll_ms before the first speech
    frame up to the end of the chunk that confirmed it, so the first word is
    not lost."""

    def __init__(
        self,
        vad: "VoiceActivityDetector",
        confirm_ms: int = BARGE_CONFIRM_MS,
        energy_margin: float = BARGE_ENERGY_MARGIN,
        min_rms: float = BARGE_MIN_RMS,
        gap_frames: int = BARGE_GAP_FRAMES,
        preroll_ms: int = BARGE_PREROLL_MS,
        echo_ref: Optional[EchoReference] = None,
    ):
        self._vad = vad
        self._need_frames = max(1, round(confirm_ms / FRAME_MS))
        self._margin = energy_margin
        self._min_rms = min_rms
        self._gap_frames = gap_frames
        self._preroll = int(SAMPLE_RATE * preroll_ms / 1000)
        self._max_history = int(SAMPLE_RATE * BARGE_HISTORY_MS / 1000)
        self._aec = AcousticEchoCanceller(echo_ref=echo_ref)
        self.reset()

    @property
    def aec(self) -> AcousticEchoCanceller:
        return self._aec

    def reset(self) -> None:
        self._history = np.zeros(0, dtype=np.float32)
        self._carry = np.zeros(0, dtype=np.float32)
        self._frames_total = 0  # frames examined since reset (for the position of the run start)
        self._run_frames = 0
        self._gap = 0
        self._run_start_frame: Optional[int] = None
        self._dropped = 0  # samples trimmed from the front of the history
        self._aec.reset()

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
        clean = self._aec.process(chunk)
        self._history = np.concatenate([self._history, clean])
        data = np.concatenate([self._carry, clean])
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
