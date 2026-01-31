from __future__ import annotations

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false

import numpy as np
import pyloudnorm as pyln
from pedalboard import (
    Pedalboard,
    Gain,
    HighpassFilter,
    Compressor,
    Limiter,
    NoiseGate,
    Deesser,
    PeakFilter,
    HighShelfFilter,
    LowShelfFilter,
    Saturation,
    Reverb,
    Delay,
)

from .tuning import apply_pitch_correction


def _pre_loudness_normalize(audio: np.ndarray, sr: int, target_lufs: float = -18.0) -> np.ndarray:
    """Normalize input to a consistent loudness before dynamics.

    This keeps the vocal hitting the compressors in a predictable way
    regardless of how it was recorded.
    """
    if audio.ndim == 1:
        mono = audio.astype(np.float32)
    else:
        # Downmix to mono for loudness measurement.
        mono = audio.mean(axis=0).astype(np.float32) if audio.shape[0] < audio.shape[1] else audio.mean(axis=1).astype(np.float32)

    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(mono)
    loudness_diff = target_lufs - loudness
    gain_linear = 10.0 ** (loudness_diff / 20.0)
    return audio * gain_linear


def _prepare_for_pedalboard(audio: np.ndarray) -> np.ndarray:
    """Ensure shape (n_samples, n_channels) float32 for Pedalboard."""
    if audio.ndim == 1:
        audio_2d = audio[np.newaxis, :]
    else:
        # Expect (channels, samples); if not, transpose.
        audio_2d = audio if audio.shape[0] <= audio.shape[1] else audio.T
    return audio_2d.T.astype(np.float32)


def _restore_shape(processed: np.ndarray, original: np.ndarray) -> np.ndarray:
    """Return audio to original shape and dtype (float32)."""
    # processed is (n_samples, n_channels)
    if original.ndim == 1:
        return processed[:, 0].astype(np.float32)
    # original was (channels, samples) or (samples, channels)
    channels_first = original.shape[0] <= original.shape[1]
    out = processed.T.astype(np.float32)
    return out if channels_first else out.T


def _process_vocal_gender(audio: np.ndarray, sr: int, gender: str | None) -> np.ndarray:
    """Dancehall vocal preset with simple gender variants.

    The core character stays the same, but female vocals get a slightly
    higher high‑pass, a touch more de‑essing, and a bit more air.
    """
    if audio.size == 0:
        return audio

    # 1) Loudness normalization (pre‑gain)
    norm = _pre_loudness_normalize(audio, sr, target_lufs=-18.0)

    # 2) Prepare for Pedalboard (samples, channels)
    pb_input = _prepare_for_pedalboard(norm)

    # Optional pitch correction using an external tuner plugin, if
    # configured via the VOCAL_TUNER_PLUGIN environment variable.
    pb_input = apply_pitch_correction(pb_input, sr)

    is_female = (gender or "male").lower() == "female"

    highpass_cutoff = 90.0 if not is_female else 110.0
    deesser_freq = 7000.0 if not is_female else 7500.0
    deesser_threshold = -28.0 if not is_female else -30.0
    compressor_threshold = -18.0 if not is_female else -20.0
    highshelf_gain = 1.5 if not is_female else 2.0

    board = Pedalboard([
        # 2) High‑pass filter – slightly higher for female voices.
        HighpassFilter(cutoff_frequency_hz=highpass_cutoff),
        # 3) Gentle noise gate to clean breaths/room between phrases.
        NoiseGate(threshold_db=-45.0, ratio=2.0, release_ms=120.0),
        # 4) De‑esser to tame harsh sibilance from bright presence boosts.
        Deesser(
            frequency=deesser_freq,
            threshold_db=deesser_threshold,
            ratio=4.0,
        ),
        # 5) Primary compressor – punchy, modern dancehall feel.
        Compressor(
            threshold_db=compressor_threshold,
            ratio=4.5,
            attack_ms=4.0,
            release_ms=80.0,
            makeup_gain_db=2.0,
        ),
        # 6) Tone‑shaping EQ
        LowShelfFilter(
            cutoff_frequency_hz=180.0,
            gain_db=-1.5,
        ),
        PeakFilter(
            center_frequency_hz=2500.0,
            gain_db=3.0,
            q=1.1,
        ),
        HighShelfFilter(
            cutoff_frequency_hz=9000.0,
            gain_db=highshelf_gain,
        ),
        # 7) Harmonic saturation for energy and edge.
        Saturation(drive_db=6.0),
        # 8) Space and vibe – short plate reverb and slap delay.
        Reverb(
            room_size=0.25,
            damping=0.4,
            wet_level=0.16,
            dry_level=0.84,
            width=1.0,
        ),
        Delay(
            delay_seconds=0.26,
            feedback=0.24,
            mix=0.18,
        ),
        # 9) Final limiter with safe streaming headroom.
        Limiter(
            threshold_db=-1.5,
            release_ms=100.0,
        ),
        Gain(gain_db=-0.5),
    ])

    processed = board(pb_input, sr)
    return _restore_shape(processed, audio)


def process_vocal_male(audio: np.ndarray, sr: int) -> np.ndarray:
    """Dancehall vocal preset tuned for male voices."""
    return _process_vocal_gender(audio, sr, "male")


def process_vocal_female(audio: np.ndarray, sr: int) -> np.ndarray:
    """Dancehall vocal preset tuned for female voices."""
    return _process_vocal_gender(audio, sr, "female")


def process_vocal(audio: np.ndarray, sr: int) -> np.ndarray:
    """Backward‑compatible entry point (defaults to male variant)."""
    return _process_vocal_gender(audio, sr, "male")
