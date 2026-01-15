from __future__ import annotations

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


def _pre_loudness_normalize(audio: np.ndarray, sr: int, target_lufs: float = -17.5) -> np.ndarray:
    """Normalize input to a slightly hotter loudness for hard rap.

    Rap often wants denser, more in‑your‑face vocals.
    """
    if audio.ndim == 1:
        mono = audio.astype(np.float32)
    else:
        mono = audio.mean(axis=0).astype(np.float32) if audio.shape[0] < audio.shape[1] else audio.mean(axis=1).astype(np.float32)

    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(mono)
    loudness_diff = target_lufs - loudness
    gain_linear = 10.0 ** (loudness_diff / 20.0)
    return audio * gain_linear


def _prepare_for_pedalboard(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        audio_2d = audio[np.newaxis, :]
    else:
        audio_2d = audio if audio.shape[0] <= audio.shape[1] else audio.T
    return audio_2d.T.astype(np.float32)


def _restore_shape(processed: np.ndarray, original: np.ndarray) -> np.ndarray:
    if original.ndim == 1:
        return processed[:, 0].astype(np.float32)
    channels_first = original.shape[0] <= original.shape[1]
    out = processed.T.astype(np.float32)
    return out if channels_first else out.T


def _process_vocal_gender(audio: np.ndarray, sr: int, gender: str | None) -> np.ndarray:
    """Rap vocal preset with male/female variants."""
    if audio.size == 0:
        return audio

    norm = _pre_loudness_normalize(audio, sr, target_lufs=-17.5)
    pb_input = _prepare_for_pedalboard(norm)

    # Optional pitch correction using an external tuner plugin if available.
    pb_input = apply_pitch_correction(pb_input, sr)

    is_female = (gender or "male").lower() == "female"

    highpass_cutoff = 100.0 if not is_female else 115.0
    deesser_freq = 7200.0 if not is_female else 7700.0
    highshelf_gain = 2.5 if not is_female else 3.0

    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=highpass_cutoff),
        NoiseGate(threshold_db=-46.0, ratio=2.2, release_ms=130.0),
        Deesser(
            frequency=deesser_freq,
            threshold_db=-30.0,
            ratio=4.0,
        ),
        Compressor(
            threshold_db=-22.0,
            ratio=5.5,
            attack_ms=4.0,
            release_ms=80.0,
            makeup_gain_db=4.0,
        ),
        LowShelfFilter(
            cutoff_frequency_hz=200.0,
            gain_db=-1.0,
        ),
        PeakFilter(
            center_frequency_hz=3000.0,
            gain_db=4.0,
            q=1.1,
        ),
        HighShelfFilter(
            cutoff_frequency_hz=10500.0,
            gain_db=highshelf_gain,
        ),
        Saturation(drive_db=7.0),
        Reverb(
            room_size=0.2,
            damping=0.45,
            wet_level=0.17,
            dry_level=0.83,
            width=1.0,
        ),
        Delay(
            delay_seconds=0.24,
            feedback=0.25,
            mix=0.18,
        ),
        Limiter(
            threshold_db=-1.5,
            release_ms=80.0,
        ),
        Gain(gain_db=-0.5),
    ])

    processed = board(pb_input, sr)
    return _restore_shape(processed, audio)


def process_vocal_male(audio: np.ndarray, sr: int) -> np.ndarray:
    """Rap vocal preset tuned for male voices."""
    return _process_vocal_gender(audio, sr, "male")


def process_vocal_female(audio: np.ndarray, sr: int) -> np.ndarray:
    """Rap vocal preset tuned for female voices."""
    return _process_vocal_gender(audio, sr, "female")


def process_vocal(audio: np.ndarray, sr: int) -> np.ndarray:
    """Backward‑compatible entry point (defaults to male variant)."""
    return _process_vocal_gender(audio, sr, "male")
