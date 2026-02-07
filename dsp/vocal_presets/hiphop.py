from __future__ import annotations

# pyright: reportGeneralTypeIssues=false, reportUnknownMemberType=false, reportUnknownArgumentType=false

import numpy as np
import pyloudnorm as pyln
from pedalboard import (
    Pedalboard,
    Gain,
    HighpassFilter,
    Compressor,
    Limiter,
    NoiseGate,
    HighShelfFilter,
    LowShelfFilter,
    Reverb,
    Delay,
)

from .tuning import apply_pitch_correction


def _pre_loudness_normalize(audio: np.ndarray, sr: int, target_lufs: float = -18.0) -> np.ndarray:
    """Normalize input to a consistent loudness before dynamics."""
    if audio.ndim == 1:
        mono = audio.astype(np.float32)
    else:
        mono = audio.mean(axis=0).astype(np.float32) if audio.shape[0] < audio.shape[1] else audio.mean(axis=1).astype(np.float32)

    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(mono)
    loudness_diff = target_lufs - loudness
    gain_linear = 10.0 ** (loudness_diff / 20.0)
    return audio * gain_linear


"""Legacy Hip-hop vocal preset.

The active implementation lives in the separate mixsmvrt-dsp service.
This module remains only to avoid import errors in older code paths.
"""

__all__: list[str] = []
