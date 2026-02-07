from __future__ import annotations

# pyright: ignore-file
from __future__ import annotations

import numpy as np

"""Legacy Dancehall vocal preset.

The active implementation lives in the separate mixsmvrt-dsp service.
This module remains only to avoid import errors in older code paths.
"""

__all__: list[str] = []


def _process_vocal_gender(audio: np.ndarray, sr: int, gender: str) -> np.ndarray:
    """Process vocal audio with specified gender."""
    # TODO: Implement or delegate to mixsmvrt-dsp service
    return audio


def process_vocal_female(audio: np.ndarray, sr: int) -> np.ndarray:
    """Dancehall vocal preset tuned for female voices."""
    return _process_vocal_gender(audio, sr, "female")


def process_vocal(audio: np.ndarray, sr: int) -> np.ndarray:
    """Backward‑compatible entry point (defaults to male variant)."""
    return _process_vocal_gender(audio, sr, "male")
