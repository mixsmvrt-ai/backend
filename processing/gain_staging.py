"""Simple, role-aware gain staging for mix inputs.

This module stays intentionally lightweight: it operates on analysis
numbers (LUFS/peak) only and returns gain offsets in dB, which can
then be applied via ffmpeg's `volume` or input `-filter_complex`.

Rules (clamped to keep things safe):
- Beat: target peak around -6 dBFS
- Lead vocal: target integrated loudness around -18 LUFS
- Background/adlibs: target integrated loudness around -22 LUFS
- Mix bus pre-master: target peak around -3 dBFS

Gain changes are clamped to +/- 6 dB to avoid drastic moves.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .analysis import TrackAnalysis

Role = Literal["beat", "vocal", "adlib", "mix-bus", "master"]


@dataclass
class GainDecision:
    role: Role
    input_gain_db: float  # to apply before processing


TARGETS = {
    "beat": {"peak_dbfs": -6.0},
    "vocal": {"lufs": -18.0},
    "adlib": {"lufs": -22.0},
    "mix-bus": {"peak_dbfs": -3.0},
}

MAX_ABS_GAIN_DB = 6.0


def _clamp_gain(db: float) -> float:
    if db > MAX_ABS_GAIN_DB:
        return MAX_ABS_GAIN_DB
    if db < -MAX_ABS_GAIN_DB:
        return -MAX_ABS_GAIN_DB
    return db


def decide_input_gain(role: Role, analysis: TrackAnalysis) -> GainDecision:
    """Compute a simple input gain decision for a given role.

    We use whichever metric is available (LUFS or peak) and never
    change gain more than +/- 6 dB.
    """

    target = TARGETS.get(role, {})

    gain_db = 0.0

    if "lufs" in target and analysis.integrated_lufs is not None:
        current = analysis.integrated_lufs
        gain_db = target["lufs"] - current
    elif "peak_dbfs" in target and analysis.peak_dbfs is not None:
        current = analysis.peak_dbfs
        gain_db = target["peak_dbfs"] - current

    gain_db = _clamp_gain(gain_db)

    return GainDecision(role=role, input_gain_db=gain_db)
