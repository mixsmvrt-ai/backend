"""Centralised production-grade preset registry for MIXSMVRT.

This module defines high-level presets that describe *intent* and
safe DSP parameter ranges for each flow (cleanup, mix, mix+master,
mastering). The actual DSP microservice and the Next.js studio UI can
both consume this registry to ensure that:

- Presets feel consistent across the app
- AI can only adjust parameters within safe, professional ranges
- Flows map cleanly onto real-world engineering workflows

The goal is for the DSP service to use these ranges as hard
constraints when mapping analysis results (Essentia, WORLD, etc.) to
concrete processor settings.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal, TypedDict, Dict, Any, List, Optional


Flow = Literal["cleanup", "mix", "mix_master", "master"]
Category = Literal["vocal", "full_mix", "master"]


class EqRanges(TypedDict, total=False):
    hpf_hz: tuple[float, float]
    low_mud_hz: tuple[float, float]
    low_mud_cut_db: tuple[float, float]
    harsh_hz: tuple[float, float]
    harsh_cut_db: tuple[float, float]
    presence_hz: tuple[float, float]
    presence_boost_db: tuple[float, float]
    air_hz: tuple[float, float]
    air_boost_db: tuple[float, float]
    sub_cut_hz: tuple[float, float]
    high_shelf_hz: tuple[float, float]
    high_shelf_boost_db: tuple[float, float]


class CompressionRanges(TypedDict, total=False):
    ratio: tuple[float, float]
    attack_ms: tuple[float, float]
    release_ms: tuple[float, float]
    gain_reduction_db: tuple[float, float]
    # For vocal cleanup expander-style behaviour
    expander_ratio: tuple[float, float]
    expander_threshold_db: tuple[float, float]


class SaturationRanges(TypedDict, total=False):
    drive_percent: tuple[float, float]


class DeesserRanges(TypedDict, total=False):
    freq_hz: tuple[float, float]
    reduction_db: tuple[float, float]


class BusRanges(TypedDict, total=False):
    ratio: tuple[float, float]
    attack_ms: tuple[float, float]
    release_ms: tuple[float, float]
    gain_reduction_db: tuple[float, float]


class LimiterRanges(TypedDict, total=False):
    true_peak_db: tuple[float, float]
    target_lufs: tuple[float, float]


class DspRanges(TypedDict, total=False):
    eq: EqRanges
    compression: CompressionRanges
    saturation: SaturationRanges
    deesser: DeesserRanges
    bus: BusRanges
    limiter: LimiterRanges


@dataclass(frozen=True)
class PresetDefinition:
    id: str
    flow: Flow
    category: Category
    name: str
    intent: str
    target_genres: list[str]
    dsp_ranges: DspRanges

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data


# ---------------------------------------------------------------------------
# Core production presets
# ---------------------------------------------------------------------------

_PRESETS: list[PresetDefinition] = [
    # ------------------------------------------------------------------
    # AUDIO CLEANUP (FINAL QUALITY)
    # ------------------------------------------------------------------
    PresetDefinition(
        id="voice_over_clean",  # aligned with existing studio preset id
        flow="cleanup",
        category="vocal",
        name="Vocal Restore  Transparent",
        intent="Broadcast-grade vocal cleanup that removes noise and mud while staying invisible.",
        target_genres=["podcast", "voiceover", "dialogue", "stream"],
        dsp_ranges={
            "eq": {
                "hpf_hz": (70.0, 100.0),
                "low_mud_hz": (200.0, 400.0),
                "low_mud_cut_db": (-3.0, -1.0),
                "harsh_hz": (3000.0, 5000.0),
                "harsh_cut_db": (-2.0, -1.0),
            },
            "compression": {
                "expander_ratio": (1.5, 2.0),
                "expander_threshold_db": (-45.0, -35.0),
            },
            "saturation": {
                "drive_percent": (0.0, 4.0),
            },
            "deesser": {
                "freq_hz": (6000.0, 8000.0),
                "reduction_db": (-4.0, -2.0),
            },
            "bus": {},
            "limiter": {},
        },
    ),
    PresetDefinition(
        id="noisy_room_cleanup",  # aligned with existing studio preset id
        flow="cleanup",
        category="vocal",
        name="Live Mic Rescue",
        intent="Aggressive but controlled rescue for noisy or reverberant live mics.",
        target_genres=["live", "event", "vlog", "phone"],
        dsp_ranges={
            "eq": {
                "hpf_hz": (90.0, 130.0),
                "low_mud_hz": (250.0, 500.0),
                "low_mud_cut_db": (-6.0, -3.0),
            },
            "compression": {
                "expander_ratio": (2.0, 3.0),
                "expander_threshold_db": (-45.0, -30.0),
            },
            "saturation": {
                "drive_percent": (0.0, 6.0),
            },
            "deesser": {
                "freq_hz": (6000.0, 8000.0),
                "reduction_db": (-6.0, -4.0),
            },
            "bus": {},
            "limiter": {},
        },
    ),
    # ------------------------------------------------------------------
    # VOCAL MIXING (FINAL DELIVERY)
    # ------------------------------------------------------------------
    PresetDefinition(
        id="clean_pop_vocal",  # existing mix preset id
        flow="mix",
        category="vocal",
        name="Modern Vocal  Clean",
        intent="Forward, polished but natural lead vocal for melodic genres.",
        target_genres=["pop", "afrobeat", "rnb"],
        dsp_ranges={
            "eq": {
                "hpf_hz": (70.0, 90.0),
                "presence_hz": (2000.0, 4000.0),
                "presence_boost_db": (1.0, 3.0),
                "air_hz": (10000.0, 14000.0),
                "air_boost_db": (1.0, 2.0),
            },
            "compression": {
                "ratio": (2.0, 3.0),
                "attack_ms": (15.0, 30.0),
                "release_ms": (60.0, 120.0),
                "gain_reduction_db": (-5.0, -3.0),
            },
            "saturation": {
                "drive_percent": (3.0, 8.0),
            },
            "deesser": {
                "freq_hz": (6000.0, 8000.0),
                "reduction_db": (-4.0, -2.0),
            },
            "bus": {},
            "limiter": {},
        },
    ),
    PresetDefinition(
        id="rap_vocal_aggressive",  # existing mix preset id
        flow="mix",
        category="vocal",
        name="Trap / Rap Vocal  Forward",
        intent="Aggressive, locked-in vocal that rides on top of hard drums.",
        target_genres=["trap", "hiphop", "drill"],
        dsp_ranges={
            "eq": {
                "hpf_hz": (80.0, 100.0),
                "presence_hz": (3000.0, 5000.0),
                "presence_boost_db": (2.0, 4.0),
            },
            "compression": {
                "ratio": (3.0, 4.0),
                "attack_ms": (5.0, 15.0),
                "release_ms": (40.0, 80.0),
                "gain_reduction_db": (-7.0, -5.0),
            },
            "saturation": {
                "drive_percent": (6.0, 12.0),
            },
            "deesser": {
                "freq_hz": (6000.0, 8000.0),
                "reduction_db": (-5.0, -3.0),
            },
            "bus": {},
            "limiter": {},
        },
    ),
    # ------------------------------------------------------------------
    # FULL MIX (STEMS)
    # ------------------------------------------------------------------
    PresetDefinition(
        id="radio_ready_mix",  # existing mix+master preset id
        flow="mix_master",
        category="full_mix",
        name="Modern Radio Mix",
        intent="Balanced, translation-first mix that travels well across devices.",
        target_genres=["pop", "hiphop", "trap", "afrobeat", "rnb"],
        dsp_ranges={
            "eq": {
                "sub_cut_hz": (30.0, 50.0),
                "high_shelf_hz": (8000.0, 10000.0),
                "high_shelf_boost_db": (0.5, 1.0),
            },
            "compression": {},
            "saturation": {},
            "deesser": {},
            "bus": {
                "ratio": (2.0, 2.0),
                "attack_ms": (30.0, 30.0),
                "release_ms": (80.0, 120.0),
                "gain_reduction_db": (-3.0, -1.0),
            },
            "limiter": {},
        },
    ),
    PresetDefinition(
        id="club_ready_mix",  # existing mix+master preset id
        flow="mix_master",
        category="full_mix",
        name="Club Ready Mix",
        intent="Punchy, loud, energetic mix that hits in clubs and cars.",
        target_genres=["trap", "dancehall", "afrobeat", "reggaeton", "edm"],
        dsp_ranges={
            "eq": {
                "sub_cut_hz": (30.0, 45.0),
                "high_shelf_hz": (8000.0, 10000.0),
                "high_shelf_boost_db": (0.5, 1.5),
            },
            "compression": {},
            "saturation": {
                "drive_percent": (5.0, 10.0),
            },
            "deesser": {},
            "bus": {
                "ratio": (3.0, 3.0),
                "attack_ms": (10.0, 20.0),
                "release_ms": (60.0, 120.0),
                "gain_reduction_db": (-4.0, -2.0),
            },
            "limiter": {},
        },
    ),
    # ------------------------------------------------------------------
    # MASTERING (FINAL, RELEASE-READY)
    # ------------------------------------------------------------------
    PresetDefinition(
        id="streaming_master_minus14",  # existing mastering preset id
        flow="master",
        category="master",
        name="Streaming Balanced Master",
        intent="Platform-safe loudness master that preserves dynamics and headroom.",
        target_genres=["any", "streaming"],
        dsp_ranges={
            "eq": {
                "sub_cut_hz": (25.0, 30.0),
                "high_shelf_hz": (10000.0, 12000.0),
                "high_shelf_boost_db": (0.5, 1.5),
            },
            "compression": {},
            "saturation": {},
            "deesser": {},
            "bus": {
                "ratio": (1.5, 2.0),
                "attack_ms": (20.0, 40.0),
                "release_ms": (80.0, 160.0),
                "gain_reduction_db": (-2.0, -1.0),
            },
            "limiter": {
                "true_peak_db": (-1.2, -1.0),
                "target_lufs": (-14.0, -13.0),
            },
        },
    ),
    PresetDefinition(
        id="loud_club_master",  # existing mastering preset id
        flow="master",
        category="master",
        name="Club Loud Master",
        intent="DJ and sound-system ready master that stays controlled at high level.",
        target_genres=["trap", "hiphop", "dancehall", "afrobeat", "edm"],
        dsp_ranges={
            "eq": {
                "sub_cut_hz": (25.0, 30.0),
                "high_shelf_hz": (10000.0, 12000.0),
                "high_shelf_boost_db": (0.0, 1.5),
            },
            "compression": {},
            "saturation": {},
            "deesser": {},
            "bus": {
                "ratio": (2.0, 3.0),
                "attack_ms": (15.0, 35.0),
                "release_ms": (60.0, 140.0),
                "gain_reduction_db": (-3.0, -2.0),
            },
            "limiter": {
                "true_peak_db": (-0.9, -0.8),
                "target_lufs": (-9.5, -8.0),
            },
        },
    ),
]


def list_presets(flow: Optional[Flow] = None) -> list[Dict[str, Any]]:
    """Return preset definitions, optionally filtered by flow.

    The returned objects are JSON-friendly and can be served directly
    by FastAPI. DSP workers should use the same registry (or fetch it
    via HTTP) to clamp analysis-driven parameters to these ranges.
    """

    items: List[PresetDefinition] = _PRESETS
    if flow is not None:
        items = [p for p in items if p.flow == flow]
    return [p.to_dict() for p in items]


def get_preset(preset_id: str) -> Dict[str, Any] | None:
    for p in _PRESETS:
        if p.id == preset_id:
            return p.to_dict()
    return None
