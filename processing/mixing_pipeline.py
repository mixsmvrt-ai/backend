
"""Mixing pipeline and shared DSP chain schema for MIXSMVRT.

Existing behaviour: simple ffmpeg-based vocal/beat mix in :func:`ai_mix`.
New: a JSON-friendly chain/preset schema that mirrors the frontend
audio-engine, so Torch/JUCE-style DSP can share the same preset metadata.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, TypedDict


def run(cmd: str) -> None:
    subprocess.run(cmd, shell=True, check=True)


def ai_mix(vocal_path: str, beat_path: str, output_path: str) -> str:
    """Mix vocal and beat using sidechain-based space creation and mix-bus glue.

    This is a ffmpeg approximation of the MIXING_ONLY_FLOW_DSP / MIX_AND_MASTER
    specs: vocal EQ+compression+parallel chain, beat EQ + sidechain ducking
    from the vocal bus, and gentle mix-bus compression.
    """

    os.makedirs("temp", exist_ok=True)

    cmd = (
        f'ffmpeg -y -i "{beat_path}" -i "{vocal_path}" '
        '-filter_complex "'
        # Vocal foreground chain: EQ, then parallel compression
        '[1:a]highpass=f=85,'
        'equalizer=f=280:t=q:w=1.2:g=-2,'
        'equalizer=f=3200:t=q:w=1.0:g=3,'
        'equalizer=f=12000:t=q:w=1.2:g=2,'
        'asplit=2[vdry][vpar];'
        '[vpar]acompressor=threshold=-24dB:ratio=6:attack=5:release=50[vcomp];'
        '[vdry][vcomp]amix=inputs=2:weights=0.8 0.2[vocal];'
        # Beat EQ for space plus broadband sidechain compression keyed by vocal
        '[0:a]equalizer=f=250:t=q:w=1.2:g=-3,'
        'equalizer=f=3500:t=q:w=1.4:g=-3[beat];'
        '[beat][vocal]sidechaincompress='
        'threshold=-24dB:ratio=3:attack=10:release=120[beat_sc][v];'
        # Sum beat and vocal, then apply gentle mix-bus glue compression
        '[beat_sc][v]amix=inputs=2:weights=1 1[mix];'
        '[mix]acompressor=threshold=-18dB:ratio=2:attack=40:release=150[out]" '
        f'-map "[out]" -c:a pcm_s16le "{output_path}"'
    )

    run(cmd)
    return output_path


class ProcessorId(str, Enum):
    EQ = "eq"
    COMPRESSOR = "compressor"
    DEESSER = "deesser"
    SATURATION = "saturation"
    STEREO = "stereo"
    LIMITER = "limiter"
    LOUDNESS = "loudness"


ChainRole = Literal["vocal", "beat", "adlib", "mix-bus", "master"]


class EqBandDict(TypedDict, total=False):
    type: Literal["highpass", "lowshelf", "peak", "highshelf"]
    frequency: float
    q: float
    gainDb: float


class ProcessorParamsDict(TypedDict, total=False):
    id: str
    # Processor-specific fields are kept open; see frontend audio-engine.


class ChainStageDict(TypedDict, total=False):
    id: str
    params: ProcessorParamsDict
    bypass: bool


class ChainDefinitionDict(TypedDict):
    id: str
    name: str
    role: ChainRole
    description: str
    stages: List[ChainStageDict]


class PresetDefinitionDict(TypedDict):
    id: str
    name: str
    role: ChainRole
    notes: str | None
    chain: ChainDefinitionDict


@dataclass
class ChainStage:
    id: ProcessorId
    params: ProcessorParamsDict = field(default_factory=dict)
    bypass: bool = False


@dataclass
class ChainDefinition:
    id: str
    name: str
    role: ChainRole
    description: str
    stages: List[ChainStage] = field(default_factory=list)


@dataclass
class PresetDefinition:
    id: str
    name: str
    role: ChainRole
    chain: ChainDefinition
    notes: str | None = None


def chain_from_dict(data: ChainDefinitionDict) -> ChainDefinition:
    """Construct a ChainDefinition from JSON-compatible data.

    This keeps the backend in sync with the frontend preset schema. A Torch or
    JUCE DSP layer can consume the resulting ChainDefinition and map each
    ProcessorId to concrete filters / dynamics processors.
    """

    stages = [
        ChainStage(
            id=ProcessorId(stage["id"]),
            params=stage.get("params", {}),
            bypass=stage.get("bypass", False),
        )
        for stage in data.get("stages", [])
    ]
    return ChainDefinition(
        id=data["id"],
        name=data["name"],
        role=data["role"],
        description=data.get("description", ""),
        stages=stages,
    )


def preset_from_dict(data: PresetDefinitionDict) -> PresetDefinition:
    return PresetDefinition(
        id=data["id"],
        name=data["name"],
        role=data["role"],
        notes=data.get("notes"),
        chain=chain_from_dict(data["chain"]),
    )


# -------------------------
# High-level DSP templates for mixing flows
# -------------------------

# These JSON-friendly structures describe the target DSP behaviour for the
# "mixing_only" and "mix_master" flows. A downstream DSP engine (ffmpeg, JUCE,
# Torch, etc.) can map these into concrete processor graphs.


MixFlowSpec = Dict[str, Any]


MIXING_ONLY_FLOW_DSP: MixFlowSpec = {
    "global": {
        "vocal_bus": {
            "role": "sidechain_source",
            "target": "beat_bus",
            "trigger_threshold_lufs": [-30.0, -24.0],
        },
    },
    "beat_processing": {
        "beat_eq_dynamic_ducking": {
            "trigger": "vocal_bus",
            "bands": {
                "low_mid": {
                    "freq_range_hz": [180.0, 350.0],
                    "reduction_db": [-4.0, -2.0],
                    "q": [1.0, 1.4],
                },
                "presence": {
                    "freq_range_hz": [2500.0, 4500.0],
                    "reduction_db": [-5.0, -2.0],
                    "q": [1.2, 1.8],
                },
            },
            "attack_ms": [5.0, 15.0],
            "release_ms": [80.0, 150.0],
        },
        "beat_multiband_sidechain_compression": {
            "trigger": "vocal_bus",
            "bands": {
                "low_mid": {
                    "freq_range_hz": [150.0, 350.0],
                    "ratio": [2.0, 3.0],
                    "gain_reduction_db": [2.0, 4.0],
                },
                "high_mid": {
                    "freq_range_hz": [2000.0, 5000.0],
                    "ratio": [2.5, 4.0],
                    "gain_reduction_db": [3.0, 6.0],
                },
            },
            "attack_ms": [10.0, 20.0],
            "release_ms": [90.0, 160.0],
        },
        "beat_stereo_control": {
            "mid_gain_db_when_vocal_active": [-1.5, -0.5],
            "side_gain_db": [0.5, 1.0],
            "automation": "vocal_presence_based",
        },
    },
    "vocal_processing": {
        "vocal_eq": {
            "hpf_hz": [70.0, 100.0],
            "low_mid_cut_db": [-3.0, -1.0],
            "low_mid_cut_hz": [200.0, 350.0],
            "presence_boost_db": [2.0, 4.0],
            "presence_boost_hz": [2500.0, 4500.0],
            "air_boost_db": [1.0, 3.0],
            "air_boost_hz": [10000.0, 16000.0],
        },
        "vocal_compression": {
            "ratio": [3.0, 4.0],
            "attack_ms": [5.0, 20.0],
            "release_ms": [60.0, 120.0],
            "gain_reduction_db": [5.0, 8.0],
        },
        "vocal_parallel_compression": {
            "blend_percent": [10.0, 25.0],
            "ratio": 6.0,
            "attack_ms": [3.0, 10.0],
            "release_ms": [40.0, 70.0],
        },
    },
    "mix_bus": {
        "mix_bus_glue": {
            "compression": {
                "ratio": 2.0,
                "attack_ms": [30.0, 50.0],
                "release_ms": [100.0, 200.0],
                "gain_reduction_db": [1.0, 2.0],
            },
            "saturation": {
                "drive_percent": [2.0, 4.0],
            },
        },
    },
    "automation": {
        "vocal_present": {
            "beat_ducking": "enabled",
        },
        "vocal_absent": {
            "beat_restore_release_ms": [150.0, 250.0],
        },
    },
}


MIX_AND_MASTER_FLOW_DSP: MixFlowSpec = {
    "global": MIXING_ONLY_FLOW_DSP["global"],
    "beat_processing": MIXING_ONLY_FLOW_DSP["beat_processing"],
    "vocal_processing": MIXING_ONLY_FLOW_DSP["vocal_processing"],
    "mix_bus": MIXING_ONLY_FLOW_DSP["mix_bus"],
    "master_bus": {
        "master_eq": {
            "low_shelf_db": [0.5, 1.5],
            "low_shelf_hz": [80.0, 120.0],
            "air_boost_db": [0.5, 1.5],
            "air_boost_hz": [12000.0, 16000.0],
        },
        "master_compression": {
            "ratio": [1.5, 2.0],
            "attack_ms": [40.0, 60.0],
            "release_ms": [120.0, 220.0],
            "gain_reduction_db": [1.0, 2.0],
        },
        "master_limiter": {
            "ceiling_dbtp": -1.0,
            "loudness_targets_lufs": {
                "streaming": [-14.0, -10.0],
                "club": [-9.0, -7.0],
            },
            "transient_preservation": True,
        },
    },
    "automation": MIXING_ONLY_FLOW_DSP["automation"],
}
