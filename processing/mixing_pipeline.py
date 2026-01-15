
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
from typing import List, Literal, TypedDict


def run(cmd: str) -> None:
    subprocess.run(cmd, shell=True, check=True)


def ai_mix(vocal_path: str, beat_path: str, output_path: str) -> str:
    os.makedirs("temp", exist_ok=True)

    run(f'ffmpeg -i {vocal_path} -af "highpass=f=80, lowpass=f=16000" temp/vocal_clean.wav')
    run(f'ffmpeg -i temp/vocal_clean.wav -af "acompressor=threshold=-18dB:ratio=3" temp/vocal_comp.wav')
    run(f'ffmpeg -i temp/vocal_comp.wav -af "equalizer=f=7000:t=q:w=1:g=-4" temp/vocal_final.wav')
    run(f'ffmpeg -i {beat_path} -af "volume=0.8" temp/beat_ready.wav')

    run(
        f'ffmpeg -i temp/beat_ready.wav -i temp/vocal_final.wav '
        f'-filter_complex "amix=inputs=2" {output_path}'
    )
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
