"""ffmpeg-based render helpers for mixing and mastering.

This module is responsible for building filter graphs and running
ffmpeg. All heavy audio processing lives inside ffmpeg so Python stays
CPU and memory light for Fly.io.

The API accepts precomputed gain staging and preset-like descriptors
and turns them into concrete ffmpeg filter graphs.
"""
from __future__ import annotations

import os
import shlex
import subprocess
from typing import Optional

from .gain_staging import GainDecision


def _run(cmd: str) -> None:
    subprocess.run(cmd, shell=True, check=True)


def build_vocal_chain_filter(gain: GainDecision) -> str:
    """Return an ffmpeg filter chain string for the vocal input.

    We keep this intentionally close to the existing ai_mix behaviour,
    but start with a role-aware input gain stage.
    """

    parts = []
    if abs(gain.input_gain_db) > 0.1:
        parts.append(f"volume={gain.input_gain_db}dB")

    parts.extend(
        [
            "highpass=f=85",
            "equalizer=f=280:t=q:w=1.2:g=-2",
            "equalizer=f=3200:t=q:w=1.0:g=3",
            "equalizer=f=12000:t=q:w=1.2:g=2",
            "asplit=2[vdry][vpar]",
            "[vpar]acompressor=threshold=-24dB:ratio=6:attack=5:release=50[vcomp]",
            "[vdry][vcomp]amix=inputs=2:weights=0.8 0.2[vocal]",
        ]
    )

    return ",".join(parts)


def build_beat_chain_filter(gain: GainDecision) -> str:
    """Return an ffmpeg filter chain for the beat input.

    Includes role-aware input gain and EQ to make room for vocal.
    """

    parts = []
    if abs(gain.input_gain_db) > 0.1:
        parts.append(f"volume={gain.input_gain_db}dB")

    parts.extend(
        [
            "equalizer=f=250:t=q:w=1.2:g=-3",
            "equalizer=f=3500:t=q:w=1.4:g=-3",
        ]
    )

    return ",".join(parts)


def render_mix_with_sidechain(
    vocal_path: str,
    beat_path: str,
    out_path: str,
    vocal_gain: GainDecision,
    beat_gain: GainDecision,
) -> str:
    """Render a vocal+beat mix via ffmpeg using sidechain ducking.

    This is a preset-driven, Fly.io-safe rework of ai_mix.
    """

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    v_chain = build_vocal_chain_filter(vocal_gain)
    b_chain = build_beat_chain_filter(beat_gain)

    # Compose a filter_complex that mirrors the old behaviour but with
    # explicit labelled chains and role-aware input gain.
    filter_complex = (
        f"[1:a]{v_chain}[vocal_prep];"
        f"[0:a]{b_chain}[beat_prep];"
        "[beat_prep][vocal_prep]sidechaincompress="
        "threshold=-24dB:ratio=3:attack=10:release=120[beat_sc][v];"
        "[beat_sc][v]amix=inputs=2:weights=1 1[mix];"
        "[mix]acompressor=threshold=-18dB:ratio=2:attack=40:release=150[out]"
    )

    cmd = (
        f"ffmpeg -y -i {shlex.quote(beat_path)} -i {shlex.quote(vocal_path)} "
        f"-filter_complex {shlex.quote(filter_complex)} "
        "-map \"[out]\" -c:a pcm_s16le "
        f"{shlex.quote(out_path)}"
    )

    _run(cmd)
    return out_path


def render_master(
    input_mix: str,
    output_master: str,
    target_lufs: str = "-14",
    extra_ceiling_db: float = -1.0,
) -> str:
    """Mastering render using ffmpeg.

    This wraps the existing ai_master behaviour with a thin function so
    analysis/gain staging can be added later if needed.
    """

    os.makedirs(os.path.dirname(output_master) or ".", exist_ok=True)

    cmd = (
        f'ffmpeg -y -i {shlex.quote(input_mix)} -af '
        f'"equalizer=f=100:t=q:w=1.1:g=1,'
        f'equalizer=f=13000:t=q:w=1.3:g=1,'
        'acompressor=threshold=-18dB:ratio=1.7:attack=50:release=180,'
        f'alimiter=limit=0.89,'
        f'loudnorm=I={target_lufs}:TP={extra_ceiling_db}:LRA=11" '
        f'{shlex.quote(output_master)}'
    )

    _run(cmd)
    return output_master
