"""Lightweight audio analysis helpers using ffmpeg.

This module runs ffmpeg/ffprobe in analysis mode so we avoid loading
large audio buffers into Python. It is designed to stay friendly to
Fly.io shared CPU and ~1 GB RAM limits.

We primarily compute:
- duration (seconds)
- sample rate
- peak level (dBFS, approx)
- integrated loudness (LUFS, via loudnorm analysis)

All heavy DSP work is delegated to ffmpeg.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrackAnalysis:
    path: str
    duration_s: float
    sample_rate: int
    peak_dbfs: Optional[float]
    integrated_lufs: Optional[float]


def _run(cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        shell=True,
        check=True,
        capture_output=True,
        text=True,
    )


def analyse_basic(path: str) -> TrackAnalysis:
    """Use ffprobe + loudnorm (analysis mode) to get basic stats.

    This keeps analysis streaming inside ffmpeg rather than pulling
    waveforms into Python.
    """

    # 1) Basic stream info (duration, sample rate)
    probe_cmd = (
        f'ffprobe -v error -select_streams a:0 '
        f'-show_entries stream=sample_rate,duration '
        f'-of json "{path}"'
    )
    probe = _run(probe_cmd)
    info = json.loads(probe.stdout or "{}")
    streams = info.get("streams") or []

    duration_s = 0.0
    sample_rate = 44100
    if streams:
        s = streams[0]
        try:
            duration_s = float(s.get("duration") or 0.0)
        except (TypeError, ValueError):
            duration_s = 0.0
        try:
            sample_rate = int(s.get("sample_rate") or 44100)
        except (TypeError, ValueError):
            sample_rate = 44100

    # 2) Loudness analysis via loudnorm in analysis-only mode
    loudnorm_cmd = (
        f'ffmpeg -y -i "{path}" -af '
        '"loudnorm=I=-14:TP=-1.0:LRA=11:print_format=json" '
        '-f null -'
    )

    peak_dbfs: Optional[float] = None
    integrated_lufs: Optional[float] = None

    try:
        loud = _run(loudnorm_cmd)
        # ffmpeg prints the JSON block to stderr
        text = loud.stderr
        start = text.rfind("{\n")
        if start != -1:
            j = json.loads(text[start:])
            integrated_lufs = float(j.get("input_i"))
            peak_dbfs = float(j.get("input_tp"))
    except subprocess.CalledProcessError:
        # Fall back to basic info only
        peak_dbfs = None
        integrated_lufs = None

    return TrackAnalysis(
        path=path,
        duration_s=duration_s,
        sample_rate=sample_rate,
        peak_dbfs=peak_dbfs,
        integrated_lufs=integrated_lufs,
    )
