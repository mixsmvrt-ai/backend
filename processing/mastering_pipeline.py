
from __future__ import annotations

from typing import Any, Dict

from .analysis import analyse_basic, TrackAnalysis
from .gain_staging import GainDecision, decide_input_gain
from .ffmpeg_render import render_master


def ai_master(
    input_mix: str,
    output_master: str,
    target_lufs: str = "-14",
) -> Dict[str, Any]:
    """Hybrid Python+ffmpeg mastering pipeline.

    Steps:
    - Analyse the input mix via ffmpeg-based loudness analysis
    - Compute a simple mix-bus gain staging decision
    - Render master via ffmpeg (EQ, compression, limiter, loudnorm)
    - Return a JSON-friendly report including analysis and gain decision
    """

    # 1) Analyse the pre-master mix
    mix_analysis: TrackAnalysis = analyse_basic(input_mix)

    # 2) Decide pre-master gain for mix-bus role
    mix_gain: GainDecision = decide_input_gain("mix-bus", mix_analysis)

    # 3) We currently fold the gain into the mastering render by
    #     letting ffmpeg handle dynamics and loudnorm. If we later
    #     want explicit input gain, we can prepend a volume filter.
    rendered_master = render_master(
        input_mix=input_mix,
        output_master=output_master,
        target_lufs=target_lufs,
        extra_ceiling_db=-1.0,
    )

    # 4) Build a JSON-friendly report
    return {
        "output_path": rendered_master,
        "input": {
            "path": input_mix,
            "analysis": {
                "duration_s": mix_analysis.duration_s,
                "sample_rate": mix_analysis.sample_rate,
                "peak_dbfs": mix_analysis.peak_dbfs,
                "integrated_lufs": mix_analysis.integrated_lufs,
            },
            "gain_decision_db": mix_gain.input_gain_db,
        },
        "target_lufs": target_lufs,
    }
