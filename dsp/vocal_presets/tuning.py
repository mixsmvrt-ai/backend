from __future__ import annotations

# pyright: strict=false, reportGeneralTypeIssues=false, reportUnknownMemberType=false, reportUnknownArgumentType=false

import os
from typing import Optional

import numpy as np
from pedalboard import load_plugin, Plugin


_TUNER_PLUGIN: Optional[Plugin] = None
_TUNER_ENV_VAR = "VOCAL_TUNER_PLUGIN"


def _get_tuner_plugin() -> Optional[Plugin]:
    """Lazily load an external pitch‑correction plugin if configured.

    If the environment variable ``VOCAL_TUNER_PLUGIN`` points to a valid
    VST3/Audio Unit effect, it will be loaded once and cached. If loading
    fails for any reason, pitch correction is silently disabled.
    """
    global _TUNER_PLUGIN

    if _TUNER_PLUGIN is not None:
        return _TUNER_PLUGIN

    plugin_path = os.getenv(_TUNER_ENV_VAR)
    if not plugin_path:
        return None

    try:
        _TUNER_PLUGIN = load_plugin(plugin_path)
    except Exception:
        # If the plugin can't be loaded, just disable pitch correction.
        _TUNER_PLUGIN = None
    return _TUNER_PLUGIN


def apply_pitch_correction(audio: np.ndarray, sr: int) -> np.ndarray:
    """Apply optional pitch correction using an external tuner plugin.

    This function is intentionally conservative:

    - If no tuner plugin is configured, the audio is returned unchanged.
    - Any error during processing causes the original audio to be returned.

    The expectation is that ``VOCAL_TUNER_PLUGIN`` points to an Auto‑Tune‑
    style VST3/Audio Unit effect that has its own internal key/scale
    configuration or default preset.
    """
    if audio.size == 0:
        return audio

    tuner = _get_tuner_plugin()
    if tuner is None:
        return audio

    try:
        # ``audio`` is already in (n_samples, n_channels) float32 shape when
        # used inside the vocal presets, so we can pass it directly.
        processed = tuner(audio.astype("float32"), sr)
    except Exception:
        return audio

    if not isinstance(processed, np.ndarray) or processed.size == 0:
        return audio
    return processed.astype("float32")
