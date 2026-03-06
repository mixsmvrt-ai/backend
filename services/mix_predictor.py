from __future__ import annotations

from typing import Any


TrackRole = str


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _base_eq_from_features(features: dict[str, float]) -> dict[str, float]:
    low_end = float(features.get("low_end_energy", 0.3))
    high_end = float(features.get("high_energy", 0.3))
    centroid = float(features.get("spectral_centroid", 3500.0))

    return {
        "highpass": _clamp(55 + low_end * 80, 40, 140),
        "low_cut_db": _clamp(-1.0 - low_end * 5.0, -8, 0),
        "low_cut_freq": _clamp(180 + low_end * 180, 120, 420),
        "presence_boost_db": _clamp(1.5 + (centroid / 10000.0) * 3.0, 0, 6),
        "presence_freq": _clamp(3500 + high_end * 3000, 2500, 8000),
    }


def _compressor_from_features(features: dict[str, float], aggressive: bool = False) -> dict[str, float]:
    dynamic_range = float(features.get("dynamic_range", 8.0))
    rms_db = float(features.get("rms_db", -18.0))

    ratio = 3.0 + (4.0 if aggressive else 2.0) * (dynamic_range / 16.0)
    threshold = rms_db - (5.0 if aggressive else 3.0)

    return {
        "ratio": _clamp(ratio, 1.5, 8.0),
        "attack": _clamp(18 - dynamic_range, 3, 40),
        "release": _clamp(60 + dynamic_range * 8.0, 30, 260),
        "threshold": _clamp(threshold, -40, -6),
    }


def _deesser_from_features(features: dict[str, float]) -> dict[str, float]:
    sib = float(features.get("sibilance_score", 0.4))
    spectral_rolloff = float(features.get("spectral_rolloff", 7000.0))

    return {
        "frequency": _clamp(spectral_rolloff * 0.75, 4500, 9000),
        "reduction": _clamp(2 + sib * 6.0, 1.0, 9.0),
    }


def _stereo_from_features(features: dict[str, float]) -> dict[str, float]:
    width = float(features.get("stereo_width", 1.0))
    return {
        "width": _clamp(0.9 + width * 0.25, 0.8, 1.45),
    }


def _saturation_from_features(features: dict[str, float]) -> dict[str, float]:
    transients = float(features.get("transient_density", 4.0))
    drive = 0.1 + min(0.3, transients / 60.0)
    return {"drive": float(_clamp(drive, 0.06, 0.4))}


def predict_plugin_chain(
    features: dict[str, float],
    track_role: TrackRole,
    genre: str | None,
    preset: str | None,
) -> dict[str, Any]:
    role = (track_role or "lead_vocal").lower().strip()
    genre_key = (genre or "unknown").lower().strip()
    preset_key = (preset or "default").lower().strip()

    eq = _base_eq_from_features(features)
    deesser = _deesser_from_features(features)
    compressor = _compressor_from_features(
        features,
        aggressive=(genre_key in {"dancehall", "trap", "drill"} or "aggressive" in preset_key),
    )
    saturation = _saturation_from_features(features)
    stereo = _stereo_from_features(features)

    if role in {"lead_vocal", "vocal"}:
        chain = [
            {"plugin": "eq", "params": eq},
            {"plugin": "deesser", "params": deesser},
            {"plugin": "compressor", "params": compressor},
            {"plugin": "saturation", "params": saturation},
        ]
    elif role in {"background_vocal", "bg_vocal", "background"}:
        chain = [
            {"plugin": "eq", "params": {**eq, "highpass": max(120.0, eq["highpass"] + 30.0)}},
            {"plugin": "compressor", "params": {**compressor, "ratio": _clamp(compressor["ratio"] + 0.5, 2, 8)}},
            {"plugin": "reverb", "params": {"size": 62, "decay": 1.9, "mix": 0.22}},
            {"plugin": "delay", "params": {"time_ms": 260, "feedback": 25, "mix": 0.12}},
        ]
    elif role in {"beat", "drums"}:
        chain = [
            {"plugin": "eq", "params": {**eq, "presence_boost_freq": 3000}},
            {"plugin": "compressor", "params": {**compressor, "attack": 12, "release": 80}},
            {"plugin": "saturation", "params": {"drive": _clamp(saturation["drive"] + 0.05, 0.08, 0.45)}},
            {"plugin": "stereo_width", "params": stereo},
        ]
    elif role == "bass":
        chain = [
            {"plugin": "eq", "params": {**eq, "highpass": 35, "presence_boost_db": 0.8, "presence_freq": 1200}},
            {"plugin": "compressor", "params": {**compressor, "ratio": _clamp(compressor["ratio"] + 1.5, 2, 8)}},
            {"plugin": "saturation", "params": {"drive": _clamp(saturation["drive"] + 0.07, 0.1, 0.45)}},
        ]
    elif role in {"melody", "instrument"}:
        chain = [
            {"plugin": "eq", "params": eq},
            {"plugin": "compressor", "params": {**compressor, "ratio": _clamp(compressor["ratio"] - 0.7, 1.5, 5)}},
            {"plugin": "stereo_width", "params": stereo},
            {"plugin": "reverb", "params": {"size": 58, "decay": 1.6, "mix": 0.16}},
        ]
    else:
        chain = [
            {"plugin": "eq", "params": eq},
            {"plugin": "compressor", "params": compressor},
            {"plugin": "saturation", "params": saturation},
        ]

    return {
        "track_role": role,
        "genre": genre_key,
        "preset": preset_key,
        "plugin_chain": chain,
    }
