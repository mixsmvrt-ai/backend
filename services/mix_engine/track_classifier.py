from __future__ import annotations


def detect_track_role(features: dict[str, float]) -> str:
    mid_energy = float(features.get("mid_energy", 0.0))
    low_energy = float(features.get("low_energy", 0.0))
    high_energy = float(features.get("high_energy", 0.0))
    dynamic_range = float(features.get("dynamic_range", 0.0))
    spectral_centroid = float(features.get("spectral_centroid", 0.0))
    stereo_width = float(features.get("stereo_width", 0.0))
    rms_stability = float(features.get("rms_stability", 99.0))
    rms_db = float(features.get("rms_db", -18.0))

    if low_energy > 0.6 and spectral_centroid < 200:
        return "bass"

    if (
        mid_energy > 0.4
        and dynamic_range > 7.0
        and 1500.0 <= spectral_centroid <= 5000.0
    ):
        return "lead_vocal"

    if (
        0.25 <= low_energy <= 0.45
        and 0.25 <= mid_energy <= 0.45
        and 0.15 <= high_energy <= 0.35
        and stereo_width > 0.5
        and rms_stability < 4.5
    ):
        return "beat"

    if (
        mid_energy > 0.34
        and dynamic_range <= 7.0
        and 1200.0 <= spectral_centroid <= 4800.0
        and rms_db < -16.0
    ):
        return "background_vocal"

    if high_energy > 0.34 and transient_density_like(features) > 4.5:
        return "drums"

    return "melody"


def transient_density_like(features: dict[str, float]) -> float:
    return float(features.get("transient_density", 0.0))
