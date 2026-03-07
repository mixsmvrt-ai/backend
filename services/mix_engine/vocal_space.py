from __future__ import annotations


def create_vocal_space(vocal_features: dict[str, float]) -> dict[str, dict[str, float]]:
    centroid = float(vocal_features.get("spectral_centroid", 3000.0))
    high_energy = float(vocal_features.get("high_energy", 0.2))
    mid_energy = float(vocal_features.get("mid_energy", 0.4))

    presence_freq = centroid
    if presence_freq < 2000.0:
        presence_freq = 2000.0 + mid_energy * 1800.0
    if presence_freq > 6000.0:
        presence_freq = 6000.0 - high_energy * 400.0

    dip_gain = -2.0 - min(2.0, (mid_energy + high_energy) * 1.8)
    dip_q = 1.2 + min(0.6, high_energy)

    return {
        "eq_dip": {
            "freq": round(presence_freq, 2),
            "gain": round(dip_gain, 2),
            "q": round(dip_q, 2),
        }
    }
