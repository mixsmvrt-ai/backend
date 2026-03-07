from __future__ import annotations


_TARGET_LUFS: dict[str, float] = {
    "lead_vocal": -16.0,
    "background_vocal": -20.0,
    "beat": -18.0,
    "bass": -18.0,
    "drums": -18.0,
    "melody": -19.0,
}


def calculate_gain_adjustment(role: str, lufs: float) -> dict[str, float]:
    target_lufs = _TARGET_LUFS.get(role, -18.0)
    adjustment = target_lufs - float(lufs)
    adjustment = max(-8.0, min(8.0, adjustment))
    return {
        "target_lufs": target_lufs,
        "gain_adjustment_db": adjustment,
    }
