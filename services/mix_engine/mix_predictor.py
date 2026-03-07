from __future__ import annotations


def predict_plugin_chain(role: str, genre: str | None, preset: str | None) -> list[dict[str, object]]:
    role_key = (role or "melody").lower()
    genre_key = (genre or "unknown").lower()
    preset_key = (preset or "default").lower()
    aggressive = genre_key in {"dancehall", "trap", "drill", "hiphop"} or "kartel" in preset_key

    vocal_chain = [
        {
            "plugin": "eq",
            "params": {
                "highpass": 80,
                "low_cut_db": -3,
                "low_cut_freq": 250,
                "presence_boost_db": 3 if aggressive else 2,
                "presence_freq": 5000 if aggressive else 4200,
            },
        },
        {
            "plugin": "deesser",
            "params": {
                "frequency": 6200,
                "reduction": 4,
            },
        },
        {
            "plugin": "compressor",
            "params": {
                "ratio": 4 if aggressive else 3,
                "attack": 15,
                "release": 80,
                "threshold": -18,
            },
        },
        {
            "plugin": "saturation",
            "params": {
                "drive": 0.18 if aggressive else 0.12,
            },
        },
        {
            "plugin": "stereo_width",
            "params": {
                "width": 1.05,
            },
        },
    ]

    if role_key == "lead_vocal":
        return vocal_chain
    if role_key == "background_vocal":
        return vocal_chain[:-1] + [
            {"plugin": "reverb", "params": {"size": 60, "decay": 1.8, "mix": 0.2}},
            {"plugin": "delay", "params": {"time_ms": 240, "feedback": 22, "mix": 0.1}},
            {"plugin": "stereo_width", "params": {"width": 1.2}},
        ]
    if role_key == "beat":
        return [
            {"plugin": "eq", "params": {"highpass": 30, "low_cut_db": -1, "low_cut_freq": 180, "presence_boost_db": 1, "presence_freq": 3000}},
            {"plugin": "compressor", "params": {"ratio": 2, "attack": 10, "release": 120, "threshold": -16}},
            {"plugin": "saturation", "params": {"drive": 0.1}},
            {"plugin": "stereo_width", "params": {"width": 1.15}},
        ]
    if role_key == "bass":
        return [
            {"plugin": "eq", "params": {"highpass": 28, "low_cut_db": 0, "low_cut_freq": 120, "presence_boost_db": 0, "presence_freq": 1200}},
            {"plugin": "compressor", "params": {"ratio": 4, "attack": 20, "release": 110, "threshold": -20}},
            {"plugin": "saturation", "params": {"drive": 0.14}},
        ]
    if role_key == "drums":
        return [
            {"plugin": "eq", "params": {"highpass": 40, "low_cut_db": -1, "low_cut_freq": 220, "presence_boost_db": 2, "presence_freq": 4500}},
            {"plugin": "compressor", "params": {"ratio": 3, "attack": 8, "release": 70, "threshold": -18}},
            {"plugin": "stereo_width", "params": {"width": 1.1}},
        ]

    return [
        {"plugin": "eq", "params": {"highpass": 60, "low_cut_db": -2, "low_cut_freq": 220, "presence_boost_db": 2, "presence_freq": 3800}},
        {"plugin": "compressor", "params": {"ratio": 2.5, "attack": 18, "release": 100, "threshold": -18}},
        {"plugin": "stereo_width", "params": {"width": 1.1}},
    ]
