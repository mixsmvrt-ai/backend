import importlib
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
DSP_ROOT = ROOT / "mixsmvrt-dsp"
if str(DSP_ROOT) not in sys.path:
    sys.path.append(str(DSP_ROOT))

pipeline = importlib.import_module("app.dsp_engine.pipeline")
process_audio_cleanup = pipeline.process_audio_cleanup
process_mixing_only = pipeline.process_mixing_only
process_mix_master = pipeline.process_mix_master
process_mastering_only = pipeline.process_mastering_only


def _to_pipeline_shape(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        return data.astype(np.float32)
    if data.ndim == 2:
        return data.T.astype(np.float32)
    raise ValueError("Unsupported audio shape")


def _to_soundfile_shape(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        return data
    if data.ndim == 2:
        return data.T
    raise ValueError("Unsupported processed audio shape")


def _looks_vocal(preset_name: str | None) -> bool:
    if not preset_name:
        return False
    key = preset_name.lower()
    return any(token in key for token in ("vocal", "lead", "rap", "rnb", "dancehall", "reggae"))


def process_audio_file(
    input_path: str,
    output_path: str,
    *,
    flow_type: str,
    preset_name: str | None,
) -> None:
    data, sr = sf.read(input_path)
    x = _to_pipeline_shape(data)

    track_name = Path(input_path).name
    is_vocal = _looks_vocal(preset_name)

    if flow_type == "audio_cleanup":
        out, _ = process_audio_cleanup(track_name, x, sr)
    elif flow_type == "mixing_only":
        out, _ = process_mixing_only(track_name, x, sr, is_vocal=is_vocal)
    elif flow_type == "mix_master":
        out, _ = process_mix_master(track_name, x, sr, is_vocal=is_vocal)
    elif flow_type == "mastering_only":
        out, _ = process_mastering_only(track_name, x, sr)
    else:
        raise ValueError(f"Unsupported flow_type '{flow_type}'")

    out_sf = _to_soundfile_shape(out)
    sf.write(output_path, out_sf, sr)
