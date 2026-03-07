from __future__ import annotations

from pathlib import Path
from typing import cast

import librosa
import numpy as np
import pyloudnorm as pyln

try:  # pragma: no cover - optional dependency
    import essentia.standard as es  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    es = None


def _to_stereo(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return np.vstack([audio, audio])
    if audio.shape[0] == 1:
        return np.vstack([audio[0], audio[0]])
    return audio[:2]


def _db(value: float, floor: float = 1e-12) -> float:
    return float(20.0 * np.log10(max(value, floor)))


def _band_energy(power_spectrum: np.ndarray, freqs: np.ndarray, low_hz: float, high_hz: float) -> float:
    mask = (freqs >= low_hz) & (freqs < high_hz)
    if not np.any(mask):
        return 0.0
    return float(np.mean(power_spectrum[mask]))


def extract_audio_features(file_path: str | Path) -> dict[str, float]:
    loaded_audio, loaded_sr = librosa.load(str(file_path), sr=None, mono=False)
    audio = np.asarray(loaded_audio, dtype=np.float32)
    sample_rate = int(loaded_sr)
    stereo = _to_stereo(audio)
    mono = np.asarray(np.mean(stereo, axis=0), dtype=np.float32)

    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0

    meter = pyln.Meter(sample_rate)
    try:
        lufs = float(meter.integrated_loudness(mono.astype(np.float64)))
    except Exception:
        lufs = _db(rms)

    frame_length = max(1024, int(sample_rate * 0.05))
    hop_length = max(256, frame_length // 2)
    rms_frames = librosa.feature.rms(y=cast(np.ndarray, mono), frame_length=frame_length, hop_length=hop_length)[0]
    rms_db_frames = librosa.amplitude_to_db(np.maximum(rms_frames, 1e-8), ref=1.0)
    dynamic_range = float(np.percentile(rms_db_frames, 95) - np.percentile(rms_db_frames, 10)) if rms_db_frames.size else 0.0
    rms_stability = float(np.std(rms_db_frames)) if rms_db_frames.size else 0.0

    centroid = librosa.feature.spectral_centroid(y=cast(np.ndarray, mono), sr=sample_rate)
    rolloff = librosa.feature.spectral_rolloff(y=cast(np.ndarray, mono), sr=sample_rate)
    flatness = librosa.feature.spectral_flatness(y=cast(np.ndarray, mono))
    contrast = librosa.feature.spectral_contrast(y=cast(np.ndarray, mono), sr=sample_rate)

    stft = np.abs(librosa.stft(cast(np.ndarray, mono), n_fft=2048, hop_length=hop_length))
    power = np.mean(np.square(stft), axis=1) if stft.size else np.zeros(1, dtype=np.float32)
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)

    low_energy = _band_energy(power, freqs, 20.0, 250.0)
    mid_energy = _band_energy(power, freqs, 250.0, 4000.0)
    high_energy = _band_energy(power, freqs, 4000.0, 16000.0)
    total_energy = max(low_energy + mid_energy + high_energy, 1e-8)

    onsets = librosa.onset.onset_detect(y=cast(np.ndarray, mono), sr=sample_rate, hop_length=hop_length, units="frames")
    transient_density = float(len(onsets) / max(len(mono) / sample_rate, 1e-6))

    left = stereo[0]
    right = stereo[1]
    mid = (left + right) * 0.5
    side = (left - right) * 0.5
    stereo_width = float(np.mean(np.square(side)) / max(np.mean(np.square(mid)), 1e-8))

    spectral_flatness = float(np.mean(flatness)) if flatness.size else 0.0
    if es is not None:
        try:  # pragma: no cover
            spectral_flatness = 0.5 * spectral_flatness + 0.5 * float(es.Flatness()(mono.astype(np.float32)))
        except Exception:
            pass

    return {
        "lufs": lufs,
        "peak_db": _db(peak),
        "rms_db": _db(rms),
        "dynamic_range": dynamic_range,
        "spectral_centroid": float(np.mean(centroid)) if centroid.size else 0.0,
        "spectral_rolloff": float(np.mean(rolloff)) if rolloff.size else 0.0,
        "spectral_flatness": spectral_flatness,
        "spectral_contrast": float(np.mean(contrast)) if contrast.size else 0.0,
        "low_energy": float(low_energy / total_energy),
        "mid_energy": float(mid_energy / total_energy),
        "high_energy": float(high_energy / total_energy),
        "stereo_width": float(np.clip(stereo_width, 0.0, 3.0)),
        "transient_density": transient_density,
        "rms_stability": rms_stability,
    }
