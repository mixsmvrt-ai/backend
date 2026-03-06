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


def _to_stereo_channels(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return np.vstack([audio, audio])
    if audio.ndim == 2:
        if audio.shape[0] == 1:
            return np.vstack([audio[0], audio[0]])
        return audio[:2]
    return np.vstack([audio.reshape(-1), audio.reshape(-1)])


def _db_from_linear(value: float, floor: float = 1e-12) -> float:
    return float(20.0 * np.log10(max(float(value), floor)))


def _band_energy(power_spectrum: np.ndarray, freqs: np.ndarray, low_hz: float, high_hz: float) -> float:
    mask = (freqs >= low_hz) & (freqs < high_hz)
    if not np.any(mask):
        return 0.0
    band_power = float(np.mean(power_spectrum[mask]))
    return max(0.0, band_power)


def analyze_audio(file_path: str | Path) -> dict[str, float]:
    loaded_audio, loaded_sr = librosa.load(str(file_path), sr=None, mono=False)
    audio = np.asarray(loaded_audio, dtype=np.float32)
    sample_rate = int(loaded_sr)

    channels = _to_stereo_channels(audio)
    mono = np.mean(channels, axis=0, dtype=np.float32)
    mono = np.asarray(mono, dtype=np.float32)

    peak_linear = float(np.max(np.abs(mono))) if mono.size else 0.0
    rms_linear = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0

    meter = pyln.Meter(int(sample_rate))
    try:
        lufs = float(meter.integrated_loudness(mono.astype(np.float64)))
    except Exception:
        lufs = _db_from_linear(rms_linear)

    frame_length = max(1024, int(sample_rate * 0.05))
    hop_length = max(256, int(frame_length // 2))
    rms_frames = librosa.feature.rms(y=cast(np.ndarray, mono), frame_length=frame_length, hop_length=hop_length)[0]
    rms_db_frames = librosa.amplitude_to_db(np.maximum(rms_frames, 1e-8), ref=1.0)
    dynamic_range = float(np.percentile(rms_db_frames, 95) - np.percentile(rms_db_frames, 10)) if rms_db_frames.size else 0.0

    centroid = librosa.feature.spectral_centroid(y=cast(np.ndarray, mono), sr=sample_rate)
    rolloff = librosa.feature.spectral_rolloff(y=cast(np.ndarray, mono), sr=sample_rate)
    flatness = librosa.feature.spectral_flatness(y=cast(np.ndarray, mono))

    onsets = librosa.onset.onset_detect(y=cast(np.ndarray, mono), sr=sample_rate, hop_length=hop_length, units="frames")
    duration_sec = max(1e-6, float(len(mono) / sample_rate))
    transient_density = float(len(onsets) / duration_sec)

    harmonic, percussive = librosa.effects.hpss(cast(np.ndarray, mono))
    harmonic_energy = float(np.mean(np.square(harmonic)))
    percussive_energy = float(np.mean(np.square(percussive)))
    harmonic_content = harmonic_energy / max(harmonic_energy + percussive_energy, 1e-8)

    left = channels[0]
    right = channels[1]
    mid = (left + right) * 0.5
    side = (left - right) * 0.5
    mid_energy = float(np.mean(np.square(mid)))
    side_energy = float(np.mean(np.square(side)))
    stereo_width = side_energy / max(mid_energy, 1e-8)

    stft = np.abs(librosa.stft(cast(np.ndarray, mono), n_fft=2048, hop_length=hop_length))
    power = np.mean(np.square(stft), axis=1) if stft.size else np.zeros(1, dtype=np.float32)
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)

    low_energy = _band_energy(power, freqs, 20.0, 250.0)
    mid_energy_band = _band_energy(power, freqs, 250.0, 4000.0)
    high_energy = _band_energy(power, freqs, 4000.0, 16000.0)
    total_energy = max(low_energy + mid_energy_band + high_energy, 1e-8)

    sibilance_band = _band_energy(power, freqs, 5000.0, 11000.0)
    sibilance_score = float(np.clip(sibilance_band / total_energy * 3.0, 0.0, 1.5))

    if es is not None:
        try:  # pragma: no cover - optional enhancement
            essentia_flatness = float(es.Flatness()(mono.astype(np.float32)))
            spectral_flatness = 0.5 * float(np.mean(flatness)) + 0.5 * essentia_flatness
        except Exception:
            spectral_flatness = float(np.mean(flatness))
    else:
        spectral_flatness = float(np.mean(flatness))

    return {
        "lufs": float(lufs),
        "peak_db": _db_from_linear(peak_linear),
        "rms_db": _db_from_linear(rms_linear),
        "dynamic_range": float(dynamic_range),
        "spectral_centroid": float(np.mean(centroid)) if centroid.size else 0.0,
        "spectral_rolloff": float(np.mean(rolloff)) if rolloff.size else 0.0,
        "spectral_flatness": float(spectral_flatness),
        "transient_density": transient_density,
        "harmonic_content": float(np.clip(harmonic_content, 0.0, 1.0)),
        "stereo_width": float(np.clip(stereo_width, 0.0, 3.0)),
        "sibilance_score": sibilance_score,
        "low_end_energy": float(low_energy / total_energy),
        "mid_energy": float(mid_energy_band / total_energy),
        "high_energy": float(high_energy / total_energy),
    }
