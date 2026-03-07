from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from s3 import generate_presigned_download_url
from services.mix_engine import (
    calculate_gain_adjustment,
    create_vocal_space,
    detect_track_role,
    extract_audio_features,
    predict_plugin_chain,
)


router = APIRouter(prefix="", tags=["analysis"])

_ALLOWED_TRACK_ROLES = {
    "lead_vocal",
    "background_vocal",
    "beat",
    "bass",
    "melody",
    "drums",
    "vocal",
    "background",
    "adlib",
    "instrument",
}

_ANALYSIS_MAX_CONCURRENT = max(1, int(os.getenv("ANALYSIS_MAX_CONCURRENT", "2")))
_ANALYSIS_ESTIMATED_SEC_PER_TRACK = float(os.getenv("ANALYSIS_ESTIMATED_SEC_PER_TRACK", "6"))
_analysis_semaphore = asyncio.Semaphore(_ANALYSIS_MAX_CONCURRENT)
_analysis_queue_lock = asyncio.Lock()
_analysis_active_jobs = 0
_analysis_waiting_jobs = 0


class AnalyzeTrackRequest(BaseModel):
    track_id: str = Field(min_length=1, max_length=128)
    role: str | None = Field(default=None, max_length=64)
    s3_url: str | None = Field(default=None, max_length=4096)
    s3_key: str | None = Field(default=None, max_length=2048)


class AnalyzeMixRequest(BaseModel):
    tracks: list[AnalyzeTrackRequest] = Field(min_length=1)
    genre: str | None = Field(default=None, max_length=64)
    preset: str | None = Field(default=None, max_length=128)


class QueueStatus(BaseModel):
    position: int
    estimated_wait: int


class AnalyzeTrackResponse(BaseModel):
    track_id: str
    role: str
    features: dict[str, float]
    gain_adjustment_db: float
    stereo_width: float
    eq_dip: dict[str, float] | None = None
    plugins: list[dict[str, Any]]


class SidechainResponse(BaseModel):
    source: str
    target: str
    ratio: float
    attack: float
    release: float
    reduction_db: float


class AnalyzeMixResponse(BaseModel):
    tracks: list[AnalyzeTrackResponse]
    queue: QueueStatus
    sidechains: list[SidechainResponse]


def _resolve_track_url(track: AnalyzeTrackRequest) -> str:
    if track.s3_url:
        return track.s3_url
    if track.s3_key:
        return generate_presigned_download_url(track.s3_key, expires=900)
    raise HTTPException(status_code=400, detail=f"Track {track.track_id} missing s3_url or s3_key")


def _normalize_role(role: str) -> str:
    key = role.strip().lower()
    if key not in _ALLOWED_TRACK_ROLES:
        return "lead_vocal"
    if key == "vocal":
        return "lead_vocal"
    if key == "background":
        return "background_vocal"
    if key == "instrument":
        return "melody"
    if key == "adlib":
        return "background_vocal"
    return key


def _maybe_build_sidechain(track_rows: list[dict[str, Any]]) -> list[SidechainResponse]:
    lead = next((row for row in track_rows if row["role"] == "lead_vocal"), None)
    beat = next((row for row in track_rows if row["role"] == "beat"), None)
    if lead is None or beat is None:
        return []

    return [
        SidechainResponse(
            source=str(lead["track_id"]),
            target=str(beat["track_id"]),
            ratio=2.0,
            attack=10.0,
            release=120.0,
            reduction_db=1.5,
        )
    ]


async def _download_track_to_temp(url: str, file_suffix: str = ".wav") -> Path:
    timeout = httpx.Timeout(connect=10.0, read=120.0, write=60.0, pool=20.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(url)
        if resp.status_code >= 400:
            raise HTTPException(status_code=400, detail=f"Failed downloading track from source URL ({resp.status_code})")

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as handle:
            handle.write(resp.content)
            return Path(handle.name)


@router.post("/analyze-mix", response_model=AnalyzeMixResponse)
async def analyze_mix(payload: AnalyzeMixRequest) -> AnalyzeMixResponse:
    global _analysis_active_jobs, _analysis_waiting_jobs

    async with _analysis_queue_lock:
        position = _analysis_active_jobs + _analysis_waiting_jobs + 1
        _analysis_waiting_jobs += 1

    estimated_wait_sec = int(max(0, position - 1) * _ANALYSIS_ESTIMATED_SEC_PER_TRACK * max(1, len(payload.tracks)))

    await _analysis_semaphore.acquire()
    async with _analysis_queue_lock:
        _analysis_waiting_jobs = max(0, _analysis_waiting_jobs - 1)
        _analysis_active_jobs += 1

    try:
        analyzed_track_rows: list[dict[str, Any]] = []
        lead_vocal_features: dict[str, float] | None = None

        for track in payload.tracks:
            resolved_role = _normalize_role(track.role) if track.role else None
            source_url = _resolve_track_url(track)
            temp_path: Path | None = None
            try:
                temp_path = await _download_track_to_temp(source_url)
                features = extract_audio_features(temp_path)
                detected_role = resolved_role or detect_track_role(features)
                gain_info = calculate_gain_adjustment(detected_role, float(features.get("lufs", -18.0)))
                plugin_chain = predict_plugin_chain(
                    role=detected_role,
                    genre=payload.genre,
                    preset=payload.preset,
                )
                eq_dip: dict[str, float] | None = None

                if detected_role == "lead_vocal":
                    lead_vocal_features = features

                analyzed_track_rows.append(
                    {
                        "track_id": track.track_id,
                        "role": detected_role,
                        "features": features,
                        "gain_adjustment_db": float(gain_info["gain_adjustment_db"]),
                        "stereo_width": float(features.get("stereo_width", 1.0)),
                        "eq_dip": eq_dip,
                        "plugins": plugin_chain,
                    }
                )
            finally:
                if temp_path is not None:
                    try:
                        temp_path.unlink(missing_ok=True)
                    except Exception:
                        pass

        if lead_vocal_features is not None:
            vocal_space = create_vocal_space(lead_vocal_features)
            eq_dip = vocal_space.get("eq_dip")
            for row in analyzed_track_rows:
                if row["role"] == "beat":
                    row["eq_dip"] = eq_dip
                    row_plugins = list(row["plugins"])
                    if eq_dip is not None:
                        row_plugins.insert(0, {"plugin": "eq", "params": {"vocal_space_dip": eq_dip}})
                    row["plugins"] = row_plugins

        analyzed_tracks = [AnalyzeTrackResponse(**row) for row in analyzed_track_rows]
        sidechains = _maybe_build_sidechain(analyzed_track_rows)

        return AnalyzeMixResponse(
            tracks=analyzed_tracks,
            queue=QueueStatus(position=position, estimated_wait=estimated_wait_sec),
            sidechains=sidechains,
        )
    finally:
        async with _analysis_queue_lock:
            _analysis_active_jobs = max(0, _analysis_active_jobs - 1)
        _analysis_semaphore.release()
