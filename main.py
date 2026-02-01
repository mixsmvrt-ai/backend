from fastapi import FastAPI, File, UploadFile, Form, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import uuid
import asyncio
import os
from datetime import datetime, timedelta, date
from typing import Literal, Optional, Any, Dict, List

import httpx

from pydantic import BaseModel

from processing.mixing_pipeline import ai_mix
from processing.mastering_pipeline import ai_master
from health import create_health_router
from supabase_client import (
    create_processing_job,
    update_processing_job,
    get_processing_job,
    supabase_select,
    supabase_patch,
    supabase_insert,
    SupabaseConfigError,
)
from progress import update_progress, mark_job_complete, mark_job_failed

app = FastAPI(title="RiddimBase Studio Backend")

# Base URL for the external DSP service. This is proxied via /dsp/* endpoints
# so the browser only talks to this backend (which already has CORS configured).
DSP_BASE_URL = os.getenv("DSP_URL", "https://mixsmvrt-dsp-1.onrender.com").rstrip("/")

# Allow local Next.js dev and the deployed studio frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
        "https://mixsmvrt.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lightweight health + readiness endpoints for uptime monitoring
health_router = create_health_router(service_name="backend", version="1.0.0")
app.include_router(health_router)

BASE_DIR = Path(__file__).resolve().parent
SESSIONS_DIR = BASE_DIR / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)


def _save_upload(upload: UploadFile | None, dest: Path | None) -> Path | None:
    if upload is None or dest is None:
        return None
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dest


class GenrePredictionResponse(BaseModel):
    genre: str
    confidence: float


@app.post("/api/genre-detect")
async def genre_detect(file: UploadFile = File(...)) -> GenrePredictionResponse:
    """Stub endpoint for genre detection.

    In a future iteration this can call a Torch / ONNX model to classify the
    uploaded audio. For now, it returns a neutral "unknown" prediction so the
    frontend can be wired up without breaking when the real model arrives.
    """

    # We intentionally ignore the contents for now; the file is read so large
    # uploads don't sit in memory unused.
    await file.read()  # pragma: no cover - placeholder behaviour
    return GenrePredictionResponse(genre="unknown", confidence=0.3)


# -------------------------
# Processing job tracking (Supabase-backed)
# -------------------------


class ProcessRequest(BaseModel):
    user_id: Optional[str] = None
    job_type: Literal["mix", "master", "mix_master"] = "mix_master"
    preset_key: Optional[str] = None
    feature_type: Literal[
        "audio_cleanup",
        "mixing_only",
        "mix_master",
        "mastering_only",
    ] = "mix_master"
    # Arbitrary JSON metadata about where the input files live
    # (e.g. Supabase storage paths, S3 URLs, or internal session IDs).
    # Optional studio preset metadata. When provided, the backend will
    # validate the preset against the studio registry and store resolved
    # details on the job for downstream DSP workers.
    preset_id: Optional[str] = None
    target: Optional[Literal["vocal", "beat", "full_mix"]] = None
    input_files: Dict[str, Any]


class StepStatus(BaseModel):
    name: str
    completed: bool


class JobStatusResponse(BaseModel):
    id: str
    status: Literal["queued", "processing", "completed", "failed"]
    progress: int
    current_stage: Optional[str] = None
    error_message: Optional[str] = None
    output_files: Optional[Dict[str, Any]] = None
    steps: Optional[list[StepStatus]] = None
    estimated_total_sec: Optional[float] = None
    elapsed_sec: Optional[float] = None


# -------------------------
# Support tickets (Supabase-backed)
# -------------------------


class SupportTicketCreate(BaseModel):
    user_id: Optional[str] = None
    email: str
    subject: str
    message: str


class SupportTicket(BaseModel):
    id: str
    user_id: Optional[str] = None
    email: str
    subject: str
    message: str
    status: str
    created_at: datetime
    created_at: datetime


@app.post("/api/support/tickets", response_model=SupportTicket)
async def create_support_ticket(payload: SupportTicketCreate):
    """Create a support ticket record in Supabase.

    The frontend passes the user's email, subject and message. If a user_id is
    provided it will be attached for easier correlation with jobs and billing.
    """

    try:
        ticket_data: Dict[str, Any] = {
            "email": payload.email,
            "subject": payload.subject,
            "message": payload.message,
        }
        if payload.user_id:
            ticket_data["user_id"] = payload.user_id

        row = await supabase_insert("support_tickets", ticket_data)
    except SupabaseConfigError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return SupportTicket(**row)


class SupportTicketListResponse(BaseModel):
    tickets: List[SupportTicket]


@app.get("/admin/support/tickets", response_model=SupportTicketListResponse)
async def list_support_tickets():
    """List recent support tickets for the admin panel.

    This endpoint is intended to be called from the admin UI using the
    backend's own Supabase service role; it is not exposed directly to
    untrusted browsers.
    """

    try:
        rows = await supabase_select(
            "support_tickets",
            params={"order": "created_at.desc"},
        )
    except SupabaseConfigError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    tickets = [SupportTicket(**row) for row in rows]
    return SupportTicketListResponse(tickets=tickets)


class SupportTicketStats(BaseModel):
    total: int
    open: int
    resolved: int
    closed: int


@app.get("/admin/support/tickets/stats", response_model=SupportTicketStats)
async def support_ticket_stats():
    """Return basic counts for support tickets for use in the admin UI."""

    try:
        rows = await supabase_select(
            "support_tickets",
            params={"select": "id,status"},
        )
    except SupabaseConfigError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    total = len(rows)
    open_count = sum(1 for row in rows if row.get("status") == "open")
    resolved_count = sum(1 for row in rows if row.get("status") == "resolved")
    closed_count = sum(1 for row in rows if row.get("status") == "closed")

    return SupportTicketStats(
        total=total,
        open=open_count,
        resolved=resolved_count,
        closed=closed_count,
    )


@app.post("/admin/support/tickets/{ticket_id}/resolve", response_model=SupportTicket)
async def resolve_support_ticket(ticket_id: str):
    """Mark a support ticket as resolved.

    This is used from the admin UI to move tickets from the "open" bucket
    into "resolved" while keeping the original record.
    """

    try:
        updated_rows = await supabase_patch(
            "support_tickets",
            {"id": f"eq.{ticket_id}"},
            {"status": "resolved"},
        )
    except SupabaseConfigError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not updated_rows:
        raise HTTPException(status_code=404, detail="Ticket not found")

    return SupportTicket(**updated_rows[0])


# -------------------------
# Studio presets registry (for UI + validation)
# -------------------------


PresetMode = Literal["audio_cleanup", "mixing_only", "mix_and_master", "mastering_only"]
PresetTarget = Literal["vocal", "beat", "full_mix"]


class StudioPreset(BaseModel):
    id: str
    name: str
    mode: PresetMode
    target: PresetTarget
    genre: Optional[str] = None
    description: str
    dsp_chain_reference: str
    tags: list[str] = []


STUDIO_PRESETS: list[StudioPreset] = [
    # Audio cleanup
    StudioPreset(
        id="podcast_clean",
        name="Podcast Clean",
        mode="audio_cleanup",
        target="vocal",
        description="Tightens spoken word, removes light noise and focuses intelligibility.",
        dsp_chain_reference="clean_vocal",
        tags=["Vocal", "Clean"],
    ),
    StudioPreset(
        id="voice_over_clean",
        name="Voice Over Clean",
        mode="audio_cleanup",
        target="vocal",
        description="Present, broadcast-style VO with controlled sibilance and proximity.",
        dsp_chain_reference="clean_vocal",
        tags=["Vocal", "Clean"],
    ),
    StudioPreset(
        id="noisy_room_cleanup",
        name="Noisy Room Cleanup",
        mode="audio_cleanup",
        target="vocal",
        description="Reduces constant background noise while keeping the voice natural.",
        dsp_chain_reference="clean_vocal",
        tags=["Vocal", "Noise"],
    ),
    StudioPreset(
        id="phone_recording_repair",
        name="Phone Recording Repair",
        mode="audio_cleanup",
        target="vocal",
        description="Evens out harsh phone captures and restores body to the voice.",
        dsp_chain_reference="clean_vocal",
        tags=["Vocal", "Fix"],
    ),
    StudioPreset(
        id="low_end_rumble_removal",
        name="Low-End Rumble Removal",
        mode="audio_cleanup",
        target="vocal",
        description="Aggressive high-pass and clean-up for HVAC, traffic and mic bumps.",
        dsp_chain_reference="clean_vocal",
        tags=["Vocal", "Clean"],
    ),
    StudioPreset(
        id="harshness_reduction",
        name="Harshness Reduction",
        mode="audio_cleanup",
        target="vocal",
        description="Tames edgy upper-mids on shouty or fatiguing recordings.",
        dsp_chain_reference="clean_vocal",
        tags=["Vocal", "Smooth"],
    ),
    StudioPreset(
        id="de_ess_cleanup",
        name="De-Ess Focused Cleanup",
        mode="audio_cleanup",
        target="vocal",
        description="Targets sibilance and breaths without over-compressing the signal.",
        dsp_chain_reference="clean_vocal",
        tags=["Vocal", "De-Ess"],
    ),
    StudioPreset(
        id="dialogue_clarity_boost",
        name="Dialogue Clarity Boost",
        mode="audio_cleanup",
        target="vocal",
        description="Lifts presence and intelligibility for interviews and dialogue.",
        dsp_chain_reference="clean_vocal",
        tags=["Vocal", "Clarity"],
    ),
    StudioPreset(
        id="room_echo_reduction",
        name="Room Echo Reduction",
        mode="audio_cleanup",
        target="vocal",
        description="Softens small-room reflections and flutter echo.",
        dsp_chain_reference="clean_vocal",
        tags=["Vocal", "Room"],
    ),
    StudioPreset(
        id="gentle_noise_reduction",
        name="Gentle Noise Reduction",
        mode="audio_cleanup",
        target="vocal",
        description="Light broadband cleanup that avoids pumping and artifacts.",
        dsp_chain_reference="clean_vocal",
        tags=["Vocal", "Subtle"],
    ),

    # Mixing only – vocals
    StudioPreset(
        id="hip_hop_vocal_pro",
        name="Hip Hop Vocal Pro",
        mode="mixing_only",
        target="vocal",
        genre="hiphop",
        description="Forward, polished hip-hop vocal designed to sit on top of the beat.",
        dsp_chain_reference="aggressive_rap",
        tags=["Vocal", "Rap", "Loud"],
    ),
    StudioPreset(
        id="trap_vocal_modern",
        name="Trap Vocal Modern",
        mode="mixing_only",
        target="vocal",
        genre="trap_dancehall",
        description="Autotune-friendly modern trap vocal with airy top and tight lows.",
        dsp_chain_reference="aggressive_rap",
        tags=["Vocal", "Trap"],
    ),
    StudioPreset(
        id="dancehall_vocal_punchy",
        name="Dancehall Vocal Punchy",
        mode="mixing_only",
        target="vocal",
        genre="dancehall",
        description="Punchy, bright dancehall vocal tuned for club systems.",
        dsp_chain_reference="dancehall",
        tags=["Vocal", "Dancehall"],
    ),
    StudioPreset(
        id="reggae_vocal_natural",
        name="Reggae Vocal Natural",
        mode="mixing_only",
        target="vocal",
        genre="reggae",
        description="Warm, rounded reggae vocal with moderate dynamics control.",
        dsp_chain_reference="reggae",
        tags=["Vocal", "Reggae", "Warm"],
    ),
    StudioPreset(
        id="rnb_vocal_smooth",
        name="R&B Vocal Smooth",
        mode="mixing_only",
        target="vocal",
        genre="rnb",
        description="Silky R&B vocal with softened edges and subtle width.",
        dsp_chain_reference="rnb",
        tags=["Vocal", "R&B", "Smooth"],
    ),
    StudioPreset(
        id="afrobeat_vocal_bright",
        name="Afrobeat Vocal Bright",
        mode="mixing_only",
        target="vocal",
        genre="afrobeat",
        description="Present afrobeat vocal chain with crisp top and controlled lows.",
        dsp_chain_reference="afrobeat",
        tags=["Vocal", "Afrobeat"],
    ),
    StudioPreset(
        id="reggaeton_vocal_wide",
        name="Reggaeton Vocal Wide",
        mode="mixing_only",
        target="vocal",
        genre="reggaeton",
        description="Wide, modern reggaeton vocal with focused midrange.",
        dsp_chain_reference="reggaeton",
        tags=["Vocal", "Reggaeton", "Wide"],
    ),
    StudioPreset(
        id="rock_vocal_grit",
        name="Rock Vocal Grit",
        mode="mixing_only",
        target="vocal",
        description="Adds saturation and bite for rock and alt vocals.",
        dsp_chain_reference="aggressive_rap",
        tags=["Vocal", "Rock"],
    ),
    StudioPreset(
        id="rap_vocal_aggressive",
        name="Rap Vocal Aggressive",
        mode="mixing_only",
        target="vocal",
        genre="rap",
        description="Hard-hitting rap vocal with dense compression and presence.",
        dsp_chain_reference="aggressive_rap",
        tags=["Vocal", "Rap", "Aggressive"],
    ),
    StudioPreset(
        id="clean_pop_vocal",
        name="Clean Pop Vocal",
        mode="mixing_only",
        target="vocal",
        description="Polished pop vocal with clean top and tight lows.",
        dsp_chain_reference="clean_vocal",
        tags=["Vocal", "Pop", "Clean"],
    ),

    # Mixing only – beat only
    StudioPreset(
        id="beat_balance_clean",
        name="Beat Balance Clean",
        mode="mixing_only",
        target="beat",
        description="Subtle balance and tone shaping for stereo beats.",
        dsp_chain_reference="streaming_master",
        tags=["Beat", "Clean"],
    ),
    StudioPreset(
        id="bass_controlled_beat",
        name="Bass-Controlled Beat",
        mode="mixing_only",
        target="beat",
        description="Focuses low-end control without flattening the groove.",
        dsp_chain_reference="streaming_master",
        tags=["Beat", "Bass"],
    ),
    StudioPreset(
        id="club_beat_punch",
        name="Club Beat Punch",
        mode="mixing_only",
        target="beat",
        description="Adds punch and clarity tailored for club playback.",
        dsp_chain_reference="streaming_master",
        tags=["Beat", "Loud"],
    ),
    StudioPreset(
        id="beat_stereo_polish",
        name="Beat Stereo Polish",
        mode="mixing_only",
        target="beat",
        description="Widens and gently shines stereo instrumentals.",
        dsp_chain_reference="streaming_master",
        tags=["Beat", "Wide"],
    ),
    StudioPreset(
        id="vintage_beat_warmth",
        name="Vintage Beat Warmth",
        mode="mixing_only",
        target="beat",
        description="Warmer, rounded tone for sample-based or lo-fi beats.",
        dsp_chain_reference="streaming_master",
        tags=["Beat", "Warm"],
    ),
    StudioPreset(
        id="minimal_beat_processing",
        name="Minimal Beat Processing",
        mode="mixing_only",
        target="beat",
        description="Safest option for leased beats – light glue, no heavy limiting.",
        dsp_chain_reference="streaming_master",
        tags=["Beat", "Safe"],
    ),
    StudioPreset(
        id="trap_beat_tight",
        name="Trap Beat Tight",
        mode="mixing_only",
        target="beat",
        description="Modern trap beat polish with tightened transients.",
        dsp_chain_reference="streaming_master",
        tags=["Beat", "Trap"],
    ),
    StudioPreset(
        id="afrobeat_groove_balance",
        name="Afrobeat Groove Balance",
        mode="mixing_only",
        target="beat",
        description="Balances groove elements in afrobeat instrumentals.",
        dsp_chain_reference="streaming_master",
        tags=["Beat", "Afrobeat"],
    ),

    # Mix + master – full mix bus
    StudioPreset(
        id="radio_ready_mix",
        name="Radio Ready Mix",
        mode="mix_and_master",
        target="full_mix",
        description="Balanced, commercial mix tuned to translate across systems.",
        dsp_chain_reference="streaming_master",
        tags=["Full Mix", "Radio"],
    ),
    StudioPreset(
        id="loud_modern_mix",
        name="Loud Modern Mix",
        mode="mix_and_master",
        target="full_mix",
        description="Competitive loudness with controlled aggression for modern genres.",
        dsp_chain_reference="streaming_master",
        tags=["Full Mix", "Loud"],
    ),
    StudioPreset(
        id="streaming_optimized_mix",
        name="Streaming Optimized",
        mode="mix_and_master",
        target="full_mix",
        description="Targets streaming services with sane loudness and dynamics.",
        dsp_chain_reference="streaming_master",
        tags=["Full Mix", "Streaming"],
    ),
    StudioPreset(
        id="club_ready_mix",
        name="Club Ready",
        mode="mix_and_master",
        target="full_mix",
        description="Club-focused tone with extra punch and sub weight.",
        dsp_chain_reference="streaming_master",
        tags=["Full Mix", "Club"],
    ),
    StudioPreset(
        id="warm_analog_mix",
        name="Warm Analog Mix",
        mode="mix_and_master",
        target="full_mix",
        description="Softer top and rounded mids inspired by analog chains.",
        dsp_chain_reference="streaming_master",
        tags=["Full Mix", "Warm"],
    ),
    StudioPreset(
        id="vocal_forward_mix",
        name="Vocal-Forward Mix",
        mode="mix_and_master",
        target="full_mix",
        description="Brings the lead vocal slightly in front of the beat.",
        dsp_chain_reference="streaming_master",
        tags=["Full Mix", "Vocal"],
    ),
    StudioPreset(
        id="bass_heavy_mix",
        name="Bass-Heavy Mix",
        mode="mix_and_master",
        target="full_mix",
        description="Emphasizes low-end impact while protecting headroom.",
        dsp_chain_reference="streaming_master",
        tags=["Full Mix", "Bass"],
    ),
    StudioPreset(
        id="clean_commercial_mix",
        name="Clean Commercial Mix",
        mode="mix_and_master",
        target="full_mix",
        description="Neutral, clean mix/master suited to many genres.",
        dsp_chain_reference="streaming_master",
        tags=["Full Mix", "Clean"],
    ),
    StudioPreset(
        id="wide_stereo_mix",
        name="Wide Stereo Mix",
        mode="mix_and_master",
        target="full_mix",
        description="Emphasizes stereo imaging while keeping mono compatibility in mind.",
        dsp_chain_reference="streaming_master",
        tags=["Full Mix", "Wide"],
    ),
    StudioPreset(
        id="punchy_urban_mix",
        name="Punchy Urban Mix",
        mode="mix_and_master",
        target="full_mix",
        description="Designed for hip-hop, trap and dancehall mixes that need extra punch.",
        dsp_chain_reference="streaming_master",
        tags=["Full Mix", "Urban", "Punchy"],
    ),

    # Mastering only
    StudioPreset(
        id="streaming_master_minus14",
        name="Streaming Master (-14 LUFS)",
        mode="mastering_only",
        target="full_mix",
        description="Streaming-safe loudness around -14 LUFS with preserved dynamics.",
        dsp_chain_reference="streaming_master",
        tags=["Master", "Streaming"],
    ),
    StudioPreset(
        id="loud_club_master",
        name="Loud Club Master",
        mode="mastering_only",
        target="full_mix",
        description="Hotter master tuned for club and DJ playback.",
        dsp_chain_reference="streaming_master",
        tags=["Master", "Club", "Loud"],
    ),
    StudioPreset(
        id="radio_master",
        name="Radio Master",
        mode="mastering_only",
        target="full_mix",
        description="Balanced radio-ready master suitable for broadcast.",
        dsp_chain_reference="streaming_master",
        tags=["Master", "Radio"],
    ),
    StudioPreset(
        id="beat_sale_master",
        name="Beat Sale Master",
        mode="mastering_only",
        target="beat",
        description="Safe master for selling beats without over-limiting.",
        dsp_chain_reference="streaming_master",
        tags=["Master", "Beat", "Safe"],
    ),
    StudioPreset(
        id="warm_analog_master",
        name="Warm Analog Master",
        mode="mastering_only",
        target="full_mix",
        description="Analog-inspired tone with softened transients and gentle saturation.",
        dsp_chain_reference="streaming_master",
        tags=["Master", "Warm"],
    ),
    StudioPreset(
        id="transparent_master",
        name="Transparent Master",
        mode="mastering_only",
        target="full_mix",
        description="Minimal coloration, focusing on level and subtle polish.",
        dsp_chain_reference="streaming_master",
        tags=["Master", "Clean"],
    ),
    StudioPreset(
        id="bass_focus_master",
        name="Bass Focus Master",
        mode="mastering_only",
        target="full_mix",
        description="Reinforces low-end while controlling boominess.",
        dsp_chain_reference="streaming_master",
        tags=["Master", "Bass"],
    ),
    StudioPreset(
        id="wide_stereo_master",
        name="Wide Stereo Master",
        mode="mastering_only",
        target="full_mix",
        description="Enhances stereo width at the mastering stage.",
        dsp_chain_reference="streaming_master",
        tags=["Master", "Wide"],
    ),
    StudioPreset(
        id="clean_hiphop_master",
        name="Clean Hip-Hop Master",
        mode="mastering_only",
        target="full_mix",
        description="Clean, punchy master tuned for hip-hop and trap records.",
        dsp_chain_reference="streaming_master",
        tags=["Master", "Hip-Hop"],
    ),
    StudioPreset(
        id="edm_loud_master",
        name="EDM Loud Master",
        mode="mastering_only",
        target="full_mix",
        description="High-energy, loud master for EDM and festival tracks.",
        dsp_chain_reference="streaming_master",
        tags=["Master", "EDM", "Loud"],
    ),
]


# High-level DSP step templates per feature type. These describe the
# conceptual processing flow and are used to drive both backend progress
# updates and frontend UX.
FLOW_STEP_TEMPLATES: dict[str, list[str]] = {
    # Audio cleanup / dialogue repair
    "audio_cleanup": [
        "Analyzing audio",
        "Noise reduction",
        "Artifact cleanup",
        "EQ cleanup",
        "Output rendering",
    ],
    # Mixing-only workflows (no mastering limiter)
    "mixing_only": [
        "Analyzing audio",
        "Gain staging",
        "Applying EQ",
        "Applying compression",
        "Adding saturation",
        "Stereo enhancement",
        "Mix render",
    ],
    # Full mix + master bus flow
    "mix_master": [
        "Analyzing audio",
        "Detecting vocal characteristics",
        "Cleaning noise & artifacts",
        "Gain staging",
        "Applying EQ",
        "Applying compression",
        "De-essing",
        "Adding saturation",
        "Stereo enhancement",
        "Bus processing",
        "Loudness normalization",
        "Finalizing output",
    ],
    # Mastering-only flow
    "mastering_only": [
        "Analyzing mix",
        "Linear EQ",
        "Multiband compression",
        "Stereo imaging",
        "Limiting",
        "Loudness normalization",
        "Final render",
    ],
}

GENERIC_STEP_TEMPLATE: list[str] = [
    "Analyzing audio",
    "Processing",
    "Finalizing output",
]


def _map_feature_type_to_preset_mode(feature_type: str) -> PresetMode | None:
    """Map a feature_type used for gating to the studio PresetMode.

    This helps keep the external credits API (audio_cleanup, mixing_only,
    mix_master, mastering_only) in sync with the UI-facing preset modes
    (audio_cleanup, mixing_only, mix_and_master, mastering_only).
    """

    if feature_type == "audio_cleanup":
        return "audio_cleanup"
    if feature_type == "mixing_only":
        return "mixing_only"
    if feature_type == "mix_master":
        return "mix_and_master"
    if feature_type == "mastering_only":
        return "mastering_only"
    return None


def _resolve_preset_for_request(
    payload: ProcessRequest,
) -> tuple[StudioPreset | None, str | None]:
    """Resolve and validate a StudioPreset for a job request.

    Behaviour:
    - If payload.preset_id is not provided, returns (None, payload.target).
    - If preset_id is provided, it must exist in STUDIO_PRESETS.
    - When feature_type is set, the preset's mode must be compatible.
    - When target is set on the payload it must match the preset.target;
      otherwise a 422 is raised.

    Returns a tuple of (resolved_preset_or_None, effective_target_or_None).
    """

    preset_id = payload.preset_id
    target = payload.target

    if not preset_id:
        # No studio preset metadata provided; fall back to legacy behaviour.
        return None, target

    mode = _map_feature_type_to_preset_mode(payload.feature_type)

    preset = next((p for p in STUDIO_PRESETS if p.id == preset_id), None)
    if preset is None:
        raise HTTPException(status_code=400, detail=f"Unknown preset_id '{preset_id}'")

    if mode is not None and preset.mode != mode:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Preset '{preset_id}' is not valid for feature_type "
                f"'{payload.feature_type}'"
            ),
        )

    effective_target = target or preset.target

    if target is not None and target != preset.target:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Preset '{preset_id}' is for target '{preset.target}' "
                f"but request specified target '{target}'."
            ),
        )

    return preset, effective_target


def _steps_for_feature_type(feature_type: str | None) -> list[str]:
    """Return the canonical step list for a given feature type.

    Falls back to a generic template when the feature type is missing or
    unknown so that progress always advances through a sensible set of
    stages.
    """

    if feature_type and feature_type in FLOW_STEP_TEMPLATES:
        return FLOW_STEP_TEMPLATES[feature_type]
    return GENERIC_STEP_TEMPLATE


def _build_step_statuses(job_row: dict[str, Any]) -> list[StepStatus] | None:
    """Build per-step completion information for a processing job.

    Steps are discovered from input_files._meta.steps when present and
    marked completed based on the job's numeric progress. This keeps the
    contract simple for the frontend while allowing different flows to
    define their own stage lists.
    """

    input_files = job_row.get("input_files") or {}
    if not isinstance(input_files, dict):
        return None

    meta = input_files.get("_meta") or {}
    if not isinstance(meta, dict):
        return None

    raw_steps = meta.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        return None

    steps: list[str] = [str(name) for name in raw_steps]
    progress = int(job_row.get("progress") or 0)
    status = (job_row.get("status") or "queued").lower()

    total = len(steps)
    if total == 0:
        return None

    # Each step completion corresponds to a progress threshold of
    # (index + 1) / total * 100.
    thresholds = [int(((index + 1) / total) * 100) for index in range(total)]

    result: list[StepStatus] = []
    for index, name in enumerate(steps):
        if status == "failed":
            completed = progress >= thresholds[index] and False
        else:
            completed = progress >= thresholds[index]
        result.append(StepStatus(name=name, completed=completed))

    return result


@app.get("/studio/presets")
async def list_studio_presets(mode: Optional[PresetMode] = None) -> list[StudioPreset]:
    """Return studio presets, optionally filtered by processing mode.

    This endpoint is consumed by the Next.js studio UI to populate the
    dynamic preset selector. The actual DSP chain is determined by the
    ``dsp_chain_reference`` field, which maps onto the DSP service
    presets (clean_vocal, streaming_master, etc.).
    """

    if mode is None:
        return STUDIO_PRESETS
    return [p for p in STUDIO_PRESETS if p.mode == mode]


# -------------------------
# Billing & plan models
# -------------------------


class CheckoutCapturePayload(BaseModel):
    user_id: Optional[str] = None
    plan_key: str
    amount_cents: int
    provider: Literal["paypal"] = "paypal"
    provider_payment_id: Optional[str] = None


class CheckoutCaptureResponse(BaseModel):
    payment_id: str
    user_id: Optional[str] = None
    plan_key: str
    amount_cents: int
    currency: str = "USD"


async def _user_has_feature_access(user_id: str | None, feature_type: str) -> bool:
    """Return True if the user is allowed to run the requested feature.

    Rules:
    - Missing user_id => no access (must be authenticated).
    - user_plans.plan_type == "subscription" with subscription_status in
      ("active", "trialing") => full access.
    - Otherwise, check user_credits for a row with matching feature_type and
      remaining_uses > 0 and (no expiry or expires_at in the future).
    """

    if not user_id:
        return False

    try:
        plans = await supabase_select(
            "user_plans", {"user_id": f"eq.{user_id}", "limit": 1}
        )
    except SupabaseConfigError:
        # If billing is misconfigured, fail closed for safety.
        return False

    plan = plans[0] if plans else None
    plan_type = (plan or {}).get("plan_type") or "free"
    subscription_status = (plan or {}).get("subscription_status") or None

    if plan_type == "subscription" and subscription_status in {"active", "trialing"}:
        return True

    # Fallback to pay-as-you-go credits for this feature type.
    try:
        credits = await supabase_select(
            "user_credits",
            {
                "user_id": f"eq.{user_id}",
                "feature_type": f"eq.{feature_type}",
            },
        )
    except SupabaseConfigError:
        return False

    now = datetime.utcnow()
    for credit in credits:
        remaining = int(credit.get("remaining_uses") or 0)
        if remaining <= 0:
            continue
        expires_at_raw = credit.get("expires_at")
        if isinstance(expires_at_raw, str):
            expires_dt = _parse_iso_datetime(expires_at_raw)
            if expires_dt is not None and expires_dt <= now:
                continue
        return True

    return False


async def _run_processing_job(job_id: str, job_type: str, input_files: Dict[str, Any]) -> None:
    """Background coroutine that simulates a multi-stage DSP pipeline.

    In production you can replace the simulated stages with real calls to
    your DSP microservice, passing along the job_id so the DSP can also
    update Supabase directly if desired.
    """

    try:
        # Determine the concrete step list for this job from metadata,
        # falling back to a generic template when necessary.
        meta = input_files.get("_meta") if isinstance(input_files, dict) else None
        feature_type = None
        if isinstance(meta, dict):
            feature_type = meta.get("feature_type") or meta.get("flow_key")
        step_names = meta.get("steps") if isinstance(meta, dict) else None
        if not isinstance(step_names, list) or not step_names:
            step_names = _steps_for_feature_type(str(feature_type) if feature_type else None)

        steps: list[str] = [str(name) for name in step_names]
        total_steps = len(steps) or 1
        last_stage_name: str | None = None

        # Execute each conceptual DSP stage; in a production system, replace
        # the sleeps with real calls into the DSP microservice, one per stage.
        for index, stage_name in enumerate(steps, start=1):
            last_stage_name = stage_name
            is_last = index == total_steps

            # Simulate DSP work for this step (replace with real DSP call).
            # await call_dsp_stage(...)
            await asyncio.sleep(0.1)

            if not is_last:
                # Non-final steps use the shared progress helper so the
                # percentage is derived from completed_steps / total_steps.
                await update_progress(
                    job_id=job_id,
                    step_name=stage_name,
                    step_index=index,
                    total_steps=total_steps,
                )
            else:
                # Final stage: mark the job as completed at 100% and attach
                # any output file metadata.
                output_files: Dict[str, Any] = {
                    "master_url": input_files.get("target_path", ""),
                }

                await update_processing_job(
                    job_id,
                    {
                        "status": "completed",
                        "progress": 100,
                        "current_stage": stage_name,
                        "output_files": output_files,
                    },
                )

        # Best-effort preset usage tracking once the job has completed.
        try:
            job_row = await get_processing_job(job_id)
            if job_row is not None:
                preset_key = job_row.get("preset_key")
                user_id = job_row.get("user_id")
                if preset_key:
                    await supabase_insert(
                        "preset_usage",
                        {
                            "user_id": user_id,
                            "job_id": job_row.get("id"),
                            "preset_key": preset_key,
                        },
                    )
        except Exception:
            # Analytics should never break job completion.
            pass
    except Exception as exc:  # pragma: no cover - defensive
        # Best-effort failure update so the job is not stuck forever.
        message = str(exc)
        try:
            step_name = last_stage_name or "pipeline"
            await mark_job_failed(job_id, step_name=step_name, error_message=message)
        except Exception:
            # If we cannot even update Supabase, just swallow – there's
            # nowhere else to report this in a background task.
            pass


async def _run_mix_master_job(
    job_id: str,
    session_id: str,
    session_dir: Path,
    beat_path: Path | None,
    lead_path: Path | None,
    adlibs_path: Path | None,
    genre: str,
    target_lufs: str,
) -> None:
    """Background job that runs the real mix+master pipeline.

    This mirrors the previous synchronous /api/mix-master behaviour but
    reports progress into Supabase as it goes.
    """

    last_stage_name: str | None = None
    try:
        await update_processing_job(
            job_id,
            {
                "status": "processing",
                "progress": 5,
                "current_stage": "Saving inputs",
            },
        )
        last_stage_name = "Saving inputs"

        loop = asyncio.get_running_loop()

        mixed_path = session_dir / "mix.wav"
        master_path = session_dir / "master.wav"

        # For now, the mixing pipeline expects a vocal + beat path. If only one
        # track exists, treat it as a full mix and only run mastering.
        if beat_path and (lead_path or adlibs_path):
            await update_processing_job(
                job_id,
                {
                    "progress": 25,
                    "current_stage": "Mixing stems",
                },
            )
            last_stage_name = "Mixing stems"
            vocal_source = lead_path or adlibs_path
            await loop.run_in_executor(
                None,
                ai_mix,
                str(vocal_source),
                str(beat_path),
                str(mixed_path),
            )
            mix_for_master: Path | None = mixed_path
        else:
            # Single-file workflow: use whichever upload is present.
            first_available = beat_path or lead_path or adlibs_path
            mix_for_master = first_available

        if mix_for_master is None:
            raise RuntimeError("No valid audio file found for mastering")

        await update_processing_job(
            job_id,
            {
                "progress": 60,
                "current_stage": "Mastering mix",
            },
        )
        last_stage_name = "Mastering mix"

        await loop.run_in_executor(
            None,
            ai_master,
            str(mix_for_master),
            str(master_path),
            target_lufs,
        )

        output_files: Dict[str, Any] = {
            "session_id": session_id,
            "master_path": str(master_path),
            "genre": genre,
            "target_lufs": target_lufs,
        }

        await update_processing_job(
            job_id,
            {
                "status": "completed",
                "progress": 100,
                "current_stage": "Finalizing output",
                "output_files": output_files,
            },
        )

        # Record preset usage for mix/master jobs when a preset_key is set
        # on the processing_jobs row.
        try:
            job_row = await get_processing_job(job_id)
            if job_row is not None:
                preset_key = job_row.get("preset_key")
                user_id = job_row.get("user_id")
                if preset_key:
                    await supabase_insert(
                        "preset_usage",
                        {
                            "user_id": user_id,
                            "job_id": job_row.get("id"),
                            "preset_key": preset_key,
                        },
                    )
        except Exception:
            pass
    except Exception as exc:  # pragma: no cover - defensive
        message = str(exc)
        try:
            stage_label = f"Error during {last_stage_name}" if last_stage_name else "error"
            await update_processing_job(
                job_id,
                {
                    "status": "failed",
                    "current_stage": stage_label,
                    "error_message": message,
                },
            )
        except Exception:
            pass


@app.post("/process", response_model=JobStatusResponse)
async def create_process_job(payload: ProcessRequest) -> JobStatusResponse:
    """Create a new processing job and start background DSP work.

    The endpoint returns immediately with the job id so the frontend can
    start polling /status/{job_id} for progress updates.
    """

    # Resolve and validate any studio preset metadata supplied by the caller.
    # This keeps the async job pipeline consistent with the studio preset
    # registry and prevents obviously incompatible combinations from running.
    resolved_preset, effective_target = _resolve_preset_for_request(payload)

    # Enrich input_files with lightweight preset metadata so downstream DSP
    # workers can call the correct chain with the intended target, and so
    # progress can be driven by a concrete list of conceptual stages.
    enriched_input_files: Dict[str, Any] = dict(payload.input_files or {})

    meta = enriched_input_files.get("_meta")
    if not isinstance(meta, dict):
        meta = {}
        enriched_input_files["_meta"] = meta

    if resolved_preset is not None:
        meta.setdefault("studio_preset_id", resolved_preset.id)
        meta.setdefault("studio_preset_mode", resolved_preset.mode)
        meta.setdefault("studio_preset_target", resolved_preset.target)
        meta.setdefault("dsp_chain_reference", resolved_preset.dsp_chain_reference)

    if effective_target is not None:
        # Target can be inferred from the preset or explicitly supplied
        # by the caller; either way it is persisted for DSP workers.
        meta.setdefault("target", effective_target)

    # Record the high-level feature_type and attach the derived step list so
    # that the background runner and /status endpoint can expose a
    # preset-aware stage breakdown for the UI.
    feature_type_value = payload.feature_type
    meta.setdefault("feature_type", feature_type_value)
    meta.setdefault("flow_key", feature_type_value)
    meta.setdefault("steps", _steps_for_feature_type(feature_type_value))

    # For analytics, prefer the canonical studio preset id when available;
    # otherwise fall back to any legacy preset_key passed by callers.
    db_preset_key = payload.preset_key
    if resolved_preset is not None:
        db_preset_key = resolved_preset.id

    # Start with base job data persisted to Supabase.
    job_data: Dict[str, Any] = {
        "user_id": payload.user_id,
        "job_type": payload.job_type,
        "preset_key": db_preset_key,
        "status": "queued",
        "progress": 0,
        "current_stage": "queued",
        "input_files": enriched_input_files,
    }

    # Derive a coarse estimated total processing time (seconds) from duration_sec and feature_type.
    # This is a heuristic that gives the frontend enough signal for a remaining-time estimate.
    try:
        duration_sec_val = float(job_data.get("duration_sec") or 0) if job_data.get("duration_sec") is not None else 0.0
    except Exception:  # pragma: no cover - defensive
        duration_sec_val = 0.0

    base_factor = 0.15  # default: ~15% of track length
    if feature_type_value in ("mix_master", "mix-master", "mix_mastering"):
        base_factor = 0.22
    elif feature_type_value in ("mastering_only", "master-only"):
        base_factor = 0.18
    elif feature_type_value in ("audio_cleanup", "cleanup"):
        base_factor = 0.10
    elif feature_type_value in ("podcast",):
        base_factor = 0.12

    estimated_total_sec = (
        max(10.0, min(300.0, duration_sec_val * base_factor)) if duration_sec_val > 0 else None
    )

    if estimated_total_sec is not None:
        job_data["estimated_total_sec"] = estimated_total_sec

    try:
        job_row = await create_processing_job(job_data)
    except SupabaseConfigError as cfg_err:
        # Configuration issue (missing URL/key) – treat as a backend error.
        raise HTTPException(status_code=500, detail=str(cfg_err)) from cfg_err
    except httpx.HTTPStatusError as http_err:
        # Surface Supabase's error payload instead of crashing the whole request.
        status_code = http_err.response.status_code if http_err.response is not None else 502
        supabase_body: str
        try:
            supabase_body = http_err.response.text if http_err.response is not None else str(http_err)
        except Exception:  # pragma: no cover - very defensive
            supabase_body = str(http_err)

        raise HTTPException(
            status_code=status_code,
            detail={
                "error": "SUPABASE_INSERT_FAILED",
                "supabase": supabase_body,
            },
        ) from http_err

    job_id = str(job_row["id"])

    # Fire-and-forget background coroutine – this process should run with
    # WEB_CONCURRENCY=1 or otherwise ensure that jobs are not duplicated.
    asyncio.create_task(
        _run_processing_job(job_id, payload.job_type, enriched_input_files)
    )

    steps = _build_step_statuses(job_row) or []

    created_at = job_row.get("created_at")
    elapsed_sec: Optional[float] = None
    if isinstance(created_at, datetime):
        elapsed_sec = (datetime.now(created_at.tzinfo) - created_at).total_seconds()

    return JobStatusResponse(
        id=job_id,
        status=job_row.get("status", "queued"),
        progress=job_row.get("progress", 0),
        current_stage=job_row.get("current_stage"),
        error_message=job_row.get("error_message"),
        output_files=job_row.get("output_files"),
        steps=steps,
        estimated_total_sec=job_row.get("estimated_total_sec"),
        elapsed_sec=elapsed_sec,
    )


@app.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """Return the current state of a processing job by id."""

    try:
        job_row = await get_processing_job(job_id)
    except SupabaseConfigError as cfg_err:
        raise HTTPException(status_code=500, detail=str(cfg_err)) from cfg_err

    if not job_row:
        raise HTTPException(status_code=404, detail="Job not found")

    steps = _build_step_statuses(job_row) or []

    created_at = job_row.get("created_at")
    elapsed_sec: Optional[float] = None
    if isinstance(created_at, datetime):
        elapsed_sec = (datetime.now(created_at.tzinfo) - created_at).total_seconds()

    return JobStatusResponse(
        id=str(job_row["id"]),
        status=job_row.get("status", "queued"),
        progress=job_row.get("progress", 0),
        current_stage=job_row.get("current_stage"),
        error_message=job_row.get("error_message"),
        output_files=job_row.get("output_files"),
        steps=steps,
        estimated_total_sec=job_row.get("estimated_total_sec"),
        elapsed_sec=elapsed_sec,
    )


@app.post("/dsp/process")
async def proxy_dsp_process(
    file: UploadFile = File(...),
    track_type: str = Form(...),
    preset: str = Form(...),
    target: str = Form(...),
    gender: Optional[str] = Form(None),
    throw_fx_mode: Optional[str] = Form(None),
    genre: Optional[str] = Form(None),
    reference_profile: Optional[str] = Form(None),
    session_key: Optional[str] = Form(None),
    session_scale: Optional[str] = Form(None),
) -> JSONResponse:
    """Proxy the studio's multipart /process call to the external DSP service.

    This keeps the browser talking only to this backend (which already has
    CORS configured for the deployed frontend), while the backend communicates
    with the DSP service without browser CORS restrictions.
    """

    # Build the form data as expected by the DSP service
    data: Dict[str, str] = {
        "track_type": track_type,
        "preset": preset,
        "target": target,
    }
    if gender is not None:
        data["gender"] = gender
    if throw_fx_mode is not None:
        data["throw_fx_mode"] = throw_fx_mode
    if genre is not None:
        data["genre"] = genre
    if reference_profile is not None:
        data["reference_profile"] = reference_profile
    if session_key is not None:
        data["session_key"] = session_key
    if session_scale is not None:
        data["session_scale"] = session_scale

    # Read uploaded file contents and forward to DSP
    file_bytes = await file.read()
    files = {
        "file": (
            file.filename or "input.wav",
            file_bytes,
            file.content_type or "audio/wav",
        )
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{DSP_BASE_URL}/process", data=data, files=files)
    except httpx.RequestError as exc:  # pragma: no cover - network errors
        raise HTTPException(status_code=502, detail=f"DSP service unreachable: {exc}") from exc

    # Bubble up DSP errors with as much detail as possible
    if resp.status_code >= 400:
        try:
            detail: Any = resp.json()
        except ValueError:
            detail = {"detail": resp.text}
        raise HTTPException(status_code=resp.status_code, detail=detail)

    try:
        payload = resp.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=f"Invalid JSON from DSP: {exc}") from exc

    return JSONResponse(status_code=resp.status_code, content=payload)


@app.post("/api/mix-master")
async def mix_master(
    beat: UploadFile | None = File(default=None),
    lead: UploadFile | None = File(default=None),
    adlibs: UploadFile | None = File(default=None),
    genre: str = Form(""),
    target_lufs: str = Form("-14"),
) -> dict[str, Any]:
    """Legacy mix+master endpoint used by some clients.

    Saves uploaded files to a session directory, enqueues a "mix_master" job,
    and returns the queued job id plus session metadata.
    """

    session_id = uuid.uuid4().hex[:12]
    session_dir = SESSIONS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    beat_path = _save_upload(beat, session_dir / "beat.wav") if beat else None
    lead_path = _save_upload(lead, session_dir / "lead.wav") if lead else None
    adlibs_path = _save_upload(adlibs, session_dir / "adlibs.wav") if adlibs else None

    # Persist a queued job in Supabase so the frontend can track it.
    # Attach a mix+master step list in _meta so the UI can render a
    # meaningful stage breakdown while the background job runs.
    input_files: Dict[str, Any] = {
        "session_id": session_id,
        "session_dir": str(session_dir),
        "beat_path": str(beat_path) if beat_path else None,
        "lead_path": str(lead_path) if lead_path else None,
        "adlibs_path": str(adlibs_path) if adlibs_path else None,
        "genre": genre,
        "target_lufs": target_lufs,
        "_meta": {
            "feature_type": "mix_master",
            "flow_key": "mix_master",
            "steps": _steps_for_feature_type("mix_master"),
        },
    }

    try:
        job_row = await create_processing_job(
            {
                "user_id": None,  # Optionally wire through the authenticated user later
                "job_type": "mix_master",
                "status": "queued",
                "progress": 0,
                "current_stage": "queued",
                "input_files": input_files,
            }
        )
    except SupabaseConfigError as cfg_err:
        raise HTTPException(status_code=500, detail=str(cfg_err)) from cfg_err

    job_id = str(job_row["id"])

    asyncio.create_task(
        _run_mix_master_job(
            job_id,
            session_id,
            session_dir,
            beat_path,
            lead_path,
            adlibs_path,
            genre,
            target_lufs,
        )
    )

    return {
        "status": "queued",
        "job_id": job_id,
        "session_id": session_id,
        "genre": genre,
        "target_lufs": target_lufs,
    }


# -------------------------
# Admin models & in-memory state
# -------------------------


class DashboardStats(BaseModel):
    total_users: int
    active_jobs: int
    jobs_today: int
    failed_jobs: int
    revenue_today: float
    revenue_month: float
    avg_processing_time: float


class TopPreset(BaseModel):
    id: str
    name: str
    uses: int


class JobsTimeseriesPoint(BaseModel):
    date: str
    jobs: int


class AdminUser(BaseModel):
    id: str
    email: str
    plan: str | None = None
    country: str | None = None
    status: Literal["active", "suspended", "banned"] = "active"


class AdminUserDetail(AdminUser):
    credits: int = 0
    jobs_count: int = 0


class AdminJob(BaseModel):
    id: str
    user_email: str
    status: Literal["active", "completed", "failed"]
    input_type: Literal["single", "stems"]
    preset: str | None = None
    duration_sec: float


class AdminJobDetail(AdminJob):
    created_at: str
    steps: list[str] = []
    input_url: str | None = None
    output_url: str | None = None
    logs: list[str] | None = None


class AdminPreset(BaseModel):
    id: str
    name: str
    role: str
    enabled: bool
    version: str


class PresetParams(BaseModel):
    hpf_cutoff: int = 100
    comp_ratio: float = 3.0
    saturation: float = 0.1
    limiter_ceiling: float = -1.0


class AdminPlan(BaseModel):
    id: str
    name: str
    price_month: float
    credits: int
    stem_limit: int | None = None


class AdminPlanUpdate(BaseModel):
	name: str | None = None
	price_month: float | None = None
	credits: int | None = None
	stem_limit: int | None = None


class AdminPayment(BaseModel):
    id: str
    user_email: str
    amount: float
    provider: str
    status: str
    created_at: str


class RevenuePoint(BaseModel):
    date: str
    amount: float


class StorageStats(BaseModel):
    total_gb: float
    avg_per_user_gb: float
    auto_delete_after_days: int | None = None


class AdminLog(BaseModel):
    id: str
    level: Literal["INFO", "WARN", "ERROR"]
    source: str
    message: str
    created_at: str


class Testimonial(BaseModel):
    id: str
    name: str
    role: str
    quote: str


class Announcement(BaseModel):
    id: str
    title: str
    body: str
    active: bool


class AdminSettings(BaseModel):
    maintenance_mode: bool = False
    dsp_version: str = "stable"
    max_concurrent_jobs: int = 4
    max_file_mb: int = 300


_admin_settings = AdminSettings()
_preset_params: dict[str, PresetParams] = {}


def _demo_datetime_series(days: int, field: str) -> list[dict[str, int | float | str]]:
    today = datetime.utcnow().date()
    series: list[dict[str, int | float | str]] = []
    for i in range(days):
        day = today - timedelta(days=days - 1 - i)
        # simple ramp for nicer charts
        value = 5 + i * 2
        series.append({"date": day.isoformat(), field: value})
    return series


def _parse_iso_datetime(value: str) -> datetime | None:
    """Best-effort parser for ISO timestamps coming back from Supabase.

    Handles optional trailing 'Z' and returns None if parsing fails.
    """

    if not value:
        return None
    try:
        cleaned = value.rstrip("Z")
        return datetime.fromisoformat(cleaned)
    except Exception:
        return None


# -------------------------
# Admin API routes
# -------------------------


@app.get("/admin/dashboard")
async def admin_dashboard():
    """Admin overview powered by live Supabase data."""

    try:
        # Fetch core datasets; for early-stage volumes it's fine to compute
        # aggregates in Python instead of specialised SQL.
        users = await supabase_select("user_profiles")
        jobs = await supabase_select("processing_jobs")
        payments = await supabase_select("billing_payments")
        preset_usage = await supabase_select("preset_usage")
    except SupabaseConfigError as cfg_err:
        raise HTTPException(status_code=500, detail=str(cfg_err)) from cfg_err

    today = datetime.utcnow().date()
    month_start = today.replace(day=1)

    total_users = len(users)
    active_jobs = len([j for j in jobs if j.get("status") in {"queued", "processing"}])

    jobs_today = 0
    failed_jobs = 0
    durations: list[float] = []

    # Derive job stats and a 14-day timeseries
    window_start = today - timedelta(days=13)
    jobs_per_day: dict[date, int] = {
        window_start + timedelta(days=i): 0 for i in range(14)
    }

    for job in jobs:
        status = job.get("status") or "queued"
        if status == "failed":
            failed_jobs += 1

        created_raw = job.get("created_at")
        created_dt = _parse_iso_datetime(created_raw) if isinstance(created_raw, str) else None
        if created_dt is not None:
            created_date = created_dt.date()
            if created_date == today:
                jobs_today += 1
            if window_start <= created_date <= today:
                jobs_per_day[created_date] = jobs_per_day.get(created_date, 0) + 1

        duration_val = job.get("duration_sec")
        if isinstance(duration_val, (int, float)) and duration_val > 0:
            durations.append(float(duration_val))

    avg_processing_time = float(sum(durations) / len(durations)) if durations else 0.0

    # Revenue (in cents) derived from successful payments
    revenue_today_cents = 0
    revenue_month_cents = 0

    for payment in payments:
        status = (payment.get("status") or "").lower()
        if status not in {"succeeded", "completed"}:
            continue

        created_raw = payment.get("created_at")
        created_dt = _parse_iso_datetime(created_raw) if isinstance(created_raw, str) else None
        if created_dt is None:
            continue

        created_date = created_dt.date()
        amount_cents = int(payment.get("amount_cents") or 0)

        if created_date == today:
            revenue_today_cents += amount_cents
        if created_date >= month_start:
            revenue_month_cents += amount_cents

    stats = DashboardStats(
        total_users=total_users,
        active_jobs=active_jobs,
        jobs_today=jobs_today,
        failed_jobs=failed_jobs,
        revenue_today=revenue_today_cents / 100.0,
        revenue_month=revenue_month_cents / 100.0,
        avg_processing_time=avg_processing_time,
    )

    # Top presets over all time, based on preset_usage events
    top_presets: list[TopPreset] = []
    if preset_usage:
        usage_counts: dict[str, int] = {}
        for row in preset_usage:
            key = str(row.get("preset_key") or "unknown")
            usage_counts[key] = usage_counts.get(key, 0) + 1

        sorted_presets = sorted(
            usage_counts.items(), key=lambda kv: kv[1], reverse=True
        )[:5]

        for key, uses in sorted_presets:
            top_presets.append(TopPreset(id=key, name=key, uses=uses))

    jobs_timeseries = [
        JobsTimeseriesPoint(date=day.isoformat(), jobs=jobs_per_day.get(day, 0))
        for day in sorted(jobs_per_day.keys())
    ]

    return {
        "stats": stats,
        "top_presets": top_presets,
        "jobs_timeseries": jobs_timeseries,
    }


@app.get("/admin/users")
async def admin_users_list():
    """List users from Supabase user_profiles with plan info."""

    try:
        profiles = await supabase_select("user_profiles")
        plans = await supabase_select("billing_plans")
    except SupabaseConfigError as cfg_err:
        raise HTTPException(status_code=500, detail=str(cfg_err)) from cfg_err

    plan_by_id = {str(p.get("id")): p.get("name") for p in plans}

    users: list[AdminUser] = []
    for row in profiles:
        user_id = str(row.get("user_id"))
        email = row.get("email") or ""
        country = row.get("country")
        status_raw = row.get("status") or "active"
        plan_id = row.get("plan_id")
        plan_name = plan_by_id.get(str(plan_id)) if plan_id is not None else None

        users.append(
            AdminUser(
                id=user_id,
                email=email,
                plan=plan_name,
                country=country,
                status=status_raw,  # constrained by DB check
            )
        )

    return {"users": users}


@app.get("/admin/users/{user_id}")
async def admin_user_detail(user_id: str):
    """Detailed view of a single user with job count and credits."""

    try:
        profiles = await supabase_select(
            "user_profiles", {"user_id": f"eq.{user_id}", "limit": 1}
        )
        if not profiles:
            raise HTTPException(status_code=404, detail="User not found")

        profile = profiles[0]
        plans = await supabase_select("billing_plans")
        jobs = await supabase_select("processing_jobs", {"user_id": f"eq.{user_id}"})
    except SupabaseConfigError as cfg_err:
        raise HTTPException(status_code=500, detail=str(cfg_err)) from cfg_err

    plan_by_id = {str(p.get("id")): p for p in plans}
    plan_id = profile.get("plan_id")
    plan = plan_by_id.get(str(plan_id)) if plan_id is not None else None

    plan_name = plan.get("name") if plan else None
    credits = int(plan.get("credits") or 0) if plan else 0

    user = AdminUserDetail(
        id=str(profile.get("user_id")),
        email=profile.get("email") or "",
        plan=plan_name,
        country=profile.get("country"),
        status=profile.get("status") or "active",
        credits=credits,
        jobs_count=len(jobs),
    )

    return {"user": user}


class UpdateUserStatusPayload(BaseModel):
    status: Literal["active", "suspended", "banned"]


@app.post("/admin/users/{user_id}/status")
async def admin_update_user_status(user_id: str, payload: UpdateUserStatusPayload):
    """Update a user's status field in user_profiles."""

    try:
        updated = await supabase_patch(
            "user_profiles",
            {"user_id": f"eq.{user_id}"},
            {"status": payload.status},
        )
    except SupabaseConfigError as cfg_err:
        raise HTTPException(status_code=500, detail=str(cfg_err)) from cfg_err

    if not updated:
        raise HTTPException(status_code=404, detail="User not found")

    return {"user_id": user_id, "status": payload.status}


class UpdateUserCreditsPayload(BaseModel):
    delta: int


@app.post("/admin/users/{user_id}/credits")
def admin_update_user_credits(user_id: str, payload: UpdateUserCreditsPayload):
    """Placeholder: accept credit adjustments without persisting them.

    The current schema does not track per-user credit balances beyond the
    plan-level credit allowance. This endpoint simply echoes the request so
    the admin UI remains functional without requiring additional schema
    changes.
    """

    return {"user_id": user_id, "delta": payload.delta}


@app.get("/admin/jobs")
async def admin_jobs_list():
    """List recent processing jobs with user email and status mapping."""

    try:
        jobs_rows = await supabase_select(
            "processing_jobs",
            {"order": "created_at.desc", "limit": 200},
        )
        profiles = await supabase_select("user_profiles")
    except SupabaseConfigError as cfg_err:
        raise HTTPException(status_code=500, detail=str(cfg_err)) from cfg_err

    email_by_user_id = {str(p.get("user_id")): p.get("email") for p in profiles}

    jobs: list[AdminJob] = []
    for row in jobs_rows:
        raw_status = (row.get("status") or "queued").lower()
        if raw_status in {"queued", "processing"}:
            status: Literal["active", "completed", "failed"] = "active"
        elif raw_status == "completed":
            status = "completed"
        else:
            status = "failed"

        input_type_val = row.get("input_type") or "single"
        preset_val = row.get("preset_key")
        duration_val = row.get("duration_sec") or 0

        user_id = row.get("user_id")
        user_email = email_by_user_id.get(str(user_id), "anonymous")

        jobs.append(
            AdminJob(
                id=str(row.get("id")),
                user_email=user_email or "anonymous",
                status=status,
                input_type=input_type_val,
                preset=preset_val,
                duration_sec=float(duration_val),
            )
        )

    return {"jobs": jobs}


@app.get("/admin/jobs/{job_id}")
async def admin_job_detail(job_id: str):
    """Detailed view of a single processing job from Supabase."""

    try:
        rows = await supabase_select(
            "processing_jobs", {"id": f"eq.{job_id}", "limit": 1}
        )
        if not rows:
            raise HTTPException(status_code=404, detail="Job not found")

        job_row = rows[0]
        profiles = await supabase_select("user_profiles")
    except SupabaseConfigError as cfg_err:
        raise HTTPException(status_code=500, detail=str(cfg_err)) from cfg_err

    email_by_user_id = {str(p.get("user_id")): p.get("email") for p in profiles}

    raw_status = (job_row.get("status") or "queued").lower()
    if raw_status in {"queued", "processing"}:
        status: Literal["active", "completed", "failed"] = "active"
    elif raw_status == "completed":
        status = "completed"
    else:
        status = "failed"

    input_type_val = job_row.get("input_type") or "single"
    preset_val = job_row.get("preset_key")
    duration_val = float(job_row.get("duration_sec") or 0)

    created_raw = job_row.get("created_at")
    created_iso = str(created_raw) if created_raw is not None else datetime.utcnow().isoformat() + "Z"

    user_id = job_row.get("user_id")
    user_email = email_by_user_id.get(str(user_id), "anonymous") or "anonymous"

    input_files = job_row.get("input_files") or {}
    output_files = job_row.get("output_files") or {}

    # Prefer the dynamic step metadata when available so this view
    # stays in sync with the studio/back-end pipeline. Fall back to a
    # simple generic template for legacy jobs that don't carry
    # input_files._meta.steps.
    step_statuses = _build_step_statuses(job_row)
    if step_statuses is not None:
        steps: list[str] = [s.name for s in step_statuses]
    else:
        job_type = job_row.get("job_type") or "mix_master"
        steps = ["Upload received"]
        if job_type in {"mix", "mix_master"}:
            steps.append("Mixing")
        if job_type in {"master", "mix_master"}:
            steps.append("Mastering")

    job = AdminJobDetail(
        id=str(job_row.get("id")),
        user_email=user_email,
        status=status,
        input_type=input_type_val,
        preset=preset_val,
        duration_sec=duration_val,
        created_at=created_iso,
        steps=steps,
        input_url=input_files.get("session_dir"),
        output_url=output_files.get("master_path"),
        logs=None,
    )

    return {"job": job}


@app.get("/admin/presets")
async def admin_presets_list():
    """List presets from the admin_presets table."""

    try:
        rows = await supabase_select("admin_presets")
    except SupabaseConfigError as cfg_err:
        raise HTTPException(status_code=500, detail=str(cfg_err)) from cfg_err

    presets: list[AdminPreset] = []
    for row in rows:
        presets.append(
            AdminPreset(
                id=str(row.get("id")),
                name=row.get("name") or "Unnamed preset",
                role=row.get("role") or "vocal",
                enabled=bool(row.get("enabled", True)),
                version=row.get("version") or "1.0.0",
            )
        )

    return {"presets": presets}


@app.get("/admin/presets/{preset_id}")
async def admin_preset_detail(preset_id: str):
    """Return parameter set for a single preset from admin_presets.params."""

    try:
        rows = await supabase_select(
            "admin_presets", {"id": f"eq.{preset_id}", "limit": 1}
        )
    except SupabaseConfigError as cfg_err:
        raise HTTPException(status_code=500, detail=str(cfg_err)) from cfg_err

    if not rows:
        raise HTTPException(status_code=404, detail="Preset not found")

    row = rows[0]
    raw_params = row.get("params") or {}

    try:
        params = PresetParams(**raw_params)
    except Exception:
        params = PresetParams()

    return {"preset_params": params}


@app.post("/admin/presets/{preset_id}")
async def admin_update_preset(preset_id: str, payload: PresetParams = Body(...)):
    """Update the JSON params for a preset in admin_presets."""

    try:
        updated = await supabase_patch(
            "admin_presets",
            {"id": f"eq.{preset_id}"},
            {"params": payload.dict()},
        )
    except SupabaseConfigError as cfg_err:
        raise HTTPException(status_code=500, detail=str(cfg_err)) from cfg_err

    if not updated:
        raise HTTPException(status_code=404, detail="Preset not found")

    return {"preset_id": preset_id, "preset_params": payload}


@app.get("/admin/plans")
async def admin_plans_list():
    """Expose billing_plans as AdminPlan objects for the admin UI."""

    try:
        rows = await supabase_select("billing_plans")
    except SupabaseConfigError as cfg_err:
        raise HTTPException(status_code=500, detail=str(cfg_err)) from cfg_err

    plans: list[AdminPlan] = []
    for row in rows:
        plans.append(
            AdminPlan(
                id=str(row.get("id")),
                name=row.get("name") or "Unnamed plan",
                price_month=float(row.get("price_month") or 0.0),
                credits=int(row.get("credits") or 0),
                stem_limit=row.get("stem_limit"),
            )
        )

    return {"plans": plans}


@app.patch("/admin/plans/{plan_id}")
async def admin_plans_update(plan_id: str, payload: AdminPlanUpdate):
    """Update a billing plan from the admin UI."""

    updates = {k: v for k, v in payload.dict().items() if v is not None}
    if not updates:
        return {"plan": None}

    try:
        rows = await supabase_patch("billing_plans", {"id": f"eq.{plan_id}"}, updates)
    except SupabaseConfigError as cfg_err:
        raise HTTPException(status_code=500, detail=str(cfg_err)) from cfg_err

    if not rows:
        raise HTTPException(status_code=404, detail="Plan not found")

    row = rows[0]
    plan = AdminPlan(
        id=str(row.get("id")),
        name=row.get("name") or "Unnamed plan",
        price_month=float(row.get("price_month") or 0.0),
        credits=int(row.get("credits") or 0),
        stem_limit=row.get("stem_limit"),
    )

    return {"plan": plan}


@app.get("/admin/payments")
async def admin_payments_list():
    """List payments from billing_payments and build a 30-day revenue series."""

    try:
        payments_rows = await supabase_select(
            "billing_payments", {"order": "created_at.desc", "limit": 200}
        )
        profiles = await supabase_select("user_profiles")
    except SupabaseConfigError as cfg_err:
        raise HTTPException(status_code=500, detail=str(cfg_err)) from cfg_err

    email_by_user_id = {str(p.get("user_id")): p.get("email") for p in profiles}

    payments: list[AdminPayment] = []
    for row in payments_rows:
        user_id = row.get("user_id")
        user_email = email_by_user_id.get(str(user_id), "unknown") or "unknown"

        amount_cents = int(row.get("amount_cents") or 0)
        created_raw = row.get("created_at")
        created_iso = str(created_raw) if created_raw is not None else datetime.utcnow().isoformat() + "Z"

        payments.append(
            AdminPayment(
                id=str(row.get("id")),
                user_email=user_email,
                amount=amount_cents / 100.0,
                provider=row.get("provider") or "unknown",
                status=row.get("status") or "pending",
                created_at=created_iso,
            )
        )

    # Build a simple 30-day revenue timeseries from successful payments
    today = datetime.utcnow().date()
    window_start = today - timedelta(days=29)
    revenue_per_day: dict[date, int] = {
        window_start + timedelta(days=i): 0 for i in range(30)
    }

    for row in payments_rows:
        status = (row.get("status") or "").lower()
        if status not in {"succeeded", "completed"}:
            continue

        created_raw = row.get("created_at")
        created_dt = _parse_iso_datetime(created_raw) if isinstance(created_raw, str) else None
        if created_dt is None:
            continue

        created_date = created_dt.date()
        if window_start <= created_date <= today:
            amount_cents = int(row.get("amount_cents") or 0)
            revenue_per_day[created_date] = revenue_per_day.get(created_date, 0) + amount_cents

    revenue_series = [
        RevenuePoint(date=day.isoformat(), amount=revenue_per_day.get(day, 0) / 100.0)
        for day in sorted(revenue_per_day.keys())
    ]

    return {"payments": payments, "revenue_timeseries": revenue_series}


@app.get("/admin/storage")
async def admin_storage_stats():
    """Read aggregate storage stats from storage_stats, or fall back to defaults."""

    try:
        rows = await supabase_select("storage_stats", {"limit": 1})
    except SupabaseConfigError as cfg_err:
        raise HTTPException(status_code=500, detail=str(cfg_err)) from cfg_err

    if not rows:
        stats = StorageStats(total_gb=0, avg_per_user_gb=0, auto_delete_after_days=30)
    else:
        row = rows[0]
        stats = StorageStats(
            total_gb=float(row.get("total_gb") or 0),
            avg_per_user_gb=float(row.get("avg_per_user_gb") or 0),
            auto_delete_after_days=row.get("auto_delete_after_days"),
        )

    return {"stats": stats}


@app.get("/admin/logs")
async def admin_logs_list():
    """List recent admin logs from the admin_logs table."""

    try:
        rows = await supabase_select(
            "admin_logs", {"order": "created_at.desc", "limit": 200}
        )
    except SupabaseConfigError as cfg_err:
        raise HTTPException(status_code=500, detail=str(cfg_err)) from cfg_err

    logs: list[AdminLog] = []
    for row in rows:
        created_raw = row.get("created_at")
        created_iso = (
            str(created_raw)
            if created_raw is not None
            else datetime.utcnow().isoformat() + "Z"
        )
        logs.append(
            AdminLog(
                id=str(row.get("id")),
                level=row.get("level") or "INFO",
                source=row.get("source") or "api",
                message=row.get("message") or "",
                created_at=created_iso,
            )
        )

    return {"logs": logs}


@app.get("/admin/content")
async def admin_content_get():
    """Fetch testimonials and announcements from Supabase tables."""

    try:
        testimonials_rows = await supabase_select("testimonials")
        announcements_rows = await supabase_select("announcements")
    except SupabaseConfigError as cfg_err:
        raise HTTPException(status_code=500, detail=str(cfg_err)) from cfg_err

    testimonials: list[Testimonial] = []
    for row in testimonials_rows:
        testimonials.append(
            Testimonial(
                id=str(row.get("id")),
                name=row.get("name") or "",
                role=row.get("role") or "",
                quote=row.get("quote") or "",
            )
        )

    announcements: list[Announcement] = []
    for row in announcements_rows:
        announcements.append(
            Announcement(
                id=str(row.get("id")),
                title=row.get("title") or "",
                body=row.get("body") or "",
                active=bool(row.get("active", True)),
            )
        )

    return {"testimonials": testimonials, "announcements": announcements}


@app.get("/admin/settings")
async def admin_settings_get():
    """Return admin settings from the admin_settings table (single row)."""

    try:
        rows = await supabase_select("admin_settings", {"limit": 1})
    except SupabaseConfigError as cfg_err:
        raise HTTPException(status_code=500, detail=str(cfg_err)) from cfg_err

    if not rows:
        return {"settings": _admin_settings}

    row = rows[0]
    settings = AdminSettings(
        maintenance_mode=bool(row.get("maintenance_mode", False)),
        dsp_version=row.get("dsp_version") or "stable",
        max_concurrent_jobs=int(row.get("max_concurrent_jobs") or 4),
        max_file_mb=int(row.get("max_file_mb") or 300),
    )

    return {"settings": settings}


@app.post("/admin/settings")
async def admin_settings_update(settings: AdminSettings):
    """Persist admin settings to the admin_settings table (id=1)."""

    global _admin_settings
    _admin_settings = settings

    try:
        await supabase_patch(
            "admin_settings",
            {"id": "eq.1"},
            {
                "maintenance_mode": settings.maintenance_mode,
                "dsp_version": settings.dsp_version,
                "max_concurrent_jobs": settings.max_concurrent_jobs,
                "max_file_mb": settings.max_file_mb,
            },
        )
    except SupabaseConfigError as cfg_err:
        raise HTTPException(status_code=500, detail=str(cfg_err)) from cfg_err

    return {"settings": _admin_settings}


# -------------------------
# Checkout capture endpoint
# -------------------------


@app.post("/billing/capture", response_model=CheckoutCaptureResponse)
async def billing_capture(payload: CheckoutCapturePayload):
    """Record a successful billing event in billing_payments.

    The frontend should call this after a PayPal payment is approved and
    captured. It will:
    - look up the plan by key
    - create a billing_payments row
    - optionally associate the payment with a known user_id
    """

    plan_key = payload.plan_key

    try:
        plans = await supabase_select(
            "billing_plans", {"key": f"eq.{plan_key}", "limit": 1}
        )
        if not plans:
            raise HTTPException(status_code=400, detail="Unknown plan key")

        plan = plans[0]
        plan_id = plan.get("id")

        payment_row = await supabase_insert(
            "billing_payments",
            {
                "user_id": payload.user_id,
                "plan_id": plan_id,
                "job_id": None,
                "amount_cents": payload.amount_cents,
                "currency": "USD",
                "status": "completed",
                "provider": payload.provider,
                "provider_payment_id": payload.provider_payment_id,
            },
        )
    except SupabaseConfigError as cfg_err:
        raise HTTPException(status_code=500, detail=str(cfg_err)) from cfg_err

    return CheckoutCaptureResponse(
        payment_id=str(payment_row.get("id")),
        user_id=payload.user_id,
        plan_key=plan_key,
        amount_cents=payload.amount_cents,
        currency="USD",
    )
