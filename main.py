from fastapi import FastAPI, File, UploadFile, Form, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import uuid
import asyncio
from datetime import datetime, timedelta, date
from typing import Literal, Optional, Any, Dict

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
    SupabaseConfigError,
)

app = FastAPI(title="RiddimBase Studio Backend")

# Allow local Next.js dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002"],
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
    # Arbitrary JSON metadata about where the input files live
    # (e.g. Supabase storage paths, S3 URLs, or internal session IDs).
    input_files: Dict[str, Any]


class JobStatusResponse(BaseModel):
    id: str
    status: Literal["queued", "processing", "completed", "failed"]
    progress: int
    current_stage: Optional[str] = None
    error_message: Optional[str] = None
    output_files: Optional[Dict[str, Any]] = None


async def _run_processing_job(job_id: str, job_type: str, input_files: Dict[str, Any]) -> None:
    """Background coroutine that simulates a multi-stage DSP pipeline.

    In production you can replace the simulated stages with real calls to
    your DSP microservice, passing along the job_id so the DSP can also
    update Supabase directly if desired.
    """

    try:
        # Mark job as processing
        await update_processing_job(
            job_id,
            {
                "status": "processing",
                "progress": 5,
                "current_stage": "starting",
            },
        )

        # Example staged updates – replace sleeps with real DSP work.
        await update_processing_job(
            job_id,
            {
                "progress": 20,
                "current_stage": "preparing_inputs",
            },
        )
        # await call_dsp_stage(...)
        await asyncio.sleep(0.1)

        await update_processing_job(
            job_id,
            {
                "progress": 50,
                "current_stage": f"running_{job_type}_pipeline",
            },
        )
        # await call_dsp_stage(...)
        await asyncio.sleep(0.1)

        await update_processing_job(
            job_id,
            {
                "progress": 80,
                "current_stage": "finalizing_outputs",
            },
        )
        # await call_dsp_stage(...)
        await asyncio.sleep(0.1)

        # In a real system, output_files would come from the DSP
        output_files: Dict[str, Any] = {
            "master_url": input_files.get("target_path", ""),
        }

        await update_processing_job(
            job_id,
            {
                "status": "completed",
                "progress": 100,
                "current_stage": "done",
                "output_files": output_files,
            },
        )
    except Exception as exc:  # pragma: no cover - defensive
        # Best-effort failure update so the job is not stuck forever.
        message = str(exc)
        try:
            await update_processing_job(
                job_id,
                {
                    "status": "failed",
                    "current_stage": "error",
                    "error_message": message,
                },
            )
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

    try:
        await update_processing_job(
            job_id,
            {
                "status": "processing",
                "progress": 5,
                "current_stage": "saving_inputs",
            },
        )

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
                    "current_stage": "mixing_stems",
                },
            )
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
                "current_stage": "mastering_mix",
            },
        )

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
                "current_stage": "done",
                "output_files": output_files,
            },
        )
    except Exception as exc:  # pragma: no cover - defensive
        message = str(exc)
        try:
            await update_processing_job(
                job_id,
                {
                    "status": "failed",
                    "current_stage": "error",
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

    try:
        job_row = await create_processing_job(
            {
                "user_id": payload.user_id,
                "job_type": payload.job_type,
                "status": "queued",
                "progress": 0,
                "current_stage": "queued",
                "input_files": payload.input_files,
            }
        )
    except SupabaseConfigError as cfg_err:
        raise HTTPException(status_code=500, detail=str(cfg_err)) from cfg_err

    job_id = str(job_row["id"])

    # Fire-and-forget background coroutine – this process should run with
    # WEB_CONCURRENCY=1 or otherwise ensure that jobs are not duplicated.
    asyncio.create_task(_run_processing_job(job_id, payload.job_type, payload.input_files))

    return JobStatusResponse(
        id=job_id,
        status=job_row.get("status", "queued"),
        progress=job_row.get("progress", 0),
        current_stage=job_row.get("current_stage"),
        error_message=job_row.get("error_message"),
        output_files=job_row.get("output_files"),
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

    return JobStatusResponse(
        id=str(job_row["id"]),
        status=job_row.get("status", "queued"),
        progress=job_row.get("progress", 0),
        current_stage=job_row.get("current_stage"),
        error_message=job_row.get("error_message"),
        output_files=job_row.get("output_files"),
    )


@app.post("/api/mix-master")
async def mix_master(
    beat: UploadFile | None = File(default=None),
    lead: UploadFile | None = File(default=None),
    adlibs: UploadFile | None = File(default=None),
    genre: str = Form(""),
    target_lufs: str = Form("-14"),
):
    """Accept audio files and enqueue an AI mix + master job.

    Files are stored in a unique session folder and a processing_jobs row
    is created in Supabase. The actual DSP work then runs in the
    background, and clients should poll /status/{job_id} for progress.
    """

    if beat is None and lead is None and adlibs is None:
        return {"status": "error", "message": "Upload at least one audio file."}

    session_id = uuid.uuid4().hex[:12]
    session_dir = SESSIONS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    beat_path = _save_upload(beat, session_dir / "beat.wav") if beat else None
    lead_path = _save_upload(lead, session_dir / "lead.wav") if lead else None
    adlibs_path = _save_upload(adlibs, session_dir / "adlibs.wav") if adlibs else None

    # Persist a queued job in Supabase so the frontend can track it.
    input_files: Dict[str, Any] = {
        "session_id": session_id,
        "session_dir": str(session_dir),
        "beat_path": str(beat_path) if beat_path else None,
        "lead_path": str(lead_path) if lead_path else None,
        "adlibs_path": str(adlibs_path) if adlibs_path else None,
        "genre": genre,
        "target_lufs": target_lufs,
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

    # Provide some generic step labels so the admin UI timeline stays useful
    job_type = job_row.get("job_type") or "mix_master"
    steps: list[str] = ["Upload received"]
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
