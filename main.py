from fastapi import FastAPI, File, UploadFile, Form, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Literal, Optional, Any, Dict

from pydantic import BaseModel

from processing.mixing_pipeline import ai_mix
from processing.mastering_pipeline import ai_master
from health import create_health_router
from supabase_client import (
    create_processing_job,
    update_processing_job,
    get_processing_job,
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
    """Accept audio files and run the AI mix + master chain.

    This endpoint is intentionally simple: it stores files in a unique
    session folder, runs your existing ffmpeg-based pipelines, and returns
    the path to the rendered master file.
    """

    if beat is None and lead is None and adlibs is None:
        return {"status": "error", "message": "Upload at least one audio file."}

    session_id = uuid.uuid4().hex[:12]
    session_dir = SESSIONS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    beat_path = _save_upload(beat, session_dir / "beat.wav") if beat else None
    lead_path = _save_upload(lead, session_dir / "lead.wav") if lead else None
    adlibs_path = _save_upload(adlibs, session_dir / "adlibs.wav") if adlibs else None

    # For now, the mixing pipeline expects a vocal + beat path. If only one
    # track exists, treat it as a full mix and only run mastering.
    mixed_path = session_dir / "mix.wav"
    master_path = session_dir / "master.wav"

    if beat_path and (lead_path or adlibs_path):
        # Prefer lead vocal; if only adlibs are present, treat them as the vocal stem.
        vocal_source = lead_path or adlibs_path
        ai_mix(str(vocal_source), str(beat_path), str(mixed_path))
        mix_for_master = mixed_path
    else:
        # Single-file workflow: use whichever upload is present.
        first_available = beat_path or lead_path or adlibs_path
        mix_for_master = first_available

    ai_master(str(mix_for_master), str(master_path), target_lufs=target_lufs)

    return {
        "status": "ok",
        "session_id": session_id,
        "genre": genre,
        "target_lufs": target_lufs,
        "master_path": str(master_path),
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


# -------------------------
# Admin API routes
# -------------------------


@app.get("/admin/dashboard")
def admin_dashboard():
    stats = DashboardStats(
        total_users=1280,
        active_jobs=7,
        jobs_today=96,
        failed_jobs=3,
        revenue_today=420.5,
        revenue_month=8920.75,
        avg_processing_time=34.2,
    )
    top_presets = [
        TopPreset(id="clean_vocal", name="Clean Vocal", uses=540),
        TopPreset(id="bg_vocal_glue", name="BG Vocal Glue", uses=312),
        TopPreset(id="streaming_master", name="Streaming Master", uses=287),
    ]
    jobs_timeseries = [JobsTimeseriesPoint(**p) for p in _demo_datetime_series(14, "jobs")]
    return {"stats": stats, "top_presets": top_presets, "jobs_timeseries": jobs_timeseries}


@app.get("/admin/users")
def admin_users_list():
    users = [
        AdminUser(id="user-demo-1", email="artist@example.com", plan="Pro", country="JM"),
        AdminUser(id="user-demo-2", email="engineer@example.com", plan="Studio", country="US"),
        AdminUser(id="user-demo-3", email="tester@example.com", plan=None, country=None, status="suspended"),
    ]
    return {"users": users}


@app.get("/admin/users/{user_id}")
def admin_user_detail(user_id: str):
    # In a real app this would query Supabase/Postgres; for now return a demo user.
    user = AdminUserDetail(
        id=user_id,
        email="artist@example.com",
        plan="Pro",
        country="JM",
        status="active",
        credits=120,
        jobs_count=48,
    )
    return {"user": user}


class UpdateUserStatusPayload(BaseModel):
    status: Literal["active", "suspended", "banned"]


@app.post("/admin/users/{user_id}/status")
def admin_update_user_status(user_id: str, payload: UpdateUserStatusPayload):
    # No-op demo: accept the change and echo it back.
    return {"user_id": user_id, "status": payload.status}


class UpdateUserCreditsPayload(BaseModel):
    delta: int


@app.post("/admin/users/{user_id}/credits")
def admin_update_user_credits(user_id: str, payload: UpdateUserCreditsPayload):
    # No-op demo: pretend credits were updated.
    return {"user_id": user_id, "delta": payload.delta}


@app.get("/admin/jobs")
def admin_jobs_list():
    jobs = [
        AdminJob(
            id="job-demo-1",
            user_email="artist@example.com",
            status="completed",
            input_type="stems",
            preset="streaming_master",
            duration_sec=185.3,
        ),
        AdminJob(
            id="job-demo-2",
            user_email="tester@example.com",
            status="failed",
            input_type="single",
            preset="clean_vocal",
            duration_sec=92.1,
        ),
    ]
    return {"jobs": jobs}


@app.get("/admin/jobs/{job_id}")
def admin_job_detail(job_id: str):
    now = datetime.utcnow().isoformat() + "Z"
    job = AdminJobDetail(
        id=job_id,
        user_email="artist@example.com",
        status="completed",
        input_type="stems",
        preset="streaming_master",
        duration_sec=185.3,
        created_at=now,
        steps=[
            "Upload received",
            "Vocal/beat alignment",
            "AI mix chain",
            "Mastering chain",
        ],
        input_url=None,
        output_url=None,
        logs=[
            "[info] job started",
            "[info] running vocal chain",
            "[info] applying streaming master preset",
            "[info] job finished",
        ],
    )
    return {"job": job}


@app.get("/admin/presets")
def admin_presets_list():
    presets = [
        AdminPreset(id="clean_vocal", name="Clean Vocal", role="vocal", enabled=True, version="1.0.0"),
        AdminPreset(id="bg_vocal_glue", name="BG Vocal Glue", role="bgv", enabled=True, version="1.0.0"),
        AdminPreset(id="streaming_master", name="Streaming Master", role="master", enabled=True, version="1.1.0"),
    ]
    return {"presets": presets}


@app.get("/admin/presets/{preset_id}")
def admin_preset_detail(preset_id: str):
    params = _preset_params.get(preset_id, PresetParams())
    return {"preset_params": params}


@app.post("/admin/presets/{preset_id}")
def admin_update_preset(preset_id: str, payload: PresetParams = Body(...)):
    _preset_params[preset_id] = payload
    return {"preset_id": preset_id, "preset_params": payload}


@app.get("/admin/plans")
def admin_plans_list():
    plans = [
        AdminPlan(id="free", name="Free", price_month=0.0, credits=20, stem_limit=2),
        AdminPlan(id="pro", name="Pro", price_month=19.0, credits=200, stem_limit=8),
        AdminPlan(id="studio", name="Studio", price_month=49.0, credits=800, stem_limit=None),
    ]
    return {"plans": plans}


@app.get("/admin/payments")
def admin_payments_list():
    now = datetime.utcnow()
    payments = [
        AdminPayment(
            id="pay-demo-1",
            user_email="artist@example.com",
            amount=19.0,
            provider="stripe",
            status="succeeded",
            created_at=(now - timedelta(days=1)).isoformat() + "Z",
        ),
        AdminPayment(
            id="pay-demo-2",
            user_email="studio@example.com",
            amount=49.0,
            provider="stripe",
            status="succeeded",
            created_at=(now - timedelta(days=3)).isoformat() + "Z",
        ),
    ]
    revenue_series = [RevenuePoint(**p) for p in _demo_datetime_series(30, "amount")]
    return {"payments": payments, "revenue_timeseries": revenue_series}


@app.get("/admin/storage")
def admin_storage_stats():
    stats = StorageStats(total_gb=42.5, avg_per_user_gb=0.5, auto_delete_after_days=30)
    return {"stats": stats}


@app.get("/admin/logs")
def admin_logs_list():
    now = datetime.utcnow()
    logs = [
        AdminLog(
            id="log-1",
            level="INFO",
            source="api",
            message="healthcheck ok",
            created_at=now.isoformat() + "Z",
        ),
        AdminLog(
            id="log-2",
            level="WARN",
            source="dsp",
            message="slow processing on job-demo-2",
            created_at=(now - timedelta(minutes=5)).isoformat() + "Z",
        ),
        AdminLog(
            id="log-3",
            level="ERROR",
            source="dsp",
            message="failed to load pitch correction plugin",
            created_at=(now - timedelta(minutes=30)).isoformat() + "Z",
        ),
    ]
    return {"logs": logs}


@app.get("/admin/content")
def admin_content_get():
    testimonials = [
        Testimonial(id="t1", name="Khalil", role="Artist", quote="Mixes hit like a record label drop."),
        Testimonial(id="t2", name="Maya", role="Engineer", quote="Cuts my rough-mix time in half."),
    ]
    announcements = [
        Announcement(
            id="a1",
            title="New vocal engine",
            body="We just shipped an upgraded vocal chain tuned for dancehall and afrobeat.",
            active=True,
        ),
        Announcement(
            id="a2",
            title="Planned maintenance",
            body="Short downtime this weekend while we roll out GPU upgrades.",
            active=False,
        ),
    ]
    return {"testimonials": testimonials, "announcements": announcements}


@app.get("/admin/settings")
def admin_settings_get():
    return {"settings": _admin_settings}


@app.post("/admin/settings")
def admin_settings_update(settings: AdminSettings):
    global _admin_settings
    _admin_settings = settings
    return {"settings": _admin_settings}
