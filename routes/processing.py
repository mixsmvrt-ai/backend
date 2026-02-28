import os
from typing import Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import select

from db import get_db
from jobs import claim_pending_job, create_job, get_job, list_jobs, update_job
from models import JobStatus, ProcessingJob
from s3 import generate_presigned_download_url

router = APIRouter(tags=["processing"])


def _require_worker_auth(authorization: str | None = Header(default=None)) -> None:
    token = os.getenv("WORKER_AUTH_TOKEN")
    if not token:
        return
    expected = f"Bearer {token}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized worker")


def _serialize_job(job: ProcessingJob) -> dict[str, str | None]:
    return {
        "job_id": job.id,
        "user_id": job.user_id,
        "genre": job.genre,
        "flow_type": job.flow_type,
        "preset_name": job.preset_name,
        "input_s3_key": job.input_s3_key,
        "output_s3_key": job.output_s3_key,
        "status": job.status,
        "error_message": job.error_message,
    }


class CreateJobRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    s3_key: str = Field(min_length=3, max_length=1024)
    genre: str | None = Field(default=None, max_length=64)
    flow_type: str = Field(min_length=3, max_length=64)
    preset_name: str | None = Field(default=None, max_length=128)


class CreateJobResponse(BaseModel):
    job_id: str


@router.post("/create-job", response_model=CreateJobResponse)
def create_processing_job(payload: CreateJobRequest, db: Session = Depends(get_db)) -> CreateJobResponse:
    job = create_job(
        db,
        user_id=payload.user_id,
        input_s3_key=payload.s3_key,
        genre=payload.genre,
        flow_type=payload.flow_type,
        preset_name=payload.preset_name,
    )
    return CreateJobResponse(job_id=job.id)


@router.get("/jobs")
def get_jobs(
    status: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    _: None = Depends(_require_worker_auth),
    db: Session = Depends(get_db),
) -> dict[str, list[dict[str, str | None]]]:
    rows = list_jobs(db, status=status, limit=limit)
    return {"jobs": [_serialize_job(row) for row in rows]}


@router.post("/jobs/claim")
def claim_job(
    _: None = Depends(_require_worker_auth),
    db: Session = Depends(get_db),
) -> dict[str, str | None]:
    job = claim_pending_job(db)
    if job is None:
        raise HTTPException(status_code=404, detail="No pending jobs")
    return _serialize_job(job)


class JobUpdateRequest(BaseModel):
    status: Literal["processing", "completed", "failed"]
    output_s3_key: str | None = Field(default=None, max_length=1024)
    error_message: str | None = Field(default=None, max_length=4000)


@router.patch("/jobs/{job_id}")
def patch_job(
    job_id: str,
    payload: JobUpdateRequest,
    _: None = Depends(_require_worker_auth),
    db: Session = Depends(get_db),
) -> dict[str, str | None]:
    updated = update_job(
        db,
        job_id=job_id,
        status=payload.status,
        output_s3_key=payload.output_s3_key,
        error_message=payload.error_message,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return _serialize_job(updated)


@router.get("/jobs/{job_id}/input-download-url")
def get_job_input_download_url(
    job_id: str,
    _: None = Depends(_require_worker_auth),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    job = get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "download_url": generate_presigned_download_url(job.input_s3_key, expires=900),
    }


@router.get("/job/{job_id}")
def get_job_status(
    job_id: str,
    user_id: str = Query(..., min_length=1),
    db: Session = Depends(get_db),
) -> dict[str, str | int | None]:
    job = get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.user_id != user_id:
        raise HTTPException(status_code=403, detail="Forbidden")

    output_download_url: str | None = None
    if job.status == JobStatus.completed.value and job.output_s3_key:
        output_download_url = generate_presigned_download_url(job.output_s3_key, expires=900)

    # Queue metadata for UI: compute position among active jobs (pending/processing)
    # for the same flow_type.
    active_statuses = {JobStatus.pending.value, JobStatus.processing.value}
    queue_position: int | None = None
    queue_size: int | None = None
    try:
        stmt = (
            select(ProcessingJob.id)
            .where(ProcessingJob.flow_type == job.flow_type)
            .where(ProcessingJob.status.in_(active_statuses))
            .order_by(ProcessingJob.created_at.asc())
        )
        ids = [row[0] for row in db.execute(stmt).all()]
        if ids:
            queue_size = len(ids)
            if job.id in ids:
                queue_position = ids.index(job.id) + 1
    except Exception:
        # Best-effort only; do not fail job status if queue calc breaks.
        queue_position = None
        queue_size = None

    return {
        "status": job.status,
        "output_download_url": output_download_url,
        "error_message": job.error_message,
        "queue_feature_type": job.flow_type,
        "queue_position": queue_position,
        "queue_size": queue_size,
    }
