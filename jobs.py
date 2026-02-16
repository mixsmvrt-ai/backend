from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from models import JobStatus, ProcessingJob


def create_job(
    db: Session,
    *,
    user_id: str,
    input_s3_key: str,
    genre: Optional[str],
    flow_type: str,
    preset_name: Optional[str],
) -> ProcessingJob:
    job = ProcessingJob(
        user_id=user_id,
        input_s3_key=input_s3_key,
        genre=genre,
        flow_type=flow_type,
        preset_name=preset_name,
        status=JobStatus.pending.value,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_job(db: Session, job_id: str) -> ProcessingJob | None:
    return db.get(ProcessingJob, job_id)


def list_jobs(
    db: Session,
    *,
    status: str | None = None,
    limit: int = 20,
) -> list[ProcessingJob]:
    stmt = select(ProcessingJob).order_by(ProcessingJob.created_at.asc()).limit(limit)
    if status:
        stmt = stmt.where(ProcessingJob.status == status)
    return list(db.execute(stmt).scalars().all())


def claim_pending_job(db: Session) -> ProcessingJob | None:
    # Atomic claim using row lock + SKIP LOCKED so multiple workers can
    # safely poll in parallel without claiming the same job.
    stmt = (
        select(ProcessingJob)
        .where(ProcessingJob.status == JobStatus.pending.value)
        .order_by(ProcessingJob.created_at.asc())
        .with_for_update(skip_locked=True)
        .limit(1)
    )

    with db.begin():
        job = db.execute(stmt).scalars().first()
        if job is None:
            return None
        job.status = JobStatus.processing.value
        job.error_message = None
        job.updated_at = datetime.utcnow()

    db.refresh(job)
    return job


def update_job(
    db: Session,
    *,
    job_id: str,
    status: str,
    output_s3_key: str | None = None,
    error_message: str | None = None,
) -> ProcessingJob | None:
    job = db.get(ProcessingJob, job_id)
    if job is None:
        return None

    job.status = status
    if output_s3_key is not None:
        job.output_s3_key = output_s3_key
    if error_message is not None:
        job.error_message = error_message
    elif status == JobStatus.completed.value:
        job.error_message = None

    job.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(job)
    return job
