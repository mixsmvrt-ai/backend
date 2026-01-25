from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from supabase_client import update_processing_job


def _now_iso() -> str:
    """Return current UTC time in ISO 8601 format."""

    return datetime.now(timezone.utc).isoformat()


def _compute_progress(completed_steps: int, total_steps: int) -> int:
    """Compute progress percentage using the canonical formula.

    progress = round((completed_steps / total_steps) * 100)
    """

    if total_steps <= 0:
        return 0
    completed = max(0, min(completed_steps, total_steps))
    return round((completed / total_steps) * 100)


async def update_progress(
    job_id: str,
    step_name: str,
    step_index: int,
    total_steps: int,
) -> Dict[str, Any]:
    """Advance a job's progress after a DSP step completes.

    - step_index is 1-based (1..total_steps)
    - status is set to "processing"
    - current_stage is the human-readable step name
    - progress only moves forward, never jumps multiple steps at once
    """

    completed_steps = step_index
    progress = _compute_progress(completed_steps, total_steps)

    payload: Dict[str, Any] = {
        "status": "processing",
        "current_stage": step_name,
        "progress": progress,
        "updated_at": _now_iso(),
    }

    return await update_processing_job(job_id, payload)


async def mark_job_complete(job_id: str, final_stage: str | None = None) -> Dict[str, Any]:
    """Mark a job as fully completed at 100%.

    If final_stage is provided, it is used as current_stage; otherwise a
    generic "Completed" label is applied.
    """

    payload: Dict[str, Any] = {
        "status": "completed",
        "progress": 100,
        "updated_at": _now_iso(),
    }
    payload["current_stage"] = final_stage or "Completed"

    return await update_processing_job(job_id, payload)


async def mark_job_failed(job_id: str, step_name: str, error_message: str) -> Dict[str, Any]:
    """Mark a job as failed after a DSP error.

    This preserves the last successful progress value by *not* touching the
    numeric progress field. Only status, current_stage, error_message, and
    updated_at are written.
    """

    payload: Dict[str, Any] = {
        "status": "failed",
        "current_stage": f"Error during {step_name}",
        "error_message": error_message,
        "updated_at": _now_iso(),
    }

    return await update_processing_job(job_id, payload)
