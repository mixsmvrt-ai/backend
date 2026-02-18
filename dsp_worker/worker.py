import asyncio
import logging
import os
import shutil
import tempfile
from pathlib import Path

import httpx

from .processor import process_audio_file
from .s3_client import download_from_presigned_url, get_output_s3_key, upload_file_to_s3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mixsmvrt_dsp_worker")

_default_local_port = os.getenv("PORT", "8080")
BACKEND_API_URL = os.getenv(
    "BACKEND_API_URL",
    f"http://127.0.0.1:{_default_local_port}",
).rstrip("/")
WORKER_AUTH_TOKEN = os.getenv("WORKER_AUTH_TOKEN")
POLL_SECONDS = float(os.getenv("WORKER_POLL_SECONDS", "5"))
MAX_RETRIES = int(os.getenv("WORKER_MAX_RETRIES", "3"))
MIN_TMP_FREE_BYTES = int(os.getenv("WORKER_MIN_TMP_FREE_BYTES", str(500 * 1024 * 1024)))


def _auth_headers() -> dict[str, str]:
    if not WORKER_AUTH_TOKEN:
        return {}
    return {"Authorization": f"Bearer {WORKER_AUTH_TOKEN}"}


async def _claim_job(client: httpx.AsyncClient) -> dict | None:
    resp = await client.post(f"{BACKEND_API_URL}/jobs/claim", headers=_auth_headers())
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


async def _get_input_download_url(client: httpx.AsyncClient, job_id: str) -> str:
    resp = await client.get(
        f"{BACKEND_API_URL}/jobs/{job_id}/input-download-url",
        headers=_auth_headers(),
    )
    resp.raise_for_status()
    payload = resp.json()
    url = payload.get("download_url")
    if not isinstance(url, str) or not url:
        raise RuntimeError("Missing download_url in backend response")
    return url


async def _patch_job(
    client: httpx.AsyncClient,
    job_id: str,
    *,
    status: str,
    output_s3_key: str | None = None,
    error_message: str | None = None,
) -> None:
    payload: dict[str, str] = {"status": status}
    if output_s3_key:
        payload["output_s3_key"] = output_s3_key
    if error_message:
        payload["error_message"] = error_message[:3900]

    resp = await client.patch(
        f"{BACKEND_API_URL}/jobs/{job_id}",
        headers=_auth_headers(),
        json=payload,
    )
    resp.raise_for_status()


def _cleanup_paths(*paths: str) -> None:
    for p in paths:
        try:
            if p and Path(p).exists():
                Path(p).unlink(missing_ok=True)
        except Exception:
            pass


def _ensure_tmp_capacity() -> None:
    usage = shutil.disk_usage("/tmp")
    if usage.free < MIN_TMP_FREE_BYTES:
        raise RuntimeError(
            f"Insufficient /tmp free space ({usage.free} bytes). "
            f"Required minimum is {MIN_TMP_FREE_BYTES} bytes."
        )


def _make_tmp_path(prefix: str, suffix: str) -> str:
    handle = tempfile.NamedTemporaryFile(prefix=prefix, suffix=suffix, dir="/tmp", delete=False)
    path = handle.name
    handle.close()
    return path


async def _process_claimed_job(client: httpx.AsyncClient, job: dict) -> None:
    job_id = str(job.get("job_id") or "")
    user_id = str(job.get("user_id") or "")
    flow_type = str(job.get("flow_type") or "")
    preset_name = job.get("preset_name")

    if not job_id or not user_id or not flow_type:
        raise RuntimeError(f"Invalid claimed job payload: {job}")

    _ensure_tmp_capacity()
    input_path = _make_tmp_path(prefix=f"mixsmvrt-in-{job_id}-", suffix=".wav")
    output_path = _make_tmp_path(prefix=f"mixsmvrt-out-{job_id}-", suffix=".wav")

    try:
        download_url = await _get_input_download_url(client, job_id)
        await download_from_presigned_url(download_url, input_path)

        process_audio_file(
            input_path,
            output_path,
            flow_type=flow_type,
            preset_name=str(preset_name) if preset_name is not None else None,
        )

        output_key = get_output_s3_key(user_id, job_id, extension="wav")
        upload_file_to_s3(output_path, output_key)

        await _patch_job(
            client,
            job_id,
            status="completed",
            output_s3_key=output_key,
        )
        logger.info("Completed job %s", job_id)
    finally:
        _cleanup_paths(input_path, output_path)


async def run_worker_loop() -> None:
    timeout = httpx.Timeout(connect=20.0, read=120.0, write=120.0, pool=20.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        while True:
            try:
                job = await _claim_job(client)
                if job is None:
                    await asyncio.sleep(POLL_SECONDS)
                    continue

                job_id = str(job.get("job_id") or "")
                logger.info("Claimed job %s", job_id)

                last_error: Exception | None = None
                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        await _process_claimed_job(client, job)
                        last_error = None
                        break
                    except Exception as exc:
                        last_error = exc
                        logger.exception("Job %s failed attempt %s/%s", job_id, attempt, MAX_RETRIES)
                        await asyncio.sleep(min(5.0 * attempt, 15.0))

                if last_error is not None:
                    await _patch_job(
                        client,
                        job_id,
                        status="failed",
                        error_message=str(last_error),
                    )
            except Exception:
                logger.exception("Worker loop error")
                await asyncio.sleep(POLL_SECONDS)


if __name__ == "__main__":
    asyncio.run(run_worker_loop())
