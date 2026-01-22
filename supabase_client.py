import os
from typing import Any, Dict, Optional

import httpx

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")


class SupabaseConfigError(RuntimeError):
    pass


def _get_base_url() -> str:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise SupabaseConfigError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in the environment for job tracking."
        )
    return SUPABASE_URL.rstrip("/") + "/rest/v1"


def _get_headers() -> Dict[str, str]:
    if not SUPABASE_SERVICE_ROLE_KEY:
        raise SupabaseConfigError(
            "SUPABASE_SERVICE_ROLE_KEY must be set in the environment for job tracking."
        )
    return {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


async def create_processing_job(job_data: Dict[str, Any]) -> Dict[str, Any]:
    """Insert a new processing_jobs row and return the inserted record."""

    base_url = _get_base_url()
    headers = _get_headers()

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{base_url}/processing_jobs",
            headers={**headers, "Prefer": "return=representation"},
            json=job_data,
        )
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list) or not data:
            raise RuntimeError("Unexpected response when creating processing job")
        return data[0]


async def update_processing_job(job_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Patch an existing processing_jobs row by id and return the updated record."""

    base_url = _get_base_url()
    headers = _get_headers()

    params = {"id": f"eq.{job_id}"}

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.patch(
            f"{base_url}/processing_jobs",
            headers={**headers, "Prefer": "return=representation"},
            params=params,
            json=updates,
        )
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list) or not data:
            raise RuntimeError("Unexpected response when updating processing job")
        return data[0]


async def get_processing_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a single processing_jobs row by id, or None if not found."""

    base_url = _get_base_url()
    headers = _get_headers()

    params = {"id": f"eq.{job_id}", "limit": 1}

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            f"{base_url}/processing_jobs",
            headers=headers,
            params=params,
        )
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list) or not data:
            return None
        return data[0]
