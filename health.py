import os
import time
from datetime import datetime, timezone

from fastapi import APIRouter

START_TIME = time.time()


def _get_environment() -> str:
    """Return a simple environment label for monitoring.

    We keep this intentionally minimal: anything starting with "stag"
    becomes "staging"; everything else is treated as "production".
    """

    raw = (os.getenv("MIXSMVRT_ENV") or "production").lower()
    if raw.startswith("stag"):
        return "staging"
    return "production"


def create_health_router(service_name: str, version: str = "1.0.0") -> APIRouter:
    """Create a lightweight health router for a FastAPI service.

    Endpoints:
    - GET /health       – liveness, no dependencies
    - GET /health/ready – readiness + optional metadata

    Both are safe for public access and suitable for uptime monitors.
    """

    router = APIRouter()
    environment = _get_environment()

    @router.get("/health", include_in_schema=False)
    async def health() -> dict[str, str]:
        return {
            "status": "ok",
            "service": service_name,
            "environment": environment,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @router.get("/health/ready", include_in_schema=False)
    async def health_ready() -> dict[str, object]:
        uptime_seconds = int(time.time() - START_TIME)
        return {
            "status": "ok",
            "service": service_name,
            "environment": environment,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": version,
            "uptime_seconds": uptime_seconds,
        }

    return router
