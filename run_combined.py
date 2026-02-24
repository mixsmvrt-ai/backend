import asyncio
import logging
import os

import uvicorn

logger = logging.getLogger("riddimbase_combined")


def _build_uvicorn_server() -> uvicorn.Server:
    host = os.getenv("HOST", "0.0.0.0")
    # Fly defaults to internal_port 8080 unless configured otherwise.
    # Use PORT when provided, else fall back to 8080 for safe deployment.
    port = int(os.getenv("PORT", "8080"))
    log_level = os.getenv("LOG_LEVEL", "info")

    config = uvicorn.Config(
        "main:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=False,
        workers=1,
    )
    logger.info("Starting combined runtime on %s:%s", host, port)
    return uvicorn.Server(config)


async def run_backend_and_worker() -> None:
    """Run FastAPI API and DSP worker loop in one process.

    This is intended for single-service deployments where you do not want a
    separate worker deploy target. The worker polls backend endpoints using
    BACKEND_API_URL and claims jobs atomically, so duplicate processing is
    still prevented.
    """

    server = _build_uvicorn_server()
    server_task = asyncio.create_task(server.serve(), name="uvicorn-server")

    worker_task: asyncio.Task | None = None
    enable_worker = os.getenv("ENABLE_DSP_WORKER", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if enable_worker:
        try:
            # Import lazily so API can still start listening even if worker
            # dependencies are missing or misconfigured.
            from dsp_worker.worker import run_worker_loop

            worker_task = asyncio.create_task(run_worker_loop(), name="dsp-worker-loop")
            logger.info("DSP worker loop started")
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to start DSP worker loop: %s", exc)
    else:
        logger.info("DSP worker loop disabled by ENABLE_DSP_WORKER")

    wait_set = {server_task}
    if worker_task is not None:
        wait_set.add(worker_task)

    done, pending = await asyncio.wait(
        wait_set,
        return_when=asyncio.FIRST_COMPLETED,
    )

    for task in done:
        exc = task.exception()
        if exc is not None:
            logger.exception("Combined runtime task failed: %s", exc)

    for task in pending:
        task.cancel()

    await asyncio.gather(*pending, return_exceptions=True)


def main() -> None:
    asyncio.run(run_backend_and_worker())


if __name__ == "__main__":
    main()
