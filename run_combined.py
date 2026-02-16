import asyncio
import logging
import os

import uvicorn

from dsp_worker.worker import run_worker_loop

logger = logging.getLogger("riddimbase_combined")


def _build_uvicorn_server() -> uvicorn.Server:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    log_level = os.getenv("LOG_LEVEL", "info")

    config = uvicorn.Config(
        "main:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=False,
        workers=1,
    )
    return uvicorn.Server(config)


async def run_backend_and_worker() -> None:
    """Run FastAPI API and DSP worker loop in one process.

    This is intended for single-service deployments where you do not want a
    separate worker deploy target. The worker polls backend endpoints using
    BACKEND_API_URL and claims jobs atomically, so duplicate processing is
    still prevented.
    """

    server = _build_uvicorn_server()
    worker_task = asyncio.create_task(run_worker_loop(), name="dsp-worker-loop")
    server_task = asyncio.create_task(server.serve(), name="uvicorn-server")

    done, pending = await asyncio.wait(
        {worker_task, server_task},
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
