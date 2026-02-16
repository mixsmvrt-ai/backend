# Backend-local DSP Worker

This worker is colocated inside the backend repo so you can deploy a single codebase.

## Single-process mode (API + worker together)

Run both FastAPI and the DSP worker loop from one entrypoint:

```bash
python run_combined.py
```

This is the recommended path when you want one deployment target.

## One-click deploy command mapping

This repo now includes:

- `backend/Procfile` with `web: python run_combined.py`
- `backend/fly.toml` with:

```toml
[processes]
	app = "python run_combined.py"
```

So the deploy process starts the combined API + worker runtime directly.

## Worker-only mode (optional)

```bash
python -m dsp_worker.worker
```

## Required environment

- `BACKEND_API_URL`
- `S3_BUCKET_NAME`
- `AWS_REGION`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `WORKER_AUTH_TOKEN` (recommended)

Optional:

- `WORKER_POLL_SECONDS` (default `5`)
- `WORKER_MAX_RETRIES` (default `3`)
- `WORKER_MIN_TMP_FREE_BYTES` (default `524288000`)
