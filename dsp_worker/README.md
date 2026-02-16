# Backend-local DSP Worker

This worker is colocated inside the backend repo so you can deploy a single codebase.

Run it as a separate process in the same deployment environment:

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
