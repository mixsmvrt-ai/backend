# S3 Job Processing Routes

This directory contains production-style routes for an S3-backed async audio processing architecture.

## Endpoints

- `POST /generate-upload-url`
- `POST /create-job`
- `GET /jobs?status=pending&limit=1` (worker/internal)
- `POST /jobs/claim` (atomic claim for worker concurrency)
- `GET /jobs/{job_id}/input-download-url` (worker/internal)
- `PATCH /jobs/{job_id}` (worker status updates)
- `GET /job/{job_id}?user_id=...` (frontend polling)

All user-facing access should use presigned URLs only. Raw bucket/object URLs are never exposed.
