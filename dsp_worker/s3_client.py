import os
import uuid
from datetime import datetime, timezone

import boto3
import httpx


def _require_bucket() -> str:
    bucket = os.getenv("S3_BUCKET_NAME")
    if not bucket:
        raise RuntimeError("S3_BUCKET_NAME is required")
    return bucket


def _s3_client():
    region = os.getenv("AWS_REGION", "us-east-1")
    return boto3.client("s3", region_name=region)


def get_output_s3_key(user_id: str, job_id: str, extension: str = "wav") -> str:
    day = datetime.now(timezone.utc).strftime("%Y/%m/%d")
    ext = extension.lstrip(".")
    token = uuid.uuid4().hex[:10]
    return f"outputs/{user_id}/{day}/{job_id}-{token}.{ext}"


async def download_from_presigned_url(url: str, destination_path: str) -> None:
    timeout = httpx.Timeout(connect=30.0, read=300.0, write=60.0, pool=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(destination_path, "wb") as handle:
                async for chunk in resp.aiter_bytes():
                    handle.write(chunk)


def upload_file_to_s3(local_path: str, key: str) -> None:
    client = _s3_client()
    bucket = _require_bucket()
    client.upload_file(local_path, bucket, key)
