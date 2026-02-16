import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.config import Config


MAX_FILE_SIZE_BYTES = 200 * 1024 * 1024


def _require_bucket_name() -> str:
    bucket = os.getenv("S3_BUCKET_NAME")
    if not bucket:
        raise RuntimeError("Missing required env var S3_BUCKET_NAME")
    return bucket


def _s3_client():
    region = os.getenv("AWS_REGION", "us-east-1")
    # boto3 reads credentials from env/instance profile automatically.
    return boto3.client(
        "s3",
        region_name=region,
        config=Config(signature_version="s3v4"),
    )


def sanitize_filename(filename: str) -> str:
    base = Path(filename).name
    # keep letters/numbers/dot/dash/underscore only
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    return cleaned[:180] or "audio.wav"


def get_s3_key_for_user(user_id: str, filename: str) -> str:
    safe_name = sanitize_filename(filename)
    date_key = datetime.now(timezone.utc).strftime("%Y/%m/%d")
    unique = uuid.uuid4().hex
    return f"uploads/{user_id}/{date_key}/{unique}-{safe_name}"


def generate_presigned_upload_url(filename: str, content_type: str) -> str:
    bucket = _require_bucket_name()
    client = _s3_client()
    return client.generate_presigned_url(
        "put_object",
        Params={
            "Bucket": bucket,
            "Key": filename,
            "ContentType": content_type,
        },
        ExpiresIn=900,
    )


def generate_presigned_download_url(key: str, expires: int = 900) -> str:
    bucket = _require_bucket_name()
    client = _s3_client()
    return client.generate_presigned_url(
        "get_object",
        Params={
            "Bucket": bucket,
            "Key": key,
        },
        ExpiresIn=expires,
    )


def upload_file_to_s3(local_path: str, key: str) -> None:
    bucket = _require_bucket_name()
    client = _s3_client()
    client.upload_file(local_path, bucket, key)
