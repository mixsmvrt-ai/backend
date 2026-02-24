import logging

from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

from s3 import (
    MAX_FILE_SIZE_BYTES,
    generate_presigned_upload_url,
    get_s3_key_for_user,
)

router = APIRouter(tags=["upload"])
logger = logging.getLogger("riddimbase_backend.upload")


class GenerateUploadUrlRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    filename: str = Field(min_length=1, max_length=255)
    content_type: str = Field(min_length=3, max_length=128)
    file_size_bytes: int | None = Field(default=None, ge=1)


class GenerateUploadUrlResponse(BaseModel):
    upload_url: str
    s3_key: str
    max_file_size_bytes: int = MAX_FILE_SIZE_BYTES


@router.post("/generate-upload-url", response_model=GenerateUploadUrlResponse)
def generate_upload_url(payload: GenerateUploadUrlRequest) -> GenerateUploadUrlResponse:
    if payload.file_size_bytes is not None and payload.file_size_bytes > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max supported size is {MAX_FILE_SIZE_BYTES} bytes",
        )

    s3_key = get_s3_key_for_user(payload.user_id, payload.filename)

    try:
        upload_url = generate_presigned_upload_url(s3_key, payload.content_type)
        return GenerateUploadUrlResponse(upload_url=upload_url, s3_key=s3_key)
    except RuntimeError as exc:
        # e.g. Missing S3_BUCKET_NAME
        logger.exception("Runtime configuration error while generating presigned upload URL")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "S3_CONFIG_ERROR",
                "message": str(exc),
            },
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive
        msg = str(exc) or "Unknown error"
        lower = msg.lower()
        if "credential" in lower or "access key" in lower or "secret" in lower:
            code = "AWS_CREDENTIALS_MISSING"
            status = 500
        elif "signature" in lower or "token" in lower or "expired" in lower:
            code = "S3_PRESIGN_FAILED"
            status = 502
        else:
            code = "UPLOAD_URL_GENERATION_FAILED"
            status = 500

        logger.exception("Error while generating presigned upload URL")
        raise HTTPException(
            status_code=status,
            detail={
                "error": code,
                "message": msg,
            },
        ) from exc
