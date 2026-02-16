from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

from s3 import (
    MAX_FILE_SIZE_BYTES,
    generate_presigned_upload_url,
    get_s3_key_for_user,
)

router = APIRouter(tags=["upload"])


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
    upload_url = generate_presigned_upload_url(s3_key, payload.content_type)
    return GenerateUploadUrlResponse(upload_url=upload_url, s3_key=s3_key)
