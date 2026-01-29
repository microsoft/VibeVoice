"""
Export endpoints for VibeVoice-Narrator
"""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, field_validator
from typing import Optional, Literal
import os
from pathlib import Path
from urllib.parse import urlparse, unquote

router = APIRouter(tags=["Export"])

class ExportRequest(BaseModel):
    """Request model for export"""
    audio_url: str
    filename: str = "output.wav"
    format: Literal["wav", "mp3", "ogg"] = "wav"

    @field_validator("filename", mode="before")
    @classmethod
    def _sanitize_filename(cls, v):
        """Sanitize and validate provided filename to prevent path traversal or separators.

        - Decode percent-encoded sequences first (e.g., %2e%2e -> ..)
        - Normalize to a safe basename using Path(...).name
        - Reject empty or special names
        """
        if not isinstance(v, str):
            raise ValueError("filename must be a string")
        decoded = unquote(v)
        # Quick rejection if it contains explicit traversal segments
        if ".." in decoded.split("/") or ".." in decoded.split("\\"):
            # Reject any inputs that contain explicit traversal segments
            raise ValueError("filename must be a plain basename without path components")
        safe = Path(decoded).name
        # Final checks: must be non-empty basename and not contain path separators
        if not safe or '/' in safe or '\\' in safe or safe in ('.', '..'):
            raise ValueError("filename must be a plain basename without path components")
        return safe

class ExportResponse(BaseModel):
    """Response model for export"""
    success: bool
    message: str
    download_url: Optional[str] = None

def _normalize_and_validate_audio_path(audio_url: str) -> str:
    """Normalize and validate an audio URL or path and return a safe, normalized leading-slash path.

    Raises HTTPException(status_code=400) for invalid inputs (path traversal, missing static prefix, etc.).
    """
    parsed_path = urlparse(audio_url).path if "://" in audio_url else audio_url
    decoded_path = unquote(parsed_path)
    audio_path = decoded_path.lstrip('/')

    segments = audio_path.split('/') if audio_path else []
    if any(seg == '..' for seg in segments) or audio_path.startswith('..'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid audio URL"
        )

    normalized = os.path.normpath(audio_path)

    if '..' in normalized.split(os.path.sep) or normalized.startswith('..'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid audio URL"
        )

    parts = normalized.split(os.path.sep)
    if parts[0] != 'static' or len(parts) < 2 or not parts[-1]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid audio URL"
        )

    return '/' + '/'.join(parts)

@router.post("/download", response_model=ExportResponse)
async def download_audio(request: ExportRequest):
    """
    Download generated audio file
    
    This is a placeholder implementation. In production, this would:
    1. Validate the audio URL
    2. Generate download response with proper headers
    3. Handle different formats (wav, mp3, ogg)
    """
    try:
        # Validate and normalize the incoming audio URL/path
        sanitized_url = _normalize_and_validate_audio_path(request.audio_url)


        # Placeholder: In production, this would serve the file
        return ExportResponse(
            success=True,
            message=f"Audio file '{request.filename}' ready for download",
            download_url=sanitized_url
        )

    except HTTPException:
        # Re-raise client errors (400s) so FastAPI preserves the status code
        raise
    except Exception as e:
        # Log the full exception server-side and return a non-revealing error to the client
        import logging
        logger = logging.getLogger(__name__)
        logger.exception("Unhandled exception while processing download: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
