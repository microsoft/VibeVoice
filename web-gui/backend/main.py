"""
FastAPI backend for VibeVoice-Narrator Web GUI
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import Iterable, Dict
import os
import uvicorn
from backend.config import settings, ensure_directories
from backend.routes import tts
from backend.routes import export as export_routes

VERSION = "1.0.0"

app = FastAPI(
    title="VibeVoice-Narrator API",
    description="Markdown-to-Speech conversion API",
    version=VERSION,
)

# CORS middleware
cors_options = {
    "allow_origins": [settings.frontend_url, "http://localhost:3000", "http://localhost:3001"],
    "allow_credentials": True,
    "allow_methods": ["*"],
    "allow_headers": ["*"],
}
# Only permit localhost:<port> origin regex in development/debug mode
if getattr(settings, "debug", False):
    cors_options["allow_origin_regex"] = r"^http://localhost:\d+$"

app.add_middleware(CORSMiddleware, **cors_options)

# Ensure the directory we will serve from exists before mounting static files to avoid RuntimeError
try:
    # Ensure both audio_dir and data_dir exist (we mount data_dir as /static)
    settings.audio_dir.mkdir(parents=True, exist_ok=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
except Exception as exc:  # pragma: no cover - defensive failure during startup
    import logging

    logger = logging.getLogger(__name__)
    logger.exception("Failed to create required directories %s or %s", settings.audio_dir, settings.data_dir)
    raise RuntimeError(f"Unable to create required directories '{settings.audio_dir}' or '{settings.data_dir}': {exc}") from exc

# Serve static files (audio, documents) from data_dir
app.mount("/static", StaticFiles(directory=str(settings.data_dir)), name="static")


@app.on_event("startup")
def warmup_preview() -> None:
    # Create any required runtime directories before starting background work
    ensure_directories()

    if not settings.warmup_preview:
        return

    import threading
    threading.Thread(target=tts.warmup_preview, daemon=True).start()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "VibeVoice-Narrator API",
        "version": VERSION,
        "status": "running",
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

def _parse_voice_filename(voice_file: Path) -> Dict[str, str]:
    stem = voice_file.stem
    language = "Unknown"
    gender = None
    name = stem

    if "-" in stem:
        lang_code, remainder = stem.split("-", 1)
        language_map = {
            "en": "English",
            "de": "German",
            "fr": "French",
            "hi": "Hindi",
            "it": "Italian",
            "ja": "Japanese",
            "ko": "Korean",
            "nl": "Dutch",
            "pl": "Polish",
            "pt": "Portuguese",
            "es": "Spanish",
        }
        language = language_map.get(lang_code.lower(), lang_code)
        name = remainder

    if "_" in name:
        base, suffix = name.rsplit("_", 1)
        gender_map = {
            "man": "male",
            "woman": "female",
            "male": "male",
            "female": "female",
        }
        gender = gender_map.get(suffix.lower())
        # Only use the base name when the suffix maps to a known gender; otherwise keep the original name
        if gender is not None:
            name = base

    return {
        "id": stem,
        "name": name,
        "language": language,
        "gender": gender,
        "path": str(voice_file),
    }


def _collect_voice_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.exists() and path.is_dir():
            files.extend(path.glob("*.pt"))
    return files


@app.get("/voices")
async def list_voices():
    """List available voices"""
    voices = []

    import logging
    logger = logging.getLogger(__name__)

    # Resolve demo voices directory: prefer explicit setting if provided
    demo_voices_dir = settings.demo_voices_dir if getattr(settings, 'demo_voices_dir', None) else settings.data_dir.parent.parent / "demo" / "voices" / "streaming_model"

    # Validate existence and fall back to configured voices_dir if necessary
    if not (demo_voices_dir and getattr(demo_voices_dir, 'exists', lambda: False)() and demo_voices_dir.is_dir()):
        logger.warning(f"Demo voices directory {demo_voices_dir} not found; falling back to configured voices_dir")
        demo_voices_dir = settings.voices_dir

    voice_files = _collect_voice_files([settings.voices_dir, demo_voices_dir])

    # Deduplicate parsed voices by id (prefer the first occurrence)
    seen_ids: set[str] = set()
    deduped: list[Dict[str, str]] = []
    for voice_file in voice_files:
        parsed = _parse_voice_filename(voice_file)
        vid = parsed.get("id")
        if vid in seen_ids:
            continue
        seen_ids.add(vid)
        deduped.append(parsed)

    return {"voices": deduped}

@app.get("/config")
async def get_config():
    """Get current configuration"""
    return {
        "default_model": settings.default_model,
        "default_device": settings.default_device,
        "sample_rate": settings.sample_rate,
        "default_pause_ms": settings.default_pause_ms,
        "default_chunk_depth": settings.default_chunk_depth,
        "max_iterations_per_request": settings.max_iterations_per_request,
    }

# Include API routes
app.include_router(tts.router, prefix="/api/tts")
app.include_router(export_routes.router, prefix="/api/export")

if __name__ == "__main__":
    # Determine reload behavior:
    # - If RELOAD environment variable is explicitly set, use that (settings.reload)
    # - Otherwise, default reload behavior to settings.debug (useful in dev)
    if os.getenv("RELOAD", "") == "":
        reload_flag = bool(getattr(settings, "debug", False))
    else:
        reload_flag = bool(getattr(settings, "reload", False))

    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=reload_flag,
    )
