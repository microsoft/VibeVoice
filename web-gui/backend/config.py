"""
Backend configuration for VibeVoice-Narrator Web GUI
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
VOICES_DIR = DATA_DIR / "voices"
DOCUMENTS_DIR = DATA_DIR / "documents"
AUDIO_DIR = DATA_DIR / "audio"

# Ensure directories exist at runtime (moved to startup to avoid import-time side effects)
def ensure_directories() -> None:
    """Create runtime-only directories used by the backend.

    This is intentionally not run at import time so tests and importers in
    read-only environments don't trigger filesystem side effects.
    """
    import logging

    logger = logging.getLogger(__name__)
    for path in (VOICES_DIR, DOCUMENTS_DIR, AUDIO_DIR):
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover - defensive logging
            # Log as error and fail fast so startup doesn't proceed with missing directories
            logger.error("Failed to ensure directory %s exists: %s", path, exc)
            raise RuntimeError(f"Failed to create required directory: {path}") from exc

# Database
# Use a POSIX-style absolute path so SQLite URLs are stable across platforms (avoid Windows backslashes)
DATABASE_URL = f"sqlite:///{DATA_DIR.resolve().as_posix()}/vibevoice.db"

# API Configuration (do not perform parsing at import time; rely on pydantic Settings for validation)

# Frontend URL (for CORS)
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# Model Configuration
DEFAULT_MODEL = "microsoft/VibeVoice-Realtime-0.5B"
DEFAULT_DEVICE = "auto"  # auto, cuda, cpu, mps

# File Upload Limits
MAX_UPLOAD_SIZE = 1024 * 1024 * 1024  # 1GB
ALLOWED_EXTENSIONS: set[str] = {".md", ".markdown", ".txt"}

# Audio Settings
SAMPLE_RATE = 24000
DEFAULT_PAUSE_MS = 500
DEFAULT_CHUNK_DEPTH = 1

# Preview warmup
WARMUP_PREVIEW = os.getenv("WARMUP_PREVIEW", "true").lower() in {"1", "true", "yes", "on"}
WARMUP_VOICE_ID = os.getenv("WARMUP_VOICE_ID")

# Preview file cleanup defaults

def _int_env(name: str, default: int) -> int:
    """Safely parse integer environment variables with a default on failure.

    Returns the default value when the environment variable is missing or cannot
    be parsed as an integer, and logs a warning so the issue is visible at
    startup rather than raising at import time.
    """
    import logging

    val = os.getenv(name)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except Exception as exc:  # pragma: no cover - defensive against bad env
        logging.getLogger(__name__).warning("Invalid integer for %s: %r; using default %s", name, val, default)
        return default

PREVIEW_TTL_MINUTES = _int_env("PREVIEW_TTL_MINUTES", 60)
PREVIEW_MAX_FILES_PER_VOICE = _int_env("PREVIEW_MAX_FILES_PER_VOICE", 10)

# Cache size defaults
MAX_MODEL_CACHE_SIZE = _int_env("MAX_MODEL_CACHE_SIZE", 4)
MAX_PROCESSOR_CACHE_SIZE = _int_env("MAX_PROCESSOR_CACHE_SIZE", 8)
MAX_VOICE_CACHE_SIZE = _int_env("MAX_VOICE_CACHE_SIZE", 16)

# Default for max iterations allowed per single request (importable constant)
MAX_ITERATIONS_PER_REQUEST = _int_env("MAX_ITERATIONS_PER_REQUEST", 10)
# Multipliers used to derive per-document and total quotas from per-request iterations
PER_ITERATION_MULTIPLIER = _int_env("PER_ITERATION_MULTIPLIER", 5)
TOTAL_MULTIPLIER = _int_env("TOTAL_MULTIPLIER", 3)


def compute_per_doc_max(iterations: int) -> int:
    return iterations * PER_ITERATION_MULTIPLIER


def compute_total_max(iterations: int) -> int:
    return compute_per_doc_max(iterations) * TOTAL_MULTIPLIER

class Settings(BaseSettings):
    """Application settings"""
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    frontend_url: str = FRONTEND_URL
    
    # Paths
    data_dir: Path = DATA_DIR
    voices_dir: Path = VOICES_DIR
    documents_dir: Path = DOCUMENTS_DIR
    audio_dir: Path = AUDIO_DIR
    
    # Database
    database_url: str = DATABASE_URL
    
    # Model settings
    default_model: str = DEFAULT_MODEL
    default_device: str = DEFAULT_DEVICE
    
    # File settings
    max_upload_size: int = MAX_UPLOAD_SIZE
    allowed_extensions: set[str] = ALLOWED_EXTENSIONS
    
    # Audio settings
    sample_rate: int = SAMPLE_RATE
    default_pause_ms: int = DEFAULT_PAUSE_MS
    default_chunk_depth: int = DEFAULT_CHUNK_DEPTH

    # Preview warmup
    warmup_preview: bool = WARMUP_PREVIEW
    warmup_voice_id: str | None = WARMUP_VOICE_ID

    # Preview cleanup configuration
    preview_ttl_minutes: int = PREVIEW_TTL_MINUTES
    preview_max_preview_files: int = PREVIEW_MAX_FILES_PER_VOICE

    # Optional demo voices directory override (can be set via DEMO_VOICES_DIR in .env)
    demo_voices_dir: Path | None = None

    # Cache sizes
    max_model_cache_size: int = MAX_MODEL_CACHE_SIZE
    max_processor_cache_size: int = MAX_PROCESSOR_CACHE_SIZE
    max_voice_cache_size: int = MAX_VOICE_CACHE_SIZE

    # Iterations limits
    max_iterations_per_request: int = MAX_ITERATIONS_PER_REQUEST

    # Debug mode: can be enabled by setting VIBEVOICE_DEBUG in environment
    # Use pydantic Field aliases so env parsing is handled by BaseSettings at instantiation time
    from pydantic import Field

    debug: bool = Field(False, alias="VIBEVOICE_DEBUG")

    # Allow uvicorn reload to be toggled via environment (default: False in production)
    # Set RELOAD=1 or RELOAD=true in environment to enable.
    reload: bool = Field(False, alias="RELOAD")

    # Pydantic v2 settings config
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
