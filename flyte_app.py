"""
VibeVoice Realtime TTS - Flyte 2.0 Deployment Wrapper

This module provides a Flyte 2.0 deployment wrapper for the VibeVoice
Realtime TTS streaming service. It wraps the existing FastAPI application
with Flyte's deployment and management capabilities.

Features:
- Real-time WebSocket audio streaming
- Multiple voice presets with speaker profiles
- Configurable CFG scale and inference steps
- Interactive web UI for text-to-speech generation
- Streaming text input with real-time synthesis
"""

import logging
import pathlib
from pathlib import Path

import flyte
from flyte.app.extras import FastAPIAppEnvironment

# Import the existing FastAPI app from demo/web
from demo.web.app import app

# Build Docker image with requirements from pyproject.toml
# The image needs CUDA support for GPU-accelerated inference
image = (
    flyte.Image.from_debian_base(name="vibevoice-realtime", python_version=(3, 12))
    .with_apt_packages("ffmpeg", "git")  # ffmpeg for audio processing, git for model downloads
    .with_uv_project(Path(__file__).parent / "pyproject.toml", pre=True)
)

# Configure Flyte environment for deployment
env = FastAPIAppEnvironment(
    name="vibevoice-realtime-tts",
    app=app,
    description="High-quality streaming text-to-speech service powered by VibeVoice Realtime model.",
    image=image,
    # Resource allocation - GPU required for model inference
    resources=flyte.Resources(
        cpu=4,
        memory="16Gi",
        gpu=1,  # NVIDIA GPU required for VibeVoice model
        disk="20Gi",
        shm="20Gi",
    ),
    # Include web demo files (index.html and app.py)
    include=["demo/web/index.html", "demo/web/app.py", "demo/voices/**/*.pt"],
    # Environment variables for model configuration
    env_vars={
        "MODEL_PATH": "microsoft/VibeVoice-Realtime-0.5b",  # HuggingFace model ID
        "MODEL_DEVICE": "cuda",  # Use GPU for inference
        "VOICE_PRESET": "en-WHTest_man",  # Default voice preset
    },
    requires_auth=False,  # Set to True if authentication is needed
)


@env.app.get("/info")
async def app_info() -> dict:
    """
    Get information about the deployed Flyte app.

    Returns deployment metadata including endpoint URL and service status.
    """
    return {
        "service": "VibeVoice Realtime TTS",
        "version": "0.0.1",
        "model": "microsoft/VibeVoice-Realtime-0.5b",
        "endpoint": getattr(env, "endpoint", "Not deployed"),
        "features": [
            "Real-time streaming text-to-speech",
            "WebSocket audio streaming",
            "Multiple voice presets",
            "Configurable CFG scale",
            "Adjustable inference steps",
            "Interactive web UI",
        ],
        "routes": {
            "index": "/",
            "websocket_stream": "/stream",
            "config": "/config",
            "info": "/info",
        },
    }


if __name__ == "__main__":
    # Initialize Flyte from configuration
    flyte.init_from_config(
        root_dir=pathlib.Path(__file__).parent,
        log_level=logging.INFO,
    )

    print("=" * 70)
    print("🎤 VibeVoice Realtime TTS Service")
    print("=" * 70)
    print("\nModel: microsoft/VibeVoice-Realtime-0.5b")
    print("Features: High-quality streaming text-to-speech with multiple voices")
    print("\n📍 Available Endpoints:")
    print("   GET  /                        - Web UI (index.html)")
    print("   WS   /stream                  - WebSocket audio streaming")
    print("   GET  /config                  - Get available voice presets")
    print("   GET  /info                    - Service information")
    print("\n" + "=" * 70)
    print("\nDeploying to Flyte...\n")

    # Deploy the application
    deployment = flyte.serve(env)
    print(f"\n✅ Deployment successful!")
    print(f"App URL: {deployment.url}")
    print(f"Endpoint: {deployment.endpoint}")
