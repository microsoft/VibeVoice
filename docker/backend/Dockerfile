# VibeVoice Backend Dockerfile
# Based on NVIDIA PyTorch Container as recommended in documentation

FROM nvcr.io/nvidia/pytorch:24.07-py3

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first for better caching
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir flash-attn --no-build-isolation || true

# Copy the entire project
COPY vibevoice/ ./vibevoice/
COPY demo/ ./demo/

# Create directory for HuggingFace cache
RUN mkdir -p /root/.cache/huggingface

# Expose the backend port
EXPOSE 8000

# Environment variables with defaults
ENV MODEL_PATH=microsoft/VibeVoice-Realtime-0.5B \
    MODEL_DEVICE=cuda \
    VOICE_PRESET=en-Carter_man

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run the FastAPI server
CMD ["python", "-m", "uvicorn", "demo.web.app:app", "--host", "0.0.0.0", "--port", "8000"]

