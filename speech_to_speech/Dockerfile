# VibeVoice Speech-to-Speech Dockerfile
# ======================================
# Optimized for RunPod GPU deployment
# Base: NVIDIA PyTorch Container (CUDA 12.x)

FROM nvcr.io/nvidia/pytorch:24.07-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Model cache directories
ENV HF_HOME=/workspace/models
ENV TRANSFORMERS_CACHE=/workspace/models/transformers
ENV TORCH_HOME=/workspace/models/torch

# Application settings
# NOTE: ASR_MODEL must be faster-whisper compatible (NOT distil-whisper/distil-small.en)
ENV MODEL_PATH=microsoft/VibeVoice-Realtime-0.5B
ENV ASR_MODEL=small.en
ENV LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct
ENV CUDA_VISIBLE_DEVICES=0

# Server ports
ENV S2S_PORT=8005
ENV TTS_PORT=8000
ENV FRONTEND_PORT=3000

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    portaudio19-dev \
    git \
    curl \
    wget \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Create workspace directories
RUN mkdir -p /workspace/models \
    && mkdir -p /workspace/app \
    && mkdir -p /workspace/logs

# Set working directory
WORKDIR /workspace/app

# Copy requirements first (for layer caching)
COPY speech_to_speech/requirements.txt /workspace/app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Install flash-attention (optional, for faster TTS)
RUN pip install flash-attn --no-build-isolation || echo "Flash attention not installed (optional)"

# Copy application code
COPY . /workspace/app/

# Install VibeVoice package
RUN pip install -e /workspace/app/

# Create start script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "========================================"\n\
echo "VibeVoice Speech-to-Speech Server"\n\
echo "========================================"\n\
echo ""\n\
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"\n\
echo "CUDA: $(nvcc --version | grep release | awk '{print $5}' | tr -d ',')"\n\
echo "PyTorch: $(python -c "import torch; print(torch.__version__)")"\n\
echo ""\n\
\n\
# Download models if not cached\n\
echo "Checking models..."\n\
python -c "\n\
from huggingface_hub import snapshot_download\n\
import os\n\
\n\
models = [\n\
    os.environ.get(\"MODEL_PATH\", \"microsoft/VibeVoice-Realtime-0.5B\"),\n\
    os.environ.get(\"LLM_MODEL\", \"Qwen/Qwen2.5-1.5B-Instruct\"),\n\
]\n\
\n\
for model in models:\n\
    print(f\"Ensuring {model} is cached...\")\n\
    try:\n\
        snapshot_download(model, cache_dir=os.environ.get(\"HF_HOME\", \"/workspace/models\"))\n\
    except Exception as e:\n\
        print(f\"Warning: Could not download {model}: {e}\")\n\
"\n\
\n\
echo "Starting S2S Pipeline Server on port $S2S_PORT..."\n\
cd /workspace/app\n\
python -m speech_to_speech.s2s_pipeline \\\n\
    --host 0.0.0.0 \\\n\
    --port ${S2S_PORT:-8005} \\\n\
    --asr-model ${ASR_MODEL:-distil-whisper/distil-small.en} \\\n\
    --llm-model ${LLM_MODEL:-Qwen/Qwen2.5-1.5B-Instruct} \\\n\
    --tts-model ${MODEL_PATH:-microsoft/VibeVoice-Realtime-0.5B} \\\n\
    --device cuda \\\n\
    --log-level INFO\n\
' > /workspace/start.sh && chmod +x /workspace/start.sh

# Expose ports
# 8005 - S2S Pipeline API & WebSocket
# 8000 - TTS WebSocket (if running separately)
# 3000 - Frontend (if running separately)
EXPOSE 8005 8000 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8005/health || exit 1

# Default command
CMD ["/workspace/start.sh"]
