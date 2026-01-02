#!/bin/bash
# VibeVoice S2S RunPod Start Script
# ==================================
# Use this script when deploying on RunPod
#
# Usage:
#   bash speech_to_speech/start_runpod.sh           # Normal start
#   bash speech_to_speech/start_runpod.sh --fresh   # Fresh install (clear cache)
#   bash speech_to_speech/start_runpod.sh --setup   # First-time setup (install apt packages)

set -e

echo "========================================"
echo "VibeVoice Speech-to-Speech Server"
echo "========================================"
echo ""

# Check for --setup flag to install system dependencies
for arg in "$@"; do
    if [ "$arg" = "--setup" ]; then
        echo "Installing system dependencies..."
        apt-get update && apt-get install -y --no-install-recommends \
            ffmpeg \
            libsndfile1 \
            libportaudio2 \
            portaudio19-dev \
            libasound2-dev \
            && rm -rf /var/lib/apt/lists/*
        echo "System dependencies installed."
        echo ""
    fi
done

# Display system info
echo "System Information:"
echo "-------------------"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "CUDA Version: $(nvcc --version 2>/dev/null | grep release | awk '{print $5}' | tr -d ',' || echo 'N/A')"
echo "PyTorch Version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')"
echo ""

# Set default environment variables
export HF_HOME="${HF_HOME:-/workspace/models}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/workspace/models/transformers}"
export TORCH_HOME="${TORCH_HOME:-/workspace/models/torch}"

# Fix cuDNN library path for ctranslate2/faster-whisper
# The pip-installed nvidia packages put libraries in non-standard locations
echo "Configuring CUDA library paths..."

# Try to find nvidia library paths from pip packages
NVIDIA_LIBS=""

# cuDNN
CUDNN_PATH=$(python -c "import nvidia.cudnn; print(nvidia.cudnn.__path__[0])" 2>/dev/null)
if [ -n "$CUDNN_PATH" ] && [ -d "$CUDNN_PATH/lib" ]; then
    NVIDIA_LIBS="${CUDNN_PATH}/lib"
    echo "  Found cuDNN: $CUDNN_PATH/lib"
fi

# cuBLAS
CUBLAS_PATH=$(python -c "import nvidia.cublas; print(nvidia.cublas.__path__[0])" 2>/dev/null)
if [ -n "$CUBLAS_PATH" ] && [ -d "$CUBLAS_PATH/lib" ]; then
    NVIDIA_LIBS="${NVIDIA_LIBS:+$NVIDIA_LIBS:}${CUBLAS_PATH}/lib"
    echo "  Found cuBLAS: $CUBLAS_PATH/lib"
fi

# cuFFT (sometimes needed)
CUFFT_PATH=$(python -c "import nvidia.cufft; print(nvidia.cufft.__path__[0])" 2>/dev/null)
if [ -n "$CUFFT_PATH" ] && [ -d "$CUFFT_PATH/lib" ]; then
    NVIDIA_LIBS="${NVIDIA_LIBS:+$NVIDIA_LIBS:}${CUFFT_PATH}/lib"
fi

# Fallback: search in common pip locations
if [ -z "$NVIDIA_LIBS" ]; then
    SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)
    if [ -d "$SITE_PACKAGES/nvidia" ]; then
        for lib_dir in "$SITE_PACKAGES/nvidia"/*/lib; do
            if [ -d "$lib_dir" ]; then
                NVIDIA_LIBS="${NVIDIA_LIBS:+$NVIDIA_LIBS:}$lib_dir"
            fi
        done
        echo "  Found nvidia libs in: $SITE_PACKAGES/nvidia/*/lib"
    fi
fi

# Set LD_LIBRARY_PATH
if [ -n "$NVIDIA_LIBS" ]; then
    export LD_LIBRARY_PATH="${NVIDIA_LIBS}:${LD_LIBRARY_PATH:-}"
fi

# Model configurations
# NOTE: ASR_MODEL must be a faster-whisper compatible model:
#   - Built-in: tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v2, large-v3, large-v3-turbo
#   - CTranslate2: Systran/faster-distil-whisper-small.en, Systran/faster-distil-whisper-medium.en
# Do NOT use distil-whisper/distil-small.en (wrong format)
export MODEL_PATH="${MODEL_PATH:-microsoft/VibeVoice-Realtime-0.5B}"
export ASR_MODEL="${ASR_MODEL:-large-v3-turbo}"  # Fast and accurate model for transcription (Whisper large-v3-turbo)
export LLM_MODEL="${LLM_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

# Web Search API (Perplexity - intelligent search with real-time info)
# Get your API key at: https://www.perplexity.ai/settings/api
export PERPLEXITY_API_KEY="${PERPLEXITY_API_KEY:-}"

# Server ports (avoid 8001, 8888 which are reserved by RunPod)
export S2S_PORT="${S2S_PORT:-8005}"
export TTS_PORT="${TTS_PORT:-8000}"
export FRONTEND_PORT="${FRONTEND_PORT:-3000}"

# Create directories
mkdir -p "$HF_HOME"
mkdir -p /workspace/logs

# Clean up any incorrectly cached ASR models (distil-whisper is wrong format)
if [ -d "$HF_HOME/hub/models--distil-whisper--distil-small.en" ]; then
    echo "Removing incorrectly cached ASR model (distil-whisper format)..."
    rm -rf "$HF_HOME/hub/models--distil-whisper--distil-small.en"
fi

# Display model configuration
echo ""
echo "Model Configuration:"
echo "-------------------"
echo "TTS Model: $MODEL_PATH"
echo "ASR Model: $ASR_MODEL (Whisper large-v3-turbo - Fast & Accurate)"
echo "LLM Model: $LLM_MODEL"
echo ""
echo "Library Path (LD_LIBRARY_PATH):"
echo "  ${LD_LIBRARY_PATH:-<not set>}"
echo ""

# Navigate to app directory
cd /workspace/app 2>/dev/null || cd "$(dirname "$0")/.."

# Check for --fresh flag to force reinstall
FRESH_INSTALL=false
for arg in "$@"; do
    if [ "$arg" = "--fresh" ]; then
        FRESH_INSTALL=true
        echo "Fresh install requested, removing cached files..."
        rm -f .deps_installed
        find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        find . -type f -name "*.pyc" -delete 2>/dev/null || true
    fi
done

# Install dependencies if needed
if [ ! -f ".deps_installed" ]; then
    echo "Installing dependencies..."
    pip install -e . --quiet
    pip install -r speech_to_speech/requirements.txt --quiet
    touch .deps_installed
    echo "Dependencies installed successfully."
fi

# Pre-download models
echo "Ensuring models are cached..."
python -c "
from huggingface_hub import snapshot_download
import os

models = [
    os.environ.get('MODEL_PATH', 'microsoft/VibeVoice-Realtime-0.5B'),
    os.environ.get('LLM_MODEL', 'Qwen/Qwen2.5-1.5B-Instruct'),
]

for model in models:
    print(f'Checking {model}...')
    try:
        snapshot_download(model, cache_dir=os.environ.get('HF_HOME', '/workspace/models'))
        print(f'  ✓ {model} ready')
    except Exception as e:
        print(f'  ⚠ Warning: {e}')
"

echo ""
echo "========================================"
echo "Starting S2S Pipeline Server"
echo "========================================"
echo "S2S API: http://0.0.0.0:$S2S_PORT"
echo "WebSocket: ws://0.0.0.0:$S2S_PORT/stream"
echo "Features: VAD (Mandatory), Conversation Display, Perplexity LLM"
echo ""
echo "RunPod Proxy URLs will be:"
echo "  https://<pod-id>-$S2S_PORT.proxy.runpod.net/"
echo "  wss://<pod-id>-$S2S_PORT.proxy.runpod.net/stream"
echo "========================================"
echo ""

# Start the server
python -m speech_to_speech.s2s_pipeline \
    --host 0.0.0.0 \
    --port "$S2S_PORT" \
    --asr-model "$ASR_MODEL" \
    --llm-model "$LLM_MODEL" \
    --tts-model "$MODEL_PATH" \
    --device cuda \
    --log-level INFO
