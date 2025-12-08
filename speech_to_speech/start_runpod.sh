#!/bin/bash
# VibeVoice S2S RunPod Start Script
# ==================================
# Use this script when deploying on RunPod

set -e

echo "========================================"
echo "VibeVoice Speech-to-Speech Server"
echo "========================================"
echo ""

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

# Model configurations
export MODEL_PATH="${MODEL_PATH:-microsoft/VibeVoice-Realtime-0.5B}"
export ASR_MODEL="${ASR_MODEL:-small.en}"
export LLM_MODEL="${LLM_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

# Server ports (avoid 8001, 8888 which are reserved by RunPod)
export S2S_PORT="${S2S_PORT:-8005}"
export TTS_PORT="${TTS_PORT:-8000}"
export FRONTEND_PORT="${FRONTEND_PORT:-3000}"

# Create directories
mkdir -p "$HF_HOME"
mkdir -p /workspace/logs

# Navigate to app directory
cd /workspace/app 2>/dev/null || cd "$(dirname "$0")/.."

# Install dependencies if needed
if [ ! -f ".deps_installed" ]; then
    echo "Installing dependencies..."
    pip install -e . --quiet
    pip install -r speech_to_speech/requirements.txt --quiet
    touch .deps_installed
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
