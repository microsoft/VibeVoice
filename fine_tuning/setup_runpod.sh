#!/bin/bash
# =============================================================================
# VEMI AI Fine-Tuning Setup Script for RunPod
# =============================================================================
# 
# This script sets up the fine-tuning environment on RunPod GPU instances.
#
# Usage:
#   1. Create a RunPod GPU Pod (A40 or A100 recommended)
#   2. SSH into the pod
#   3. Run: bash setup_runpod.sh
#
# =============================================================================

set -e

echo "========================================"
echo "VEMI AI Fine-Tuning Setup"
echo "========================================"
echo ""

# Check GPU
echo "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Navigate to workspace
cd /workspace

# Clone or update repository
if [ -d "VibeVoice" ]; then
    echo "Updating VibeVoice repository..."
    cd VibeVoice
    git pull origin main
else
    echo "Cloning VibeVoice repository..."
    git clone https://github.com/your-repo/VibeVoice.git
    cd VibeVoice
fi

# Navigate to fine-tuning directory
cd fine_tuning

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install Unsloth (optimized for speed)
echo ""
echo "Installing Unsloth..."
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Create data directories
mkdir -p data/medical data/automobile models/medical models/automobile

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Prepare datasets:"
echo "   python prepare_data.py --domain all"
echo ""
echo "2. Train Medical model:"
echo "   python train_medical.py"
echo ""
echo "3. Train Automobile model:"
echo "   python train_automobile.py"
echo ""
echo "4. Copy models to main project:"
echo "   cp -r models/medical/merged_model ../speech_to_speech/models/medical_llm"
echo "   cp -r models/automobile/merged_model ../speech_to_speech/models/automobile_llm"
echo ""
echo "========================================"
