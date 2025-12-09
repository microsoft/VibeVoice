#!/bin/bash
# =============================================================================
# VEMI AI Fine-Tuning Environment Setup
# =============================================================================
# This script sets up the correct Python environment for fine-tuning.
# Run this FIRST before any training scripts.
#
# Usage: bash setup_env.sh
# =============================================================================

set -e

echo "========================================"
echo "VEMI AI Fine-Tuning Environment Setup"
echo "========================================"
echo ""

# Check GPU
echo "[1/5] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Uninstall conflicting packages
echo "[2/5] Removing conflicting packages..."
pip uninstall -y transformers peft trl bitsandbytes accelerate torchao 2>/dev/null || true

# Install specific compatible versions
echo ""
echo "[3/5] Installing compatible package versions..."
pip install --upgrade pip

# These versions are tested to work together
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.44.2
pip install peft==0.12.0
pip install trl==0.9.6
pip install accelerate==0.33.0
pip install bitsandbytes==0.43.1
pip install datasets==2.20.0
pip install scipy>=1.10.0

echo ""
echo "[4/5] Installing additional dependencies..."
pip install safetensors>=0.4.0
pip install sentencepiece>=0.1.99
pip install protobuf>=4.25.0

echo ""
echo "[5/5] Verifying installation..."
python -c "
import torch
import transformers
import peft
import trl
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'PEFT: {peft.__version__}')
print(f'TRL: {trl.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "========================================"
echo "Environment Setup Complete!"
echo "========================================"
echo ""
echo "Now run:"
echo "  python expand_automobile_data.py"
echo "  python train_medical_peft.py --epochs 3 --batch_size 4"
echo "  python train_automobile_peft.py --epochs 3 --batch_size 4"
echo "========================================"
