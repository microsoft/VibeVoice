#!/usr/bin/env python3
"""
Quick dependency installer for VEMI AI Fine-Tuning
Run this first: python install_deps.py
"""
import subprocess
import sys

print("="*60)
print("Installing compatible dependencies for fine-tuning...")
print("="*60)

# Uninstall conflicting packages first
print("\n[1/3] Removing conflicting packages...")
for pkg in ["transformers", "peft", "trl", "accelerate", "bitsandbytes", "torchao"]:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", pkg], 
                   capture_output=True)

# Install compatible versions
print("\n[2/3] Installing compatible versions...")
packages = [
    "transformers==4.44.2",
    "peft==0.12.0",
    "trl==0.9.6",
    "accelerate==0.33.0",
    "bitsandbytes==0.43.1",
    "datasets>=2.16.0",
    "safetensors>=0.4.0",
    "scipy>=1.10.0",
]

for pkg in packages:
    print(f"  Installing {pkg}...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg])

# Verify installation
print("\n[3/3] Verifying installation...")
try:
    import transformers
    import peft
    import trl
    import torch
    
    print(f"\n  ✓ transformers: {transformers.__version__}")
    print(f"  ✓ peft: {peft.__version__}")
    print(f"  ✓ trl: {trl.__version__}")
    print(f"  ✓ torch: {torch.__version__}")
    print(f"  ✓ CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
    
    print("\n" + "="*60)
    print("SUCCESS! Dependencies installed correctly.")
    print("="*60)
    print("\nNow run:")
    print("  python train_medical_peft.py --epochs 3 --batch_size 4")
    print("  python train_automobile_peft.py --epochs 3 --batch_size 4")
    
except Exception as e:
    print(f"\nERROR: {e}")
    print("Please try running: bash setup_env.sh")
