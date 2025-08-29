#!/usr/bin/env python
# coding=utf-8
"""
MPS Diagnostic Utility for VibeVoice.
This module provides diagnostic functions to quickly assess PyTorch installation,
MPS backend availability, bfloat16 support, and SDPA functionality on Apple Silicon.
"""
import torch
import sys
from typing import Tuple
def diagnose_mps_setup() -> None:
    """
    Diagnose common MPS setup issues and print a clear status report.
    
    This function checks:
    - PyTorch version
    - MPS backend availability and build status
    - Basic MPS operations (matrix multiplication)
    - BFloat16 tensor support on MPS
    - Scaled Dot Product Attention (SDPA) functionality
    """
    print("=== VibeVoice MPS Diagnostic Tool ===")
    print()
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Check MPS availability and build status
    mps_available = torch.backends.mps.is_available()
    mps_built = torch.backends.mps.is_built()
    
    print(f"MPS available: {mps_available}")
    print(f"MPS built: {mps_built}")
    
    if not mps_available:
        if not mps_built:
            print("❌ MPS is not built in this PyTorch installation.")
            print("   Recommendation: Install a PyTorch version with MPS support.")
        else:
            print("❌ MPS is built but not available.")
            print("   Recommendation: Check if you're running on Apple Silicon (M1/M2/M3) macOS 12.3+.")
        print()
        return
    
    print("✅ MPS is available and built.")
    print()
    
    # Test basic MPS operations
    print("Testing basic MPS operations...")
    try:
        x = torch.randn(100, 100, device="mps")
        y = torch.randn(100, 100, device="mps")
        z = torch.matmul(x, y)
        print("✅ Basic MPS operations work (matrix multiplication successful)")
    except Exception as e:
        print(f"❌ Basic MPS operations failed: {e}")
        print("   Recommendation: Check your PyTorch installation and macOS version.")
        print()
        return
    
    # Test bfloat16 tensor allocation
    print("\nTesting bfloat16 tensor support...")
    try:
        # Try to create a bfloat16 tensor on MPS
        bf16_tensor = torch.randn(10, 10, device="mps", dtype=torch.bfloat16)
        print("✅ BFloat16 tensors supported on MPS")
        bf16_supported = True
    except Exception as e:
        print(f"⚠️  BFloat16 tensors not supported on MPS: {e}")
        print("   This is normal on some PyTorch versions or macOS versions.")
        bf16_supported = False
    
    # Test float16 tensor allocation
    print("\nTesting float16 tensor support...")
    try:
        # Try to create a float16 tensor on MPS
        fp16_tensor = torch.randn(10, 10, device="mps", dtype=torch.float16)
        print("✅ Float16 tensors supported on MPS")
        fp16_supported = True
    except Exception as e:
        print(f"❌ Float16 tensors not supported on MPS: {e}")
        print("   This is unexpected. Check your PyTorch installation.")
        fp16_supported = False
    
    # Test SDPA functionality
    print("\nTesting Scaled Dot Product Attention (SDPA)...")
    dtypes_to_test = []
    if bf16_supported:
        dtypes_to_test.append((torch.bfloat16, "BFloat16"))
    if fp16_supported:
        dtypes_to_test.append((torch.float16, "Float16"))
    
    # Always test float32 as a fallback
    dtypes_to_test.append((torch.float32, "Float32"))
    
    sdpa_works = False
    best_dtype = None
    
    for dtype, dtype_name in dtypes_to_test:
        try:
            # Use small tensors to avoid heavy allocations
            q = torch.randn(1, 4, 16, 32, device="mps", dtype=dtype)
            k = torch.randn(1, 4, 16, 32, device="mps", dtype=dtype)
            v = torch.randn(1, 4, 16, 32, device="mps", dtype=dtype)
            
            # Test if scaled_dot_product_attention works
            output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            
            print(f"✅ SDPA works with {dtype_name}")
            sdpa_works = True
            best_dtype = dtype_name
            break
        except Exception as e:
            print(f"⚠️  SDPA failed with {dtype_name}: {type(e).__name__}")
    
    if not sdpa_works:
        print("❌ SDPA not working with any tested dtype")
        print("   Recommendation: Update PyTorch to a version with better MPS support")
    else:
        print(f"✅ SDPA is functional (best performance with {best_dtype})")
    
    print()
    print("=== Summary ===")
    if mps_available and sdpa_works:
        print("✅ MPS setup looks good for VibeVoice!")
        print("   You should be able to run inference on Apple Silicon.")
    else:
        print("⚠️  MPS setup has issues that may affect VibeVoice performance.")
        print("   See recommendations above.")
    
    print()
    print("Note: For best performance, ensure you're using:")
    print("  - Apple Silicon (M1/M2/M3) Mac")
    print("  - macOS 12.3 or later")
    print("  - PyTorch 2.0 or later with MPS support")
def main() -> None:
    """Main entry point when running as a module."""
    diagnose_mps_setup()
if __name__ == "__main__":
    main()
