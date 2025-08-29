"""Minimal device detection utilities for VibeVoice."""

import torch


def detect_device(prefer_cuda: bool = True) -> torch.device:
    """Detect the optimal device for inference.
    
    Args:
        prefer_cuda: If True, prefer CUDA over MPS when both are available.
                    This preserves default behavior on machines with NVIDIA GPUs.
    
    Returns:
        torch.device: The detected device (cuda, mps, or cpu)
    """
    # Prefer CUDA if available (preserves default behavior on machines with NVIDIA GPUs)
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")

    # Use MPS if available (Apple Silicon)
    has_mps = getattr(torch.backends, "mps", None) is not None
    if has_mps:
        try:
            if torch.backends.mps.is_available():
                return torch.device("mps")
        except Exception:
            pass

    # Fallback: CPU
    return torch.device("cpu")


def recommended_dtype_for(device: torch.device) -> torch.dtype:
    """Get recommended dtype for a given device.
    
    Args:
        device: The target device
        
    Returns:
        torch.dtype: Recommended dtype for the device
    """
    # Conservative, stable defaults
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16  # More stable than bfloat16 on MPS
    return torch.float32


def get_attention_implementation(device: torch.device) -> str:
    """Get the recommended attention implementation for a device.
    
    Args:
        device: The target device
        
    Returns:
        str: Either "flash_attention_2" or "sdpa"
    """
    if device.type == "cuda":
        try:
            import flash_attn
            return "flash_attention_2"
        except ImportError:
            return "sdpa"
    else:
        # MPS and CPU use SDPA
        return "sdpa"