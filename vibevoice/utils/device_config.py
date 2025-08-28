"""Device configuration utilities for VibeVoice."""

import os
import platform
import torch
import warnings
from typing import Dict, Any


def setup_mps_environment_early() -> None:
    """Set environment variables for MPS fallback and memory management.

    Must be called before torch initialization on macOS to ensure proper MPS behavior.
    Sets PYTORCH_ENABLE_MPS_FALLBACK=1 and PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
    """
    if platform.system() == "Darwin":  # macOS
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


def _flash_attn_available() -> bool:
    """Check if flash attention is available using importlib.util.find_spec for safer checking."""
    try:
        import importlib.util
        return importlib.util.find_spec("flash_attn") is not None
    except Exception:
        return False


def _sdpa_supported_on_mps(dtype: torch.dtype) -> bool:
    """Probe scaled_dot_product_attention support on MPS with given dtype.

    Args:
        dtype: The torch dtype to test SDPA with

    Returns:
        bool: True if SDPA works on MPS with the given dtype without crashing
    """
    if not torch.backends.mps.is_available():
        return False

    try:
        # Use small tensors to avoid heavy allocations
        q = torch.randn(1, 1, 8, 8, device="mps", dtype=dtype)
        k = torch.randn(1, 1, 8, 8, device="mps", dtype=dtype)
        v = torch.randn(1, 1, 8, 8, device="mps", dtype=dtype)

        # Test if scaled_dot_product_attention works
        torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return True
    except Exception:
        return False


def resolve_config_from_args(args) -> Dict[str, Any]:
    """Resolve device, dtype, and attention implementation configuration from command line args.
    
    This function centralizes the duplicated logic from demo scripts.
    
    Args:
        args: Parsed command line arguments with device, dtype, and attn_impl attributes
        
    Returns:
        dict: Configuration with keys:
        - device: device type as string ("cuda", "mps", or "cpu")
        - dtype: optimal torch dtype
        - attn_implementation: attention implementation to use
    """
    # Get optimal config which provides defaults
    config = get_optimal_config()

    # Override with command line args if specified
    device = args.device if args.device != "auto" else config["device"]
    
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if args.dtype == "auto":
        torch_dtype = config["dtype"]
    else:
        torch_dtype = dtype_map[args.dtype]

    attn_impl = args.attn_impl if args.attn_impl != "auto" else config["attn_implementation"]

    return {
        "device": device,
        "dtype": torch_dtype,
        "attn_implementation": attn_impl,
    }


def get_optimal_config() -> Dict[str, Any]:
    """Get optimal device, dtype, and attention implementation configuration.

    Returns:
        dict: Configuration with keys:
        - device: device type as string ("cuda", "mps", or "cpu")
        - dtype: optimal torch dtype
        - attn_implementation: attention implementation to use
    """
    # CUDA path
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
        attn_implementation = "flash_attention_2" if _flash_attn_available() else "sdpa"
        return {
            "device": device,
            "dtype": dtype,
            "attn_implementation": attn_implementation,
        }

    # MPS path
    if torch.backends.mps.is_available():
        device = "mps"

        # Check if bfloat16 is supported with SDPA
        if _sdpa_supported_on_mps(torch.bfloat16):
            dtype = torch.bfloat16
        elif _sdpa_supported_on_mps(torch.float16):
            # bfloat16 not supported, try float16
            dtype = torch.float16
            warnings.warn(
                "bfloat16 not supported on MPS with SDPA. Using float16.",
                UserWarning,
                stacklevel=2,
            )
        else:
            # Neither works, fallback to float16 with warning
            dtype = torch.float16
            warnings.warn(
                "SDPA not supported on MPS with bfloat16 or float16. Defaulting to float16.",
                UserWarning,
                stacklevel=2,
            )

        attn_implementation = "sdpa"
        return {
            "device": device,
            "dtype": dtype,
            "attn_implementation": attn_implementation,
        }

    # CPU path
    device = "cpu"
    dtype = torch.float32
    attn_implementation = "sdpa"

    return {
        "device": device,
        "dtype": dtype,
        "attn_implementation": attn_implementation,
    }
