"""VRAM detection and quantization recommendation utilities."""

import torch
from transformers.utils import logging

logger = logging.get_logger(__name__)


def get_available_vram_gb() -> float:
    """
    Get available VRAM in GB.
    
    Returns:
        float: Available VRAM in GB, or 0 if no CUDA device available
    """
    if not torch.cuda.is_available():
        return 0.0
    
    try:
        # Get first CUDA device
        device = torch.device("cuda:0")

        # Prefer direct CUDA mem info if available (free, total in bytes)
        if hasattr(torch.cuda, "mem_get_info"):
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            available_gb = free_bytes / (1024 ** 3)
        else:
            # Fallback: estimate free memory from total minus reserved/allocated
            props = torch.cuda.get_device_properties(device)
            total_bytes = props.total_memory
            reserved_bytes = torch.cuda.memory_reserved(device)
            allocated_bytes = torch.cuda.memory_allocated(device)
            used_bytes = max(reserved_bytes, allocated_bytes)
            free_bytes = max(total_bytes - used_bytes, 0)
            available_gb = free_bytes / (1024 ** 3)

        return available_gb
    except Exception as e:
        logger.warning(f"Could not detect VRAM: {e}")
        return 0.0


def suggest_quantization(available_vram_gb: float, model_name: str = "VibeVoice-7B") -> str:
    """
    Suggest quantization level based on available VRAM.
    
    Args:
        available_vram_gb: Available VRAM in GB
        model_name: Name of the model being loaded
        
    Returns:
        str: Suggested quantization level ("fp16", "8bit", or "4bit")
    """
    # Parse model size from name (e.g., "0.5B", "1.5B", "7B")
    import re
    size_match = re.search(r'(\d+\.?\d*)B', model_name)
    
    if size_match:
        size_b = float(size_match.group(1))
    else:
        # Default to 7B if size cannot be determined
        size_b = 7.0
    
    # Adjust thresholds based on model size
    if size_b <= 0.5:
        # 0.5B model
        if available_vram_gb >= 4:
            return "fp16"
        elif available_vram_gb >= 3:
            return "8bit"
        else:
            return "4bit"
    elif size_b <= 1.5:
        # 1.5B model
        if available_vram_gb >= 8:
            return "fp16"
        elif available_vram_gb >= 6:
            return "8bit"
        else:
            return "4bit"
    else:
        # 7B or larger model
        if available_vram_gb >= 22:
            return "fp16"
        elif available_vram_gb >= 14:
            return "8bit"
        else:
            return "4bit"


def print_vram_info(available_vram_gb: float, model_name: str, quantization: str = "fp16"):
    """
    Print VRAM information and quantization recommendation.
    
    Args:
        available_vram_gb: Available VRAM in GB
        model_name: Name of the model being loaded
        quantization: Current quantization setting
    """
    logger.info(f"Available VRAM: {available_vram_gb:.1f}GB")
    
    suggested = suggest_quantization(available_vram_gb, model_name)
    
    if suggested != quantization and quantization == "fp16":
        logger.warning(
            f"⚠️  Low VRAM detected ({available_vram_gb:.1f}GB). "
            f"Recommended: --quantization {suggested}"
        )
        logger.warning(
            f"   Example: python demo/realtime_model_inference_from_file.py "
            f"--model_path {model_name} --quantization {suggested} ..."
        )
