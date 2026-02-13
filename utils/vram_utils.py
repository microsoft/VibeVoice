"""VRAM detection and quantization recommendation utilities."""

import torch
import logging

logger = logging.getLogger(__name__)


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
        # Get total and allocated memory
        total = torch.cuda.get_device_properties(device).total_memory
        allocated = torch.cuda.memory_allocated(device)
        available = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
        return available
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
    # VibeVoice-7B memory requirements (approximate)
    # Full precision (fp16/bf16): ~20GB
    # 8-bit quantization: ~12GB
    # 4-bit quantization: ~7GB
    
    if "1.5B" in model_name:
        # 1.5B model is smaller, adjust thresholds
        if available_vram_gb >= 8:
            return "fp16"
        elif available_vram_gb >= 6:
            return "8bit"
        else:
            return "4bit"
    else:
        # Assume 7B model
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
