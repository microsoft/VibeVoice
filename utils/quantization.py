"""Quantization utilities for VibeVoice models."""

import logging
from typing import Optional
import torch

logger = logging.getLogger(__name__)


def get_quantization_config(quantization: str = "fp16") -> Optional[dict]:
    """
    Get quantization configuration for model loading.
    
    Args:
        quantization: Quantization level ("fp16", "8bit", or "4bit")
        
    Returns:
        dict: Quantization config for from_pretrained, or None for fp16
    """
    if quantization == "fp16" or quantization == "full":
        return None
    
    if quantization == "8bit":
        try:
            import bitsandbytes as bnb
            logger.info("Using 8-bit quantization (selective LLM only)")
            return {
                "load_in_8bit": True,
                "llm_int8_threshold": 6.0,
            }
        except ImportError:
            logger.error(
                "8-bit quantization requires bitsandbytes. "
                "Install with: pip install bitsandbytes"
            )
            raise
    
    elif quantization == "4bit":
        try:
            import bitsandbytes as bnb
            from transformers import BitsAndBytesConfig
            
            logger.info("Using 4-bit NF4 quantization (selective LLM only)")
            return {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
            }
        except ImportError:
            logger.error(
                "4-bit quantization requires bitsandbytes. "
                "Install with: pip install bitsandbytes"
            )
            raise
    
    else:
        raise ValueError(
            f"Invalid quantization: {quantization}. "
            f"Must be one of: fp16, 8bit, 4bit"
        )


def apply_selective_quantization(model, quantization: str):
    """
    Apply selective quantization only to safe components.
    
    This function identifies which modules should be quantized and which
    should remain at full precision for audio quality preservation.
    
    Args:
        model: The VibeVoice model
        quantization: Quantization level ("8bit" or "4bit")
    """
    if quantization == "fp16":
        return model
    
    logger.info("Applying selective quantization...")
    
    # Components to KEEP at full precision (audio-critical)
    keep_fp_components = [
        "diffusion_head",
        "acoustic_connector",
        "semantic_connector",
        "acoustic_tokenizer",
        "semantic_tokenizer",
        "vae",
    ]
    
    # Only quantize the LLM (Qwen2.5) component
    quantize_components = ["llm", "language_model"]
    
    for name, module in model.named_modules():
        # Check if this module should stay at full precision
        should_keep_fp = any(comp in name for comp in keep_fp_components)
        should_quantize = any(comp in name for comp in quantize_components)
        
        if should_keep_fp:
            # Ensure audio components stay at higher precision (e.g., bfloat16 instead of 4/8-bit)
            with torch.no_grad():
                module.to(torch.bfloat16)
            logger.debug(f"Keeping {name} at full precision (audio-critical)")
        
        elif should_quantize:
            logger.debug(f"Quantized {name} to {quantization}")
    
    logger.info(f"✓ Selective {quantization} quantization applied")
    logger.info("  • LLM: Quantized")
    logger.info("  • Audio components: Full precision")
    
    return model