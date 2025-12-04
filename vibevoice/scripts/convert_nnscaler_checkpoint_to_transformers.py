#!/usr/bin/env python
# coding=utf-8

import argparse
import json
import os
from pathlib import Path
import re
import torch
from typing import Dict, List, Tuple

from vibevoice.modular.configuration_vibevoice import (
    VibeVoiceConfig
)
from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration
from transformers.utils import logging

logger = logging.get_logger(__name__)

def convert_vibevoice_nnscaler_checkpoint_to_hf(
    checkpoint_path: str,
    pytorch_dump_folder_path: str,
    config_path: str = None,
):
    """
    Convert a nnscaler VibeVoice checkpoint to HuggingFace format.
    Supports both regular checkpoints and tensor parallel checkpoints.
    """
    
    # Load regular checkpoint
    logger.info(f"Loading regular checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu") # ['model', 'optimizer', 'lr_scheduler', 'train_status', 'train_args', 'rng_states', 'nnscaler', 'dataloader']
    
    # config = checkpoint['train_args']
    init_config_name = checkpoint['train_args']['vars']['model_args']['config_path']['relative_path']
    pretrained_name = checkpoint['train_args']['vars']['data_args']['tokenizer_path']
    
    init_config_path = Path(__file__).parent.parent / 'configs' / init_config_name.split('/')[-1]
    if init_config_path.exists():
        logger.info(f"Loading initial config from {init_config_path}")
        with open(init_config_path, 'r') as f:
            init_config = json.load(f)
    else:
        raise FileNotFoundError(f"Initial config file {init_config_path} not found. Please provide a valid path.")

    tie_word_embeddings = init_config['decoder_config'].get('tie_word_embeddings', True)
    logger.info(f"Tie word embeddings: {tie_word_embeddings}")

    init_config['decoder_config']['use_cache'] = True
    config = VibeVoiceConfig(**init_config, tie_word_embeddings=tie_word_embeddings)

    # # Extract the model state dict
    model_state_dict = {k.replace('model.model.', 'model.'): v for k, v in checkpoint["model"].items() if k.startswith('model.model.')}
    if not tie_word_embeddings and 'model.lm_head.weight' in checkpoint["model"].keys():
        # If not tying weights, we need to add the lm_head weight separately
        model_state_dict['lm_head.weight'] = checkpoint["model"]['model.lm_head.weight']
    
    # Override with provided config if available
    if config_path:
        logger.info(f"Loading config from {config_path}")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = VibeVoiceConfig.from_dict(config_dict)
    
    # Set the default dtype to bfloat16 before creating the model
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)

    # Create the HuggingFace model
    logger.info("Creating HuggingFace VibeVoiceForConditionalGeneration model")
    model = VibeVoiceForConditionalGeneration(config)
    
    # Restore original dtype
    torch.set_default_dtype(original_dtype)

    # Load the state dict
    logger.info("Loading weights into model")
    missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
    
    if missing_keys:
        logger.warning(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys: {unexpected_keys}")
    
    # Create output directory
    os.makedirs(pytorch_dump_folder_path, exist_ok=True)
    
    # Save the model and config
    logger.info(f"Saving model to {pytorch_dump_folder_path}")
    
    # Save config
    config.save_pretrained(pytorch_dump_folder_path)
    
    # Save VibeVoiceProcessor configuration
    logger.info("Saving VibeVoiceProcessor configuration")
    processor_config = {
        "processor_class": "VibeVoiceProcessor",
        "speech_tok_compress_ratio": 3200,
        "db_normalize": True,
        # Audio processor configuration
        "audio_processor": {
            "feature_extractor_type": "VibeVoiceTokenizerProcessor",
            "sampling_rate": 24000,
            "normalize_audio": True,
            "target_dB_FS": -25,
            "eps": 1e-6,
        },
        "language_model_pretrained_name": pretrained_name,
    }
    
    processor_config_path = os.path.join(pytorch_dump_folder_path, "preprocessor_config.json")
    with open(processor_config_path, 'w') as f:
        json.dump(processor_config, f, indent=2)
    logger.info(f"Saved processor config to {processor_config_path}")
    
    # Save model with sharding
    # save_pretrained handles tied weights automatically
    logger.info("Saving model weights with sharding...")
    model.save_pretrained(
        pytorch_dump_folder_path,
        max_shard_size="2GB",  # Set maximum size for each shard
        safe_serialization=True  # Ensure saving in .safetensors format
    )
    logger.info(f"Model weights saved to {pytorch_dump_folder_path}")
    
    logger.info("Conversion complete!")
    
    # Verify the saved model can be loaded
    logger.info("Verifying saved model...")
    loaded_model = VibeVoiceForConditionalGeneration.from_pretrained(pytorch_dump_folder_path)
    logger.info("Model successfully loaded from saved checkpoint!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nnscaler_checkpoint_path",
        type=str,
        required=True,
        help="Path to the fairseq checkpoint (.pt file). For tensor parallel checkpoints, "
             "provide any one of the part files (e.g., checkpoint_1_5000-model_part-0.pt), "
             "and the script will automatically detect and merge all parts.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", 
        type=str,
        required=True,
        help="Path to the output PyTorch model directory",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Optional path to a config JSON file to override extracted config",
    )
    
    args = parser.parse_args()
    
    convert_vibevoice_nnscaler_checkpoint_to_hf(
        args.nnscaler_checkpoint_path,
        args.pytorch_dump_folder_path,
        args.config_path,
    )


if __name__ == "__main__":
    main()