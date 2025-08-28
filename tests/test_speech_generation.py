#!/usr/bin/env python
# coding=utf-8
"""
Simple test script to verify that speech generation works correctly after our fixes.
"""

import torch
import numpy as np
from vibevoice.model import load_vibevoice_model

def test_speech_generation():
    """Test that speech generation works with our fixes."""
    print("Loading model...")
    model, processor = load_vibevoice_model(
        "microsoft/VibeVoice-1.5B",
        device="cpu",
        torch_dtype=torch.float32,
        attn_implementation="eager"
    )
    print("Model loaded successfully.")
    
    # Create a test input with voice sample
    text_input = "Speaker 1: This is a test of the speech generation system."
    dummy_voice_sample = np.zeros(24000, dtype=np.float32)
    
    print("Processing input...")
    inputs = processor(
        text=[text_input],
        voice_samples=[[dummy_voice_sample]],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    
    print("Generating speech...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=16,
            tokenizer=processor.tokenizer,
            do_sample=False,
        )
    
    print("Speech generation completed successfully!")
    print(f"Output sequences shape: {outputs.sequences.shape}")
    print(f"Speech outputs: {outputs.speech_outputs is not None}")

if __name__ == "__main__":
    test_speech_generation()