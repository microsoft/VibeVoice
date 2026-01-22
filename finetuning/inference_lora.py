#!/usr/bin/env python
"""
Inference with LoRA Fine-tuned VibeVoice ASR Model

This script loads a LoRA fine-tuned model and runs inference.

Usage:
    python inference_lora.py \
        --base_model microsoft/VibeVoice-ASR \
        --lora_path ./output \
        --audio_file ./toy_dataset/0.mp3
"""

import argparse
import torch

from peft import PeftModel

from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor


def load_lora_model(
    base_model_path: str,
    lora_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """
    Load base model and merge with LoRA weights.
    
    Args:
        base_model_path: Path to base pretrained model
        lora_path: Path to LoRA adapter weights
        device: Device to load model on
        dtype: Data type for model
        
    Returns:
        Tuple of (model, processor)
    """
    print(f"Loading base model from {base_model_path}")
    
    # Load processor
    processor = VibeVoiceASRProcessor.from_pretrained(
        base_model_path,
        language_model_pretrained_name="Qwen/Qwen2.5-7B"
    )
    
    # Load base model
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        base_model_path,
        dtype=dtype,
        device_map=device if device == "auto" else None,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    
    if device != "auto":
        model = model.to(device)
    
    # Load LoRA adapter
    print(f"Loading LoRA adapter from {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)
    
    # Optionally merge LoRA weights into base model for faster inference
    # model = model.merge_and_unload()
    
    model.eval()
    print("Model loaded successfully")
    
    return model, processor


def transcribe(
    model,
    processor,
    audio_path: str,
    max_new_tokens: int = 4096,
    temperature: float = 0.0,
    context_info: str = None,
    device: str = "cuda",
):
    """
    Transcribe an audio file using the LoRA fine-tuned model.
    
    Args:
        model: The LoRA fine-tuned model
        processor: The processor
        audio_path: Path to audio file
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 = greedy)
        context_info: Optional context info (e.g., hotwords)
        device: Device
        
    Returns:
        Transcription result
    """
    print(f"\nTranscribing: {audio_path}")
    
    # Process audio
    inputs = processor(
        audio=audio_path,
        sampling_rate=None,
        return_tensors="pt",
        padding=True,
        add_generation_prompt=True,
        context_info=context_info,
    )
    
    # Move to device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
              for k, v in inputs.items()}
    
    # Generation config
    gen_config = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": processor.pad_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        gen_config["temperature"] = temperature
        gen_config["top_p"] = 0.9
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_config)
    
    # Decode
    input_length = inputs['input_ids'].shape[1]
    generated_ids = output_ids[0, input_length:]
    generated_text = processor.decode(generated_ids, skip_special_tokens=True)
    
    # Parse structured output
    try:
        segments = processor.post_process_transcription(generated_text)
    except Exception as e:
        print(f"Warning: Failed to parse structured output: {e}")
        segments = []
    
    return {
        "raw_text": generated_text,
        "segments": segments,
    }


def main():
    parser = argparse.ArgumentParser(description="Inference with LoRA Fine-tuned VibeVoice ASR")
    
    parser.add_argument(
        "--base_model",
        type=str,
        default="microsoft/VibeVoice-ASR",
        help="Path to base pretrained model"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to LoRA adapter weights"
    )
    parser.add_argument(
        "--audio_file",
        type=str,
        required=True,
        help="Path to audio file to transcribe"
    )
    parser.add_argument(
        "--context_info",
        type=str,
        default=None,
        help="Optional context info (e.g., 'Hotwords: Tea Brew, Aiden Host')"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=4096,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    # Load model
    dtype = torch.bfloat16 if args.device != "cpu" else torch.float32
    model, processor = load_lora_model(
        base_model_path=args.base_model,
        lora_path=args.lora_path,
        device=args.device,
        dtype=dtype,
    )
    
    # Transcribe
    result = transcribe(
        model=model,
        processor=processor,
        audio_path=args.audio_file,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        context_info=args.context_info,
        device=args.device,
    )
    
    # Print results
    print("\n" + "="*60)
    print("Transcription Result")
    print("="*60)
    
    print("\n--- Raw Output ---")
    raw_text = result['raw_text']
    print(raw_text[:2000] + "..." if len(raw_text) > 2000 else raw_text)
    
    if result['segments']:
        print(f"\n--- Structured Output ({len(result['segments'])} segments) ---")
        for seg in result['segments'][:20]:
            print(f"[{seg.get('start_time', 'N/A')} - {seg.get('end_time', 'N/A')}] "
                  f"Speaker {seg.get('speaker_id', 'N/A')}: {seg.get('text', '')[:80]}...")
        if len(result['segments']) > 20:
            print(f"  ... and {len(result['segments']) - 20} more segments")


if __name__ == "__main__":
    main()
