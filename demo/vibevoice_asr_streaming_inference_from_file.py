#!/usr/bin/env python
"""
VibeVoice ASR Streaming Inference Demo Script

This script transcribes audio files chunk by chunk, printing each chunk as
soon as the model emits it instead of waiting for the whole file.
"""

import argparse
import json
import os
import time

import torch

from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.audio_utils import load_audio_use_ffmpeg
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor


def load_frame_config(model_path: str) -> dict:
    """Read the chunk and lookahead the checkpoint was trained on.

    Takes a local checkpoint directory or a HF repo id, the same as the
    ``from_pretrained`` calls below.
    """
    name = "preprocessor_config.json"
    if os.path.isdir(model_path):
        config_path = os.path.join(model_path, name)
    else:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(model_path, name)

    with open(config_path) as f:
        cfg = json.load(f)

    missing = [k for k in ("chunk_frames", "lookahead_frames") if k not in cfg]
    if missing:
        raise SystemExit(
            f"{config_path} has no {', '.join(missing)}, so {model_path} is not "
            "a streaming checkpoint. Transcribe it with "
            "demo/vibevoice_asr_inference_from_file.py instead."
        )

    sample_rate = cfg["target_sample_rate"]
    frame_seconds = cfg["speech_tok_compress_ratio"] / sample_rate
    return {
        "sample_rate": sample_rate,
        "chunk_duration": cfg["chunk_frames"] * frame_seconds,
        "text_audio_delay": cfg["lookahead_frames"] * frame_seconds,
    }


def main():
    parser = argparse.ArgumentParser(description="VibeVoice ASR Streaming Inference Demo")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the streaming model checkpoint"
    )
    parser.add_argument(
        "--audio_files",
        type=str,
        nargs='+',
        required=True,
        help="Paths to audio files for transcription"
    )
    parser.add_argument(
        "--context_info",
        type=str,
        default=None,
        help="Hotwords to bias recognition, e.g. 'Microsoft,VibeVoice'"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu", "mps", "xpu"],
        help="Device to run inference on"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate per chunk"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for sampling (0 = greedy decoding)"
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Attention implementation to use"
    )

    args = parser.parse_args()

    frame_config = load_frame_config(args.model_path)
    sample_rate = frame_config["sample_rate"]

    print(f"Loading VibeVoice ASR streaming model from {args.model_path}")
    processor = VibeVoiceASRProcessor.from_pretrained(args.model_path)
    if processor.tokenizer.text_chunk_end_id is None:
        raise SystemExit(
            f"{args.model_path} is not a streaming checkpoint: its tokenizer has no "
            "<|text_chunk_end|>, so no chunk would ever end."
        )

    model_dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=model_dtype,
        attn_implementation=args.attn_implementation,
    ).to(args.device).eval()

    print(f"Model loaded successfully on {args.device}")
    print(f"Chunk: {frame_config['chunk_duration']:.4f}s, "
          f"lookahead: {frame_config['text_audio_delay']:.4f}s")

    for audio_file in args.audio_files:
        print("\n" + "=" * 80)
        print(f"File: {audio_file}")
        print("=" * 80)

        audio, _ = load_audio_use_ffmpeg(audio_file, resample=True, target_sr=sample_rate)
        duration = len(audio) / sample_rate

        start_time = time.time()
        chunks = []
        for chunk_index, total_chunks, chunk_text in model.streaming_generate(
            audio_tensor=torch.from_numpy(audio),
            tokenizer=processor.tokenizer,
            chunk_duration=frame_config["chunk_duration"],
            text_audio_delay=frame_config["text_audio_delay"],
            sample_rate=sample_rate,
            max_new_tokens_per_chunk=args.max_new_tokens,
            temperature=args.temperature,
            context_info=args.context_info,
        ):
            chunks.append(chunk_text)
            print(f"[{chunk_index + 1}/{total_chunks}] {chunk_text}", flush=True)

        elapsed = time.time() - start_time
        print("\n--- Transcription ---")
        print("".join(chunks))
        print(f"\nAudio: {duration:.2f}s, generation time: {elapsed:.2f}s, "
              f"RTF: {elapsed / duration:.3f}")


if __name__ == "__main__":
    main()
