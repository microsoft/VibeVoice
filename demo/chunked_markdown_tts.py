#!/usr/bin/env python3
"""
Demo script for processing markdown documents with chunked TTS.
"""

import argparse
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vibevoice.processor import VibeVoiceProcessor
from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration
from vibevoice.processor.chunking import ChunkedTTSProcessor


def resolve_device(device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate TTS audio from markdown documents using heading-based chunking."
    )
    parser.add_argument("--markdown", "-m", required=True, help="Path to markdown (.md) file")
    parser.add_argument("--voice", "-v", required=True, help="Path to voice sample audio file")
    parser.add_argument("--output", "-o", default="output.wav", help="Output audio file path")
    parser.add_argument("--depth", "-d", type=int, default=1, help="Heading depth to chunk on")
    parser.add_argument("--model", default="microsoft/VibeVoice-1.5B", help="Model name or path")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"], help="Device")
    parser.add_argument("--pause-ms", type=int, default=500, help="Pause duration between chunks")
    parser.add_argument("--include-heading", action="store_true", help="Speak headings in each chunk")
    parser.add_argument("--no-strip", action="store_true", help="Keep markdown formatting in content")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Max tokens to generate")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=None, help="Nucleus sampling top-p")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")

    args = parser.parse_args()

    md_path = Path(args.markdown)
    if not md_path.exists():
        print(f"Markdown file not found: {args.markdown}")
        sys.exit(1)

    voice_path = Path(args.voice)
    if not voice_path.exists():
        print(f"Voice sample not found: {args.voice}")
        sys.exit(1)

    device = resolve_device(args.device)

    if not args.quiet:
        print(f"Loading model: {args.model}")
        print(f"Device: {device}")

    processor = VibeVoiceProcessor.from_pretrained(args.model)
    model = VibeVoiceForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype="auto" if device == "cuda" else None,
    )
    model.to(device)
    model.eval()

    generation_kwargs = {}
    if args.max_new_tokens is not None:
        generation_kwargs["max_new_tokens"] = args.max_new_tokens
    if args.do_sample:
        generation_kwargs["do_sample"] = True
    if args.temperature is not None:
        generation_kwargs["temperature"] = args.temperature
    if args.top_p is not None:
        generation_kwargs["top_p"] = args.top_p

    chunked_processor = ChunkedTTSProcessor(
        processor=processor,
        model=model,
        chunk_depth=args.depth,
        strip_markdown=not args.no_strip,
        include_heading=args.include_heading,
    )

    try:
        chunked_processor.process_markdown_file(
            markdown_file=str(md_path),
            voice_sample=str(voice_path),
            output_file=args.output,
            device=device,
            pause_duration_ms=args.pause_ms,
            verbose=not args.quiet,
            generation_kwargs=generation_kwargs or None,
        )
    except Exception as exc:
        print(f"Error during processing: {exc}")
        raise


if __name__ == "__main__":
    main()
