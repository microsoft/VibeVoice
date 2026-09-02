#!/usr/bin/env python3
"""
VibeVoice vLLM Streaming ASR Server Launcher

One-click deployment script that handles:
1. Installing system dependencies (FFmpeg, etc.)
2. Installing VibeVoice Python package
3. Downloading model from HuggingFace
4. Refusing a checkpoint that cannot be streamed
5. Generating tokenizer files
6. Starting the streaming ASR server

Steps 1-3 are shared with start_server.py; the rest differ. A streaming
checkpoint transcribes one chunk at a time off a growing sequence, so a request
is a session rather than a single generate call and stock `vllm serve` has
nowhere to keep it. This launches vllm_plugin.asr_streaming_server instead,
which holds the same engine in-process and adds a WebSocket for live audio.

Usage:
    python3 start_streaming_server.py [--model MODEL_ID] [--port PORT]
"""

import argparse
import json
import os
import sys

# Same directory, run as a script rather than a package -- make the sibling
# importable so the four bootstrap steps stay in one place instead of drifting.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# And vllm_plugin/ itself, for a bare import of asr_streaming: the package
# __init__ imports vLLM, while the checks below exist to run before the engine
# does -- including on a box where vLLM is not importable yet.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asr_streaming import ChunkGeometry, TEXT_CHUNK_END_ID  # noqa: E402
from start_server import (download_model, install_system_deps,  # noqa: E402
                          install_vibevoice, run_command)


def resolve_model(model_id: str) -> str:
    """A local checkpoint directory is served as-is; anything else is downloaded."""
    if os.path.isdir(model_id):
        print(f"\n{'='*60}")
        print(f"  Using local model")
        print(f"  📁 Path: {model_id}")
        print(f"{'='*60}\n")
        return model_id
    return download_model(model_id)


def generate_streaming_tokenizer(model_path: str) -> None:
    """Generate tokenizer files, unless the checkpoint already ships them.

    A streaming checkpoint holds <|text_chunk_end|> at 151665 and its audio
    tokens one id higher. Writing the non-streaming layout over that would take
    away the token every chunk ends on, so the layout is asked for explicitly
    and a checkpoint that already has it is left alone.
    """
    added_tokens = os.path.join(model_path, "added_tokens.json")
    if os.path.isfile(added_tokens):
        with open(added_tokens) as f:
            if json.load(f).get("<|text_chunk_end|>") == TEXT_CHUNK_END_ID:
                print("\nTokenizer files already have <|text_chunk_end|>, skipping\n")
                return
    run_command(
        [sys.executable, "-m", "vllm_plugin.tools.generate_tokenizer_files",
         "--output", model_path, "--streaming"],
        "Generating tokenizer files"
    )


def check_streaming_checkpoint(model_path: str) -> None:
    """Refuse a checkpoint that cannot be streamed, before the engine starts.

    Serving a non-streaming checkpoint here does not raise on its own: the
    weights load, the first chunk transcribes, and every later chunk comes back
    empty because nothing ever emits <|text_chunk_end|>.
    """
    geometry = ChunkGeometry.from_pretrained(model_path)
    print(f"\n{'='*60}")
    print(f"  ✅ Streaming checkpoint")
    print(f"  📁 Path: {model_path}")
    print(f"  🎚️  {geometry.describe()}")
    print(f"{'='*60}\n")


def start_streaming_server(model_path: str, port: int,
                           tensor_parallel_size: int = 1,
                           max_model_len: int = 16384,
                           max_audio_windows: int = 512,
                           mm_processor_cache_gb: float = 16.0,
                           gpu_memory_utilization: float = 0.85) -> None:
    """Start the streaming ASR server (replaces current process)."""
    print(f"\n{'='*60}")
    print(f"  Starting streaming ASR server on port {port}")
    print(f"  Tensor Parallel (TP): {tensor_parallel_size}")
    print(f"  Max Model Len:        {max_model_len}")
    print(f"  Max Audio Windows:    {max_audio_windows}")
    print(f"  MM Cache (GB):        {mm_processor_cache_gb}")
    print(f"  GPU Mem Utilization:  {gpu_memory_utilization}")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, "-m", "vllm_plugin.asr_streaming_server",
        "--model", model_path,
        "--served-model-name", "vibevoice",
        "--port", str(port),
        "--dtype", "bfloat16",
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--max-model-len", str(max_model_len),
        "--max-audio-windows", str(max_audio_windows),
        "--mm-processor-cache-gb", str(mm_processor_cache_gb),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
    ]
    os.execvp(sys.executable, cmd)


def main():
    parser = argparse.ArgumentParser(
        description="VibeVoice vLLM Streaming ASR Server - One-Click Deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start with default settings (single GPU)
    python3 start_streaming_server.py

    # Use custom port
    python3 start_streaming_server.py --port 8080

    # Tensor parallel: split model across 2 GPUs
    python3 start_streaming_server.py --tp 2

    # Longer sessions (the context is what caps how long a session can run)
    python3 start_streaming_server.py --max-model-len 32768

    # Skip dependency installation (if already installed)
    python3 start_streaming_server.py --skip-deps

The chunk and lookahead are read from the checkpoint's preprocessor_config.json,
so a checkpoint always runs at the geometry it was trained on.
        """
    )
    parser.add_argument(
        "--model", "-m",
        default="microsoft/VibeVoice-ASR-Streaming",
        help="HuggingFace model ID or local path (default: microsoft/VibeVoice-ASR-Streaming)"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Server port (default: 8000)"
    )
    parser.add_argument(
        "--skip-deps",
        action="store_true",
        help="Skip installing system dependencies"
    )
    parser.add_argument(
        "--skip-tokenizer",
        action="store_true",
        help="Skip generating tokenizer files"
    )
    parser.add_argument(
        "--tp", "--tensor-parallel-size",
        type=int,
        default=1,
        dest="tensor_parallel_size",
        help="Tensor parallel size: split one model across N GPUs (default: 1)"
    )
    parser.add_argument(
        "--dp", "--data-parallel-size",
        type=int,
        default=1,
        dest="data_parallel_size",
        help="Not supported for streaming: a session's KV cache lives on one "
             "replica, and round-robin would send its later chunks to replicas "
             "that never saw its audio (default: 1)"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=16384,
        dest="max_model_len",
        help="Context per session, which is what caps session length (default: 16384)"
    )
    parser.add_argument(
        "--max-audio-windows",
        type=int,
        default=512,
        dest="max_audio_windows",
        help="Audio windows per session, ~17 min at a 2s chunk (default: 512)"
    )
    parser.add_argument(
        "--mm-processor-cache-gb",
        type=float,
        default=16.0,
        dest="mm_processor_cache_gb",
        help="Multimodal cache; must hold every in-flight session's windows "
             "(default: 16)"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        dest="gpu_memory_utilization",
        help="GPU memory utilization fraction (default: 0.85)"
    )
    args = parser.parse_args()

    if args.data_parallel_size > 1:
        parser.error(
            "--dp is not supported for streaming. Each session keeps its audio "
            "windows in one replica's prefix cache, and an nginx round-robin "
            "would send its later chunks to a replica that never saw them. Run "
            "one server per GPU on separate ports and route whole sessions, or "
            "use --tp to split the model across GPUs."
        )

    print("\n" + "="*60)
    print("  VibeVoice vLLM Streaming ASR Server - One-Click Deployment")
    print("="*60)

    # Step 1: Install system dependencies
    if not args.skip_deps:
        install_system_deps()

    # Step 2: Install VibeVoice
    install_vibevoice()

    # Step 3: Download model
    model_path = resolve_model(args.model)

    # Step 4: Refuse a checkpoint that cannot be streamed -- before step 5,
    # which rewrites the tokenizer files in place and would leave a
    # non-streaming checkpoint unusable by start_server.py too.
    check_streaming_checkpoint(model_path)

    # Step 5: Generate tokenizer files
    if not args.skip_tokenizer:
        generate_streaming_tokenizer(model_path)

    # Step 6: Start server
    start_streaming_server(
        model_path, args.port,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_audio_windows=args.max_audio_windows,
        mm_processor_cache_gb=args.mm_processor_cache_gb,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )


if __name__ == "__main__":
    main()
