#!/usr/bin/env python3
"""
VibeVoice vLLM ASR Server Launcher

One-click deployment script that handles:
1. Installing system dependencies (FFmpeg, etc.)
2. Installing VibeVoice Python package
3. Downloading model from HuggingFace
4. Generating tokenizer files
5. Starting vLLM server

Usage:
    python3 start_server.py [--model MODEL_ID] [--port PORT]
"""

import argparse
import os
import subprocess
import sys


def run_command(cmd: list[str], description: str, shell: bool = False) -> None:
    """Run a command with logging."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}\n")
    if shell:
        subprocess.run(cmd, shell=True, check=True)
    else:
        subprocess.run(cmd, check=True)


def install_system_deps() -> None:
    """Install system dependencies (FFmpeg, etc.)."""
    run_command(["apt-get", "update"], "Updating package list")
    run_command(
        ["apt-get", "install", "-y", "ffmpeg", "libsndfile1"],
        "Installing FFmpeg and audio libraries"
    )


def install_vibevoice() -> None:
    """Install VibeVoice Python package."""
    run_command(
        [sys.executable, "-m", "pip", "install", "-e", "/app[vllm]"],
        "Installing VibeVoice with vLLM support"
    )


def download_model(model_id: str) -> str:
    """Download model from HuggingFace using default cache."""
    print(f"\n{'='*60}")
    print(f"  Downloading model: {model_id}")
    print(f"{'='*60}\n")
    
    import warnings
    from huggingface_hub import snapshot_download
    
    # Suppress deprecation warnings from huggingface_hub
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_path = snapshot_download(model_id)
    
    print(f"\n{'='*60}")
    print(f"  ✅ Model downloaded successfully!")
    print(f"  📁 Path: {model_path}")
    print(f"{'='*60}\n")
    return model_path


def generate_tokenizer(model_path: str) -> None:
    """Generate tokenizer files for the model."""
    run_command(
        [sys.executable, "-m", "vllm_plugin.tools.generate_tokenizer_files", 
         "--output", model_path],
        "Generating tokenizer files"
    )


def start_vllm_server(model_path: str, port: int) -> None:
    """Start vLLM server (replaces current process)."""
    print(f"\n{'='*60}")
    print(f"  Starting vLLM server on port {port}")
    print(f"{'='*60}\n")
    
    vllm_cmd = [
        "vllm", "serve", model_path,
        "--served-model-name", "vibevoice",
        "--trust-remote-code",
        "--dtype", "bfloat16",
        "--max-num-seqs", "64",
        "--max-model-len", "65536",
        # "--max-num-batched-tokens", "32768",
        "--gpu-memory-utilization", "0.8",
        # "--enforce-eager",
        "--no-enable-prefix-caching",
        "--enable-chunked-prefill",
        "--chat-template-content-format", "openai",
        "--tensor-parallel-size", "1",
        "--allowed-local-media-path", "/app",
        "--port", str(port),
    ]
    
    os.execvp("vllm", vllm_cmd)


def main():
    parser = argparse.ArgumentParser(
        description="VibeVoice vLLM ASR Server - One-Click Deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start with default settings
    python3 start_server.py

    # Use custom port
    python3 start_server.py --port 8080

    # Skip dependency installation (if already installed)
    python3 start_server.py --skip-deps
        """
    )
    parser.add_argument(
        "--model", "-m",
        default="microsoft/VibeVoice-ASR",
        help="HuggingFace model ID (default: microsoft/VibeVoice-ASR)"
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
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  VibeVoice vLLM ASR Server - One-Click Deployment")
    print("="*60)

    # Step 1: Install system dependencies
    if not args.skip_deps:
        install_system_deps()

    # Step 2: Install VibeVoice
    install_vibevoice()

    # Step 3: Download model
    model_path = download_model(args.model)

    # Step 4: Generate tokenizer files
    if not args.skip_tokenizer:
        generate_tokenizer(model_path)

    # Step 5: Start vLLM server
    start_vllm_server(model_path, args.port)


if __name__ == "__main__":
    main()
