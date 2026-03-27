#!/usr/bin/env python3
"""
VibeVoice vLLM ASR Server Launcher

One-click deployment script that handles:
1. Installing system dependencies (FFmpeg, etc.)
2. Installing VibeVoice Python package
3. Downloading model from HuggingFace
4. Generating tokenizer files
5. Starting vLLM server

For DP > 1, launches N independent vLLM processes behind an nginx
reverse proxy for optimal throughput (avoids single-process HTTP
bottleneck of vLLM's built-in DP coordinator).

Usage:
    python3 start_server.py [--model MODEL_ID] [--port PORT]
"""

import argparse
import os
import signal
import subprocess
import sys
import textwrap
import time


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


def _build_vllm_cmd(model_path: str, port: int,
                     tensor_parallel_size: int = 1,
                     data_parallel_size: int = 1,
                     max_num_seqs: int = 64,
                     max_model_len: int = 65536,
                     gpu_memory_utilization: float = 0.8) -> list[str]:
    """Build the vllm serve command."""
    return [
        "vllm", "serve", model_path,
        "--served-model-name", "vibevoice",
        "--trust-remote-code",
        "--dtype", "bfloat16",
        "--max-num-seqs", str(max_num_seqs),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--no-enable-prefix-caching",
        "--enable-chunked-prefill",
        "--chat-template-content-format", "openai",
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--data-parallel-size", str(data_parallel_size),
        "--allowed-local-media-path", "/app",
        "--port", str(port),
    ]


def start_vllm_server(model_path: str, port: int,
                      tensor_parallel_size: int = 1,
                      data_parallel_size: int = 1,
                      max_num_seqs: int = 64,
                      max_model_len: int = 65536,
                      gpu_memory_utilization: float = 0.8) -> None:
    """Start a single vLLM server (replaces current process)."""
    print(f"\n{'='*60}")
    print(f"  Starting vLLM server on port {port}")
    print(f"  Tensor Parallel (TP): {tensor_parallel_size}")
    print(f"  Data Parallel   (DP): {data_parallel_size}")
    print(f"  Max Num Seqs:         {max_num_seqs}")
    print(f"  Max Model Len:        {max_model_len}")
    print(f"  GPU Mem Utilization:  {gpu_memory_utilization}")
    print(f"{'='*60}\n")
    
    vllm_cmd = _build_vllm_cmd(
        model_path, port,
        tensor_parallel_size=tensor_parallel_size,
        data_parallel_size=data_parallel_size,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    os.execvp("vllm", vllm_cmd)


def _install_nginx() -> None:
    """Install nginx if not already available."""
    if subprocess.run(["which", "nginx"], capture_output=True).returncode != 0:
        run_command(["apt-get", "update"], "Updating package list for nginx")
        run_command(
            ["apt-get", "install", "-y", "nginx"],
            "Installing nginx for load balancing"
        )


def _write_nginx_config(frontend_port: int, backend_ports: list[int],
                        num_workers: int = 0) -> str:
    """Write nginx config for round-robin load balancing.
    
    Args:
        num_workers: Number of nginx worker processes. 0 = auto (2 × num backends).
    """
    if num_workers <= 0:
        num_workers = len(backend_ports) * 2
    backends = "\n".join(f"        server 127.0.0.1:{p};" for p in backend_ports)
    config = textwrap.dedent(f"""\
        worker_processes {num_workers};
        worker_rlimit_nofile 65536;
        error_log /dev/stderr warn;
        pid /tmp/nginx.pid;

        events {{
            worker_connections 8192;
        }}

        http {{
            access_log off;

            upstream vllm_backends {{
                least_conn;
        {backends}
            }}

            server {{
                listen {frontend_port};
                client_max_body_size 200m;
                client_body_buffer_size 10m;
                proxy_buffering on;
                proxy_buffer_size 64k;
                proxy_buffers 16 64k;

                location / {{
                    proxy_pass http://vllm_backends;
                    proxy_read_timeout 600s;
                    proxy_connect_timeout 10s;
                    proxy_send_timeout 600s;
                    proxy_http_version 1.1;
                    proxy_set_header Connection "";
                }}
            }}
        }}
    """)
    config_path = "/tmp/nginx_vllm.conf"
    with open(config_path, "w") as f:
        f.write(config)
    return config_path


def start_dp_server(model_path: str, frontend_port: int,
                    data_parallel_size: int,
                    tensor_parallel_size: int = 1,
                    max_num_seqs: int = 64,
                    max_model_len: int = 65536,
                    gpu_memory_utilization: float = 0.8) -> None:
    """Start multiple vLLM workers behind nginx for data parallelism.
    
    Launches N independent vLLM processes (one per GPU group) on internal
    ports, with an nginx reverse proxy on the frontend port for load
    balancing. This avoids the single-process HTTP bottleneck of vLLM's
    built-in DP coordinator when handling large audio payloads.
    """
    import torch
    num_gpus = torch.cuda.device_count()
    gpus_per_replica = tensor_parallel_size
    total_gpus_needed = data_parallel_size * gpus_per_replica
    assert num_gpus >= total_gpus_needed, (
        f"Need {total_gpus_needed} GPUs (dp={data_parallel_size} × tp={tensor_parallel_size}) "
        f"but only {num_gpus} available"
    )

    # Auto-tune per-worker env vars based on dp size
    ffmpeg_concurrency = max(
        64, int(os.environ.get("VIBEVOICE_FFMPEG_MAX_CONCURRENCY", "64"))
    )
    media_threads = max(
        8, int(os.environ.get("VLLM_MEDIA_LOADING_THREAD_COUNT", "8"))
    )

    _install_nginx()

    # Assign internal ports: frontend_port + 100, +101, ...
    backend_ports = [frontend_port + 100 + i for i in range(data_parallel_size)]

    print(f"\n{'='*60}")
    print(f"  Starting DP server with nginx load balancing")
    print(f"  Frontend port:     {frontend_port} (nginx)")
    print(f"  Backend ports:     {backend_ports}")
    print(f"  Data Parallel:     {data_parallel_size}")
    print(f"  Tensor Parallel:   {tensor_parallel_size}")
    print(f"  GPUs per replica:  {gpus_per_replica}")
    print(f"  Max Num Seqs:      {max_num_seqs}")
    print(f"  Max Model Len:     {max_model_len}")
    print(f"  FFmpeg concurrency (per worker): {ffmpeg_concurrency}")
    print(f"  Media loading threads (per worker): {media_threads}")
    print(f"{'='*60}\n")

    # Write nginx config
    nginx_conf = _write_nginx_config(frontend_port, backend_ports)

    # Launch vLLM workers
    workers: list[subprocess.Popen] = []
    for rank in range(data_parallel_size):
        gpu_start = rank * gpus_per_replica
        gpu_ids = ",".join(str(gpu_start + j) for j in range(gpus_per_replica))
        port = backend_ports[rank]
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids
        env["VIBEVOICE_FFMPEG_MAX_CONCURRENCY"] = str(ffmpeg_concurrency)
        env["VLLM_MEDIA_LOADING_THREAD_COUNT"] = str(media_threads)

        vllm_cmd = _build_vllm_cmd(
            model_path, port,
            tensor_parallel_size=tensor_parallel_size,
            data_parallel_size=1,  # Each worker is dp=1
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )

        print(f"  Launching worker rank={rank} on GPU(s) {gpu_ids}, port {port}")
        proc = subprocess.Popen(vllm_cmd, env=env)
        workers.append(proc)

    # Start nginx
    print(f"\n  Starting nginx on port {frontend_port} ...")
    nginx_proc = subprocess.Popen(
        ["nginx", "-c", nginx_conf, "-g", "daemon off;"]
    )

    # Wait for all backends to be ready
    print("  Waiting for all backends to be ready ...")
    import urllib.request
    for port in backend_ports:
        url = f"http://127.0.0.1:{port}/v1/models"
        for attempt in range(600):  # up to 10 minutes
            try:
                urllib.request.urlopen(url, timeout=2)
                print(f"    ✅ Backend on port {port} is ready")
                break
            except Exception:
                time.sleep(1)
        else:
            print(f"    ❌ Backend on port {port} failed to start")

    print(f"\n{'='*60}")
    print(f"  ✅ VibeVoice DP server ready on port {frontend_port}")
    print(f"     {data_parallel_size} replicas behind nginx load balancer")
    print(f"{'='*60}\n")

    # Handle shutdown: forward signals to all children
    def _shutdown(signum, frame):
        print("\nShutting down ...")
        nginx_proc.terminate()
        for w in workers:
            w.terminate()
        for w in workers:
            w.wait(timeout=10)
        nginx_proc.wait(timeout=5)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    # Wait for any child to exit (indicates a failure)
    while True:
        for i, w in enumerate(workers):
            ret = w.poll()
            if ret is not None:
                print(f"  ❌ Worker {i} exited with code {ret}")
                _shutdown(None, None)
        if nginx_proc.poll() is not None:
            print(f"  ❌ nginx exited with code {nginx_proc.returncode}")
            _shutdown(None, None)
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(
        description="VibeVoice vLLM ASR Server - One-Click Deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start with default settings (single GPU)
    python3 start_server.py

    # Use custom port
    python3 start_server.py --port 8080

    # Data parallel: 4 replicas on 4 GPUs (nginx load balancing)
    python3 start_server.py --dp 4

    # Tensor parallel: split model across 2 GPUs
    python3 start_server.py --tp 2

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
        help="Data parallel size: run N independent model replicas for load balancing (default: 1)"
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=64,
        dest="max_num_seqs",
        help="Maximum number of sequences per batch (default: 64)"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=65536,
        dest="max_model_len",
        help="Maximum model context length (default: 65536)"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        dest="gpu_memory_utilization",
        help="GPU memory utilization fraction (default: 0.8)"
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

    # Step 5: Start server
    if args.data_parallel_size > 1:
        start_dp_server(
            model_path, args.port,
            data_parallel_size=args.data_parallel_size,
            tensor_parallel_size=args.tensor_parallel_size,
            max_num_seqs=args.max_num_seqs,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    else:
        start_vllm_server(
            model_path, args.port,
            tensor_parallel_size=args.tensor_parallel_size,
            data_parallel_size=1,
            max_num_seqs=args.max_num_seqs,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )


if __name__ == "__main__":
    main()
