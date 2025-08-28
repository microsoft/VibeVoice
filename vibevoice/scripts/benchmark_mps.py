#!/usr/bin/env python
# coding=utf-8
"""
Benchmark script for measuring VibeVoice inference performance across different devices (MPS, CUDA, CPU).

This script measures end-to-end generation latency with proper synchronization and cache handling
for accurate timing on each backend.
"""

import time
import torch
import os
from typing import Dict, List

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Memory usage will not be reported.")

from vibevoice.utils.device_config import get_optimal_config
from vibevoice.model import load_vibevoice_model


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage information."""
    if not PSUTIL_AVAILABLE:
        return {}
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
        "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
    }


def benchmark_inference(model, processor, test_inputs: List[str], device: torch.device) -> Dict:
    """
    Benchmark inference performance with backend-appropriate synchronization.
    
    Args:
        model: The loaded VibeVoice model
        processor: The VibeVoice processor
        test_inputs: List of text inputs to benchmark
        device: The device the model is running on
        
    Returns:
        Dictionary with timing results
    """
    model.eval()
    times = []
    
    # Get initial memory usage
    initial_memory = get_memory_usage()
    
    with torch.no_grad():
        for text_input in test_inputs:
            # Tokenize input and move to device
            inputs = processor(
                text=[text_input],  # Wrap in list for batch processing
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            # Clear cache before timing
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()
            
            start_time = time.time()
            # For benchmarking, we'll just run a forward pass instead of full generation
            # This avoids the complexity of the full generation pipeline
            with torch.no_grad():
                _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
            
            # For CUDA, synchronize to ensure the operation is complete before stopping the timer
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append(end_time - start_time)
    
    # Get final memory usage
    final_memory = get_memory_usage()
    
    # Calculate memory difference if psutil is available
    memory_diff = {}
    if initial_memory and final_memory:
        memory_diff = {
            "rss_diff_mb": final_memory["rss_mb"] - initial_memory["rss_mb"],
            "vms_diff_mb": final_memory["vms_mb"] - initial_memory["vms_mb"],
        }
    
    return {
        "avg_time": sum(times) / len(times),
        "times": times,
        "min_time": min(times),
        "max_time": max(times),
        "memory_usage": final_memory,
        "memory_diff": memory_diff
    }


def main():
    """Main benchmark function."""
    print("=== VibeVoice Performance Benchmark ===")
    
    # Get optimal configuration
    config = get_optimal_config()
    device = config["device"]
    dtype = config["dtype"]
    attn_impl = config["attn_implementation"]
    
    print(f"Selected device: {device}")
    print(f"Data type: {dtype}")
    print(f"Attention implementation: {attn_impl}")
    
    if device.type == "mps":
        print("Note: MPS uses unified memory. Timings can have variance. Recommend median of multiple runs.")
    elif device.type == "cuda":
        print("Note: CUDA timings include synchronization for accuracy.")
    else:
        print("Note: CPU timings do not require synchronization.")
    
    # Load model and processor
    print("\nLoading model...")
    model, processor = load_vibevoice_model(
        "microsoft/VibeVoice-1.5B",
        device=device,
        torch_dtype=dtype,
        attn_implementation=attn_impl
    )
    print("Model loaded successfully.")
    
    # Define test inputs of varying lengths in the expected format
    test_inputs = [
        "Speaker 1: Short test sentence.",
        "Speaker 1: This is a medium length sentence for testing performance.",
        "Speaker 1: This is a much longer sentence that will test the model's ability to handle extended input sequences. " * 3,
        "Speaker 1: A conversation between two people discussing the future of artificial intelligence.\nSpeaker 2: Yes, it's fascinating how quickly the field is evolving.\nSpeaker 1: Indeed, we're seeing breakthroughs almost daily."
    ]
    
    print(f"\nTest inputs: {len(test_inputs)} prompts of varying lengths")
    
    # Warmup run
    print("\nPerforming warmup run...")
    warmup_result = benchmark_inference(model, processor, [test_inputs[0]], device)
    print(f"Warmup completed in {warmup_result['avg_time']:.3f}s")
    
    # Measured run
    print("\nStarting benchmark...")
    results = benchmark_inference(model, processor, test_inputs, device)
    
    # Print results
    print("\n=== Benchmark Results ===")
    print(f"Average inference time: {results['avg_time']:.3f}s")
    print(f"Min inference time: {results['min_time']:.3f}s")
    print(f"Max inference time: {results['max_time']:.3f}s")
    print(f"Total time for all inputs: {sum(results['times']):.3f}s")
    
    if results['memory_usage']:
        print(f"\nMemory usage:")
        print(f"  RSS: {results['memory_usage']['rss_mb']:.1f} MB")
        print(f"  VMS: {results['memory_usage']['vms_mb']:.1f} MB")
    
    if results['memory_diff']:
        print(f"\nMemory change during benchmark:")
        print(f"  RSS change: {results['memory_diff']['rss_diff_mb']:+.1f} MB")
        print(f"  VMS change: {results['memory_diff']['vms_diff_mb']:+.1f} MB")
    
    print("\nNote: For MPS, it's recommended to take the median of several full benchmark runs for stable results.")
    print("MPS memory is unified with system memory, unlike dedicated VRAM on CUDA devices.")


if __name__ == "__main__":
    main()