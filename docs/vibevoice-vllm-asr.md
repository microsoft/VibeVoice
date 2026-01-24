# VibeVoice vLLM ASR Deployment

<a href="https://huggingface.co/microsoft/VibeVoice-ASR"><img alt="Huggingface" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-VibeVoice--ASR-blue"></a>

Deploy VibeVoice ASR model as a high-performance API service using [vLLM](https://github.com/vllm-project/vllm). This plugin provides OpenAI-compatible API endpoints for speech-to-text transcription with streaming support.

## 🔥 Key Features

- **🚀 High-Performance Serving**: Optimized for high-throughput ASR inference with vLLM's continuous batching
- **📡 OpenAI-Compatible API**: Standard `/v1/chat/completions` endpoint with streaming support
- **🎵 Long Audio Support**: Process up to 60+ minutes of audio in a single request
- **🔌 Plugin Architecture**: No vLLM source code modification required - just install and run

## 🛠️ Installation

Using Official vLLM Docker Image (Recommended)

```bash
# 1. Pull the official vLLM image
docker pull vllm/vllm-openai:latest

# 2. Start an interactive container
docker run -it --gpus all --name vibevoice-vllm \
  --ipc=host \
  -p 8000:8000 \
  -e VIBEVOICE_FFMPEG_MAX_CONCURRENCY=64 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -v /path/to/models:/models \
  -v /path/to/VibeVoice:/app \
  -w /app \
  --entrypoint bash \
  vllm/vllm-openai:latest

# 3. Inside container: Install system dependencies
bash vllm_plugin/scripts/install_deps.sh

# 4. Inside container: Install VibeVoice with vLLM support
pip install -e .[vllm]

# 5. Inside container: (Optional) Generate tokenizer files if needed
python3 -m vllm_plugin.tools.generate_tokenizer_files --output /models/your_model

# 6. Inside container: Start vLLM server
vllm serve /models/your_model \
  --served-model-name vibevoice \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-num-seqs 64 \
  --max-model-len 65536 \
  --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.8 \
  --enforce-eager \
  --no-enable-prefix-caching \
  --enable-chunked-prefill \
  --chat-template-content-format openai \
  --tensor-parallel-size 1 \
  --allowed-local-media-path /app \
  --port 8000
```

> **Note**: This approach allows you to switch models, adjust parameters, and debug issues without rebuilding the container.


## 🚀 Quick Start

### Test the API

Once the vLLM server is running, test it with the provided script:

```bash
# Run the test script (inside container)
python3 vllm_plugin/tests/test_api.py /path/to/audio.wav
```


### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VIBEVOICE_FFMPEG_MAX_CONCURRENCY` | Maximum FFmpeg processes for audio decoding | `64` |
| `PYTORCH_CUDA_ALLOC_CONF` | CUDA memory allocator config | `expandable_segments:True` |



## 📊 Performance Tips

1. **GPU Memory**: Use `--gpu-memory-utilization 0.9` for maximum throughput if you have dedicated GPU
2. **Batch Size**: Increase `--max-num-seqs` for higher concurrency (requires more GPU memory)
3. **FFmpeg Concurrency**: Tune `VIBEVOICE_FFMPEG_MAX_CONCURRENCY` based on CPU cores

## 🚨 Troubleshooting

### Common Issues

1. **"CUDA out of memory"**
   - Reduce `--gpu-memory-utilization`
   - Reduce `--max-num-seqs`
   - Use smaller `--max-model-len`

2. **"Audio decoding failed"**
   - Ensure FFmpeg is installed: `ffmpeg -version`
   - Check audio file format is supported 

3. **"Model not found"**
   - Ensure model path contains `config.json` and model weights
   - Generate tokenizer files if missing

4. **"Plugin not loaded"**
   - Verify installation: `pip show vibevoice`
   - Check entry point: `pip show -f vibevoice | grep entry`


