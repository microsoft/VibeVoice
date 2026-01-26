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

1. Clone the repository
```bash
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice
```

2. Launch the server (background mode)
```bash
docker run -d --gpus all --name vibevoice-vllm \
  --ipc=host \
  -p 8000:8000 \
  -e VIBEVOICE_FFMPEG_MAX_CONCURRENCY=64 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -v $(pwd):/app \
  -w /app \
  --entrypoint bash \
  vllm/vllm-openai:latest \
  -c "python3 /app/vllm_plugin/scripts/start_server.py"
```

3. View logs
```bash
docker logs -f vibevoice-vllm
```

> **Note**: 
> - The `-d` flag runs the container in background (detached mode)
> - Use `docker stop vibevoice-vllm` to stop the service
> - The model will be downloaded to HuggingFace cache (`~/.cache/huggingface`) inside the container

## 🚀 Usages

### Test the API

Once the vLLM server is running, test it with the provided script:

```bash
# Run the test (use container path /app/...)
docker exec -it vibevoice-vllm python3 vllm_plugin/tests/test_api.py /app/audio.wav
```
> **Note**: The audio file must be inside the mounted directory (`/app` in the container). Copy your audio to the VibeVoice folder before testing.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VIBEVOICE_FFMPEG_MAX_CONCURRENCY` | Maximum FFmpeg processes for audio decoding | `64` |
| `PYTORCH_ALLOC_CONF` | PyTorch memory allocator config | `expandable_segments:True` |



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


