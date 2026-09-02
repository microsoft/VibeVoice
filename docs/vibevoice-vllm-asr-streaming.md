# VibeVoice vLLM Streaming ASR Deployment

<a href="https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-7B"><img alt="Huggingface" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-VibeVoice--ASR--Streaming-blue"></a>

Deploy VibeVoice ASR Streaming model as a high-performance API service using [vLLM](https://github.com/vllm-project/vllm). A session pushes audio while it is still being spoken and gets one transcript back per chunk, so text appears before the audio ends.

**Non-streaming:** [vLLM-asr](./vibevoice-vllm-asr.md)<br>

## 🔥 Key Features

- **🎤 True Streaming**: Transcribe from a live microphone or a growing stream, one chunk at a time
- **🚀 Prefix-Cached Sessions**: Each chunk reuses the KV cache of every chunk before it, so cost stays flat as a session grows
- **📡 WebSocket + REST**: `/v1/stream` for live audio, `/v1/transcribe` and `/v1/chat/completions` for whole files
- **🔌 Plugin Architecture**: No vLLM source code modification required - just install and run
- **🎚️ Checkpoint-Defined Chunking**: Chunk and lookahead are read from the checkpoint, so a model always runs at the geometry it was trained on

## 🛠️ Installation

Using Official vLLM Docker Image (Recommended)

1. Clone the repository
```bash
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice
```

2. Launch the server (background mode)
```bash
docker run -d --gpus all --name vibevoice-vllm-streaming \
  --ipc=host \
  -p 8000:8000 \
  -e VIBEVOICE_FFMPEG_MAX_CONCURRENCY=64 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -v $(pwd):/app \
  -w /app \
  --entrypoint bash \
  vllm/vllm-openai:v0.14.1 \
  -c "python3 /app/vllm_plugin/scripts/start_streaming_server.py"
```

## ⚡ Multi-GPU Deployment

### Tensor Parallel

Split a single model across 2 GPUs (useful if GPU memory is limited):

```bash
docker run -d --gpus '"device=0,1"' --name vibevoice-vllm-streaming \
  --ipc=host \
  -p 8000:8000 \
  -e VIBEVOICE_FFMPEG_MAX_CONCURRENCY=64 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -v $(pwd):/app \
  -w /app \
  --entrypoint bash \
  vllm/vllm-openai:v0.14.1 \
  -c "python3 /app/vllm_plugin/scripts/start_streaming_server.py --tp 2"
```

### Data Parallel

`--dp` is not available here. A session's audio lives in one replica's prefix cache, and round-robin would send its later chunks to a replica that never saw the earlier ones. To use N GPUs for throughput, run one server per GPU on its own port and route **whole sessions**, not individual chunks:

```bash
for i in 0 1 2 3; do
  docker run -d --gpus "\"device=$i\"" --name vibevoice-vllm-streaming-$i \
    --ipc=host \
    -p $((8000 + i)):8000 \
    -e PYTORCH_ALLOC_CONF=expandable_segments:True \
    -v $(pwd):/app \
    -w /app \
    --entrypoint bash \
    vllm/vllm-openai:v0.14.1 \
    -c "python3 /app/vllm_plugin/scripts/start_streaming_server.py"
done
```

3. View logs
```bash
docker logs -f vibevoice-vllm-streaming
```

> **Note**:
> - The `-d` flag runs the container in background (detached mode)
> - Use `docker stop vibevoice-vllm-streaming` to stop the service
> - The model will be downloaded to HuggingFace cache (`~/.cache/huggingface`) inside the container

## 🚀 Usages

### Usage 1: Test the API

Once the streaming server is running, test it with the provided script:

```bash
# Basic streaming transcription
docker exec -it vibevoice-vllm-streaming python3 vllm_plugin/tests/test_api_streaming.py /app/audio.wav

# With hotwords for better recognition of specific terms
docker exec -it vibevoice-vllm-streaming python3 vllm_plugin/tests/test_api_streaming.py /app/audio.wav --hotwords "Microsoft,VibeVoice"
```

Whole-file endpoints are available too, for callers that do not stream:

```bash
docker exec -it vibevoice-vllm-streaming python3 vllm_plugin/tests/test_api.py /app/audio.wav
```

> **Note**:
> - The audio/video file must be inside the mounted directory (`/app` in the container). Copy your files to the VibeVoice folder before testing.
> - Hotwords help improve recognition of domain-specific terms like proper nouns, technical terms, and speaker names.

### Usage 2: Launch the demo page

Start the demo in a second container on the host network, so that `--api_url` reaches the server already running there:

```bash
docker run -d --network host --name vibevoice-vllm-streaming-demo \
  -v $(pwd):/app \
  -w /app \
  --entrypoint bash \
  vllm/vllm-openai:v0.14.1 \
  -c "python3 /app/vllm_plugin/scripts/fastapi_asr_streaming_demo_api.py --api_url http://localhost:8000"
```

Open `http://localhost:7860`, then record from the microphone or pick a file. The page keeps a WebSocket open for the whole recording, so text appears while you are still speaking.

Nothing extra to install: the demo needs only `fastapi`, `uvicorn`, `httpx` and `websockets`, all of which the vLLM image already ships.

### Usage 3: Share the demo publicly

Add `--cloudflared` to get a public link. The flag downloads the `cloudflared` binary to `~/.local/bin` on first use, so the container needs outbound network:

```bash
docker run -d --network host --name vibevoice-vllm-streaming-demo \
  -v $(pwd):/app \
  -w /app \
  --entrypoint bash \
  vllm/vllm-openai:v0.14.1 \
  -c "python3 /app/vllm_plugin/scripts/fastapi_asr_streaming_demo_api.py --api_url http://localhost:8000 --cloudflared"
```

The link is printed once the tunnel is up:

```bash
docker logs vibevoice-vllm-streaming-demo
```

```
relaying to http://localhost:8000, open http://localhost:7860

  public URL: https://xxxx-xxxx-xxxx.trycloudflare.com
```

> **Note**: use the `https` link rather than a plain `http` address when the page is not on `localhost` — browsers only grant microphone access in a secure context.

### Demo Options

| Flag | Description | Default |
|------|-------------|---------|
| `--api_url URL` | Streaming server URL | `http://localhost:8000` |
| `--port PORT` | Local demo port | `7860` |
| `--host HOST` | Address to bind | `0.0.0.0` |
| `--cloudflared` | Create a public link using cloudflared tunnel | off |
| `--max_new_tokens N` | Default max new tokens per chunk | `256` |

### Endpoints

| Endpoint | Description |
|----------|-------------|
| `WS /v1/stream` | Live session: JSON config, then float32 PCM frames, then `"end"` |
| `POST /v1/transcribe` | One whole file, chunked internally, full transcript in the reply |
| `POST /v1/transcribe_batch` | Several files in one request |
| `POST /v1/chat/completions` | OpenAI-compatible, for existing clients |
| `GET /v1/config` | Sample rate and chunk geometry of the loaded checkpoint |

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VIBEVOICE_FFMPEG_MAX_CONCURRENCY` | Cap on simultaneous FFmpeg decodes | unset — uncapped; the `docker run` commands above set `64` |
| `PYTORCH_ALLOC_CONF` | PyTorch memory allocator config | unset; the `docker run` commands above set `expandable_segments:True` |

Unlike `start_server.py`, the streaming launcher imposes no FFmpeg limit of its own, so an unset `VIBEVOICE_FFMPEG_MAX_CONCURRENCY` really does mean unlimited.

## 📊 Performance Tips

1. **GPU Memory**: Use `--gpu-memory-utilization 0.9` for maximum throughput if you have dedicated GPU
2. **Session Length**: `--max-model-len` is what caps how long one session can run; raise it for hour-long streams
3. **Concurrent Sessions**: Raise `--mm-processor-cache-gb` before raising concurrency — it must hold every in-flight session's audio windows

## 🚨 Troubleshooting

### Common Issues

1. **"CUDA out of memory"**
   - Reduce `--gpu-memory-utilization`
   - Reduce `--mm-processor-cache-gb`
   - Use smaller `--max-model-len`

2. **"is not a streaming checkpoint"**
   - The checkpoint has no `chunk_frames` in `preprocessor_config.json`
   - Serve it with `start_server.py` instead

3. **"tokenizer layout mismatch"**
   - `<|text_chunk_end|>` is missing or sits at another id, so nothing would ever end a chunk. The server refuses to start rather than serve a first chunk and then empty ones
   - Regenerate: `python3 -m vllm_plugin.tools.generate_tokenizer_files --output MODEL_PATH --streaming`

4. **"Cache miss" or a session stalling after many chunks**
   - Raise `--mm-processor-cache-gb`, or lower the number of concurrent sessions

5. **"Plugin not loaded"**
   - Verify installation: `pip show vibevoice`
   - Check entry point: `pip show -f vibevoice | grep entry`
