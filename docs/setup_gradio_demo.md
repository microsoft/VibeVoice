# VibeVoice Gradio Demo Setup Guide

End-to-end instructions to deploy the VibeVoice ASR server and launch the Gradio web demo.

## Prerequisites

- CUDA-capable GPU(s)
- Docker with GPU support (`nvidia-docker`)
- VibeVoice repository cloned locally

```bash
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice
```

---

## Step 1 — Start the ASR Server

Launch a Docker container running the vLLM ASR server. The launcher script handles everything automatically (system deps, pip install, model download, tokenizer generation, server start).

### Single GPU (default)

```bash
docker run -d --gpus '"device=0"' --name vibevoice-asr-demo \
  --ipc=host \
  -p 6001:6001 \
  -e VIBEVOICE_FFMPEG_MAX_CONCURRENCY=64 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -v $(pwd):/app \
  -w /app \
  --entrypoint bash \
  vllm/vllm-openai:v0.14.1 \
  -c "python3 /app/vllm_plugin/scripts/start_server.py --port 6001"
```

### Multi-GPU with Data Parallel (load balancing)

Run 4 independent replicas, one per GPU. vLLM distributes requests automatically:

```bash
docker run -d --gpus '"device=0,1,2,3"' --name vibevoice-asr-demo \
  --ipc=host \
  -p 6001:6001 \
  -e VIBEVOICE_FFMPEG_MAX_CONCURRENCY=64 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -v $(pwd):/app \
  -w /app \
  --entrypoint bash \
  vllm/vllm-openai:v0.14.1 \
  -c "python3 /app/vllm_plugin/scripts/start_server.py --port 6001 --dp 4"
```

> **Tip**: Use `--dp N` for N-way data parallel (throughput scaling). Use `--tp N` for tensor parallel (large models). See `docs/vibevoice-vllm-asr.md` for details.

### Check Logs

```bash
docker logs -f vibevoice-asr-demo
```

Wait until you see `Application startup complete.` — this means the server is ready.

---

## Step 2 — Verify the Server

```bash
# Check the model is loaded
curl http://localhost:6001/v1/models
```

Expected output:
```json
{
  "data": [{ "id": "vibevoice", ... }]
}
```

### Quick Test with Audio File

```bash
docker exec -it vibevoice-asr-demo \
  python3 /app/vllm_plugin/tests/test_api.py /app/en-Alice_woman.wav \
  --url http://localhost:6001
```

---

## Step 3 — Launch the Gradio Demo

### Install tmux inside the container (to keep it running)

```bash
docker exec vibevoice-asr-demo apt-get install -y tmux
```

### Start Gradio in tmux

```bash
docker exec vibevoice-asr-demo bash -c \
  "PYTHONUNBUFFERED=1 tmux new-session -d -s gradio \
  'PYTHONUNBUFFERED=1 python3 /app/vllm_plugin/scripts/gradio_asr_demo_api_video.py \
  --api_url http://localhost:6001 --share \
  2>&1 | tee /tmp/gradio.log'"
```

### Get the Share Link

Wait ~20 seconds, then:

```bash
docker exec vibevoice-asr-demo cat /tmp/gradio.log
```

You should see:
```
✅ Connected to API: http://localhost:6001 | Model: vibevoice
🚀 Starting VibeVoice ASR Demo
* Running on local URL:  http://0.0.0.0:7860
* Running on public URL: https://xxxxxx.gradio.live
```

The `gradio.live` link is publicly accessible (valid for 1 week).

### Gradio Options

| Flag | Description | Default |
|------|-------------|---------|
| `--api_url URL` | vLLM server URL | `http://localhost:8000` |
| `--share` | Create a public Gradio link | off |
| `--port PORT` | Local Gradio port | `7860` |
| `--cloudflared` | Use Cloudflare tunnel instead of Gradio share | off |
| `--max_video_size MB` | Max upload video size | `50` |

---

## Managing the Service

### Stop Gradio (keep ASR server running)

```bash
docker exec vibevoice-asr-demo tmux kill-session -t gradio
```

### Restart Gradio

Re-run the tmux command from Step 3.

### Stop Everything

```bash
docker stop vibevoice-asr-demo
docker rm vibevoice-asr-demo
```

---

## Example: Full Setup on GPU 0 with Port 6001

```bash
# 1. Start server
docker run -d --gpus '"device=0"' --name vibevoice-asr-demo \
  --ipc=host -p 6001:6001 \
  -e VIBEVOICE_FFMPEG_MAX_CONCURRENCY=64 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -v $(pwd):/app -w /app \
  --entrypoint bash \
  vllm/vllm-openai:v0.14.1 \
  -c "python3 /app/vllm_plugin/scripts/start_server.py --port 6001"

# 2. Wait for startup (~2 min), then verify
docker logs -f vibevoice-asr-demo  # wait for "Application startup complete."
curl http://localhost:6001/v1/models

# 3. Install tmux and launch Gradio
docker exec vibevoice-asr-demo apt-get install -y tmux
docker exec vibevoice-asr-demo bash -c \
  "PYTHONUNBUFFERED=1 tmux new-session -d -s gradio \
  'PYTHONUNBUFFERED=1 python3 /app/vllm_plugin/scripts/gradio_asr_demo_api_video.py \
  --api_url http://localhost:6001 --share \
  2>&1 | tee /tmp/gradio.log'"

# 4. Get the public link
sleep 20 && docker exec vibevoice-asr-demo cat /tmp/gradio.log
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `CUDA out of memory` | Use a different GPU (`device=X`) or reduce `--gpu-memory-utilization 0.7` in `start_server.py` |
| Gradio log is empty | Wait longer (~30s); Gradio buffers output. Use `PYTHONUNBUFFERED=1` as shown above |
| `Port already in use` | Pick a different port or stop the existing container: `docker stop <name> && docker rm <name>` |
| Share link shows "No interface" | Gradio is still loading. Wait for `Application startup complete` in the log |
| `tmux: command not found` | Run `docker exec <container> apt-get install -y tmux` first |
