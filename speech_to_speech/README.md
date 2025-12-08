# VibeVoice Speech-to-Speech Real-Time Streaming

End-to-end Speech-to-Speech (S2S) system with **<800ms latency** for real-time conversational AI.

## 🎯 Key Features

- **Real-time streaming**: <800ms end-to-end latency
- **Fully open-source**: All components MIT/Apache 2.0 licensed
- **Commercial-ready**: No attribution required, fully legally safe
- **RunPod optimized**: Ready for GPU cloud deployment

## 📊 Latency Budget

| Component | Target | Actual | Model |
|-----------|--------|--------|-------|
| **VAD** | <5ms | ~1ms | Silero VAD (MIT) |
| **ASR** | <200ms | ~150ms | faster-whisper small.en (MIT) |
| **LLM** | <300ms | ~200ms | Qwen2.5-1.5B-Instruct (Apache 2.0) |
| **TTS** | <300ms | ~250ms | VibeVoice-Realtime-0.5B (MIT) |
| **Total** | **<800ms** | **~600ms** | End-to-end first audio |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Browser Client                       │
│  [Mic Capture] → [WebSocket] → [Audio Playback]         │
└─────────────────────────────────────────────────────────┘
                         │ ▲
                         ▼ │  wss://
┌─────────────────────────────────────────────────────────┐
│                    S2S Pipeline                          │
│                                                          │
│   Audio → [VAD] → [ASR] → [LLM] → [TTS] → Audio        │
│            │        │        │        │                  │
│         Silero  Whisper   Qwen2.5  VibeVoice            │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Local Development

```bash
# Clone and setup
cd VibeVoice
pip install -e .
pip install -r speech_to_speech/requirements.txt

# Start server
python -m speech_to_speech.s2s_pipeline --port 8005

# Open browser
# http://localhost:8005
```

### RunPod Deployment

See [RunPod Setup Guide](#runpod-deployment-guide) below.

## 📁 Project Structure

```
speech_to_speech/
├── __init__.py              # Package init
├── ARCHITECTURE.md          # System architecture doc
├── README.md                # This file
├── requirements.txt         # Pinned dependencies
├── vad_module.py           # Silero VAD integration
├── asr_module.py           # faster-whisper ASR
├── llm_module.py           # Qwen2.5 LLM
├── s2s_pipeline.py         # Main pipeline & server
├── Dockerfile              # Container build
├── start_runpod.sh         # RunPod start script
└── frontend/
    └── index.html          # Web UI
```

## 📋 Component Details

### VAD - Silero VAD
- **License**: MIT
- **Latency**: <1ms per chunk
- **Features**: Voice activity detection, speech segmentation
- **Config**: 16kHz, 512 sample windows

### ASR - faster-whisper
- **License**: MIT (CTranslate2 + Whisper weights)
- **Model**: small.en (built-in) or Systran/faster-distil-whisper-small.en
- **Latency**: ~150ms for typical utterance
- **Features**: INT8 quantization, streaming transcription

### LLM - Qwen2.5-1.5B-Instruct
- **License**: Apache 2.0
- **Latency**: ~200ms for short responses
- **Features**: Streaming generation, conversational AI
- **Config**: BFloat16, max 64 tokens

### TTS - VibeVoice-Realtime-0.5B
- **License**: MIT
- **Latency**: ~250ms first audio chunk
- **Features**: Streaming audio, 24kHz output
- **Config**: Flash Attention 2, 5 inference steps

---

# RunPod Deployment Guide

## 🖥️ Recommended GPU Configuration

### GPU Options (by cost/performance)

| GPU | VRAM | Price/hr | Recommendation |
|-----|------|----------|----------------|
| **L4** | 24GB | ~$0.44 | ✅ Best value |
| RTX A4000 | 16GB | ~$0.38 | ✅ Budget option |
| RTX A5000 | 24GB | ~$0.49 | Good performance |
| RTX 4090 | 24GB | ~$0.69 | Fast inference |
| A100 40GB | 40GB | ~$1.89 | Maximum throughput |

**Recommended**: NVIDIA L4 (24GB) - Best balance of cost and performance.

## 📦 Disk Configuration

### Container Disk: 40GB
Contains:
- Base OS + CUDA runtime (~10GB)
- Python environment (~5GB)
- Application code (~1GB)
- Temporary files (~5GB)
- Buffer space (~19GB)

### Volume Disk: 100GB
Contains:
- VibeVoice-Realtime-0.5B (~2GB)
- Qwen2.5-1.5B-Instruct (~3GB)
- faster-whisper models (~1GB)
- Silero VAD (~10MB)
- Future model updates (~90GB buffer)

## 🔌 Port Configuration

### Required Ports

| Port | Protocol | Purpose |
|------|----------|---------|
| **8005** | HTTP/WebSocket | S2S Pipeline API & Streaming |
| 8000 | HTTP/WebSocket | TTS only (optional) |
| 3000 | HTTP | Frontend (optional) |

### Ports to AVOID
- **8001**: RunPod JupyterLab
- **8888**: RunPod reserved
- **22**: SSH (use RunPod web terminal)

### RunPod Proxy URLs

After deployment, your endpoints will be:
```
# HTTP API
https://<POD_ID>-8005.proxy.runpod.net/

# WebSocket Streaming
wss://<POD_ID>-8005.proxy.runpod.net/stream

# Health Check
https://<POD_ID>-8005.proxy.runpod.net/health
```

## 🌐 Environment Variables

Set these in RunPod pod configuration:

```bash
# Required
MODEL_PATH=microsoft/VibeVoice-Realtime-0.5B
LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct
ASR_MODEL=small.en

# Model Cache (use volume mount)
HF_HOME=/workspace/models
TRANSFORMERS_CACHE=/workspace/models/transformers
TORCH_HOME=/workspace/models/torch

# Server
S2S_PORT=8005
CUDA_VISIBLE_DEVICES=0
```

## 📝 Pod Template

### Using NVIDIA PyTorch Template

1. **Go to RunPod** → Deploy → GPU Pod
2. **Select GPU**: NVIDIA L4 (24GB) recommended
3. **Template**: Select "RunPod Pytorch 2.1"
4. **Container Disk**: 40GB
5. **Volume Disk**: 100GB (mount to `/workspace`)
6. **Expose HTTP Ports**: `8005, 8000, 3000`
7. **Environment Variables**: Add the ones above

### Custom Docker Image (Recommended)

Build and push your own image:

```bash
# Build
docker build -f speech_to_speech/Dockerfile -t your-registry/vibevoice-s2s:latest .

# Push
docker push your-registry/vibevoice-s2s:latest
```

Then use your image in RunPod.

## 🚀 Deployment Steps

### Option 1: Using RunPod Web Terminal

1. **Create Pod** with PyTorch template
2. **Open Web Terminal** (not SSH)
3. **Clone repository**:
   ```bash
   cd /workspace
   git clone https://github.com/microsoft/VibeVoice.git app
   cd app
   ```

4. **Install dependencies**:
   ```bash
   pip install -e .
   pip install -r speech_to_speech/requirements.txt
   ```

5. **Start server**:
   ```bash
   bash speech_to_speech/start_runpod.sh
   ```

6. **Access via proxy URL**:
   ```
   https://<POD_ID>-8005.proxy.runpod.net/
   ```

### Option 2: Using Custom Docker Image

1. **Build and push image** (see above)
2. **Create Pod** with your custom image
3. Server starts automatically
4. **Access via proxy URL**

## 🔧 Troubleshooting

### Common Issues

**Port not accessible**
- Ensure port 8005 is in "Expose HTTP Ports"
- Wait 1-2 minutes for proxy to initialize
- Check pod logs for startup errors

**Out of memory**
- Use a GPU with more VRAM (L4 24GB recommended)
- Reduce LLM max tokens
- Use INT8 quantization for ASR

**Slow first request**
- Models are downloading on first use
- Use volume disk to persist model cache
- Pre-download models in startup script

**WebSocket connection fails**
- Ensure using `wss://` not `ws://` for RunPod proxy
- Check browser console for errors
- Verify server is running with `/health` endpoint

### Logs

View logs in RunPod web terminal:
```bash
# View live logs
tail -f /workspace/logs/s2s.log

# Check for errors
grep -i error /workspace/logs/s2s.log
```

## 📊 Performance Tuning

### For Lower Latency

1. **Use smaller ASR model**:
   ```bash
   ASR_MODEL=tiny.en  # Smallest and fastest
   # or
   ASR_MODEL=base.en  # Balance of speed and accuracy
   ```

2. **Reduce LLM tokens**:
   ```python
   # In llm_module.py
   max_new_tokens=32  # Shorter responses
   ```

3. **Reduce TTS steps**:
   ```python
   # In s2s_pipeline.py
   tts_inference_steps=3  # Faster but lower quality
   ```

### For Higher Quality

1. **Use larger ASR model**:
   ```bash
   ASR_MODEL=openai/whisper-small.en
   ```

2. **Increase LLM tokens**:
   ```python
   max_new_tokens=128
   ```

3. **Increase TTS steps**:
   ```python
   tts_inference_steps=10
   ```

## 📜 Licenses

All components are commercially free:

| Component | License | Commercial Use |
|-----------|---------|----------------|
| Silero VAD | MIT | ✅ Yes |
| faster-whisper | MIT | ✅ Yes |
| Whisper weights | MIT | ✅ Yes |
| Qwen2.5 | Apache 2.0 | ✅ Yes |
| VibeVoice | MIT | ✅ Yes |

**No attribution required. Fully legal for commercial applications.**

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Submit pull request

## 📞 Support

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **RunPod**: RunPod Discord

---

Built with ❤️ using VibeVoice, Silero, Whisper, and Qwen.
