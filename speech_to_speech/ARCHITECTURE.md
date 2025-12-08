# VibeVoice Speech-to-Speech Real-Time Streaming Architecture

## Overview
End-to-end Speech-to-Speech (S2S) system with **<800ms latency** target for real-time conversational AI.

## Latency Budget Breakdown

| Component | Target Latency | Description |
|-----------|---------------|-------------|
| **VAD** | <5ms | Silero VAD voice activity detection |
| **ASR** | <200ms | faster-whisper with small.en model |
| **LLM** | <300ms | Qwen2.5-1.5B-Instruct (short responses) |
| **TTS** | <300ms | VibeVoice-Realtime-0.5B first chunk |
| **Total** | **<800ms** | End-to-end first audio response |

## Component Selection & Licensing

### All components are fully commercial-free:

| Component | Model | License | Notes |
|-----------|-------|---------|-------|
| VAD | Silero VAD | MIT | <1ms per chunk, CPU-based |
| ASR | faster-whisper (small.en) | MIT | CTranslate2 backend, int8 quantization |
| LLM | Qwen2.5-1.5B-Instruct | Apache 2.0 | Fast inference, multilingual |
| TTS | VibeVoice-Realtime-0.5B | MIT | Streaming audio, ~300ms first chunk |

### Rejected Options:
- **OmniASR**: CC-BY-NC 4.0 (non-commercial) - NOT suitable
- **Whisper-large-v3**: Too slow for <200ms ASR target
- **Llama-3.2**: Requires Meta license agreement

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client (Browser)                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │ Mic Capture │───>│ Audio Chunk │───>│ WebSocket Send      │  │
│  │ (16kHz PCM) │    │ (200-400ms) │    │ (Binary Audio)      │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Audio Playback (24kHz PCM16) <── WebSocket Receive (Binary) ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ WebSocket (wss://)
┌─────────────────────────────────────────────────────────────────┐
│                     RunPod GPU Server                            │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                 S2S Pipeline Server                       │   │
│  │                                                           │   │
│  │  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐  │   │
│  │  │  VAD    │──>│   ASR   │──>│   LLM   │──>│   TTS   │  │   │
│  │  │ Silero  │   │ Whisper │   │ Qwen2.5 │   │VibeVoice│  │   │
│  │  │ (<5ms)  │   │(<200ms) │   │(<300ms) │   │(<300ms) │  │   │
│  │  └─────────┘   └─────────┘   └─────────┘   └─────────┘  │   │
│  │       │              │             │             │       │   │
│  │       ▼              ▼             ▼             ▼       │   │
│  │  [Voice Det]   [Transcript]   [Response]   [Audio Chunks]│   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Ports: 8000 (TTS WS), 8005 (Pipeline), 3000 (Frontend)         │
└─────────────────────────────────────────────────────────────────┘
```

## Streaming Flow

1. **Client Audio Capture**: 16kHz PCM, 200-400ms chunks
2. **VAD Processing**: Detect speech activity, buffer speech segments
3. **ASR Streaming**: Transcribe speech chunks incrementally
4. **LLM Generation**: Stream tokens as they're generated
5. **TTS Streaming**: Convert text to audio chunks in real-time
6. **Client Playback**: Play audio chunks as they arrive

## RunPod Deployment Configuration

### Recommended GPU
- **Primary**: NVIDIA L4 (24GB VRAM) - Best cost/performance
- **Alternative**: RTX A4000 (16GB) or RTX A5000 (24GB)
- **High-end**: A100 40GB for maximum throughput

### Disk Configuration
- **Container Disk**: 40GB (OS + Python + deps + app code)
- **Volume Disk**: 100GB (model weights cached)

### Ports Configuration
- **8000**: TTS WebSocket Server (VibeVoice streaming)
- **8005**: S2S Pipeline API Server
- **3000**: Web Frontend Server
- **Avoid**: 8001, 8888 (RunPod reserved)

### Container Template
- **Base**: `nvcr.io/nvidia/pytorch:24.07-py3`
- **CUDA**: 12.x
- **Python**: 3.10+

### Environment Variables
```bash
HF_HOME=/workspace/models
TRANSFORMERS_CACHE=/workspace/models/transformers
MODEL_PATH=microsoft/VibeVoice-Realtime-0.5B
ASR_MODEL=small.en
LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct
CUDA_VISIBLE_DEVICES=0
```

## Key Optimizations

1. **VAD Gating**: Only process audio when speech detected
2. **Streaming ASR**: Incremental transcription with short chunks
3. **LLM Streaming**: Token-by-token generation with early TTS trigger
4. **TTS Prefill**: Voice embeddings cached for instant generation
5. **Async Pipeline**: All components run asynchronously
6. **GPU Memory**: Models loaded in bfloat16/int8 for efficiency
