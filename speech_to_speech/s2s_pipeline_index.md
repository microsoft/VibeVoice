# S2S Pipeline Code Index

## Overview
Speech-to-Speech Pipeline with WebSocket Streaming - Unified pipeline integrating Silero VAD, faster-whisper ASR, Qwen2.5-1.5B LLM, and VibeVoice TTS with <800ms end-to-end latency.

## File Structure Index

### 1. Module Documentation & Imports
- **Docstring**: Pipeline description and component overview
- **Imports**: Standard library, async, ML libraries, FastAPI, WebSocket support

### 2. System Prompts (Lines ~50-200)
- **medical**: VEMI AI Medical Assistant prompt
- **automobile**: VEMI AI Automobile Assistant prompt  
- **general**: VEMI AI general assistant prompt
- **viva**: Medical Viva Examiner prompt
- **aviation**: Aviation Viva Examiner prompt

### 3. Configuration Classes
- **PipelineConfig**: Main configuration dataclass
- **PipelineMetrics**: Performance metrics tracking

### 4. Utility Functions
- **clean_llm_response()**: Text cleaning for voice output
- **user_is_greeting()**: Greeting detection
- **PipelineState**: Enum for processing states

### 5. Core Pipeline Classes
- **S2SPipeline**: Main speech-to-speech pipeline
- **TTSClient**: VibeVoice TTS service client

### 6. FastAPI Application
- **create_app()**: Application factory
- **WebSocket endpoints**: Real-time streaming
- **HTTP endpoints**: Health, status, configuration

### 7. Main Entry Point
- **main()**: Command-line interface and server startup

## Detailed Component Index

### System Prompts
Each agent has specialized prompts optimized for:
- Natural conversation flow
- Domain-specific knowledge
- Voice interaction patterns
- Context awareness

### Pipeline Configuration
Configurable parameters for:
- Model selection (ASR, LLM, TTS)
- Device settings (CUDA/CPU)
- Audio processing (sample rates, VAD thresholds)
- Performance tuning (latency, quality trade-offs)

### Core Processing Pipeline
1. **VAD**: Voice Activity Detection
2. **ASR**: Speech-to-Text transcription
3. **LLM**: Response generation with streaming
4. **TTS**: Text-to-Speech synthesis

### WebSocket Streaming
Real-time bidirectional communication:
- Audio input/output streaming
- Status updates and metrics
- Control messages (reset, cancel, agent switching)
- Barge-in cancellation support

### Performance Features
- Sub-800ms latency target
- Sentence-level streaming
- Echo suppression
- Barge-in cancellation
- Metrics tracking

## Key Features

### Multi-Agent Support
- Medical assistant
- Automobile technician
- General AI assistant
- Medical viva examiner
- Aviation viva examiner

### Real-time Capabilities
- WebSocket streaming
- Barge-in interruption
- Echo cancellation
- Voice switching

### Quality Assurance
- Response cleaning
- Greeting detection
- Help request handling
- Disconnect detection

## Dependencies
- FastAPI + WebSocket
- PyTorch + VibeVoice models
- faster-whisper ASR
- Silero VAD
- NumPy for audio processing
- AsyncIO for streaming

## Usage
```bash
python s2s_pipeline.py --host 0.0.0.0 --port 8005
```

## API Endpoints
- `GET /`: Agent selection page
- `GET /chat`: Chat interface
- `GET /health`: Health check
- `GET /status`: Pipeline status
- `GET /config`: Configuration
- `GET /voices`: Available TTS voices
- `WS /stream`: WebSocket streaming endpoint
