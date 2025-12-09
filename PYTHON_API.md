# VibeVoice Python API

Easy-to-use Python API for real-time text-to-speech with VibeVoice.

## Quick Start

### One-Line Synthesis (Easiest!)

```python
from vibevoice import synthesize_speech

# Simplest possible - automatically uses default voice!
synthesize_speech("Hello world!", device="cuda")
```

### Class-Based API

```python
from vibevoice import VibeVoiceStreamingTTS, AudioPlayer

# Initialize TTS (automatically loads default voice)
tts = VibeVoiceStreamingTTS(
    model_path="microsoft/VibeVoice-Realtime-0.5B",
    device="cuda"
)

# Initialize audio player
player = AudioPlayer()

# Generate text
def text_gen():
    for word in ["Hello", "world"]:
        yield word

# Generate and play
audio = tts.text_to_speech_streaming(text_gen())
player.play_stream(audio, realtime=True)
```

## Installation

```bash
# Install VibeVoice
pip install -e .

# Install audio playback support
pip install sounddevice
```

## Features

- ✅ **One-line synthesis** - `synthesize_speech("Hello!")`
- ✅ **Automatic voice loading** - 7 default voices included, no setup needed!
- ✅ **Real-time streaming** - ~100ms latency
- ✅ **Voice cloning** - Use voice prompts for speaker cloning
- ✅ **Speaker selection** - Choose output device
- ✅ **Easy-to-use API** - Simple high-level interface
- ✅ **GPU acceleration** - CUDA, Apple Silicon (MPS), CPU support
- ✅ **Iterator-based** - Works with LLM token streams

## Documentation

- **[Python Inference Guide](docs/python_inference.md)** - Complete API reference
- **[Examples](examples/)** - Code examples

## API Overview

### High-Level Functions

```python
from vibevoice import synthesize_speech, list_default_voices

# One-line synthesis (easiest!)
synthesize_speech("Hello world!", device="cuda")

# List available default voices
voices = list_default_voices()
# ['en-Mike_man', 'en-Emma_woman', 'en-Carter_man', ...]

# With iterator (LLM streaming)
def text_gen():
    for word in ["Hello", "world"]:
        yield word
synthesize_speech(text_gen(), device="cuda")
```

### Class-Based API

```python
from vibevoice import VibeVoiceStreamingTTS, AudioPlayer

# TTS with streaming (automatically loads default voice)
tts = VibeVoiceStreamingTTS(
    model_path="microsoft/VibeVoice-Realtime-0.5B",
    voice_prompt_path="path/to/voice.pt",  # Optional - uses default if None
    device="cuda",
    inference_steps=5
)

# Audio player with device selection
player = AudioPlayer(device_id=None)  # None = default device
player.play_stream(audio_iterator, realtime=True)
```

### Low-Level API

```python
from vibevoice import (
    VibeVoiceStreamingForConditionalGenerationInference,
    VibeVoiceStreamingProcessor,
    AudioStreamer
)

# Direct model access for advanced users
processor = VibeVoiceStreamingProcessor.from_pretrained(model_path)
model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
```

## Examples

### One-Line Synthesis

```python
from vibevoice import synthesize_speech

# Simplest possible
synthesize_speech("Hello world!", device="cuda")

# With iterator
def text_gen():
    for word in ["Hello", "world"]:
        yield word
synthesize_speech(text_gen(), device="cuda")
```

### Basic TTS with Classes

```python
from vibevoice import VibeVoiceStreamingTTS, AudioPlayer

tts = VibeVoiceStreamingTTS("microsoft/VibeVoice-Realtime-0.5B", device="cuda")
player = AudioPlayer()

def text_gen():
    yield "Hello world!"

audio = tts.text_to_speech_streaming(text_gen())
player.play_stream(audio, realtime=True)
```

### Save to File

```python
import numpy as np
from vibevoice import VibeVoiceStreamingTTS

tts = VibeVoiceStreamingTTS("microsoft/VibeVoice-Realtime-0.5B", device="cuda")

chunks = list(tts.text_to_speech_streaming(text_gen()))
full_audio = np.concatenate(chunks)
tts.save_audio(full_audio, "output.wav")
```

### Voice Cloning

```python
from vibevoice import VibeVoiceStreamingTTS

tts = VibeVoiceStreamingTTS(
    model_path="microsoft/VibeVoice-Realtime-0.5B",
    voice_prompt_path="voices/speaker.pt",  # Speaker embedding
    device="cuda"
)

audio = tts.text_to_speech_streaming(text_gen())
player.play_stream(audio, realtime=True)
```

### LLM Integration

```python
from vibevoice import VibeVoiceStreamingTTS, AudioPlayer

tts = VibeVoiceStreamingTTS("microsoft/VibeVoice-Realtime-0.5B", device="cuda")
player = AudioPlayer()

def llm_stream():
    """Your LLM generates tokens here"""
    for token in llm.generate():
        yield token

# Real-time TTS as LLM generates
audio = tts.text_to_speech_streaming(llm_stream())
player.play_stream(audio, realtime=True)
```

## Performance

- **First audio chunk**: ~100-300ms (CUDA)
- **Audio quality**: 24kHz sample rate
- **Devices**: CUDA (fastest), MPS (Apple), CPU (slower)
- **Inference steps**: 5 (fast) to 50 (high quality)

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- sounddevice (for audio playback)
- CUDA toolkit (optional, for GPU)

## License

See [LICENSE](LICENSE) for details.

## Citation

If you use VibeVoice in your research, please cite:

```bibtex
@article{vibevoice2025,
  title={VibeVoice: Real-time Streaming Text-to-Speech},
  author={VibeVoice Team},
  journal={Microsoft Research},
  year={2025}
}
```
