# VibeVoice Python Inference Guide

Complete API reference for VibeVoice text-to-speech.

## Table of Contents

- [Quick Start](#quick-start)
- [API Reference](#api-reference)
  - [synthesize_speech()](#synthesize_speech)
  - [list_default_voices()](#list_default_voices)
  - [VibeVoiceStreamingTTS](#vibevoicestreamingtts)
  - [AudioPlayer](#audioplayer)

---

## Quick Start

```python
from vibevoice import synthesize_speech

# Simplest
synthesize_speech("Hello world!")

# With device
synthesize_speech(text="Hello world!", device="cuda")
```

---

## API Reference

### synthesize_speech()

One-line function for text-to-speech.

```python
synthesize_speech(
    text: str | Iterator[str],
    device: str = "cuda",
    output_file: str = None,
    voice_prompt_path: str = None,
    inference_steps: int = 5,
    cfg_scale: float = 1.5,
    **kwargs
)
```

**Key Parameters:**

- `text` - Text or iterator
- `device` - "cuda", "mps", or "cpu"
- `output_file` - Save path (optional)
- `inference_steps` - 5 (fast) to 50 (quality)
- `cfg_scale` - 1.0-2.0 (quality)

**Examples:**

```python
# Basic
synthesize_speech(text="Hello", device="cuda")

# Iterator (LLM streaming)
synthesize_speech(text=["Hello", "world"], device="cuda")

# Save file
synthesize_speech(text="Hello", device="cuda", output_file="out.wav")

# Custom voice
synthesize_speech(
    text="Hello",
    device="cuda",
    voice_prompt_path="voices/custom.pt"
)

# High quality
synthesize_speech(text="Hello", device="cuda", inference_steps=50, cfg_scale=2.0)
```

---

### list_default_voices()

List available voice presets.

```python
voices = list_default_voices()
# Returns: ['en-Carter_man', 'en-Davis_man', 'en-Emma_woman', ...]
```

---

### VibeVoiceStreamingTTS

High-level TTS class for advanced usage.

**Constructor:**

```python
tts = VibeVoiceStreamingTTS(
    model_path="microsoft/VibeVoice-Realtime-0.5B",
    device="cuda",
    voice_prompt_path=None,  # Auto-loads default
    inference_steps=5
)
```

**Parameters:**

- `model_path` - HuggingFace model ID
- `device` - "cuda", "mps", "cpu"
- `voice_prompt_path` - Voice file (optional, auto-loads if None)
- `inference_steps` - 5-50 (speed vs quality)

**Methods:**

#### `text_to_speech_streaming(text_iterator, cfg_scale=1.5)`

Generate speech from iterator.

```python
def text_gen():
    yield "Hello world"

audio = tts.text_to_speech_streaming(text_gen())
# Returns: Iterator[np.ndarray]
```

#### `save_audio(audio, output_path)`

Save audio to WAV file.

```python
import numpy as np

chunks = list(tts.text_to_speech_streaming(text_gen()))
audio = np.concatenate(chunks)
tts.save_audio(audio, "output.wav")
```

---

### AudioPlayer

Audio playback with speaker selection.

**Constructor:**

```python
player = AudioPlayer(device_id=None, sample_rate=24000)
```

**Methods:**

#### `list_devices()` [static]

```python
AudioPlayer.list_devices()
# Shows available speakers
```

#### `play_stream(audio_iterator, realtime=True)`

```python
player.play_stream(audio, realtime=True)  # Streaming
player.play_stream(audio, realtime=False)  # Buffered
```

---

## Quick Reference

| Function | Purpose |
|----------|---------|
| `synthesize_speech()` | One-line TTS |
| `list_default_voices()` | See available voices |
| `VibeVoiceStreamingTTS` | Advanced TTS class |
| `AudioPlayer` | Audio playback |

**Devices:**
- `"cuda"` - NVIDIA GPU (fastest)
- `"mps"` - Apple Silicon
- `"cpu"` - CPU (slower)

**Quality Settings:**
- Fast: `inference_steps=5`, `cfg_scale=1.5`
- Quality: `inference_steps=50`, `cfg_scale=2.0`

**Default Voices:**
- en-Mike_man, en-Emma_woman, en-Carter_man, en-Davis_man, en-Frank_man, en-Grace_woman, in-Samuel_man

---

## License

See [LICENSE](../LICENSE) for details.
