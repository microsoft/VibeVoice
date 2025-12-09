# VibeVoice Python Inference Guide

Complete guide for using VibeVoice text-to-speech in Python with streaming support.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
  - [synthesize_speech()](#synthesize_speech) - High-level function (easiest!)
  - [list_default_voices()](#list_default_voices) - List available voices
  - [VibeVoiceStreamingTTS](#vibevoicestreamingtts)
  - [AudioPlayer](#audioplayer)
- [Advanced Usage](#advanced-usage)
- [Examples](#examples)

---

## Installation

```bash
# Install VibeVoice
pip install -e /path/to/VibeVoice

# Install audio playback support (optional, required for AudioPlayer)
pip install sounddevice
```

---

## Quick Start

### One-Line Synthesis (Easiest!)

```python
from vibevoice import synthesize_speech

# Simplest possible - automatically uses default voice!
synthesize_speech("Hello world!", device="cuda")
```

### Basic Text-to-Speech

```python
from vibevoice import VibeVoiceStreamingTTS, AudioPlayer

# Initialize TTS (automatically loads default voice)
tts = VibeVoiceStreamingTTS(
    model_path="microsoft/VibeVoice-Realtime-0.5B",
    device="cuda"  # or "cpu" or "mps"
)

# Initialize audio player
player = AudioPlayer()

# Generate text
def text_generator():
    for word in ["Hello", "world", "from", "VibeVoice"]:
        yield word

# Generate and play audio in real-time
audio_stream = tts.text_to_speech_streaming(text_generator())
player.play_stream(audio_stream, realtime=True)
```

### With Voice Cloning

```python
# Load with voice prompt for voice cloning
tts = VibeVoiceStreamingTTS(
    model_path="microsoft/VibeVoice-Realtime-0.5B",
    voice_prompt_path="path/to/voice.pt",  # Speaker embedding
    device="cuda"
)

# Generate with cloned voice
audio_stream = tts.text_to_speech_streaming(text_generator())
player.play_stream(audio_stream, realtime=True)
```

---

## API Reference

### synthesize_speech()

**Easiest way to use VibeVoice!** One-line function for text-to-speech synthesis.

```python
synthesize_speech(
    text: str | Iterator[str],
    model_path: str = "microsoft/VibeVoice-Realtime-0.5B",
    voice_prompt_path: Optional[str] = None,
    device: str = "cuda",
    output_file: Optional[str] = None,
    play_audio: bool = True,
    speaker_device_id: Optional[int] = None,
    inference_steps: int = 5,
    cfg_scale: float = 1.5,
    realtime: bool = True
) -> Optional[np.ndarray]
```

**Parameters:**

- `text` (str or Iterator[str]): Text to synthesize or iterator yielding text chunks
- `model_path` (str): HuggingFace model ID (default: "microsoft/VibeVoice-Realtime-0.5B")
- `voice_prompt_path` (str, optional): Custom voice prompt path. If None, uses default voice.
- `device` (str): Device ("cuda", "mps", "cpu")
- `output_file` (str, optional): Path to save WAV file
- `play_audio` (bool): Whether to play audio (default: True)
- `speaker_device_id` (int, optional): Speaker device ID (None for default)
- `inference_steps` (int): Diffusion steps (5=fast, 50=quality)
- `cfg_scale` (float): Guidance scale (1.0-2.0)
- `realtime` (bool): Use streaming playback (default: True)

**Returns:**
- `np.ndarray` or `None`: Audio array if `output_file` specified, else None

**Examples:**

```python
from vibevoice import synthesize_speech

# Simple usage
synthesize_speech("Hello world!", device="cuda")

# With iterator (LLM streaming)
def text_gen():
    for word in ["Hello", "streaming", "world"]:
        yield word
synthesize_speech(text_gen(), device="cuda")

# Save to file
synthesize_speech("Save this", output_file="output.wav", device="cuda")

# Custom voice
synthesize_speech(
    "Custom voice",
    voice_prompt_path="voices/custom.pt",
    device="cuda"
)

# High quality
synthesize_speech(
    "High quality",
    inference_steps=50,
    cfg_scale=2.0,
    device="cuda"
)
```

---

### list_default_voices()

List available default voice prompts included with VibeVoice.

```python
list_default_voices() -> list[str]
```

**Returns:**
- `list[str]`: List of available voice names (without .pt extension)

**Example:**

```python
from vibevoice import list_default_voices

voices = list_default_voices()
print(voices)
# ['en-Carter_man', 'en-Davis_man', 'en-Emma_woman', 'en-Frank_man',
#  'en-Grace_woman', 'en-Mike_man', 'in-Samuel_man']

# Use a specific default voice
voice_path = f"demo/voices/streaming_model/{voices[2]}.pt"  # en-Emma_woman
synthesize_speech("Hello", voice_prompt_path=voice_path)
```

---

### VibeVoiceStreamingTTS

High-level wrapper for VibeVoice streaming text-to-speech.

#### Constructor

```python
VibeVoiceStreamingTTS(
    model_path: str,
    voice_prompt_path: Optional[str] = None,
    device: str = "cuda",
    inference_steps: int = 5
)
```

**Parameters:**

- `model_path` (str): Path to VibeVoice model or HuggingFace model ID
  - Example: `"microsoft/VibeVoice-Realtime-0.5B"`
- `voice_prompt_path` (str, optional): Path to voice prompt file (.pt) for voice cloning
  - **If None (default):** Automatically uses a default voice from `demo/voices/streaming_model/`
  - **7 default voices available:** en-Mike_man, en-Emma_woman, en-Carter_man, en-Davis_man, en-Frank_man, en-Grace_woman, in-Samuel_man
  - Use `list_default_voices()` to see available voices
- `device` (str): Device to run on
  - `"cuda"` - NVIDIA GPU (fastest, requires flash-attention-2)
  - `"mps"` - Apple Silicon GPU
  - `"cpu"` - CPU (slower)
- `inference_steps` (int): Number of diffusion steps
  - `5` - Fast, good quality (default, ~100ms latency)
  - `50` - High quality (~500ms latency)

#### Methods

##### `text_to_speech_streaming(text_iterator, cfg_scale=1.5)`

Generate speech from text iterator with real-time streaming.

**Parameters:**

- `text_iterator` (Iterator[str]): Iterator yielding text tokens/chunks
- `cfg_scale` (float): Classifier-free guidance scale (1.0-2.0)
  - `1.0` - Faster, lower quality
  - `1.5` - Balanced (default)
  - `2.0` - Better quality, slower

**Returns:**

- Iterator[np.ndarray]: Audio chunks as float32 numpy arrays

**Example:**

```python
def text_gen():
    for word in ["Hello", "world"]:
        yield word

for audio_chunk in tts.text_to_speech_streaming(text_gen()):
    # audio_chunk is np.ndarray, shape (N,), dtype float32
    # values are normalized to [-1.0, 1.0]
    print(f"Received {len(audio_chunk)} samples")
```

##### `save_audio(audio, output_path)`

Save generated audio to WAV file.

**Parameters:**

- `audio` (np.ndarray): Audio data
- `output_path` (str): Path to save WAV file

**Example:**

```python
import numpy as np

# Collect all chunks
chunks = list(tts.text_to_speech_streaming(text_gen()))

# Concatenate and save
full_audio = np.concatenate(chunks)
tts.save_audio(full_audio, "output.wav")
```

---

### AudioPlayer

Audio player with speaker selection and streaming support.

#### Constructor

```python
AudioPlayer(device_id: Optional[int] = None, sample_rate: int = 24000)
```

**Parameters:**

- `device_id` (int, optional): Speaker device ID (None for default)
- `sample_rate` (int): Audio sample rate in Hz (default 24000)

#### Methods

##### `list_devices(show_all=False)` [static]

List available audio output devices.

**Parameters:**

- `show_all` (bool): If True, show all devices including duplicates

**Example:**

```python
AudioPlayer.list_devices()
# Output:
# Available Audio Output Devices:
# [3] Microsoft Sound Mapper - Output ⭐ DEFAULT
# [4] Speakers (USB Audio Device)
```

##### `get_default_output_device()` [static]

Get default output device ID.

**Returns:**

- int: Default device ID

**Example:**

```python
device_id = AudioPlayer.get_default_output_device()
player = AudioPlayer(device_id=device_id)
```

##### `play_stream(audio_iterator, realtime=True)`

Play audio from an iterator of chunks.

**Parameters:**

- `audio_iterator` (Iterator[np.ndarray]): Iterator yielding audio chunks
- `realtime` (bool): Streaming mode
  - `True` - Real-time streaming with ~100ms latency
  - `False` - Buffered playback (waits for all chunks, smooth)

**Example:**

```python
# Real-time streaming (low latency)
audio_stream = tts.text_to_speech_streaming(text_gen())
player.play_stream(audio_stream, realtime=True)

# Buffered playback (smooth, no gaps)
audio_stream = tts.text_to_speech_streaming(text_gen())
player.play_stream(audio_stream, realtime=False)
```

##### `stop()`

Stop current playback.

---

## Advanced Usage

### Select Specific Speaker

```python
# List available devices
AudioPlayer.list_devices()

# Use specific device
player = AudioPlayer(device_id=4)  # Use device 4
audio_stream = tts.text_to_speech_streaming(text_gen())
player.play_stream(audio_stream, realtime=True)
```

### Custom Quality Settings

```python
# High quality (slower)
tts = VibeVoiceStreamingTTS(
    model_path="microsoft/VibeVoice-Realtime-0.5B",
    device="cuda",
    inference_steps=50  # More steps = better quality
)

audio_stream = tts.text_to_speech_streaming(
    text_gen(),
    cfg_scale=2.0  # Higher CFG = better quality
)
player.play_stream(audio_stream, realtime=True)
```

### Process Audio Chunks

```python
import soundfile as sf

audio_chunks = []
for audio_chunk in tts.text_to_speech_streaming(text_gen()):
    # Process each chunk as it arrives
    audio_chunks.append(audio_chunk)

    # You can also apply effects here
    # audio_chunk = apply_effects(audio_chunk)

# Save all chunks
full_audio = np.concatenate(audio_chunks)
sf.write("output.wav", full_audio, tts.sample_rate)
```

### Streaming from LLM Output

```python
def llm_token_stream():
    """Simulate LLM generating tokens"""
    llm_output = [
        "The", "weather", "today", "is", "sunny", "and", "warm."
    ]
    for token in llm_output:
        yield token

# Convert LLM output to speech in real-time
audio_stream = tts.text_to_speech_streaming(llm_token_stream())
player.play_stream(audio_stream, realtime=True)
```

---

## Examples

### Example 1: Simple TTS

```python
from vibevoice import VibeVoiceStreamingTTS, AudioPlayer

tts = VibeVoiceStreamingTTS(
    model_path="microsoft/VibeVoice-Realtime-0.5B",
    device="cuda"
)

player = AudioPlayer()

def text_gen():
    yield "Hello world!"

audio = tts.text_to_speech_streaming(text_gen())
player.play_stream(audio, realtime=True)
```

### Example 2: Voice Cloning

```python
from vibevoice import VibeVoiceStreamingTTS, AudioPlayer

# Load with voice prompt
tts = VibeVoiceStreamingTTS(
    model_path="microsoft/VibeVoice-Realtime-0.5B",
    voice_prompt_path="voices/emma.pt",  # Clone Emma's voice
    device="cuda"
)

player = AudioPlayer()

def text_gen():
    yield "This is a cloned voice speaking!"

audio = tts.text_to_speech_streaming(text_gen())
player.play_stream(audio, realtime=True)
```

### Example 3: Save to File

```python
import numpy as np
from vibevoice import VibeVoiceStreamingTTS

tts = VibeVoiceStreamingTTS(
    model_path="microsoft/VibeVoice-Realtime-0.5B",
    device="cuda"
)

def text_gen():
    for sentence in ["Hello.", "How are you?", "Goodbye!"]:
        yield sentence

# Collect all chunks
chunks = list(tts.text_to_speech_streaming(text_gen()))
full_audio = np.concatenate(chunks)

# Save to file
tts.save_audio(full_audio, "output.wav")
print("Audio saved to output.wav")
```

### Example 4: Multiple Speaker Devices

```python
from vibevoice import VibeVoiceStreamingTTS, AudioPlayer

# List available devices
AudioPlayer.list_devices()

# Use multiple devices
player1 = AudioPlayer(device_id=3)  # Default speaker
player2 = AudioPlayer(device_id=4)  # External speaker

tts = VibeVoiceStreamingTTS(
    model_path="microsoft/VibeVoice-Realtime-0.5B",
    device="cuda"
)

def text_gen():
    yield "Hello from VibeVoice!"

# Play on device 1
audio1 = tts.text_to_speech_streaming(text_gen())
player1.play_stream(audio1, realtime=True)

# Play on device 2
audio2 = tts.text_to_speech_streaming(text_gen())
player2.play_stream(audio2, realtime=True)
```

### Example 5: Real-time LLM Integration

```python
from vibevoice import VibeVoiceStreamingTTS, AudioPlayer
import threading

tts = VibeVoiceStreamingTTS(
    model_path="microsoft/VibeVoice-Realtime-0.5B",
    device="cuda"
)

player = AudioPlayer()

def llm_stream():
    """Your LLM generates tokens here"""
    tokens = [
        "Once", "upon", "a", "time", "there", "was",
        "a", "voice", "assistant", "that", "could", "speak."
    ]
    for token in tokens:
        yield token

# Generate and play simultaneously
audio_stream = tts.text_to_speech_streaming(llm_stream())
player.play_stream(audio_stream, realtime=True)
```

---

## Performance Tips

1. **Use CUDA** - GPU is much faster than CPU
   ```python
   tts = VibeVoiceStreamingTTS(model_path="...", device="cuda")
   ```

2. **Lower inference steps** for lower latency
   ```python
   tts = VibeVoiceStreamingTTS(model_path="...", inference_steps=5)
   ```

3. **Use real-time streaming** for lowest latency
   ```python
   player.play_stream(audio, realtime=True)
   ```

4. **Prebuffer for smoother playback**
   - Real-time mode prebuffers 100ms automatically
   - Buffered mode collects all audio first

---

## Troubleshooting

### No audio output

```python
# Check available devices
AudioPlayer.list_devices()

# Try default device
player = AudioPlayer(device_id=None)
```

### CUDA out of memory

```python
# Use CPU instead
tts = VibeVoiceStreamingTTS(model_path="...", device="cpu")
```

### Import errors

```bash
# Reinstall VibeVoice
pip install -e /path/to/VibeVoice

# Install sounddevice for audio playback
pip install sounddevice
```

### Distorted audio

Audio is automatically normalized and clipped to [-1.0, 1.0] range. If you still hear distortion, try:

```python
# Lower CFG scale
audio = tts.text_to_speech_streaming(text_gen(), cfg_scale=1.0)
```

---

## License

See the [LICENSE](../LICENSE) file for details.
