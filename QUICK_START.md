# VibeVoice Quick Start Guide

Get up and running with VibeVoice in minutes!

## 🚀 Installation

### Option 1: Install from PyPI (when available)
```bash
pip install vibevoice
```

### Option 2: Install from source
```bash
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice
pip install -e .
```

### Option 3: Install with requirements.txt
```bash
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice
pip install -r requirements.txt
```

## ✅ Verify Installation

Run the validation script:
```bash
python validate_installation.py
```

## 🎯 Basic Usage

### 1. Web Interface (Easiest)

Launch the web demo:
```bash
python demo/vibevoice_realtime_demo.py \
    --model_path microsoft/VibeVoice-Realtime-0.5B \
    --device cuda
```

Open http://localhost:3000 in your browser and start generating speech!

### 2. Python API

```python
from vibevoice import (
    VibeVoiceStreamingForConditionalGenerationInference,
    VibeVoiceStreamingProcessor
)
import torch

# Load model and processor
model_path = "microsoft/VibeVoice-Realtime-0.5B"
processor = VibeVoiceStreamingProcessor.from_pretrained(model_path)
model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

# Load voice preset
voice_preset = torch.load("demo/voices/streaming_model/en-WHTest_man.pt")

# Prepare input
text = "Hello, this is a test of VibeVoice text to speech synthesis."
inputs = processor.process_input_with_cached_prompt(
    text=text,
    cached_prompt=voice_preset,
    padding=True,
    return_tensors="pt"
)

# Generate speech
outputs = model.generate(
    **inputs,
    cfg_scale=1.5,
    tokenizer=processor.tokenizer
)

# Save audio
processor.save_audio(
    outputs.speech_outputs[0],
    output_path="output.wav"
)
```

### 3. Command Line (Batch Processing)

Generate audio from a text file:
```bash
python demo/realtime_model_inference_from_file.py \
    --model_path microsoft/VibeVoice-Realtime-0.5B \
    --txt_path demo/text_examples/1p_vibevoice.txt \
    --speaker_name Wayne \
    --output_dir ./outputs
```

## 🎛️ Device Selection

### CUDA (NVIDIA GPU)
```bash
--device cuda
```
Best performance, requires NVIDIA GPU with CUDA support.

### MPS (Apple Silicon)
```bash
--device mps
```
For M1/M2/M3 Macs. Good performance on Apple Silicon.

### CPU
```bash
--device cpu
```
Works everywhere but slower. Good for testing.

## 🎤 Voice Presets

Available voice presets are in `demo/voices/streaming_model/`:
- `en-WHTest_man` - English male
- `en-WHTest_woman` - English female
- Language-specific voices: DE, FR, IT, JP, KR, NL, PL, PT, ES

Use with `--speaker_name` or `--voice` parameter.

## ⚙️ Common Parameters

### cfg_scale (Classifier-Free Guidance)
- Default: `1.5`
- Range: `1.0` to `3.0`
- Higher values = more expressive but less stable
- Lower values = more stable but less expressive

### inference_steps
- Default: `5`
- Range: `1` to `50`
- More steps = better quality but slower
- Fewer steps = faster but lower quality

## 🐛 Troubleshooting

### "CUDA out of memory"
```bash
# Use CPU instead
--device cpu

# Or use MPS on Mac
--device mps
```

### "Flash Attention not available"
The model will automatically fall back to SDPA. This is normal and expected on some systems.

### "Model not found"
Make sure you have internet connection for first-time download, or specify a local path:
```bash
--model_path /path/to/local/model
```

### "Voice preset not found"
Check available voices:
```bash
ls demo/voices/streaming_model/
```

## 📚 Next Steps

- Read the full [README.md](README.md)
- Check [demo/README.md](demo/README.md) for advanced usage
- See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute
- Visit the [Project Page](https://microsoft.github.io/VibeVoice) for examples

## 🆘 Getting Help

- 📖 Documentation: Check the docs/ folder
- 🐛 Issues: https://github.com/microsoft/VibeVoice/issues
- 💬 Discussions: GitHub Discussions
- 📧 Email: vibepod@microsoft.com

## ⚠️ Important Notes

- VibeVoice is for research purposes
- Always disclose AI-generated content
- Do not use for deepfakes or misinformation
- See [SECURITY.md](SECURITY.md) for security guidelines

## 🎉 You're Ready!

Start generating expressive speech with VibeVoice. Have fun and use responsibly!
