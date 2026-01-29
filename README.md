<div align="center">

## 🎙️ VibeVoice-Narrator: Markdown-to-Speech Conversion

[![Project Page](https://img.shields.io/badge/Project-Page-blue?logo=githubpages)](https://github.com/Dazlarus/VibeVoice-Narrator)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Collection-orange?logo=huggingface)](https://huggingface.co/collections/microsoft/vibevoice-68a2ef24a875c44be47b034f)
[![TTS Report](https://img.shields.io/badge/TTS-Report-red?logo=arxiv)](https://arxiv.org/pdf/2508.19205)

</div>

<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="Figures/VibeVoice_logo_white.png">
  <img src="Figures/VibeVoice_logo.png" alt="VibeVoice Logo" width="300">
</picture>

</div>

<div align="left">

<h3>📰 Overview</h3>

VibeVoice-Narrator is a **Markdown-to-Speech conversion tool** built on top of the VibeVoice TTS model. It enables users to convert Markdown documents into natural-sounding speech audio with support for heading-based chunking and multiple speakers.

<h3>🎯 Purpose</h3>

This fork focuses exclusively on **Text-to-Speech (TTS)** functionality for converting Markdown documents to audio. The original VibeVoice repository included ASR (Speech-to-Text) capabilities which have been removed to create a focused, efficient architecture.

<h3>🔧 Key Features</h3>

- **📝 Markdown Chunking**: Automatically splits Markdown documents by headings for natural speech segmentation
- **🎭 Multi-speaker Support**: Supports multiple voice presets for different speakers
- **⏱️ Long-form Generation**: Handles documents up to 90 minutes in a single pass
- **🌐 Multi-lingual**: Supports English, Chinese, and other languages
- **🎵 Real-time Streaming**: Low-latency streaming TTS for interactive applications
- **📊 Configurable Parameters**: Adjustable chunk depth, pause duration, and voice selection

<h3>🏗️ Architecture</h3>

```
┌─────────────────────────────────────────────────────────────────┐
│                    Markdown Document Input                    │
│                           │                               │
│                           ▼                               │
│                    Markdown Chunker                        │
│                           │                               │
│                           ▼                               │
│                    VibeVoice Streaming TTS Model              │
│                           │                               │
│                           ▼                               │
│                    Audio Output (WAV)                       │
└─────────────────────────────────────────────────────────────────┘
```

<h3>📦 Installation</h3>

```bash
# Clone the repository
git clone https://github.com/Dazlarus/VibeVoice-Narrator.git
cd VibeVoice-Narrator

# Install dependencies
pip install -e ".[tts]"
```

<h3>🚀 Quick Start</h3>

```bash
# Convert a Markdown file to speech
python demo/chunked_markdown_tts_realtime.py \
    --markdown demo/example_scripts/simple.md \
    --speaker Carter \
    --output output.wav \
    --depth 3 \
    --device cuda
```

<h3>📚 Demo Scripts</h3>

- **[`demo/chunked_markdown_tts.py`](demo/chunked_markdown_tts.py)** - Standard TTS with heading-based chunking
- **[`demo/chunked_markdown_tts_realtime.py`](demo/chunked_markdown_tts_realtime.py)** - Real-time streaming TTS with lower latency

<h3>🎤 Voice Files</h3>

Voice presets are required for TTS generation. For local development you can use the demo voices bundled in this repo (`demo/voices/streaming_model/`), or download official voice checkpoints from release assets. Below are concrete, reproducible steps and a small helper script to automate downloads.

Required vs Optional
- REQUIRED: one or more voice checkpoint files (e.g., `en-Carter_man.pt`, `en-Emma_woman.pt`) — needed to run TTS.
- OPTIONAL: `extra-voices.zip` or additional checkpoint bundles — add more voices.

Approximate file sizes (varies by model):
- Single voice checkpoint (`*.pt`): ~150 MB — 1 GB (expect ~30s–3m on typical broadband)
- Voice bundle / ZIP (`extra-voices.zip`): ~1–4 GB (expect several minutes)

Quick copy (use the bundled demo voices):

```bash
# Copy demo voices into data/voices for quick start
mkdir -p data/voices
cp -v demo/voices/streaming_model/*.pt data/voices/
```

Automated download helper (create `demo/voices/streaming_model/download_voices.sh`):

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p ../../../data/voices

# REQUIRED: single voice examples (replace version/tag and filenames as needed)
# Examples below reference GitHub release assets on microsoft/VibeVoice — update tags/filenames for the release you want
curl -L -o en-Carter_man.pt \
  https://github.com/microsoft/VibeVoice/releases/download/v1.0/en-Carter_man.pt
curl -L -o en-Emma_woman.pt \
  https://github.com/microsoft/VibeVoice/releases/download/v1.0/en-Emma_woman.pt

# OPTIONAL: download an extra bundle (large)
# curl -L -o extra-voices.zip https://github.com/microsoft/VibeVoice/releases/download/v1.0/extra-voices.zip
# unzip -d extra-voices extra-voices.zip

# Copy downloaded files into project data dir
mkdir -p ../../../data/voices
cp -v *.pt ../../../data/voices/
```

Usage:

```bash
# Make the helper executable and run it from repo root
chmod +x demo/voices/streaming_model/download_voices.sh
./demo/voices/streaming_model/download_voices.sh
```

Direct wget example:

```bash
wget -O data/voices/en-Carter_man.pt \
  https://github.com/microsoft/VibeVoice/releases/download/v1.0/en-Carter_man.pt
```

On Windows (PowerShell):

```powershell
# Copy demo voices into data\voices (quick start) — run from repository root
New-Item -ItemType Directory -Path data\voices -Force
Copy-Item demo\voices\streaming_model\*.pt data\voices\
```

Placement and expected layout

```
data/voices/
├── en-Carter_man.pt   # REQUIRED
├── en-Emma_woman.pt   # OPTIONAL (example)
└── extra-voices/      # OPTIONAL (unzipped bundle)
```

Notes
- You may also symlink the demo voices instead of copying, e.g. `ln -s ../../demo/voices/streaming_model data/voices` (or a junction on Windows).
- Replace `v1.0` and filenames in the script/commands with the actual release tag and asset names if they differ.
- If you host large models internally, prefer internal artifact storage instead of GitHub releases to avoid rate limits.

<h3>📖 Documentation</h3>

- **[`docs/markdown-chunking.md`](docs/markdown-chunking.md)** - Markdown chunking documentation
- **[`docs/web-gui-implementation.md`](docs/web-gui-implementation.md)** - Web GUI implementation status and API documentation

<h3>🧪 Testing</h3>

```bash
# Run tests
python -m pytest tests/chunking/test_markdown_chunker.py -v
```

<h3>📊 Project Structure</h3>

```
vibevoice/
├── __init__.py                 # Main package exports
├── configs/                    # Model configurations
├── modular/                     # Core TTS models
│   ├── configuration_vibevoice.py
│   ├── configuration_vibevoice_streaming.py
│   ├── modeling_vibevoice.py
│   ├── modeling_vibevoice_streaming.py
│   ├── modeling_vibevoice_streaming_inference.py
│   ├── modular_vibevoice_diffusion_head.py
│   ├── modular_vibevoice_text_tokenizer.py
│   └── modular_vibevoice_tokenizer.py
├── processor/                   # Text and audio processing
│   ├── vibevoice_streaming_processor.py
│   ├── vibevoice_tokenizer_processor.py
│   ├── audio_utils.py
│   └── chunking/
│       ├── markdown_chunker.py
│       └── chunked_tts_processor.py
└── schedule/                    # Diffusion scheduling
    ├── dpm_solver.py
    └── timestep_sampler.py

demo/
├── chunked_markdown_tts.py
├── chunked_markdown_tts_realtime.py
├── example_scripts/
└── voices/
    └── streaming_model/
        └── README.md

tests/
└── chunking/
    └── test_markdown_chunker.py
```

<h3>🎓 Usage Examples</h3>

<h4>Basic Markdown to Speech</h4>

```bash
# Simple conversion with default settings
python demo/chunked_markdown_tts_realtime.py \
    --markdown my_document.md \
    --speaker Emma \
    --output speech.wav
```

<h4>Advanced Options</h4>

```bash
# Custom chunk depth (split by H2 headings)
python demo/chunked_markdown_tts_realtime.py \
    --markdown my_document.md \
    --speaker Davis \
    --depth 2 \
    --output speech.wav \
    --device cuda

# Include headings in speech
python demo/chunked_markdown_tts_realtime.py \
    --markdown my_document.md \
    --speaker Grace \
    --include-heading \
    --output speech.wav

# Adjust pause between chunks
python demo/chunked_markdown_tts_realtime.py \
    --markdown my_document.md \
    --speaker Frank \
    --pause-ms 1000 \
    --output speech.wav
```

<h3>⚙️ Configuration</h3>

The demo scripts support the following configuration options:

| Option | Default | Description |
|---------|-----------|-------------|
| `--markdown` | Required | Path to Markdown (.md) file |
| `--speaker` | Required | Speaker name (e.g., Carter, Davis, Emma) |
| `--output` | `output.wav` | Output audio file path |
| `--depth` | `1` | Heading depth to chunk on (1-3) |
| `--model` | `microsoft/VibeVoice-Realtime-0.5B` | Model name or path |
| `--device` | `auto` | Device: auto, cuda, cpu, mps |
| `--pause-ms` | `500` | Pause duration between chunks (ms) |
| `--include-heading` | False | Speak headings in each chunk |
| `--no-strip` | False | Keep markdown formatting in content |
| `--cfg-scale` | `1.25` | CFG scale for generation |
| `--quiet` | False | Suppress progress output |

<h3>🤝 Contributing</h3>

This is a focused fork of VibeVoice. Contributions should focus on:

1. Improving Markdown-to-Speech conversion quality
2. Enhancing chunking algorithms for better speech segmentation
3. Adding support for additional languages
4. Optimizing performance and reducing latency
5. Improving documentation and examples

<h3>📜 License</h3>

This project is licensed under the **MIT License**.

See the top-level `LICENSE` file for the full text.

<h3>🔗 Links</h3>

- **Original VibeVoice**: [https://github.com/microsoft/VibeVoice](https://github.com/microsoft/VibeVoice)
- **VibeVoice TTS Paper**: [https://arxiv.org/pdf/2508.19205](https://arxiv.org/pdf/2508.19205)
- **Hugging Face Collection**: [https://huggingface.co/collections/microsoft/vibevoice](https://huggingface.co/collections/microsoft/vibevoice)

<h3>⚠️ Important Notes</h3>

- This fork focuses **exclusively on Text-to-Speech (TTS)** functionality
- ASR (Speech-to-Text) components have been removed to create a focused architecture
- Voice files are required for TTS generation - download them from the official repository
- The codebase has been refactored to remove bloat and improve maintainability
- All tests pass successfully

</div>
