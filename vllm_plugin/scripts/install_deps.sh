#!/bin/bash
# Install system dependencies for VibeVoice vLLM plugin
# Run this script inside the vLLM container before using the plugin

set -e

echo "Installing system dependencies for VibeVoice vLLM plugin..."

# Update package list
apt-get update

# Install FFmpeg and audio processing libraries
apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git

echo "✅ System dependencies installed successfully!"
echo ""
echo "Next steps:"
echo "  1. Install VibeVoice: pip install -e .[vllm]"
echo "  2. Generate tokenizer files (if needed): python -m vllm_plugin.tools.generate_tokenizer_files -o /path/to/model"
echo "  3. Start vLLM server: vllm serve <model_path> --trust-remote-code --enforce-eager --no-enable-prefix-caching"
