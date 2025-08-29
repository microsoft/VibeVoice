#!/bin/bash

# Docker Container Setup Script for VibeVoice News Podcast
# This script sets up the environment for news podcast generation

set -e

echo "🐳 VibeVoice News Podcast Docker Setup"
echo "========================================"

# Check if we're inside a container
if [ -f /.dockerenv ]; then
    echo "✓ Running inside Docker container"
    
    # Verify required environment variables
    echo "📋 Checking environment variables..."
    echo "   MODEL_PATH: ${MODEL_PATH:-Not set}"
    echo "   OLLAMA_URL: ${OLLAMA_URL:-Not set}" 
    echo "   OLLAMA_MODEL: ${OLLAMA_MODEL:-Not set}"
    
    # Test Ollama connection if configured
    if [ -n "$OLLAMA_URL" ]; then
        echo "🔗 Testing Ollama connection..."
        if curl -s -o /dev/null -w "%{http_code}" "$OLLAMA_URL/api/version" | grep -q "200"; then
            echo "✓ Ollama connection successful"
        else
            echo "⚠️  Warning: Cannot connect to Ollama at $OLLAMA_URL"
            echo "   Make sure Ollama is running and accessible"
        fi
    fi
    
    # Verify news podcast files exist
    echo "📁 Checking news podcast files..."
    if [ -d "/app/news_podcast" ]; then
        echo "✓ News podcast module found"
    else
        echo "⚠️  Warning: News podcast module not found"
        echo "   This might be expected if using the base VibeVoice image"
    fi
    
    if [ -f "/app/generate_news_podcast.py" ]; then
        echo "✓ News podcast generator script found"
    else
        echo "⚠️  Warning: News podcast generator script not found"
    fi
    
    # Create output directory
    mkdir -p /app/podcast_output
    echo "✓ Output directory created at /app/podcast_output"
    
    # Test basic Python imports
    echo "🐍 Testing Python environment..."
    python -c "import torch; print(f'✓ PyTorch {torch.__version__} available')" || echo "❌ PyTorch import failed"
    python -c "import transformers; print(f'✓ Transformers available')" || echo "❌ Transformers import failed"
    python -c "import requests; print('✓ Requests available')" || echo "❌ Requests import failed"
    
    # Test GPU availability
    echo "🖥️  Checking GPU availability..."
    if python -c "import torch; print('✓ GPU available' if torch.cuda.is_available() else '⚠️  GPU not available (CPU mode)')"; then
        if python -c "import torch; print(f'   Device count: {torch.cuda.device_count()}')"; then
            python -c "import torch; print(f'   Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
        fi
    fi
    
    echo ""
    echo "🎉 Setup complete!"
    echo ""
    echo "📖 Usage Examples:"
    echo "   # Test pipeline (no audio):"
    echo "   python test_news_pipeline.py"
    echo ""
    echo "   # Generate complete podcast:"
    echo "   python generate_news_podcast.py"
    echo ""
    echo "   # List available voices:"
    echo "   python generate_news_podcast.py --list-voices"
    echo ""
    echo "   # Custom generation:"
    echo "   python generate_news_podcast.py --speakers 3 --output-dir /app/podcast_output"
    echo ""
    
else
    echo "❌ This script should be run inside a Docker container"
    echo "   Use: docker compose exec vibevoice bash"
    echo "   Then: ./setup_container.sh"
    exit 1
fi