#!/bin/bash

# Script to prepare Docker build with news podcast functionality
# Run this before docker compose up --build

set -e

echo "🔧 Preparing VibeVoice Docker build with News Podcast functionality"
echo "=================================================================="

# Check if we're in the right directory
if [ ! -f "docker/Dockerfile" ]; then
    echo "❌ Please run this script from the VibeVoice root directory"
    exit 1
fi

# Check if news podcast files exist
echo "📋 Checking news podcast files..."

REQUIRED_FILES=(
    "news_podcast/__init__.py"
    "news_podcast/news_fetcher.py"
    "news_podcast/ollama_processor.py"
    "news_podcast/audio_generator.py"
    "news_podcast/main.py"
    "generate_news_podcast.py"
    "test_news_pipeline.py"
)

MISSING_FILES=()

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✓ $file"
    else
        echo "   ❌ $file (missing)"
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    echo ""
    echo "⚠️  Warning: Some news podcast files are missing."
    echo "   The Docker container will still build but news podcast functionality may not work."
    echo "   Missing files: ${MISSING_FILES[*]}"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Build cancelled."
        exit 1
    fi
else
    echo "✅ All news podcast files found"
fi

# Update Dockerfile to include news podcast files
echo ""
echo "📝 Updating Dockerfile to include news podcast functionality..."

# Uncomment the COPY lines in Dockerfile
sed -i.bak 's/# COPY news_podcast\//COPY news_podcast\//' docker/Dockerfile
sed -i.bak 's/# COPY generate_news_podcast.py/COPY generate_news_podcast.py/' docker/Dockerfile  
sed -i.bak 's/# COPY test_news_pipeline.py/COPY test_news_pipeline.py/' docker/Dockerfile

echo "✅ Dockerfile updated"

# Create output directory for mounting
mkdir -p docker/podcast_output
echo "✅ Created docker/podcast_output directory"

echo ""
echo "🎉 Preparation complete!"
echo ""
echo "📖 Next steps:"
echo "   1. Build and start the container:"
echo "      cd docker && docker compose up --build -d"
echo ""
echo "   2. Enter the container:"
echo "      docker compose exec vibevoice bash"
echo ""
echo "   3. Run the setup check:"
echo "      ./setup_container.sh"
echo ""
echo "   4. Generate a podcast:"
echo "      python generate_news_podcast.py"
echo ""
echo "📁 Generated files will be available in docker/podcast_output/"