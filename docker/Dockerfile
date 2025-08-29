# Based on NVIDIA PyTorch container image
FROM nvcr.io/nvidia/pytorch:24.07-py3

# Set working directory
WORKDIR /app

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=WestZhang/VibeVoice-Large-pt

# Update system and install required packages
RUN apt-get update && \
    apt-get install -y ffmpeg git curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install flash-attn
RUN pip install flash-attn --no-build-isolation

# Clone VibeVoice repository
RUN git clone https://github.com/brianxiadong/VibeVoice.git .

# Install VibeVoice dependencies
RUN pip install -e .

# Install additional dependencies for news podcast functionality
RUN pip install soundfile requests

# Copy news podcast functionality (this will be included in the repo)
# COPY news_podcast/ /app/news_podcast/
# COPY generate_news_podcast.py /app/
# COPY test_news_pipeline.py /app/

# Copy container setup script
COPY ./setup_container.sh /app/setup_container.sh
RUN chmod +x /app/setup_container.sh

# Create output directory
RUN mkdir -p /app/podcast_output

# Expose Gradio default port
EXPOSE 7860

# Set startup command using shell form to properly expand environment variables
CMD python demo/gradio_demo.py --model_path "$MODEL_PATH" --share