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
RUN git clone https://github.com/microsoft/VibeVoice.git .

# Install VibeVoice dependencies
RUN pip install -e .

# Expose Gradio default port
EXPOSE 7860

# Set startup command using shell form to properly expand environment variables
CMD python demo/gradio_demo.py --model_path "$MODEL_PATH" --share