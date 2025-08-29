# VibeVoice Docker Deployment Guide

## Project Overview

VibeVoice is an AI-powered voice synthesis system that can generate high-quality speech from text. This Docker setup provides an easy way to deploy VibeVoice with GPU acceleration using NVIDIA containers.

## Quick Start (Recommended)

### Prerequisites
- Docker and Docker Compose installed
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit installed

### Using Docker Compose

1. **Clone the repository:**
   ```bash
   git clone https://github.com/microsoft/VibeVoice.git
   cd VibeVoice
   ```

2. **Build and start the container:**
   ```bash
   docker-compose -f docker/docker-compose.yml up --build
   ```

3. **Access the application:**
   Open your browser and navigate to `http://localhost:7860`

4. **Stop the service:**
   ```bash
   docker-compose -f docker/docker-compose.yml down
   ```

## Direct Docker Commands

If you prefer using Docker commands directly:

1. **Build the image:**
   ```bash
   docker build -f docker/Dockerfile -t vibevoice:latest .
   ```

2. **Run the container:**
   ```bash
   docker run --gpus all -p 7860:7860 \
     -e MODEL_PATH=WestZhang/VibeVoice-Large-pt \
     vibevoice:latest
   ```

## Configuration

### Model Selection

You can choose between different models by setting the `MODEL_PATH` environment variable:

- **Default 1.5B Model:** `microsoft/VibeVoice-1.5B`
- **Large Model:** `WestZhang/VibeVoice-Large-pt` (recommended)

Edit the `MODEL_PATH` in `docker-compose.yml` or pass it as an environment variable.

### Volume Mounting

The setup includes a cache volume mount (`./cache:/root/.cache`) to persist model downloads and improve startup times.

## Common Issues

### GPU Support
- Ensure NVIDIA Container Toolkit is properly installed
- Verify GPU access with: `docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi`

### Memory Requirements
- Large models require significant GPU memory (8GB+ recommended)
- If you encounter OOM errors, try using the smaller 1.5B model

### Port Conflicts
- If port 7860 is in use, modify the port mapping in `docker-compose.yml`:
  ```yaml
  ports:
    - "8080:7860"  # Use port 8080 instead
  ```

### Container Logs
View logs for troubleshooting:
```bash
docker-compose -f docker/docker-compose.yml logs -f
```

## Notes

- The container automatically downloads the specified model on first run
- GPU acceleration is enabled by default for optimal performance
- The Gradio interface provides an easy-to-use web UI for voice synthesis