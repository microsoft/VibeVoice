# VibeVoice Docker Deployment Guide

This directory contains Docker configuration files for deploying the VibeVoice project.

## Project Overview

VibeVoice is a deep learning-based voice processing project that supports multiple model configurations. This Docker setup provides a complete containerized deployment solution with GPU acceleration and flexible model switching capabilities.

## Prerequisites

Before getting started, ensure your system meets the following requirements:

- **Docker**: Version 20.10 or higher
- **Docker Compose**: Version 2.0 or higher
- **NVIDIA GPU**: CUDA-compatible GPU (recommended)
- **NVIDIA Container Toolkit**: For GPU support
- **System Memory**: At least 8GB RAM
- **Storage Space**: At least 10GB available space

### Installing NVIDIA Container Toolkit

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd VibeVoice
```

### 2. Build and Start Container

```bash
cd docker
docker compose up --build
```

Once the container is running, the VibeVoice service will be available at `http://localhost:7860`.

**Note**: No additional scripts are required. The container starts directly with the configured command.

## Model Configuration

The project supports two model configurations:

### Default Model (microsoft/VibeVoice-1.5B)

```bash
docker compose up --build
```

### Large Model (microsoft/VibeVoice-7B)

Modify the environment variable in `docker-compose.yml`:

```yaml
environment:
  - MODEL_PATH=microsoft/VibeVoice-7B
```

Or specify it at runtime:

```bash
MODEL_PATH=microsoft/VibeVoice-7B docker compose up --build
```

## Usage

### Start the Service (Detached Mode)

```bash
cd docker
docker compose up -d
```

### Start the Service (Interactive Mode)

```bash
cd docker
docker compose up
```

### View Logs

```bash
docker compose logs -f
```

### Stop the Service

```bash
docker compose down
```

### Rebuild Container

```bash
docker compose up --build
```

### Direct Docker Commands (Alternative)

You can also use direct Docker commands:

```bash
# Build the image
docker build -t vibevoice .

# Run the container
docker run --gpus all -p 7860:7860 -e MODEL_PATH=microsoft/VibeVoice-1.5B vibevoice
```

## Troubleshooting

### GPU Support Issues

If you encounter GPU-related problems, please check:

1. NVIDIA drivers are properly installed
2. NVIDIA Container Toolkit is installed
3. Docker has been restarted

```bash
# Check GPU availability
nvidia-smi

# Test Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### Out of Memory

If you encounter memory issues:

1. Ensure sufficient RAM (16GB+ recommended)
2. Close other memory-intensive applications
3. Consider using a smaller model

### Port Conflicts

If port 7860 is occupied:

```yaml
ports:
  - "8080:7860"  # Use a different host port
```

## Important Notes

- First run will download model files, which may take considerable time
- Ensure sufficient disk space for model files
- Will automatically fallback to CPU mode if GPU memory is insufficient
- Recommended to use fixed model versions in production environments
```