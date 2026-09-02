# VibeVoice-ASR-Streaming

**VibeVoice-ASR-Streaming** transcribes while the audio is still arriving, instead of waiting for it to end. It emits text once per audio chunk, so a transcript appears as the speaker talks.

**Non-streaming:** [VibeVoice-ASR](./vibevoice-asr.md)<br>
**vLLM:** [vLLM-asr-streaming](./vibevoice-vllm-asr-streaming.md)<br>

## Installation
We recommend using NVIDIA Deep Learning Container to manage the CUDA environment. 

1. Launch docker
```bash
# NVIDIA PyTorch Container 24.07 ~ 25.12 verified. 
# Previous versions are also compatible.
sudo docker run --privileged --net=host --ipc=host --ulimit memlock=-1:-1 --ulimit stack=-1:-1 --gpus all --rm -it  nvcr.io/nvidia/pytorch:25.12-py3

## If flash attention is not included in your docker environment, you need to install it manually
## Refer to https://github.com/Dao-AILab/flash-attention for installation instructions
# pip install flash-attn --no-build-isolation
```

2. Install from github 
```bash
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice

pip install -e .
```

3. Install ffmpeg, which both usages below need to decode audio
```bash
apt update && apt install ffmpeg -y
```

## Usages

### Usage 1: Launch the FastAPI demo
```bash
python demo/vibevoice_asr_streaming_fastapi_demo.py --model_path [add the checkpoint path here]
```

Open `http://localhost:7870`, then record from the microphone or pick a file. The page keeps a WebSocket open for the whole recording, so text appears while you are still speaking.

### Usage 2: Inference from files directly
```bash
python demo/vibevoice_asr_streaming_inference_from_file.py --model_path [add the checkpoint path here] --audio_files [add an audio path here]
```

Each chunk is printed as soon as the model emits it. Add `--context_info "Microsoft,VibeVoice"` to bias recognition toward specific terms, the same way hotwords work on the non-streaming model.

The chunk and lookahead are read from the checkpoint's `preprocessor_config.json`, so a checkpoint always runs at the chunk it was trained on.

## 📄 License

This project is licensed under the [MIT License](../LICENSE).
