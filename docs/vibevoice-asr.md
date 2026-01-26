# VibeVoice-ASR

[![Hugging Face](https://img.shields.io/badge/HuggingFace-Collection-orange?logo=huggingface)](https://huggingface.co/microsoft/VibeVoice-ASR)
[![Live Playground](https://img.shields.io/badge/Live-Playground-green?logo=gradio)](https://aka.ms/vibevoice-asr)

**VibeVoice-ASR** is a unified speech-to-text model designed to handle **60-minute long-form audio** in a single pass, generating structured transcriptions containing **Who (Speaker), When (Timestamps), and What (Content)**, with support for **Customized Hotwords**.

**Model:** [VibeVoice-ASR-7B](https://huggingface.co/microsoft/VibeVoice-ASR)<br>
**Demo:** [VibeVoice-ASR-Demo](https://aka.ms/vibevoice-asr)<br>
**Report:** [VibeVoice-ASR-Report](./VibeVoice-ASR-Report.pdf)<br>
**Finetuning:** [finetune-guide](../finetuning-asr/README.md)<br>
**vllm:** [vllm-asr](./vibevoice-vllm-asr.md)<br>


## 🔥 Key Features

- **🕒 60-minute Single-Pass Processing**:
  Unlike conventional ASR models that slice audio into short chunks (often losing global context), VibeVoice ASR accepts up to **60 minutes** of continuous audio input within 64K token length. This ensures consistent speaker tracking and semantic coherence across the entire hour.

- **👤 Customized Hotwords**:
  Users can provide customized hotwords (e.g., specific names, technical terms, or background info) to guide the recognition process, significantly improving accuracy on domain-specific content.

- **📝 Rich Transcription (Who, When, What)**:
  The model jointly performs ASR, diarization, and timestamping, producing a structured output that indicates *who* said *what* and *when*.


## 🏗️ Model Architecture

<p align="center">
  <img src="../Figures/VibeVoice_ASR_archi.png" alt="VibeVoice ASR Architecture" width="80%">
</p>

# Demo

<div align="center" id="vibevoice-asr">

https://github.com/user-attachments/assets/acde5602-dc17-4314-9e3b-c630bc84aefa

</div>

## Evaluation
<p align="center">
  <img src="../Figures/DER.jpg" alt="DER" width="50%"><br>
  <img src="../Figures/cpWER.jpg" alt="cpWER" width="50%"><br>
  <img src="../Figures/tcpWER.jpg" alt="tcpWER" width="50%">
</p>



## Installation
We recommend to use NVIDIA Deep Learning Container to manage the CUDA environment. 

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

## Usages

### Usage 1: Launch Gradio demo
```bash
apt update && apt install ffmpeg -y # for demo

python demo/vibevoice_asr_gradio_demo.py --model_path microsoft/VibeVoice-ASR --share
```

### Usage 2: Inference from files directly
```bash
python demo/vibevoice_asr_inference_from_file.py --model_path microsoft/VibeVoice-ASR --audio_files [add a audio path here] 
```

## Results

### Multilingual
| Dataset        | Language  | DER  | cpWER | tcpWER | WER  |
|----------------|-----------|------|-------|--------|------|
| MLC-Challenge  | English   | 4.28 | 11.48 | 13.02  | 7.99  |
| MLC-Challenge  | French    | 3.80 | 18.80 | 19.64  | 15.21 |
| MLC-Challenge  | German    | 1.04 | 17.10 | 17.26  | 16.30 |
| MLC-Challenge  | Italian   | 2.08 | 15.76 | 15.91  | 13.91 |
| MLC-Challenge  | Japanese  | 0.82 | 15.33 | 15.41  | 14.69 |
| MLC-Challenge  | Korean    | 4.52 | 15.35 | 16.07  | 9.65  |
| MLC-Challenge  | Portuguese| 7.98 | 29.91 | 31.65  | 21.54 |
| MLC-Challenge  | Russian   | 0.90 | 12.94 | 12.98  | 12.40 |
| MLC-Challenge  | Spanish   | 2.67 | 10.51 | 11.71  | 8.04  |
| MLC-Challenge  | Thai      | 4.09 | 14.91 | 15.57  | 13.61 |
| MLC-Challenge  | Vietnamese| 0.16 | 14.57 | 14.57  | 14.43 |

---

| Dataset        | Language  | DER  | cpWER | tcpWER | WER  |
|----------------|-----------|------|-------|--------|------|
| AISHELL-4      | Chinese   | 6.77 | 24.99 | 25.35  | 21.40 |
| AMI-IHM        | English   | 11.92| 20.41 | 20.82  | 18.81 |
| AMI-SDM        | English   | 13.43| 28.82 | 29.80  | 24.65 |
| AliMeeting     | Chinese   | 10.92| 29.33 | 29.51  | 27.40 |
| MLC-Challenge  | Average   | 3.42 | 14.81 | 15.66  | 12.07|


## Finetuning
LoRA (Low-Rank Adaptation) fine-tuning is supported. See [Finetuning](../finetuning-asr/README.md) for detailed guide.


## 📄 License

This project is licensed under the [MIT License](../LICENSE).


## Language Distribution
<p align="center">
  <img src="../Figures/language_distribution_horizontal.png" alt="Language Distribution" width="80%">
</p>


