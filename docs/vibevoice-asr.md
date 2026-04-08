# VibeVoice-ASR

[![Hugging Face](https://img.shields.io/badge/HuggingFace-Collection-orange?logo=huggingface)](https://huggingface.co/microsoft/VibeVoice-ASR)
[![Live Playground](https://img.shields.io/badge/Live-Playground-green?logo=gradio)](https://aka.ms/vibevoice-asr)

**VibeVoice-ASR** is a unified speech-to-text model designed to handle **60-minute long-form audio** in a single pass, generating structured transcriptions containing **Who (Speaker), When (Timestamps), and What (Content)**, with support for **Customized Hotwords** and over **50 languages**.

**Model:** [VibeVoice-ASR-7B](https://huggingface.co/microsoft/VibeVoice-ASR)<br>
**Demo:** [VibeVoice-ASR-Demo](https://aka.ms/vibevoice-asr)<br>
**Report:** [VibeVoice-ASR-Report](https://arxiv.org/pdf/2601.18184)<br>
**Finetuning:** [finetune-guide](../finetuning-asr/README.md)<br>
**vLLM:** [vLLM-asr](./vibevoice-vllm-asr.md)<br>


## 🔥 Key Features

- **🕒 60-minute Single-Pass Processing**:
  Unlike conventional ASR models that slice audio into short chunks (often losing global context), VibeVoice ASR accepts up to **60 minutes** of continuous audio input within 64K token length. This ensures consistent speaker tracking and semantic coherence across the entire hour.

- **👤 Customized Hotwords**:
  Users can provide customized hotwords (e.g., specific names, technical terms, or background info) to guide the recognition process, significantly improving accuracy on domain-specific content.

- **📝 Rich Transcription (Who, When, What)**:
  The model jointly performs ASR, diarization, and timestamping, producing a structured output that indicates *who* said *what* and *when*.
  
- **🌍 Multilingual & Code-Switching Support**:
  It supports over 50 languages, requires no explicit language setting, and natively handles code-switching within and across utterances. Language distribution can be found [here](#language-distribution).


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

### Usage 3: Python API — direct inference with hotwords

For programmatic integration, you can use `VibeVoiceASRProcessor` and `VibeVoiceASRForConditionalGeneration` directly:

```python
import torch
from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

model_path = "microsoft/VibeVoice-ASR"

# Load model and processor
model = VibeVoiceASRForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = VibeVoiceASRProcessor.from_pretrained(model_path)

# --- Basic transcription ---
inputs = processor(
    audio="path/to/audio.wav",
    return_tensors="pt",
).to(model.device)

with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=32768)

transcription = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
print(transcription)

# --- With customized hotwords ---
# Hotwords improve recognition of proper nouns, technical terms, or speaker names.
# Pass them as a comma-separated string via context_info.
inputs = processor(
    audio="path/to/audio.wav",
    context_info="Microsoft, Azure, VibeVoice",  # domain-specific terms
    return_tensors="pt",
).to(model.device)

with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=32768)

transcription = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
```

### Output format

The model produces a JSON array where each element represents one speech segment, containing:

- **`Start time`** — segment start time in seconds
- **`End time`** — segment end time in seconds
- **`Speaker ID`** — speaker identifier (e.g., `"SPEAKER_00"`, `"SPEAKER_01"`)
- **`Content`** — transcribed text for this segment

Example output for a two-speaker recording:

```json
[
  {"Start time": 0.0, "End time": 12.5, "Speaker ID": "SPEAKER_00", "Content": "Welcome to our podcast. Today we're discussing AI research."},
  {"Start time": 12.8, "End time": 25.3, "Speaker ID": "SPEAKER_01", "Content": "Thanks for having me. I've been working on speech models for five years."},
  {"Start time": 25.5, "End time": 38.0, "Speaker ID": "SPEAKER_00", "Content": "Let's start with the basics. How does automatic speech recognition work?"}
]
```

The `post_process_transcription` helper on the processor can parse this JSON into a list of Python dicts:

```python
segments = processor.post_process_transcription(transcription)
for seg in segments:
    print(f"[{seg['start_time']:.1f}s – {seg['end_time']:.1f}s] {seg['speaker_id']}: {seg['text']}")
```


## Finetuning
LoRA (Low-Rank Adaptation) fine-tuning is supported. See [Finetuning](../finetuning-asr/README.md) for detailed guide.



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


## Language Distribution
<p align="center">
  <img src="../Figures/language_distribution_horizontal.png" alt="Language Distribution" width="80%">
</p>

## 📄 License

This project is licensed under the [MIT License](../LICENSE).



