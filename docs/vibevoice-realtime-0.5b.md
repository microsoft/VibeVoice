<div align="center">

## 🎙️ VibeVoice-Realtime: Real-time Long‑Form Text‑to‑Speech with Streaming Input
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Collection-orange?logo=huggingface)](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B)
[![Colab](https://img.shields.io/badge/Run-Colab-orange?logo=googlecolab)](https://colab.research.google.com/github/microsoft/VibeVoice/blob/main/demo/vibevoice_realtime_colab.ipynb)
</div>

VibeVoice-Realtime is a **lightweight real‑time** text-to-speech model supporting **streaming text input** and **robust long-form speech generation**. It can be used to build real-time TTS services, narrate live data streams, and let different LLMs start speaking from their very first tokens (plug in your preferred model) long before a full answer is generated. It produces initial audible speech in **~300 milliseconds** (hardware dependent).

<div align="center">

| Model | Context Length | Generation Length |  Weight |
|-------|----------------|----------|----------|
| VibeVoice-Realtime-0.5B | 8K | ~10 min | [HF link](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B) |

</div>

The model uses an interleaved, windowed design: it incrementally encodes incoming text chunks while, in parallel, continuing diffusion-based acoustic latent generation from prior context. Unlike the full multi-speaker long-form variants, this streaming model removes the semantic tokenizer and relies solely on an efficient acoustic tokenizer operating at an ultra-low frame rate (7.5 Hz).

<div align="center">
	<picture>
		<source media="(prefers-color-scheme: dark)" srcset="../Figures/VibeVoice_logo_white.png">
		<img src="../Figures/VibeVoice_Realtime.png" alt="VibeVoice Realtime Overview" width="800" />
	</picture>
	<br>
	<em>Overview of VibeVoice Realtime Model.</em>
</div>

Key features:
- Parameter size: 0.5B (deployment-friendly)
- Real-time TTS (~300 milliseconds first audible latency)
- Streaming text input
- Robust long-form speech generation

This real-time variant supports only a single speaker. For multi‑speaker conversational speech generation, please use other VibeVoice models (long‑form multi‑speaker variants). The model is currently intended for English speech only; other languages may produce unpredictable results.

To mitigate deepfake risks and ensure low latency for the first speech chunk, voice prompts are provided in an embedded format. For users requiring voice customization, please reach out to our team. We will also be expanding the range of available speakers.


### 📋 TODO

- [ ] Add more voices (expand available speakers/voice timbres)
- [ ] Implement streaming text input function to feed new tokens while audio is still being generated
- [ ] Merge models into official HuggingFace's `transformers` repository 


### 🎵 Demo Examples

<div align="center" id="generated-example-audio-vibevoice-realtime">

https://github.com/user-attachments/assets/9aa8ab3c-681d-4a02-b9ea-3f54ffd180b2

</div>


## Results

The model achieves satisfactory performance on short-sentence benchmarks, while the model is more focused on long‑form speech generation.

### Zero-shot TTS performance on LibriSpeech test-clean set

| Model | WER (%) ↓ | Speaker Similarity ↑ |
|:--------------------|:---------:|:----------------:|
| VALL-E 2            | 2.40      | 0.643            |
| Voicebox            | 1.90      | 0.662            |
| MELLE               | 2.10      | 0.625            |
| **VibeVoice-Realtime-0.5B** | 2.00 | 0.695            |

### Zero-shot TTS performance on SEED test-en set

| Model | WER (%) ↓ | Speaker Similarity ↑ |
|:--------------------|:---------:|:----------------:|
| MaskGCT             | 2.62      | 0.714            |
| Seed-TTS            | 2.25      | 0.762            |
| FireRedTTS          | 3.82      | 0.460            |
| SparkTTS            | 1.98      | 0.584            |
| CosyVoice2          | 2.57      | 0.652            |
| **VibeVoice-Realtime-0.5B** | 2.05 | 0.633            | 


## Installation
We recommend to use NVIDIA Deep Learning Container to manage the CUDA environment. 

1. Launch docker
```bash
# NVIDIA PyTorch Container 24.07 / 24.10 / 24.12 verified. 
# Later versions are also compatible.
sudo docker run --privileged --net=host --ipc=host --ulimit memlock=-1:-1 --ulimit stack=-1:-1 --gpus all --rm -it  nvcr.io/nvidia/pytorch:24.07-py3

## If flash attention is not included in your docker environment, you need to install it manually
## Refer to https://github.com/Dao-AILab/flash-attention for installation instructions
# pip install flash-attn --no-build-isolation
```

2. Install from github
```bash
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice/

pip install -e .
```

## Usages


### Usage 1: Launch real-time websocket demo
Note: NVIDIA T4 / Mac M4 Pro achieve realtime in our tests; other devices with weaker inference capability may require further testing and speed optimizations. 

Due to network latency, the time when audio playback is heard may exceed the ~300 ms first speech chunk generation latency.
```bash
python demo/vibevoice_realtime_demo.py --model_path microsoft/VibeVoice-Realtime-0.5B
```

Tip: Just try it on [Colab](https://colab.research.google.com/github/microsoft/VibeVoice/blob/main/demo/vibevoice_realtime_colab.ipynb).

### Usage 2: Inference from files directly
```bash
# We provide some example scripts under demo/text_examples/ for demo
python demo/realtime_model_inference_from_file.py --model_path microsoft/VibeVoice-Realtime-0.5B --txt_path demo/text_examples/1p_vibevoice.txt --speaker_name Carter
```


## Risks and limitations

While efforts have been made to optimize it through various techniques, it may still produce outputs that are unexpected, biased, or inaccurate. VibeVoice inherits any biases, errors, or omissions produced by its base model (specifically, Qwen2.5 0.5b in this release).

Potential for Deepfakes and Disinformation: High-quality synthetic speech can be misused to create convincing fake audio content for impersonation, fraud, or spreading disinformation. Users must ensure transcripts are reliable, check content accuracy, and avoid using generated content in misleading ways. Users are expected to use the generated content and to deploy the models in a lawful manner, in full compliance with all applicable laws and regulations in the relevant jurisdictions. It is best practice to disclose the use of AI when sharing AI-generated content.

English only: Transcripts in languages other than English may result in unexpected audio outputs.

Non-Speech Audio: The model focuses solely on speech synthesis and does not handle background noise, music, or other sound effects.

Code, formulas, and special symbols: The model does not currently support reading code, mathematical formulas, or uncommon symbols. Please pre‑process input text to remove or normalize such content to avoid unpredictable results. 

Very short inputs: When the input text is extremely short (three words or fewer), the model’s stability may degrade.

We do not recommend using VibeVoice in commercial or real-world applications without further testing and development. This model is intended for research and development purposes only. Please use responsibly.
