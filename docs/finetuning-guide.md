# VibeVoice-ASR Fine-tuning Guide

> **Prerequisites**: Python 3.10+, CUDA 11.8+, 24GB+ VRAM GPU

## Installation

```bash
pip install -e .
pip install peft transformers torch torchaudio
```

## Data Preparation

Organize audio files and JSON labels in the same directory:

```
your_dataset/
├── audio_001.mp3
├── audio_001.json
├── audio_002.mp3
├── audio_002.json
└── ...
```

### JSON Label Format

```json
{
  "audio_duration": 351.73,
  "audio_path": "audio_001.mp3",
  "segments": [
    {
      "speaker": 0,
      "text": "Hey everyone, welcome back to the show.",
      "start": 0.0,
      "end": 38.68
    },
    {
      "speaker": 1,
      "text": "Thanks for having me. Today we're discussing AI.",
      "start": 38.75,
      "end": 77.88
    }
  ],
  "customized_context": ["Tea Brew", "Aiden Host"]  // optional
}
```

- `customized_context` is optional; use it for domain-specific terms, names, or product names to boost recognition accuracy.

## Training (Single GPU)

```bash
torchrun --nproc_per_node=1 finetune.py \
    --model_path microsoft/VibeVoice-ASR \
    --data_dir ./your_dataset \
    --output_dir ./output \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --bf16 \
    --report_to none
```

## Training (Multi-GPU)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 finetune.py \
    --model_path microsoft/VibeVoice-ASR \
    --data_dir ./your_dataset \
    --output_dir ./output \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --bf16 \
    --report_to none
```

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lora_r` | 16 | LoRA rank (higher = more expressive, more VRAM) |
| `--lora_alpha` | 32 | LoRA scaling (typically 2× rank) |
| `--lora_dropout` | 0.05 | Dropout for LoRA layers |
| `--per_device_train_batch_size` | 8 | Batch size per GPU |
| `--gradient_accumulation_steps` | 1 | Effective batch = batch × grad_accum |
| `--learning_rate` | 1e-4 | 1e-4 to 2e-4 is typical for LoRA |
| `--gradient_checkpointing` | false | Enable to reduce VRAM usage |
| `--max_audio_length` | none | Skip audio longer than N seconds |

## Inference with Fine-tuned LoRA

```bash
python inference_lora.py \
    --base_model microsoft/VibeVoice-ASR \
    --lora_path ./output \
    --audio_file ./your_dataset/audio_001.mp3 \
    --context_info "Tea Brew, Aiden Host"
```

## Merging LoRA into Base Model

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("microsoft/VibeVoice-ASR")
model = PeftModel.from_pretrained(base, "./output")
merged = model.merge_and_unload()
merged.save_pretrained("./merged_model")
```

Then use `./merged_model` with the standard inference pipeline — no LoRA loader needed.

## Running in Google Colab

See the official Colab notebook:

- **VibeVoice Realtime Demo**: [`demo/vibevoice_realtime_colab.ipynb`](https://github.com/microsoft/VibeVoice/blob/main/demo/vibevoice_realtime_colab.ipynb)

## GPU Memory Guide

| Setup | VRAM | Notes |
|-------|------|-------|
| 1 GPU (bf16, no gradient checkpointing) | ~24 GB | Default settings |
| 1 GPU (bf16 + gradient checkpointing) | ~16 GB | Use `--gradient_checkpointing` |
| 4 GPU (tensor parallelism) | ~8 GB per GPU | Use `--nproc_per_node=4` |

## Common Issues

**CUDA OOM**: Reduce `--per_device_train_batch_size` to 1, enable `--gradient_checkpointing`, or use multi-GPU.

**Poor transcription quality**: Ensure your training JSON has accurate timestamps and speaker labels. Add domain-specific terms via `customized_context`.

**Language detection issues**: VibeVoice-ASR auto-detects language; no explicit setting needed. Ensure audio files are >1 second.