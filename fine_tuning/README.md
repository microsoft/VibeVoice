# LLM Fine-Tuning for VEMI AI Voice Agents

## Overview
Fine-tune Qwen2.5-1.5B-Instruct for domain-specific voice agents:
- **Medical Agent**: Healthcare Q&A, symptoms, treatments, medications
- **Automobile Agent**: Car repair, diagnostics, maintenance, specifications

## Recommended Setup

### Hardware Requirements
| GPU | VRAM | Training Time | Cost/Hour |
|-----|------|---------------|-----------|
| A40 | 48GB | ~4-6 hours | $0.39 |
| A100 | 80GB | ~2-3 hours | $1.89 |
| H100 | 80GB | ~1-2 hours | $3.89 |

**Recommendation**: A40 for budget, A100 for balanced speed/cost

### Dataset Sizes
- **Medical**: ~15,000-20,000 Q&A pairs from curated datasets
- **Automobile**: ~10,000-15,000 Q&A pairs (hybrid: real + synthetic)

## Quick Start on RunPod

```bash
# 1. Create a GPU Pod (A40 or A100 recommended)
# 2. SSH into the pod and run:

cd /workspace
git clone https://github.com/your-repo/VibeVoice.git
cd VibeVoice/fine_tuning

# Install dependencies
pip install -r requirements.txt

# Prepare datasets
python prepare_data.py --domain medical
python prepare_data.py --domain automobile

# Fine-tune (choose one)
python train_medical.py   # For medical agent
python train_automobile.py  # For automobile agent

# Or train both
python train_all.py
```

## Directory Structure
```
fine_tuning/
├── README.md
├── requirements.txt
├── prepare_data.py          # Data preparation script
├── train_medical.py         # Medical fine-tuning
├── train_automobile.py      # Automobile fine-tuning
├── train_all.py             # Train both models
├── data/
│   ├── medical/             # Medical datasets
│   └── automobile/          # Automobile datasets
├── models/
│   ├── medical/             # Fine-tuned medical model
│   └── automobile/          # Fine-tuned automobile model
└── prompts/
    ├── medical_system.txt   # Medical system prompt
    └── automobile_system.txt # Automobile system prompt
```

## Datasets Used

### Medical (From lavita/medical-qa-datasets)
1. **MedQuAD** - 47K medical Q&A pairs
2. **PubMedQA** - Research-based medical QA
3. **MedMCQA** - Medical exam questions
4. **HealthCareMagic** - Patient-doctor conversations

### Automobile (Hybrid Approach)
1. **AutoAIQnA** - Kaggle automotive Q&A
2. **Synthetic from Car Specs** - Generated from vehicle databases
3. **Reddit r/MechanicAdvice** - Community Q&A (curated)
4. **OBD-II Diagnostic Codes** - Standardized codes Q&A

## Training Configuration (LoRA)
```python
lora_config = {
    "r": 16,                    # LoRA rank
    "lora_alpha": 32,           # Scaling factor
    "lora_dropout": 0.05,       # Dropout
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

training_args = {
    "learning_rate": 2e-4,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "warmup_ratio": 0.03,
    "max_seq_length": 2048,
}
```

## After Training
1. Models saved to `models/medical/` and `models/automobile/`
2. Copy to main project: `cp -r models/ ../speech_to_speech/`
3. Update `s2s_pipeline.py` to load domain-specific models
4. Restart the S2S server

## Cost Estimate
- **A40**: ~$2-3 total (4-6 hours × $0.39/hr)
- **A100**: ~$4-6 total (2-3 hours × $1.89/hr)
- **Data Storage**: Minimal (~500MB)
