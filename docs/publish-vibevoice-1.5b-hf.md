# Publishing VibeVoice 1.5B for Transformers

This release path creates `microsoft/VibeVoice-1.5B-HF`, the official
Transformers-native artifact for the original TTS checkpoint. The original
[`microsoft/VibeVoice-1.5B`](https://huggingface.co/microsoft/VibeVoice-1.5B)
remains the legacy source/provenance repository; link it to `-HF` after the
owner publication rather than replacing its main branch.

## Pinned inputs

| Input | Pinned revision | Use |
| --- | --- | --- |
| [`microsoft/VibeVoice-1.5B`](https://huggingface.co/microsoft/VibeVoice-1.5B) | `c00898d257e6b46004e3e2866a47534085fb685a` | Official weight source |
| [Transformers](https://github.com/huggingface/transformers) | `640a08a597034221ca1c4fc0c129cf0118179225` | Canonical VibeVoice converter and native implementation |
| [`Qwen/Qwen2.5-1.5B`](https://huggingface.co/Qwen/Qwen2.5-1.5B) | `8faed761d45a263340a0528343f099c05c9a4323` | Canonical tokenizer input |
| [`vibevoice/VibeVoice-1.5B-hf`](https://huggingface.co/vibevoice/VibeVoice-1.5B-hf) | `edc39f80f5cae656da37baf8faa8f5502bf7081f` | Independent 1,204-key layout evidence only |

The Transformers commit contains `VibeVoiceForConditionalGeneration`,
`VibeVoiceProcessor`, `AutoModelForTextToWaveform` registration, and the
canonical `convert_vibevoice_to_hf.py` converter. It postdates the v5.16.1
release, so use the pinned source commit rather than a released package.

The sidecar reference above is never downloaded by the converter or used at
runtime. The converter downloads only the official Microsoft checkpoint and
the pinned Qwen tokenizer, invokes the canonical Transformers converter, and
fails unless all 1,204 source-to-native keys, shapes, and dtypes agree.

## Owner conversion and publication

Run this in a clean Python environment with sufficient disk and memory for the
multi-GB checkpoint:

```bash
git clone https://github.com/huggingface/transformers.git /tmp/transformers
git -C /tmp/transformers checkout 640a08a597034221ca1c4fc0c129cf0118179225
python -m pip install -e "/tmp/transformers[torch]"

python tools/release/convert_vibevoice_1_5b_hf.py \
  --transformers-source /tmp/transformers \
  --output-dir /tmp/VibeVoice-1.5B-HF
```

The output includes native safetensor shards, config, generation config,
tokenizer, processor, chat template, model card, and
`conversion-manifest.json`. The manifest records every pinned input and the
strict tensor-alignment digest, including the clean VibeVoice release-tool
commit. The script has no upload option.

Only an owner authorized for the Microsoft Hugging Face namespace should
publish the reviewed local artifact:

```bash
huggingface-cli upload microsoft/VibeVoice-1.5B-HF /tmp/VibeVoice-1.5B-HF . \
  --commit-message "Publish Transformers-native VibeVoice 1.5B"
```

## Acceptance

The unit checks do not download model weights:

```bash
python -m unittest discover -s tests -p 'test_publish_vibevoice_1_5b_hf.py' -v
```

The conversion command performs the optional real-weight validation. After the
owner publishes, this native-only load confirms the public artifact has no
remote code or sidecar dependency:

```python
from transformers import AutoModelForTextToWaveform, AutoProcessor

model_id = "microsoft/VibeVoice-1.5B-HF"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=False)
model = AutoModelForTextToWaveform.from_pretrained(
    model_id,
    dtype="auto",
    trust_remote_code=False,
)

assert processor.__class__.__name__ == "VibeVoiceProcessor"
assert model.__class__.__name__ == "VibeVoiceForConditionalGeneration"
```
