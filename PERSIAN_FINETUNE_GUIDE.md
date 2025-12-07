# راهنمای جامع Fine-tuning مدل VibeVoice-Realtime بر روی دیتاست فارسی

این سند حاوی تمام اطلاعات استخراج شده از کدبیس VibeVoice است که برای fine-tuning بر روی دیتاست فارسی نیاز دارید.

---

## ۱. پیدا کردن مدل Realtime و کلاس اصلی

### کلاس‌های اصلی

#### الف) کلاس مدل پایه
```python
from vibevoice.modular.modeling_vibevoice_streaming import VibeVoiceStreamingModel
```

- **مسیر فایل**: `/vibevoice/modular/modeling_vibevoice_streaming.py`
- **توضیح**: این کلاس شامل معماری اصلی مدل است (بدون generation logic)

#### ب) کلاس مدل Inference
```python
from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
```

- **مسیر فایل**: `/vibevoice/modular/modeling_vibevoice_streaming_inference.py`
- **توضیح**: این کلاس شامل منطق generation و inference است (برای استفاده واقعی)

### Configuration
```python
from vibevoice.modular.configuration_vibevoice_streaming import VibeVoiceStreamingConfig
```

#### پارامترهای مهم Configuration:
- `acoustic_vae_dim`: 64 (ابعاد latent space برای acoustic tokens)
- `tts_backbone_num_hidden_layers`: 20 (تعداد لایه‌های بالایی برای TTS)
- `decoder_config`: Qwen2Config (configuration مدل زبانی پایه)
- `diffusion_head_config`: VibeVoiceDiffusionHeadConfig

---

## ۲. شناسایی Submodule‌های آکوستیک/صوتی

مدل `VibeVoiceStreamingModel` دارای کامپوننت‌های زیر است:

### الف) Language Models

```python
# Lower Transformer layers (فقط برای text encoding)
self.language_model = AutoModel.from_config(lm_config)
# تعداد لایه: num_hidden_layers - tts_backbone_num_hidden_layers

# Upper Transformer layers (برای TTS generation)
self.tts_language_model = AutoModel.from_config(tts_lm_config)
# تعداد لایه: tts_backbone_num_hidden_layers (معمولاً 20)
```

**نکته مهم**:
- `language_model.norm` به `nn.Identity()` تبدیل شده (استفاده نمی‌شود)
- هر دو مدل از Qwen2 استفاده می‌کنند

### ب) Acoustic Components

```python
# 1. Acoustic Tokenizer (Decoder)
self.acoustic_tokenizer = AutoModel.from_config(config.acoustic_tokenizer_config)
# کلاس: VibeVoiceAcousticTokenizerModel
# نقش: تبدیل latent representations به waveform

# 2. Acoustic Connector
self.acoustic_connector = SpeechConnector(
    config.acoustic_vae_dim,  # 64
    lm_config.hidden_size
)
# نقش: تبدیل acoustic features به hidden states مدل زبانی
```

### ج) Diffusion Head

```python
# Prediction Head (Diffusion Model)
self.prediction_head = AutoModel.from_config(config.diffusion_head_config)
# کلاس: VibeVoiceDiffusionHead
# نقش: تولید acoustic latents با استفاده از diffusion process

# Noise Scheduler
self.noise_scheduler = DPMSolverMultistepScheduler(
    num_train_timesteps=config.diffusion_head_config.ddpm_num_steps,  # 1000
    beta_schedule=config.diffusion_head_config.ddpm_beta_schedule,    # "cosine"
    prediction_type=config.diffusion_head_config.prediction_type      # "v_prediction"
)
```

### د) سایر کامپوننت‌ها

```python
# TTS Input Type Embeddings
self.tts_input_types = nn.Embedding(
    num_embeddings=2,
    embedding_dim=config.decoder_config.hidden_size
)

# Scaling factors (برای normalization)
self.speech_scaling_factor = torch.tensor(float('nan'))  # buffer
self.speech_bias_factor = torch.tensor(float('nan'))     # buffer
```

---

## ۳. پیدا کردن Tokenizer/Processor رسمی برای متن

### کلاس Processor

```python
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
```

**مسیر فایل**: `/vibevoice/processor/vibevoice_streaming_processor.py`

### استفاده

```python
# بارگذاری
processor = VibeVoiceStreamingProcessor.from_pretrained(
    "microsoft/VibeVoice-Realtime-0.5B"
)

# پردازش ورودی
inputs = processor.process_input_with_cached_prompt(
    text="متن شما اینجا",
    cached_prompt=all_prefilled_outputs,  # voice prompt embedding
    padding=True,
    return_tensors="pt",
    return_attention_mask=True,
)
```

### Tokenizer داخلی

Processor از یکی از این tokenizer ها استفاده می‌کند:

```python
from vibevoice.modular.modular_vibevoice_text_tokenizer import (
    VibeVoiceTextTokenizer,       # نسخه معمولی
    VibeVoiceTextTokenizerFast    # نسخه سریع
)
```

**پیش‌فرض**: Tokenizer از `Qwen/Qwen2.5-1.5B` بارگذاری می‌شود

### خروجی Processor

```python
# inputs شامل:
{
    'input_ids': torch.LongTensor,           # token IDs برای language_model
    'attention_mask': torch.LongTensor,
    'tts_lm_input_ids': torch.LongTensor,   # token IDs برای tts_language_model
    'tts_lm_attention_mask': torch.LongTensor,
    'tts_text_ids': torch.LongTensor,       # متن اصلی برای TTS
    'speech_input_mask': torch.BoolTensor,  # ماسک برای speech tokens
}
```

---

## ۴. کشف مسیر Audio → Acoustic Tokens

### Acoustic Tokenizer Model

```python
from vibevoice.modular.modular_vibevoice_tokenizer import VibeVoiceAcousticTokenizerModel
```

**مسیر فایل**: `/vibevoice/modular/modular_vibevoice_tokenizer.py`

### Configuration

```python
from vibevoice.modular.configuration_vibevoice import VibeVoiceAcousticTokenizerConfig
```

**پارامترهای کلیدی**:
- `vae_dim`: 64 (ابعاد latent space)
- `causal`: True
- `encoder_ratios`: [8, 5, 5, 4, 2, 2] (نرخ کاهش در encoder)
- `decoder_ratios`: [8, 5, 5, 4, 2, 2] (نرخ افزایش در decoder)
- `channels`: 1 (mono audio)

### استفاده در Inference

```python
# Decode: latents → waveform
audio = model.model.acoustic_tokenizer.decode(
    latents,           # shape: [batch, vae_dim, time] یا [batch, time, vae_dim]
    cache=cache,
    sample_indices=sample_indices,
    use_cache=use_cache
)
```

**نکته مهم**: این مدل فقط **decoder** دارد (نه encoder). در training، شما باید:
1. یک encoder جداگانه برای تبدیل audio به acoustic latents داشته باشید
2. یا از pre-computed latents استفاده کنید

### Compression Ratio

```python
# از processor:
speech_tok_compress_ratio = 3200  # پیش‌فرض

# به این معنی که:
# اگر audio 24kHz باشد:
# 24000 samples/sec ÷ 3200 = 7.5 tokens/sec
# این همان "ultra-low frame rate of 7.5 Hz" است که در documentation ذکر شده
```

---

## ۵. پیدا کردن Training Code و Loss Functions

### ⚠️ نکته مهم

**کد training رسمی در این ریپوزیتوری موجود نیست**. این ریپو فقط شامل کد inference است.

با این حال، می‌توانیم از کد inference، منطق training را استنباط کنیم:

### الف) Diffusion Loss (استنباط شده)

از تابع `sample_speech_tokens` در inference:

```python
def sample_speech_tokens(self, condition, neg_condition, cfg_scale=3.0):
    """
    نمونه‌برداری از acoustic tokens با استفاده از diffusion

    Args:
        condition: hidden states از TTS LM (شرط مثبت)
        neg_condition: hidden states برای unconditional (شرط منفی)
        cfg_scale: مقیاس classifier-free guidance
    """
    self.model.noise_scheduler.set_timesteps(self.ddpm_inference_steps)
    condition = torch.cat([condition, neg_condition], dim=0)

    # شروع از نویز تصادفی
    speech = torch.randn(condition.shape[0], self.config.acoustic_vae_dim)

    # حلقه denoising
    for t in self.model.noise_scheduler.timesteps:
        half = speech[: len(speech) // 2]
        combined = torch.cat([half, half], dim=0)

        # پیش‌بینی نویز
        eps = self.model.prediction_head(
            combined,
            t.repeat(combined.shape[0]),
            condition=condition
        )

        # Classifier-free guidance
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)

        # به‌روزرسانی
        speech = self.model.noise_scheduler.step(eps, t, speech).prev_sample

    return speech[: len(speech) // 2]
```

### ب) Training Loop (پیشنهادی)

بر اساس inference code، training loop باید شامل موارد زیر باشد:

```python
# 1. پردازش ورودی‌ها
text_inputs = processor.tokenizer(text, return_tensors="pt")
# فرض: شما acoustic latents از قبل دارید یا با encoder جداگانه‌ای می‌سازید
target_acoustic_latents = your_acoustic_encoder(audio)  # shape: [B, T, 64]

# 2. گذر از language models
lm_outputs = model.model.language_model(
    input_ids=text_inputs['input_ids'],
    attention_mask=text_inputs['attention_mask'],
)
hidden_states = lm_outputs.last_hidden_state

tts_lm_outputs = model.model.tts_language_model(
    inputs_embeds=hidden_states,  # یا ترکیب با speech embeddings
    attention_mask=...,
)
condition = tts_lm_outputs.last_hidden_state  # [B, T, hidden_size]

# 3. Diffusion training (v-prediction)
# نمونه‌برداری timestep تصادفی
timesteps = torch.randint(
    0,
    model.config.diffusion_head_config.ddpm_num_steps,
    (batch_size,)
)

# اضافه کردن نویز به target
noise = torch.randn_like(target_acoustic_latents)
noisy_latents = model.noise_scheduler.add_noise(
    target_acoustic_latents,
    noise,
    timesteps
)

# پیش‌بینی با diffusion head
predicted = model.model.prediction_head(
    noisy_latents,
    timesteps,
    condition=condition
)

# محاسبه loss
if model.config.diffusion_head_config.prediction_type == "v_prediction":
    # v-prediction: مدل velocity را پیش‌بینی می‌کند
    velocity = model.noise_scheduler.get_velocity(
        target_acoustic_latents,
        noise,
        timesteps
    )
    loss = F.mse_loss(predicted, velocity)
elif model.config.diffusion_head_config.prediction_type == "epsilon":
    # epsilon-prediction: مدل نویز را پیش‌بینی می‌کند
    loss = F.mse_loss(predicted, noise)
else:
    raise ValueError(f"Unknown prediction type")

# 4. Backpropagation
loss.backward()
optimizer.step()
```

### ج) Additional Losses (اختیاری)

ممکن است بخواهید loss های اضافی اضافه کنید:

```python
# 1. EOS Classifier Loss
eos_logits = model.tts_eos_classifier(hidden_states)
eos_labels = ...  # True برای آخرین token، False برای بقیه
eos_loss = F.binary_cross_entropy_with_logits(eos_logits, eos_labels)

# 2. Reconstruction Loss (اگر acoustic encoder دارید)
reconstructed = model.model.acoustic_tokenizer.decode(predicted_latents)
recon_loss = F.l1_loss(reconstructed, target_audio)

# Total loss
total_loss = diffusion_loss + 0.1 * eos_loss + 0.01 * recon_loss
```

---

## ۶. تحلیل معماری و تعیین Freeze Strategy

### الف) تعداد پارامترها

مدل VibeVoice-Realtime-0.5B دارای حدود **500 میلیون** پارامتر است.

توزیع تقریبی:
- Language Model (Qwen2): ~70% پارامترها
- Acoustic Tokenizer Decoder: ~20% پارامترها
- Diffusion Head: ~5% پارامترها
- Connectors و سایر: ~5% پارامترها

### ب) استراتژی Freeze (پیشنهاد ۱: محافظه‌کارانه)

```python
for name, param in model.named_parameters():
    if any(k in name for k in [
        "model.language_model",           # Freeze: lower text encoder
        "model.acoustic_tokenizer",       # Freeze: audio decoder
    ]):
        param.requires_grad = False
    else:
        # Fine-tune:
        # - model.tts_language_model (upper TTS layers)
        # - model.acoustic_connector
        # - model.prediction_head (diffusion)
        # - model.tts_eos_classifier
        # - model.tts_input_types
        param.requires_grad = True
```

**مزایا**:
- حفظ دانش text encoding
- حفظ کیفیت audio decoder

**معایب**:
- ممکن است adaptation به فارسی کامل نباشد

### ج) استراتژی Freeze (پیشنهاد ۲: متعادل با LoRA)

```python
from peft import LoraConfig, get_peft_model

# Freeze همه چیز
for param in model.parameters():
    param.requires_grad = False

# اعمال LoRA فقط به لایه‌های خاص
lora_config = LoraConfig(
    r=16,                    # rank
    lora_alpha=32,
    target_modules=[
        # TTS language model attention layers
        "model.tts_language_model.layers.*.self_attn.q_proj",
        "model.tts_language_model.layers.*.self_attn.k_proj",
        "model.tts_language_model.layers.*.self_attn.v_proj",
        "model.tts_language_model.layers.*.self_attn.o_proj",
        # Diffusion head
        "model.prediction_head.*",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Fine-tune connectors به صورت full (خارج از LoRA)
for name, param in model.named_parameters():
    if "acoustic_connector" in name or "tts_eos_classifier" in name:
        param.requires_grad = True
```

**مزایا**:
- حافظه و سرعت بالاتر
- کاهش overfitting
- قابل ادغام با مدل اصلی

### د) استراتژی Freeze (پیشنهاد ۳: جسورانه)

```python
# Freeze فقط acoustic tokenizer
for name, param in model.named_parameters():
    if "model.acoustic_tokenizer" in name:
        param.requires_grad = False
    else:
        param.requires_grad = True
```

**مزایا**:
- بیشترین adaptation به فارسی
- یادگیری pronunciation patterns فارسی

**معایب**:
- نیاز به دیتاست بزرگ‌تر
- ریسک overfitting
- ریسک catastrophic forgetting

### ه) توصیه نهایی

**برای شروع**:
1. از استراتژی ۲ (LoRA) شروع کنید
2. با 2000-5000 step تست کنید
3. کیفیت audio را بررسی کنید
4. در صورت نیاز، به استراتژی ۱ یا ۳ بروید

---

## ۷. Hyperparameter ها

### الف) Learning Rate

```python
# برای LoRA
learning_rate = 1e-4  # یا 5e-5

# برای Full Fine-tuning
learning_rate = 5e-5  # یا 1e-5
```

### ب) Batch Size و Gradient Accumulation

```python
batch_size = 2              # یا 4، بسته به GPU
gradient_accumulation_steps = 4  # یا 8

# Effective batch size = batch_size × gradient_accumulation_steps
# مثال: 2 × 4 = 8
```

### ج) Audio Duration

```python
max_duration_seconds = 10   # برای شروع
# بعداً می‌توانید به 15 یا 20 افزایش دهید
```

### د) Training Steps

```python
# برای Proof of Concept
num_steps = 2000

# برای training جدی
num_steps = 10000  # یا بیشتر
```

### ه) Diffusion Parameters

```python
# در inference
ddpm_inference_steps = 5    # برای سرعت
# یا
ddpm_inference_steps = 20   # برای کیفیت

# در training
ddpm_num_steps = 1000       # تعداد timesteps (ثابت نگه دارید)
```

### و) CFG Scale

```python
cfg_scale = 1.5  # در inference
# مقدار بالاتر = پایبندی بیشتر به متن
# مقدار پایین‌تر = تنوع بیشتر
```

### ز) Optimizer

```python
optimizer = torch.optim.AdamW(
    trainable_params,
    lr=learning_rate,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01
)
```

### ح) Scheduler

```python
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=500,      # 10% از total steps
    num_training_steps=num_steps
)
```

---

## ۸. جمع‌بندی و Checklist نهایی

### ✅ قبل از Training

- [ ] ریپو را clone کرده‌اید
- [ ] اسکریپت `inspect_realtime.py` را اجرا کرده و ساختار مدل را بررسی کرده‌اید
- [ ] دیتاست فارسی خود را آماده کرده‌اید (متن + audio)
- [ ] acoustic encoder دارید یا pre-computed latents (یا روش دیگری برای تولید target)

### ✅ Import های لازم

```python
import torch
from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference
)
from vibevoice.processor.vibevoice_streaming_processor import (
    VibeVoiceStreamingProcessor
)
from peft import LoraConfig, get_peft_model  # اگر از LoRA استفاده می‌کنید
```

### ✅ بارگذاری مدل

```python
model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
    "microsoft/VibeVoice-Realtime-0.5B",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="flash_attention_2",
)
processor = VibeVoiceStreamingProcessor.from_pretrained(
    "microsoft/VibeVoice-Realtime-0.5B"
)
```

### ✅ اعمال Freeze Strategy

```python
# مثال: استراتژی ۱
for name, param in model.named_parameters():
    if "language_model" in name or "acoustic_tokenizer" in name:
        param.requires_grad = False
    else:
        param.requires_grad = True

# بررسی
trainable = [n for n, p in model.named_parameters() if p.requires_grad]
print(f"Trainable parameters: {len(trainable)}")
```

### ✅ ساخت Training Loop

```python
# TODO: پیاده‌سازی مطابق بخش ۵
```

### ✅ Inference و تست

```python
# بعد از training، تست کنید:
model.eval()
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=None,
        cfg_scale=1.5,
        tokenizer=processor.tokenizer,
    )
    processor.save_audio(outputs.speech_outputs[0], "output.wav")
```

### ✅ معیارهای ارزیابی

- گوش دادن به خروجی (مهم‌ترین!)
- WER (Word Error Rate) با یک ASR model
- Speaker Similarity (اگر voice cloning دارید)
- Naturalness MOS (Mean Opinion Score)

---

## ۹. نکات تکمیلی

### الف) Voice Prompts

مدل نیاز به **cached voice prompt** دارد (embedded format).

در دمو، از فایل‌های `.pt` در `demo/voices/streaming_model/` استفاده می‌شود:

```python
voice_sample = torch.load("demo/voices/streaming_model/Carter.pt")
all_prefilled_outputs = voice_sample

inputs = processor.process_input_with_cached_prompt(
    text=text,
    cached_prompt=all_prefilled_outputs,
    ...
)
```

**برای training**: شما باید voice prompts فارسی بسازید یا از موجودی‌ها استفاده کنید.

### ب) Text Normalization

برای فارسی، احتمالاً نیاز دارید:
- اعداد را به حروف تبدیل کنید
- علائم خاص را normalize کنید
- کلمات انگلیسی را به فارسی تبدیل یا transliterate کنید

### ج) Audio Preprocessing

```python
# Sample rate باید 24000 Hz باشد
target_sample_rate = 24000

# Audio normalization
# processor.audio_processor دارای این قابلیت است:
# - normalize_audio = True
# - target_dB_FS = -25
```

### د) Streaming vs Non-streaming

مدل realtime برای **streaming text input** طراحی شده، اما می‌توانید از آن برای non-streaming هم استفاده کنید.

در training، معمولاً non-streaming ساده‌تر است.

---

## ۱۰. منابع و مراجع

### کدهای مرجع

- **مدل اصلی**: `vibevoice/modular/modeling_vibevoice_streaming.py`
- **Inference**: `vibevoice/modular/modeling_vibevoice_streaming_inference.py`
- **Processor**: `vibevoice/processor/vibevoice_streaming_processor.py`
- **Diffusion Head**: `vibevoice/modular/modular_vibevoice_diffusion_head.py`
- **Acoustic Tokenizer**: `vibevoice/modular/modular_vibevoice_tokenizer.py`
- **Configuration**: `vibevoice/modular/configuration_vibevoice_streaming.py`

### دمو و مثال‌ها

- **Inference از فایل**: `demo/realtime_model_inference_from_file.py`
- **WebSocket Demo**: `demo/vibevoice_realtime_demo.py`
- **Colab**: [vibevoice_realtime_colab.ipynb](https://colab.research.google.com/github/microsoft/VibeVoice/blob/main/demo/vibevoice_realtime_colab.ipynb)

### Documentation

- **README**: `README.md`
- **Realtime Docs**: `docs/vibevoice-realtime-0.5b.md`
- **Technical Report**: https://arxiv.org/pdf/2508.19205

### اسکریپت‌های کمکی

- **Inspection Script**: `inspect_realtime.py` (ایجاد شده توسط این راهنما)

---

## پایان

این راهنما تمام اطلاعات لازم برای شروع fine-tuning مدل VibeVoice-Realtime را فراهم می‌کند.

**مراحل بعدی**:
1. اسکریپت `inspect_realtime.py` را اجرا کنید
2. دیتاست خود را آماده کنید
3. یک training loop ساده بنویسید
4. با تعداد کم step تست کنید
5. نتایج را ارزیابی کنید
6. hyperparameter ها را tune کنید
7. training کامل را انجام دهید

**موفق باشید!** 🎙️🇮🇷
