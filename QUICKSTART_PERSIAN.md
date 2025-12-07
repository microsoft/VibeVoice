# شروع سریع: Fine-tuning VibeVoice-Realtime برای فارسی

## فایل‌های ایجاد شده

این repo حالا شامل سه فایل جدید است:

### 1. `PERSIAN_FINETUNE_GUIDE.md`
**راهنمای جامع فارسی** که شامل:
- تمام کلاس‌ها و import های لازم
- ساختار معماری مدل
- توضیح کامل تمام submodule ها
- استراتژی‌های freeze مختلف
- hyperparameter ها
- checklist کامل

👉 **شروع کنید با خواندن این فایل**

### 2. `inspect_realtime.py`
**اسکریپت بررسی مدل** که نشان می‌دهد:
- ساختار کامل مدل
- تمام لایه‌های قابل fine-tune
- پارامترهای configuration
- مثال‌های freeze strategy

#### استفاده:
```bash
python inspect_realtime.py
```

### 3. `example_training.py`
**مثال کامل training loop** شامل:
- کلاس Dataset برای دیتای فارسی
- تابع collate_fn
- تابع محاسبه diffusion loss
- حلقه training کامل
- ذخیره checkpoints

#### استفاده:
```bash
# قبل از اجرا، باید دیتاست خود را آماده کنید
# فرمت دیتاست در کد توضیح داده شده

python example_training.py
```

## مراحل سریع

### مرحله 1: بررسی مدل
```bash
python inspect_realtime.py
```

### مرحله 2: آماده‌سازی دیتاست
فرمت مورد نیاز:
```
persian_tts_data/
├── metadata.json         # لیست samples
├── audio/
│   ├── sample001.wav
│   └── sample002.wav
└── latents/              # اختیاری
    ├── sample001.pt
    └── sample002.pt
```

فرمت `metadata.json`:
```json
[
  {
    "id": "sample001",
    "text": "این یک متن نمونه فارسی است.",
    "audio_file": "sample001.wav"
  },
  ...
]
```

### مرحله 3: تنظیم پارامترها
در فایل `example_training.py`:
```python
# خط ~580
data_dir = "./persian_tts_data"  # مسیر دیتاست شما
freeze_strategy = "conservative"  # یا "lora" یا "aggressive"
batch_size = 2
learning_rate = 1e-4
num_steps = 5000
```

### مرحله 4: شروع training
```bash
python example_training.py
```

## نکات مهم

### ⚠️ Acoustic Latents
کد مثال فرض می‌کند که شما **pre-computed acoustic latents** دارید.

چون این repo فقط acoustic **decoder** دارد (نه encoder), شما باید:
1. یک acoustic encoder جداگانه داشته باشید، یا
2. از روش دیگری برای تولید target latents استفاده کنید

برای اطلاعات بیشتر، بخش ۴ از `PERSIAN_FINETUNE_GUIDE.md` را بخوانید.

### 🎯 شروع پیشنهادی

1. **اول**: با تعداد کم sample شروع کنید (50-100)
2. **دوم**: با 500-1000 step تست کنید
3. **سوم**: خروجی audio را گوش کنید
4. **چهارم**: hyperparameter ها را tune کنید
5. **پنجم**: با دیتاست کامل train کنید

### 📊 ارزیابی

بعد از training:
```python
from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference
)
from vibevoice.processor.vibevoice_streaming_processor import (
    VibeVoiceStreamingProcessor
)

# بارگذاری fine-tuned model
model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
    "./checkpoints/final_model"
)
processor = VibeVoiceStreamingProcessor.from_pretrained(
    "./checkpoints/final_model"
)

# تست
# (از کد demo استفاده کنید)
```

## کمک و پشتیبانی

- **مشکلات فنی**: issue در GitHub
- **سوالات**: بخش discussions
- **مستندات**: `PERSIAN_FINETUNE_GUIDE.md`

## منابع اضافی

- [Technical Report](https://arxiv.org/pdf/2508.19205)
- [Hugging Face Model](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B)
- [Colab Demo](https://colab.research.google.com/github/microsoft/VibeVoice/blob/main/demo/vibevoice_realtime_colab.ipynb)

## Disclaimer

این کدها برای **آموزش و تحقیق** هستند.

قبل از استفاده در production:
- تست کامل کنید
- کیفیت را ارزیابی کنید
- مسائل اخلاقی (deepfake) را در نظر بگیرید

---

**موفق باشید!** 🎙️🇮🇷
