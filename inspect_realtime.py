#!/usr/bin/env python3
"""
اسکریپت بررسی ساختار مدل VibeVoice-Realtime
این اسکریپت به شما کمک می‌کند تا ساختار دقیق مدل و لایه‌های آن را مشاهده کنید
"""

import torch
from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference

def main():
    print("=" * 80)
    print("بارگذاری مدل VibeVoice-Realtime-0.5B...")
    print("=" * 80)

    # بارگذاری مدل (فقط با config، بدون وزن‌ها برای سرعت بیشتر)
    try:
        model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            "microsoft/VibeVoice-Realtime-0.5B",
            torch_dtype=torch.float32,
            device_map="cpu",
            attn_implementation="sdpa",
        )
        model.eval()
    except Exception as e:
        print(f"خطا در بارگذاری مدل: {e}")
        print("در حال امتحان بدون flash attention...")
        model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            "microsoft/VibeVoice-Realtime-0.5B",
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        model.eval()

    print("\n" + "=" * 80)
    print("۱. ساختار کلی مدل")
    print("=" * 80)
    print(f"کلاس اصلی: {model.__class__.__name__}")
    print(f"Base model: {model.model.__class__.__name__}")

    print("\n" + "=" * 80)
    print("۲. کامپوننت‌های اصلی مدل (model.model)")
    print("=" * 80)
    for name in dir(model.model):
        if not name.startswith('_'):
            attr = getattr(model.model, name)
            if hasattr(attr, '__class__') and 'vibevoice' in str(type(attr)).lower() or \
               hasattr(attr, '__class__') and any(x in str(type(attr)).lower() for x in ['qwen', 'acoustic', 'connector', 'diffusion', 'scheduler']):
                print(f"  - {name}: {type(attr).__name__}")

    print("\n" + "=" * 80)
    print("۳. لایه‌های آکوستیک و diffusion (برای fine-tuning)")
    print("=" * 80)

    acoustic_layers = []
    diffusion_layers = []
    text_layers = []

    for name, module in model.named_modules():
        # فیلتر کردن لایه‌های مرتبط با acoustic
        if any(k in name.lower() for k in ["acoustic", "codec"]):
            acoustic_layers.append((name, module.__class__.__name__))
        # فیلتر کردن لایه‌های مرتبط با diffusion
        elif any(k in name.lower() for k in ["diffusion", "prediction_head"]):
            diffusion_layers.append((name, module.__class__.__name__))
        # فیلتر کردن لایه‌های text encoder
        elif any(k in name.lower() for k in ["language_model", "tts_language_model"]) and \
             any(x in name.lower() for x in ["layers.", "embed", "norm"]):
            text_layers.append((name, module.__class__.__name__))

    print("\nلایه‌های Acoustic Tokenizer:")
    for name, cls in acoustic_layers[:10]:  # نمایش ۱۰ اولی
        print(f"  {name} → {cls}")
    if len(acoustic_layers) > 10:
        print(f"  ... و {len(acoustic_layers) - 10} لایه دیگر")

    print(f"\nلایه‌های Diffusion/Prediction Head:")
    for name, cls in diffusion_layers[:15]:
        print(f"  {name} → {cls}")
    if len(diffusion_layers) > 15:
        print(f"  ... و {len(diffusion_layers) - 15} لایه دیگر")

    print(f"\nنمونه از لایه‌های Language Model (text encoder):")
    for name, cls in text_layers[:10]:
        print(f"  {name} → {cls}")
    if len(text_layers) > 10:
        print(f"  ... و {len(text_layers) - 10} لایه دیگر")

    print("\n" + "=" * 80)
    print("۴. پارامترهای Configuration")
    print("=" * 80)
    config = model.config
    print(f"  - acoustic_vae_dim: {config.acoustic_vae_dim}")
    print(f"  - tts_backbone_num_hidden_layers: {config.tts_backbone_num_hidden_layers}")
    print(f"  - decoder hidden_size: {config.decoder_config.hidden_size}")
    print(f"  - decoder num_hidden_layers: {config.decoder_config.num_hidden_layers}")
    print(f"  - diffusion hidden_size: {config.diffusion_head_config.hidden_size}")
    print(f"  - diffusion head_layers: {config.diffusion_head_config.head_layers}")
    print(f"  - diffusion ddpm_num_steps: {config.diffusion_head_config.ddpm_num_steps}")
    print(f"  - diffusion prediction_type: {config.diffusion_head_config.prediction_type}")

    print("\n" + "=" * 80)
    print("۵. پارامترهای قابل آموزش (Trainable Parameters)")
    print("=" * 80)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  کل پارامترها: {total_params:,}")
    print(f"  پارامترهای قابل آموزش: {trainable_params:,}")
    print(f"  درصد قابل آموزش: {100 * trainable_params / total_params:.2f}%")

    print("\n" + "=" * 80)
    print("۶. استراتژی پیشنهادی برای Fine-tuning")
    print("=" * 80)
    print("""
برای fine-tuning بر روی دیتاست فارسی:

الف) Freeze کردن:
  - model.model.language_model (text encoder - لایه‌های پایینی Qwen)
  - می‌توانید تصمیم بگیرید که آیا tts_language_model را نیز freeze کنید یا نه

ب) Fine-tune کردن:
  - model.model.acoustic_connector (SpeechConnector)
  - model.model.prediction_head (VibeVoiceDiffusionHead)
  - اختیاری: model.tts_eos_classifier
  - اختیاری: model.model.tts_language_model (لایه‌های بالایی)

ج) نگه‌داری به صورت Frozen:
  - model.model.acoustic_tokenizer (decoder برای تبدیل latents به audio)

د) روش LoRA (پیشنهادی):
  target_modules باید شامل لایه‌های زیر باشد:
  - "prediction_head.*" (diffusion head)
  - "acoustic_connector.*"
  - اختیاری: "tts_language_model.layers.*" (فقط attention و mlp)
    """)

    print("\n" + "=" * 80)
    print("۷. مثال کد برای Freeze")
    print("=" * 80)
    print("""
# مثال freeze کردن text encoder
for name, param in model.named_parameters():
    if any(k in name.lower() for k in ["language_model", "embed_tokens"]):
        # Freeze text encoder
        param.requires_grad = False
    elif "acoustic_tokenizer" in name.lower():
        # Freeze acoustic tokenizer decoder
        param.requires_grad = False
    else:
        # Fine-tune بقیه (acoustic_connector, prediction_head, tts_eos_classifier)
        param.requires_grad = True

# بررسی پارامترهای قابل آموزش
trainable = [n for n, p in model.named_parameters() if p.requires_grad]
print(f"Trainable params: {len(trainable)}")
for n in trainable[:10]:
    print(f"  {n}")
    """)

    print("\n" + "=" * 80)
    print("اتمام بررسی")
    print("=" * 80)

if __name__ == "__main__":
    main()
