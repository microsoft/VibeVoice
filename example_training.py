#!/usr/bin/env python3
"""
مثال کد Training برای Fine-tuning VibeVoice-Realtime بر روی دیتاست فارسی

این کد یک نمونه اولیه است و باید برای دیتاست واقعی شما تنظیم شود.

نکات مهم:
1. این کد فرض می‌کند که شما acoustic latents از قبل محاسبه شده دارید
2. اگر acoustic encoder ندارید، باید از روش دیگری استفاده کنید
3. این کد برای آموزش است و باید برای production بهینه‌سازی شود
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import json

from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference
)
from vibevoice.processor.vibevoice_streaming_processor import (
    VibeVoiceStreamingProcessor
)


# ============================================================================
# 1. تعریف Dataset
# ============================================================================

class PersianTTSDataset(Dataset):
    """
    Dataset برای training TTS فارسی

    فرمت دیتا:
    - data_dir/
      - metadata.json  # لیست samples
      - audio/
        - sample001.wav
        - sample002.wav
      - latents/  # اختیاری: pre-computed acoustic latents
        - sample001.pt
        - sample002.pt
    """

    def __init__(
        self,
        data_dir,
        processor,
        max_duration_sec=10.0,
        sample_rate=24000,
        use_precomputed_latents=True
    ):
        self.data_dir = data_dir
        self.processor = processor
        self.max_duration_sec = max_duration_sec
        self.sample_rate = sample_rate
        self.use_precomputed_latents = use_precomputed_latents

        # بارگذاری metadata
        metadata_path = os.path.join(data_dir, "metadata.json")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)

        print(f"Loaded {len(self.samples)} samples from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # متن
        text = sample['text']

        # Acoustic latents
        if self.use_precomputed_latents:
            # بارگذاری latents از قبل محاسبه شده
            latent_path = os.path.join(
                self.data_dir,
                "latents",
                f"{sample['id']}.pt"
            )
            acoustic_latents = torch.load(latent_path)
        else:
            # TODO: محاسبه latents از audio
            # این قسمت به acoustic encoder نیاز دارد
            audio_path = os.path.join(
                self.data_dir,
                "audio",
                sample['audio_file']
            )
            # audio = load_audio(audio_path, self.sample_rate)
            # acoustic_latents = self.acoustic_encoder(audio)
            raise NotImplementedError(
                "Real-time latent computation not implemented. "
                "Please use pre-computed latents."
            )

        # پردازش متن
        # نکته: در training واقعی، شما نیاز به voice prompt هم دارید
        # اینجا فرض می‌کنیم که cached_prompt را جداگانه مدیریت می‌کنید
        text_encoding = self.processor.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=512,
            truncation=True
        )

        return {
            'text': text,
            'input_ids': text_encoding['input_ids'].squeeze(0),
            'attention_mask': text_encoding['attention_mask'].squeeze(0),
            'acoustic_latents': acoustic_latents,
            'sample_id': sample['id']
        }


# ============================================================================
# 2. Collate Function
# ============================================================================

def collate_fn(batch):
    """Collate function برای DataLoader"""
    # پیدا کردن max lengths
    max_text_len = max(item['input_ids'].size(0) for item in batch)
    max_latent_len = max(item['acoustic_latents'].size(0) for item in batch)

    # Padding
    input_ids = []
    attention_masks = []
    acoustic_latents = []

    for item in batch:
        # Pad text
        text_len = item['input_ids'].size(0)
        pad_len = max_text_len - text_len
        input_ids.append(
            F.pad(item['input_ids'], (0, pad_len), value=0)
        )
        attention_masks.append(
            F.pad(item['attention_mask'], (0, pad_len), value=0)
        )

        # Pad latents
        latent_len = item['acoustic_latents'].size(0)
        latent_dim = item['acoustic_latents'].size(-1)
        pad_len = max_latent_len - latent_len
        padded_latent = F.pad(
            item['acoustic_latents'],
            (0, 0, 0, pad_len),
            value=0
        )
        acoustic_latents.append(padded_latent)

    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_masks),
        'acoustic_latents': torch.stack(acoustic_latents),
        'texts': [item['text'] for item in batch],
        'sample_ids': [item['sample_id'] for item in batch]
    }


# ============================================================================
# 3. Training Functions
# ============================================================================

def setup_model_for_training(model, freeze_strategy="lora"):
    """
    تنظیم مدل برای training

    Args:
        model: مدل VibeVoice
        freeze_strategy: یکی از ["conservative", "lora", "aggressive"]
    """
    if freeze_strategy == "conservative":
        # استراتژی ۱: freeze language_model و acoustic_tokenizer
        for name, param in model.named_parameters():
            if any(k in name for k in [
                "model.language_model",
                "model.acoustic_tokenizer",
            ]):
                param.requires_grad = False
            else:
                param.requires_grad = True

    elif freeze_strategy == "lora":
        # استراتژی ۲: LoRA
        try:
            from peft import LoraConfig, get_peft_model

            # Freeze همه چیز
            for param in model.parameters():
                param.requires_grad = False

            # اعمال LoRA
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=[
                    "model.tts_language_model.layers.*.self_attn.q_proj",
                    "model.tts_language_model.layers.*.self_attn.k_proj",
                    "model.tts_language_model.layers.*.self_attn.v_proj",
                    "model.tts_language_model.layers.*.self_attn.o_proj",
                ],
                lora_dropout=0.05,
                bias="none",
            )
            model = get_peft_model(model, lora_config)

            # Fine-tune connectors
            for name, param in model.named_parameters():
                if any(k in name for k in [
                    "acoustic_connector",
                    "prediction_head",
                    "tts_eos_classifier"
                ]):
                    param.requires_grad = True

        except ImportError:
            print("PEFT not installed. Falling back to conservative strategy.")
            return setup_model_for_training(model, "conservative")

    elif freeze_strategy == "aggressive":
        # استراتژی ۳: فقط acoustic_tokenizer را freeze کن
        for name, param in model.named_parameters():
            if "model.acoustic_tokenizer" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    else:
        raise ValueError(f"Unknown freeze strategy: {freeze_strategy}")

    # شمارش پارامترهای trainable
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())

    print(f"\nTrainable parameters: {trainable_params:,} / {total_params:,}")
    print(f"Percentage: {100 * trainable_params / total_params:.2f}%\n")

    return model


def compute_diffusion_loss(
    model,
    condition,
    target_latents,
    timesteps=None,
    batch_size=None
):
    """
    محاسبه diffusion loss

    Args:
        model: مدل VibeVoice
        condition: hidden states از TTS LM [B, T, D]
        target_latents: acoustic latents هدف [B, T, 64]
        timesteps: اختیاری، timesteps برای diffusion
        batch_size: اختیاری

    Returns:
        loss: diffusion loss
    """
    if batch_size is None:
        batch_size = target_latents.size(0)

    # نمونه‌برداری timestep تصادفی
    if timesteps is None:
        timesteps = torch.randint(
            0,
            model.config.diffusion_head_config.ddpm_num_steps,
            (batch_size,),
            device=target_latents.device
        )

    # اضافه کردن نویز
    noise = torch.randn_like(target_latents)
    noisy_latents = model.model.noise_scheduler.add_noise(
        target_latents,
        noise,
        timesteps
    )

    # پیش‌بینی با diffusion head
    # نکته: diffusion head انتظار دارد [B, latent_dim] نه [B, T, latent_dim]
    # باید هر frame را جداگانه پردازش کنیم یا reshape کنیم

    # روش ساده: فرض می‌کنیم T=1 (یک frame در هر بار)
    # در عمل، باید این را برای sequence ها تعمیم دهید

    # TODO: این قسمت باید بر اساس معماری دقیق تنظیم شود
    # اینجا یک نسخه ساده‌شده است

    # فرض: condition و noisy_latents هر دو [B, T, D]
    # باید به [B*T, D] reshape کنیم

    B, T_cond, D_cond = condition.shape
    _, T_lat, D_lat = noisy_latents.shape

    # برای سادگی، فرض می‌کنیم T_cond == T_lat
    if T_cond != T_lat:
        # TODO: handle alignment
        raise ValueError(
            f"Condition length {T_cond} != latent length {T_lat}"
        )

    # Reshape
    condition_flat = condition.view(B * T_cond, D_cond)
    noisy_latents_flat = noisy_latents.view(B * T_lat, D_lat)
    timesteps_flat = timesteps.unsqueeze(1).repeat(1, T_lat).view(-1)

    # پیش‌بینی
    predicted = model.model.prediction_head(
        noisy_latents_flat,
        timesteps_flat,
        condition=condition_flat
    )

    # محاسبه target بر اساس prediction_type
    if model.config.diffusion_head_config.prediction_type == "v_prediction":
        # v-prediction
        velocity = model.model.noise_scheduler.get_velocity(
            target_latents.view(B * T_lat, D_lat),
            noise.view(B * T_lat, D_lat),
            timesteps_flat
        )
        loss = F.mse_loss(predicted, velocity)
    elif model.config.diffusion_head_config.prediction_type == "epsilon":
        # epsilon-prediction
        loss = F.mse_loss(predicted, noise.view(B * T_lat, D_lat))
    else:
        raise ValueError(
            f"Unknown prediction type: "
            f"{model.config.diffusion_head_config.prediction_type}"
        )

    return loss


def training_step(model, batch, device):
    """
    یک step training

    Args:
        model: مدل VibeVoice
        batch: batch از dataloader
        device: cuda یا cpu

    Returns:
        loss: total loss
        loss_dict: dictionary از losses مختلف
    """
    # انتقال به device
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    target_latents = batch['acoustic_latents'].to(device)

    # 1. Forward pass از language_model
    lm_outputs = model.model.language_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
        output_hidden_states=True
    )
    hidden_states = lm_outputs.last_hidden_state

    # 2. Forward pass از tts_language_model
    # نکته: در واقع، باید speech embeddings را هم ترکیب کنیم
    # اینجا یک نسخه ساده‌شده است
    tts_lm_outputs = model.model.tts_language_model(
        inputs_embeds=hidden_states,
        attention_mask=attention_mask,
        return_dict=True,
        output_hidden_states=True
    )
    condition = tts_lm_outputs.last_hidden_state  # [B, T, hidden_size]

    # 3. محاسبه diffusion loss
    diffusion_loss = compute_diffusion_loss(
        model,
        condition,
        target_latents,
        batch_size=input_ids.size(0)
    )

    # 4. سایر losses (اختیاری)
    # TODO: اضافه کردن EOS classifier loss

    total_loss = diffusion_loss

    loss_dict = {
        'total_loss': total_loss.item(),
        'diffusion_loss': diffusion_loss.item(),
    }

    return total_loss, loss_dict


# ============================================================================
# 4. Main Training Loop
# ============================================================================

def train(
    model,
    train_dataloader,
    optimizer,
    scheduler,
    device,
    num_steps=5000,
    gradient_accumulation_steps=4,
    log_every=100,
    save_every=1000,
    output_dir="./checkpoints"
):
    """
    حلقه اصلی training

    Args:
        model: مدل VibeVoice
        train_dataloader: DataLoader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: cuda یا cpu
        num_steps: تعداد کل steps
        gradient_accumulation_steps: gradient accumulation
        log_every: هر چند step لاگ کند
        save_every: هر چند step checkpoint ذخیره کند
        output_dir: مسیر ذخیره checkpoints
    """
    os.makedirs(output_dir, exist_ok=True)

    model.train()
    global_step = 0
    epoch = 0

    # برای logging
    running_loss = 0.0
    running_loss_dict = {}

    progress_bar = tqdm(total=num_steps, desc="Training")

    while global_step < num_steps:
        epoch += 1

        for batch in train_dataloader:
            # Training step
            loss, loss_dict = training_step(model, batch, device)

            # Backward
            loss = loss / gradient_accumulation_steps
            loss.backward()

            # Accumulate loss for logging
            running_loss += loss.item()
            for k, v in loss_dict.items():
                running_loss_dict[k] = running_loss_dict.get(k, 0.0) + v

            # Optimizer step
            if (global_step + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping (اختیاری)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            progress_bar.update(1)

            # Logging
            if global_step % log_every == 0:
                avg_loss = running_loss / log_every
                progress_bar.set_postfix({
                    'loss': f"{avg_loss:.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })

                # پرینت losses
                print(f"\nStep {global_step}:")
                for k, v in running_loss_dict.items():
                    print(f"  {k}: {v / log_every:.4f}")

                # Reset
                running_loss = 0.0
                running_loss_dict = {}

            # Checkpoint
            if global_step % save_every == 0:
                checkpoint_path = os.path.join(
                    output_dir,
                    f"checkpoint-{global_step}"
                )
                print(f"\nSaving checkpoint to {checkpoint_path}")
                model.save_pretrained(checkpoint_path)

            # بررسی اتمام
            if global_step >= num_steps:
                break

    progress_bar.close()

    # ذخیره final model
    final_path = os.path.join(output_dir, "final_model")
    print(f"\nSaving final model to {final_path}")
    model.save_pretrained(final_path)


# ============================================================================
# 5. Main Function
# ============================================================================

def main():
    # Hyperparameters
    model_name = "microsoft/VibeVoice-Realtime-0.5B"
    data_dir = "./persian_tts_data"  # تغییر دهید
    output_dir = "./checkpoints"
    freeze_strategy = "conservative"  # یا "lora" یا "aggressive"

    batch_size = 2
    gradient_accumulation_steps = 4
    learning_rate = 1e-4
    num_steps = 5000
    warmup_steps = 500

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # بارگذاری model و processor
    print(f"Loading model from {model_name}...")
    model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)

    processor = VibeVoiceStreamingProcessor.from_pretrained(model_name)

    # تنظیم برای training
    print(f"\nSetting up model with freeze strategy: {freeze_strategy}")
    model = setup_model_for_training(model, freeze_strategy)

    # Dataset و DataLoader
    print(f"\nLoading dataset from {data_dir}...")
    train_dataset = PersianTTSDataset(
        data_dir=data_dir,
        processor=processor,
        max_duration_sec=10.0,
        use_precomputed_latents=True
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # تغییر دهید برای سرعت بیشتر
    )

    # Optimizer و Scheduler
    print("\nSetting up optimizer...")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_steps
    )

    # Training
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)

    train(
        model=model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_steps=num_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        log_every=100,
        save_every=1000,
        output_dir=output_dir
    )

    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
