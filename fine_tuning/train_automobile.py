#!/usr/bin/env python3
"""
Automobile LLM Fine-Tuning with Unsloth
=======================================

Fine-tunes Qwen2.5-1.5B-Instruct on automobile Q&A data using LoRA.
Optimized for RunPod A40/A100 GPUs.

Usage:
    python train_automobile.py
    
    # With custom settings
    python train_automobile.py --epochs 3 --batch_size 4 --lr 2e-4
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Set environment variables before imports
os.environ["WANDB_DISABLED"] = "true"  # Disable wandb by default

def main():
    parser = argparse.ArgumentParser(description="Fine-tune automobile LLM")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--output_dir", type=str, default="models/automobile", help="Output directory")
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization (QLoRA)")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("VEMI AI Automobile LLM Fine-Tuning")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check for GPU
    import torch
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script requires a GPU.")
        sys.exit(1)
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # Install unsloth if needed
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("\nInstalling Unsloth...")
        os.system("pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'")
        from unsloth import FastLanguageModel
    
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments
    
    # Load data
    data_path = Path(__file__).parent / "data" / "automobile" / "automobile_qa.json"
    if not data_path.exists():
        print(f"\nERROR: Data file not found: {data_path}")
        print("Run 'python prepare_data.py --domain automobile' first.")
        sys.exit(1)
    
    print(f"\nLoading data from: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    print(f"Loaded {len(raw_data)} examples")
    
    # Convert to dataset format
    def format_prompt(example):
        """Format for Qwen2.5 instruction format"""
        return f"""<|im_start|>system
{example['system']}<|im_end|>
<|im_start|>user
{example['instruction']}<|im_end|>
<|im_start|>assistant
{example['output']}<|im_end|>"""
    
    formatted_data = [{"text": format_prompt(ex)} for ex in raw_data]
    dataset = Dataset.from_list(formatted_data)
    
    # Split into train/eval
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    
    print(f"Train examples: {len(train_dataset)}")
    print(f"Eval examples: {len(eval_dataset)}")
    
    # Load model with Unsloth
    print("\n" + "-"*40)
    print("Loading model with Unsloth...")
    print("-"*40)
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        max_seq_length=args.max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=args.use_4bit,
    )
    
    # Add LoRA adapters
    print("\nConfiguring LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    # Training arguments
    output_dir = Path(__file__).parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        report_to="none",
        seed=42,
    )
    
    # Create trainer
    print("\n" + "-"*40)
    print("Starting training...")
    print("-"*40)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"LoRA rank: {args.lora_r}")
    print(f"Output: {output_dir}")
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=training_args,
    )
    
    # Train
    train_result = trainer.train()
    
    # Save model
    print("\n" + "-"*40)
    print("Saving model...")
    print("-"*40)
    
    # Save LoRA adapters
    model.save_pretrained(output_dir / "lora_adapter")
    tokenizer.save_pretrained(output_dir / "lora_adapter")
    
    # Merge and save full model
    print("Merging LoRA weights into base model...")
    model.save_pretrained_merged(
        output_dir / "merged_model",
        tokenizer,
        save_method="merged_16bit",
    )
    
    # Save training info
    info = {
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "domain": "automobile",
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "lora_rank": args.lora_r,
        "max_seq_length": args.max_seq_length,
        "train_loss": train_result.training_loss,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(output_dir / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Final loss: {train_result.training_loss:.4f}")
    print(f"Model saved to: {output_dir}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nTo use this model, copy to:")
    print(f"  cp -r {output_dir}/merged_model ../speech_to_speech/models/automobile_llm")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
