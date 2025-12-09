#!/usr/bin/env python3
"""
Medical LLM Fine-Tuning with PEFT
=================================

Fine-tunes Qwen2.5-1.5B-Instruct on medical Q&A data using LoRA.

IMPORTANT: Run setup_env.sh first to install compatible dependencies!
    bash setup_env.sh

Usage:
    python train_medical_peft.py
    python train_medical_peft.py --epochs 3 --batch_size 4 --lr 2e-4
"""

import os
import sys
import subprocess

# Set environment variables BEFORE importing anything else
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"


def check_and_install_deps():
    """Check and install compatible dependencies"""
    try:
        import transformers
        import peft
        # Check versions
        trans_ver = tuple(map(int, transformers.__version__.split('.')[:2]))
        if trans_ver > (4, 45):
            raise ImportError("transformers version too new")
    except (ImportError, Exception) as e:
        print(f"Dependency issue detected: {e}")
        print("Installing compatible versions...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            "transformers==4.44.2",
            "peft==0.12.0",
            "trl==0.9.6",
            "accelerate==0.33.0",
            "bitsandbytes==0.43.1"
        ])
        print("Dependencies installed. Please restart the script.")
        sys.exit(0)


def main():
    import argparse
    import json
    from pathlib import Path
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description="Fine-tune medical LLM with PEFT")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--output_dir", type=str, default="models/medical", help="Output directory")
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization (QLoRA)")
    parser.add_argument("--skip_deps_check", action="store_true", help="Skip dependency check")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("VEMI AI Medical LLM Fine-Tuning (PEFT)")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check dependencies unless skipped
    if not args.skip_deps_check:
        check_and_install_deps()
    
    # Now import the ML libraries
    import torch
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script requires a GPU.")
        sys.exit(1)
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # Import libraries after dependency check
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, get_peft_model
    from datasets import Dataset
    from trl import SFTTrainer
    
    # Import BitsAndBytesConfig only if using 4bit
    if args.use_4bit:
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
    
    # Load data
    data_path = Path(__file__).parent / "data" / "medical" / "medical_qa.json"
    if not data_path.exists():
        print(f"\nERROR: Data file not found: {data_path}")
        print("Run 'python prepare_data.py --domain medical' first.")
        sys.exit(1)
    
    print(f"\nLoading data from: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    print(f"Loaded {len(raw_data)} examples")
    
    # Format prompts
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
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    
    print(f"Train examples: {len(train_dataset)}")
    print(f"Eval examples: {len(eval_dataset)}")
    
    # Load tokenizer
    print("\n" + "-"*40)
    print("Loading model and tokenizer...")
    print("-"*40)
    
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model with quantization if requested
    if args.use_4bit:
        print("Using 4-bit quantization (QLoRA)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        print("Using full precision (FP32)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        # Enable gradient computation
        model.enable_input_require_grads()
    
    # Configure LoRA
    print("\nConfiguring LoRA adapters...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Training arguments
    output_dir = Path(__file__).parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure model is in training mode
    model.train()
    
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
        eval_steps=200,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        fp16=False,  # Disable mixed precision to avoid gradient issues
        bf16=False,
        optim="adamw_torch",
        report_to="none",
        seed=42,
        gradient_checkpointing=False,  # Disable - causes issues with PEFT
        max_grad_norm=1.0,
        dataloader_pin_memory=False,
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
    
    # Save merged model
    print("Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_dir / "merged_model")
    tokenizer.save_pretrained(output_dir / "merged_model")
    
    # Save training info
    info = {
        "base_model": model_name,
        "domain": "medical",
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
    print("\nTo use this model:")
    print(f"  cp -r {output_dir}/merged_model ../speech_to_speech/models/medical_llm")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
