#!/usr/bin/env python
"""
VibeVoice ASR LoRA Fine-tuning Script

This script implements LoRA (Low-Rank Adaptation) fine-tuning for the VibeVoice ASR model.
It uses PEFT (Parameter-Efficient Fine-Tuning) library for efficient training.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from transformers import (
    TrainingArguments,
    Trainer,
    HfArgumentParser,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    model_path: str = field(
        default="microsoft/VibeVoice-ASR",
        metadata={"help": "Path to pretrained model (HuggingFace model ID or local path)"}
    )


@dataclass
class DataArguments:
    """Arguments for data configuration."""
    data_dir: str = field(
        default="./toy_dataset",
        metadata={"help": "Directory containing training data"}
    )
    max_audio_length: Optional[float] = field(
        default=None,
        metadata={"help": "Maximum audio length in seconds (default: no limit)"}
    )
    use_customized_context: bool = field(
        default=True,
        metadata={"help": "Whether to use customized_context from JSON as additional context"}
    )


@dataclass
class LoraArguments:
    """Arguments for LoRA configuration."""
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha (scaling factor)"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout"}
    )


@dataclass
class VibeVoiceASRDataCollator:
    """
    Data collator for VibeVoice ASR fine-tuning.
    Handles batching of variable-length audio and text sequences.
    """
    processor: VibeVoiceASRProcessor
    pad_token_id: int
    label_pad_token_id: int = -100
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of features into model inputs.
        """
        # Separate inputs and labels
        input_ids_list = [f["input_ids"] for f in features]
        labels_list = [f["labels"] for f in features]
        acoustic_mask_list = [f["acoustic_input_mask"] for f in features]
        speech_list = [f["speech"] for f in features]
        vae_tok_lens = [f["vae_tok_len"] for f in features]
        
        # Determine max lengths
        max_seq_len = max(len(ids) for ids in input_ids_list)
        max_speech_len = max(len(s) for s in speech_list)
        max_vae_len = max(vae_tok_lens)
        
        batch_size = len(features)
        
        # Initialize padded tensors
        input_ids = torch.full((batch_size, max_seq_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        labels = torch.full((batch_size, max_seq_len), self.label_pad_token_id, dtype=torch.long)
        acoustic_input_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
        speech_tensors = torch.zeros((batch_size, max_speech_len), dtype=torch.float32)
        speech_masks = torch.zeros((batch_size, max_vae_len), dtype=torch.bool)
        
        # Fill in the tensors (right padding for training)
        # Note: processor uses left padding for inference/generation, but training uses right padding
        for i, (ids, lbls, amask, speech, vae_len) in enumerate(
            zip(input_ids_list, labels_list, acoustic_mask_list, speech_list, vae_tok_lens)
        ):
            seq_len = len(ids)
            
            # Right padding for input_ids and labels
            input_ids[i, :seq_len] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :seq_len] = 1
            labels[i, :seq_len] = torch.tensor(lbls, dtype=torch.long)
            acoustic_input_mask[i, :seq_len] = torch.tensor(amask, dtype=torch.bool)
            
            # Speech tensors (right padding, zeros work as padding)
            speech_len = len(speech)
            speech_tensors[i, :speech_len] = torch.tensor(speech, dtype=torch.float32)
            speech_masks[i, :vae_len] = True
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "acoustic_input_mask": acoustic_input_mask,
            "speech_tensors": speech_tensors,
            "speech_masks": speech_masks,
        }


class VibeVoiceASRDataset(Dataset):
    """
    Dataset for VibeVoice ASR fine-tuning.
    
    Expected data format:
        - Audio files: .mp3, .wav, .flac, etc.
        - Label files: .json with matching name
        
    JSON format:
        {
            "audio_path": "0.mp3",
            "audio_duration": 351.73,
            "segments": [
                {
                    "speaker": 0,
                    "text": "Hey everyone, welcome back...",
                    "start": 0.0,
                    "end": 38.68
                },
                ...
            ],
            "customized_context": ["Tea Brew", "The property is near Meter Street."]  # optional
        }
    """
    
    def __init__(
        self,
        data_dir: str,
        processor: VibeVoiceASRProcessor,
        max_audio_length: Optional[float] = None,  # in seconds
        use_customized_context: bool = True,
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing audio files and JSON labels
            processor: VibeVoice ASR processor
            max_audio_length: Maximum audio length in seconds (None = no limit)
            use_customized_context: Whether to include customized_context in prompt
        """
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.max_audio_length = max_audio_length
        self.use_customized_context = use_customized_context
        
        # Find all JSON files
        self.samples = self._load_samples()
        logger.info(f"Loaded {len(self.samples)} samples from {data_dir}")
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load and validate all samples from data directory."""
        samples = []
        
        for json_path in sorted(self.data_dir.glob("*.json")):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Get audio path from JSON
                audio_filename = data.get("audio_path")
                if not audio_filename:
                    logger.warning(f"No audio_path specified in {json_path}")
                    continue
                    
                audio_path = self.data_dir / audio_filename
                if not audio_path.exists():
                    logger.warning(f"Audio file not found: {audio_path}")
                    continue
                
                # Optional: filter by duration
                if self.max_audio_length is not None:
                    duration = data.get("audio_duration", float("inf"))
                    if duration > self.max_audio_length:
                        logger.info(f"Skipping {json_path.stem}: duration {duration:.1f}s > max {self.max_audio_length}s")
                        continue
                
                samples.append({
                    "audio_path": str(audio_path),
                    "json_path": str(json_path),
                    "data": data,
                })
                
            except Exception as e:
                logger.warning(f"Error loading {json_path}: {e}")
                continue
        
        return samples
    
    def _format_transcription(self, segments: List[Dict], audio_duration: float) -> str:
        """
        Format transcription segments into JSON output format.
        
        This matches the expected model output format used in training.
        """
        formatted_segments = []
        
        for seg in segments:
            formatted_seg = {}
            # Add timestamp
            formatted_seg["Start"] = round(seg['start'], 2)
            formatted_seg["End"] = round(seg['end'], 2)
            # Add speaker if available
            if "speaker" in seg:
                formatted_seg["Speaker"] = seg["speaker"]
            # Add content
            formatted_seg["Content"] = seg.get("text", "")
            formatted_segments.append(formatted_seg)
        
        # Return as compact JSON string (no spaces after separators)
        return json.dumps(formatted_segments, ensure_ascii=False, separators=(',', ':'))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample for training.
        
        Returns:
            Dict with:
                - input_ids: Token IDs for input (system + user + assistant prompt)
                - labels: Token IDs for labels (-100 for non-predicted tokens)
                - acoustic_input_mask: Mask for speech token positions
                - speech: Raw audio array
                - vae_tok_len: Number of speech tokens
        """
        sample = self.samples[idx]
        data = sample["data"]
        audio_path = sample["audio_path"]
        
        # Prepare context info (customized_context)
        context_info = None
        if self.use_customized_context and "customized_context" in data:
            customized_context = data["customized_context"]
            if customized_context:
                context_info = "\n".join(customized_context)
        
        # Process audio using the processor's internal method
        encoding = self.processor._process_single_audio(
            audio_path,
            sampling_rate=None,
            add_generation_prompt=True,
            use_streaming=True,
            context_info=context_info,
        )
        
        # Get the input tokens (system + user + generation prompt)
        input_ids = encoding["input_ids"]
        acoustic_input_mask = encoding["acoustic_input_mask"]
        speech = encoding["speech"]
        vae_tok_len = encoding["vae_tok_len"]
        
        # Format the target transcription
        target_text = self._format_transcription(
            data["segments"],
            data.get("audio_duration", len(speech) / 24000)
        )
        
        # Encode target using apply_chat_template to match training format
        # This adds the assistant role tokens (e.g., <|im_start|>assistant\n...<|im_end|>)
        target_tokens = self.processor.tokenizer.apply_chat_template(
            [{"role": "assistant", "content": target_text}],
            tokenize=True,
            add_generation_prompt=False,
        )
        
        # Combine input and target
        full_input_ids = input_ids + target_tokens
        full_acoustic_mask = acoustic_input_mask + [0] * len(target_tokens)
        
        # Create labels: -100 for input tokens, actual tokens for target
        # We mask the input portion so loss is only computed on the response
        labels = [-100] * len(input_ids) + target_tokens
        
        return {
            "input_ids": full_input_ids,
            "labels": labels,
            "acoustic_input_mask": full_acoustic_mask,
            "speech": speech,
            "vae_tok_len": vae_tok_len,
        }


def get_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
) -> LoraConfig:
    """
    Create LoRA configuration for VibeVoice ASR model.
    
    We apply LoRA to the language model's attention layers and MLP,
    following common practices for LLM fine-tuning.
    
    Args:
        r: LoRA rank
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout for LoRA layers
        target_modules: List of module names to apply LoRA to
        
    Returns:
        LoraConfig object
    """
    if target_modules is None:
        # Target Qwen2 attention and MLP layers
        # These are the common targets for language model fine-tuning
        target_modules = [
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def setup_model_for_training(
    model_path: str,
    lora_config: LoraConfig,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    gradient_checkpointing: bool = True,
) -> Tuple[nn.Module, VibeVoiceASRProcessor]:
    """
    Load and prepare model for LoRA training.
    
    Args:
        model_path: Path to pretrained model
        lora_config: LoRA configuration
        device: Device to use
        dtype: Data type for model
        gradient_checkpointing: Whether to use gradient checkpointing
        
    Returns:
        Tuple of (model, processor)
    """
    logger.info(f"Loading model from {model_path}")
    
    # Load processor
    processor = VibeVoiceASRProcessor.from_pretrained(
        model_path,
        language_model_pretrained_name="Qwen/Qwen2.5-7B"
    )
    
    # Load model
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=device if device == "auto" else None,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    
    if device != "auto":
        model = model.to(device)
    
    # Freeze speech tokenizers (we only want to fine-tune the language model)
    for name, param in model.named_parameters():
        if "acoustic_tokenizer" in name or "semantic_tokenizer" in name:
            param.requires_grad = False
            logger.debug(f"Frozen: {name}")
    
    # Apply LoRA
    logger.info(f"Applying LoRA with config: r={lora_config.r}, alpha={lora_config.lora_alpha}")
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Enable gradient checkpointing if requested
    if gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    
    return model, processor


def train(
    model_args: ModelArguments,
    data_args: DataArguments,
    lora_args: LoraArguments,
    training_args: TrainingArguments,
    gradient_checkpointing: bool = True,
):
    """
    Main training function for LoRA fine-tuning.
    
    Args:
        model_args: Model configuration arguments
        data_args: Data configuration arguments
        lora_args: LoRA configuration arguments
        training_args: HuggingFace TrainingArguments
        gradient_checkpointing: Whether to use gradient checkpointing
    """
    # Set seed
    torch.manual_seed(training_args.seed)
    np.random.seed(training_args.seed)
    
    # Setup LoRA config
    lora_config = get_lora_config(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
    )
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model and processor
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    model, processor = setup_model_for_training(
        model_path=model_args.model_path,
        lora_config=lora_config,
        device=device,
        dtype=dtype,
        gradient_checkpointing=gradient_checkpointing,
    )
    
    # Create dataset
    train_dataset = VibeVoiceASRDataset(
        data_dir=data_args.data_dir,
        processor=processor,
        max_audio_length=data_args.max_audio_length,
        use_customized_context=data_args.use_customized_context,
    )
    
    if len(train_dataset) == 0:
        logger.error("No training samples found!")
        return
    
    # Create data collator
    data_collator = VibeVoiceASRDataCollator(
        processor=processor,
        pad_token_id=processor.pad_id,
    )
    
    # Set some sensible defaults for audio training
    training_args.dataloader_num_workers = 0  # Audio loading can be tricky with multiprocessing
    training_args.remove_unused_columns = False  # Keep all columns
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Train
    logger.info("Starting training...")
    logger.info(f"  Num samples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {training_args.num_train_epochs}")
    logger.info(f"  Batch size = {training_args.per_device_train_batch_size}")
    logger.info(f"  Gradient accumulation steps = {training_args.gradient_accumulation_steps}")
    total_steps = len(train_dataset) * int(training_args.num_train_epochs) // (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    )
    logger.info(f"  Total optimization steps = {total_steps}")
    
    train_result = trainer.train()
    
    # Save final model
    logger.info(f"Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Save processor config
    processor.save_pretrained(training_args.output_dir)
    
    logger.info("Training complete!")
    
    return model, processor


def main():
    # Use HfArgumentParser to parse all argument dataclasses
    parser = HfArgumentParser((ModelArguments, DataArguments, LoraArguments, TrainingArguments))
    model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()
    
    # Run training
    train(
        model_args=model_args,
        data_args=data_args,
        lora_args=lora_args,
        training_args=training_args,
    )


if __name__ == "__main__":
    main()
