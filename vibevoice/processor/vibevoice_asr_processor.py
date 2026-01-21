"""
Processor class for VibeVoice ASR models.
"""

import os
import json
import math
import warnings
from typing import List, Optional, Union, Dict, Any, Tuple

import numpy as np
import torch

from transformers.tokenization_utils_base import BatchEncoding
from transformers.utils import TensorType, logging
from .vibevoice_tokenizer_processor import VibeVoiceTokenizerProcessor, AudioNormalizer

try:
    from .audio_utils import load_audio_use_ffmpeg
    HAS_FFMPEG_UTILS = True
except ImportError:
    HAS_FFMPEG_UTILS = False
    warnings.warn("audio_utils not available, will fall back to soundfile for audio loading")

logger = logging.get_logger(__name__)

SYSTEM_PROMPT = "You are a helpful assistant that transcribes audio input into text output in JSON format."


class VibeVoiceASRProcessor: 
    """
    Processor for VibeVoice ASR (Automatic Speech Recognition) models.
    
    This processor handles audio preprocessing and tokenization for ASR tasks,
    following the exact format used in training with proper chat templates.
    
    Args:
        tokenizer: The text tokenizer for processing text
        audio_processor: The audio processor for processing speech
        speech_tok_compress_ratio (int): Compression ratio for speech tokenization
        target_sample_rate (int): Target sample rate for audio
        normalize_audio (bool): Whether to normalize audio input
    """
    
    def __init__(
        self,
        tokenizer=None,
        audio_processor=None,
        speech_tok_compress_ratio=320,
        target_sample_rate=24000,
        normalize_audio=True,
        **kwargs
    ):
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor or VibeVoiceTokenizerProcessor(
            sampling_rate=target_sample_rate,
            normalize_audio=normalize_audio
        )
        self.speech_tok_compress_ratio = speech_tok_compress_ratio
        self.target_sample_rate = target_sample_rate
        self.normalize_audio = normalize_audio
        
        if normalize_audio:
            self.audio_normalizer = AudioNormalizer()
        else:
            self.audio_normalizer = None
        
        # Cache special token IDs
        self._cache_special_tokens()
    
    def _cache_special_tokens(self):
        """Cache special token IDs for efficiency."""
        # Add safety checks for special tokens
        if hasattr(self.tokenizer, 'speech_start_id'):
            self.speech_start_id = self.tokenizer.speech_start_id
        else:
            self.speech_start_id = self.tokenizer.convert_tokens_to_ids("<|speech_start|>")
            
        if hasattr(self.tokenizer, 'speech_end_id'):
            self.speech_end_id = self.tokenizer.speech_end_id
        else:
            self.speech_end_id = self.tokenizer.convert_tokens_to_ids("<|speech_end|>")
            
        if hasattr(self.tokenizer, 'speech_pad_id'):
            self.speech_pad_id = self.tokenizer.speech_pad_id
        else:
            self.speech_pad_id = self.tokenizer.convert_tokens_to_ids("<|speech_pad|>")
            
        if hasattr(self.tokenizer, 'pad_id'):
            self.pad_id = self.tokenizer.pad_id
        elif hasattr(self.tokenizer, 'pad_token_id'):
            self.pad_id = self.tokenizer.pad_token_id
        else:
            self.pad_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Load processor from a pretrained model path.
        
        Args:
            pretrained_model_name_or_path: Path to the pretrained model
            **kwargs: Additional keyword arguments
            
        Returns:
            VibeVoiceASRProcessor: The loaded processor
        """
        import json
        from transformers.utils import cached_file
        from vibevoice.modular.modular_vibevoice_text_tokenizer import VibeVoiceASRTextTokenizerFast
        
        # Try to load configuration
        config_path = os.path.join(pretrained_model_name_or_path, "preprocessor_config.json")
        config = {}
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            try:
                config_file = cached_file(
                    pretrained_model_name_or_path,
                    "preprocessor_config.json",
                    **kwargs
                )
                with open(config_file, 'r') as f:
                    config = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load preprocessor_config.json: {e}")
                logger.warning("Using default configuration")
        
        # Extract parameters
        speech_tok_compress_ratio = config.get("speech_tok_compress_ratio", 3200)
        target_sample_rate = config.get("target_sample_rate", 24000)
        normalize_audio = config.get("normalize_audio", True)
        
        # Load tokenizer
        language_model_pretrained_name = config.get("language_model_pretrained_name", None) or kwargs.pop("language_model_pretrained_name", "Qwen/Qwen2.5-1.5B")
        logger.info(f"Loading tokenizer from {language_model_pretrained_name}")
        
        if 'qwen' in language_model_pretrained_name.lower():
            tokenizer = VibeVoiceASRTextTokenizerFast.from_pretrained(
                language_model_pretrained_name,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported tokenizer type for {language_model_pretrained_name}")
        
        # Load audio processor
        audio_processor = VibeVoiceTokenizerProcessor(
            sampling_rate=target_sample_rate,
            normalize_audio=normalize_audio,
            target_dB_FS=config.get("target_dB_FS", -25),
            eps=config.get("eps", 1e-6),
        )
        
        return cls(
            tokenizer=tokenizer,
            audio_processor=audio_processor,
            speech_tok_compress_ratio=speech_tok_compress_ratio,
            target_sample_rate=target_sample_rate,
            normalize_audio=normalize_audio,
        )
    
    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        """
        Save processor configuration to a directory.
        
        Args:
            save_directory: Directory to save the configuration
            **kwargs: Additional keyword arguments
        """
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save processor configuration
        processor_config = {
            "processor_class": "VibeVoiceASRProcessor",
            "speech_tok_compress_ratio": self.speech_tok_compress_ratio,
            "target_sample_rate": self.target_sample_rate,
            "normalize_audio": self.normalize_audio,
            "target_dB_FS": -25,
            "eps": 1e-6,
        }
        
        config_path = os.path.join(save_directory, "preprocessor_config.json")
        with open(config_path, 'w') as f:
            json.dump(processor_config, f, indent=2)
        
        logger.info(f"Processor configuration saved in {config_path}")
    
    def __call__(
        self,
        audio: Optional[Union[str, np.ndarray, torch.Tensor, List[Union[str, np.ndarray, torch.Tensor]]]] = None,
        sampling_rate: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        padding: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
        add_generation_prompt: bool = True,
        use_streaming: bool = True,
        context_info: Optional[str] = None,
        **kwargs
    ) -> BatchEncoding:
        """
        Process audio input for ASR model.
        
        Args:
            audio: Audio input(s). Can be:
                - str: Path to audio file
                - np.ndarray: Audio array
                - torch.Tensor: Audio tensor
                - List of the above for batch processing
            sampling_rate: Sampling rate of input audio
            return_tensors: Output format ('pt' for PyTorch, 'np' for NumPy)
            padding: Whether to pad batch inputs
            max_length: Maximum sequence length
            truncation: Whether to truncate long sequences
            add_generation_prompt: Whether to add generation prompt for inference
            use_streaming: Whether to use streaming mode (True by default, auto False if <60s)
            context_info: Optional context information (e.g., hotwords, metadata) to help transcription
            
        Returns:
            BatchEncoding with:
                - input_ids: Token IDs for the model
                - attention_mask: Attention mask
                - acoustic_input_mask: Mask indicating speech token positions
                - speech_tensors: Processed speech features
                - speech_masks: Valid speech masks
                - vae_tok_seqlens: Length of each speech segment in tokens
        """
        if audio is None:
            raise ValueError("Audio input is required for ASR processing")
        
        # Handle single vs batch input
        if isinstance(audio, list):
            is_batched = True
            audio_list = audio
        else:
            is_batched = False
            audio_list = [audio]
        
        # Process each audio input
        all_encodings = []
        for audio_input in audio_list:
            encoding = self._process_single_audio(
                audio_input,
                sampling_rate=sampling_rate,
                add_generation_prompt=add_generation_prompt,
                use_streaming=use_streaming,
                context_info=context_info,
            )
            all_encodings.append(encoding)
        
        # Combine into batch
        batch_encoding = self._batch_encode(
            all_encodings,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        
        return batch_encoding
    
    def _process_single_audio(
        self,
        audio: Union[str, np.ndarray, torch.Tensor],
        sampling_rate: Optional[int] = None,
        add_generation_prompt: bool = True,
        use_streaming: bool = True,
        context_info: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a single audio input.
        
        Args:
            audio: Single audio input
            sampling_rate: Audio sampling rate
            add_generation_prompt: Whether to add generation prompt
            context_info: Optional context information (e.g., hotwords, metadata) to help transcription
            
        Returns:
            Dictionary with processed tokens and audio features
        """
        # Process audio through audio processor
        if isinstance(audio, str):
            # Load from file using ffmpeg for better format support
            if HAS_FFMPEG_UTILS:
                try:
                    audio_array, file_sr = load_audio_use_ffmpeg(audio, resample=False)
                except Exception as e:
                    # Fall back to soundfile if ffmpeg fails
                    warnings.warn(f"ffmpeg loading failed, falling back to soundfile: {e}")
                    import soundfile as sf
                    audio_array, file_sr = sf.read(audio)
                    if audio_array.ndim > 1:
                        audio_array = audio_array.mean(axis=1)  # Convert to mono
            else:
                import soundfile as sf
                audio_array, file_sr = sf.read(audio)
                if audio_array.ndim > 1:
                    audio_array = audio_array.mean(axis=1)  # Convert to mono
            
            # Resample if needed
            if file_sr != self.target_sample_rate:
                import librosa
                audio_array = librosa.resample(
                    audio_array, 
                    orig_sr=file_sr, 
                    target_sr=self.target_sample_rate
                )
        elif isinstance(audio, torch.Tensor):
            audio_array = audio.cpu().numpy()
            if audio_array.ndim > 1:
                audio_array = audio_array.squeeze()
        else:
            audio_array = np.array(audio, dtype=np.float32)
            if audio_array.ndim > 1:
                audio_array = audio_array.squeeze()
        
        # Ensure float32
        audio_array = audio_array.astype(np.float32)
        
        # Normalize if needed
        if self.normalize_audio and self.audio_normalizer:
            audio_array = self.audio_normalizer(audio_array)
        
        # Calculate audio duration
        audio_duration = len(audio_array) / self.target_sample_rate
        
        # Auto-disable streaming for short audio (<60s)
        if use_streaming and audio_duration < 60.0:
            use_streaming = False
        
        # Calculate token length based on streaming mode
        # Non-streaming: uses ceil (encoder adds extra_padding for stride alignment)
        # Streaming: uses floor (segments processed independently, no global alignment)
        # if use_streaming:
        #     vae_tok_len = len(audio_array) // self.speech_tok_compress_ratio
        # else:
        vae_tok_len = math.ceil(len(audio_array) / self.speech_tok_compress_ratio)
        
        # Build token sequence following training format
        # 1. System prompt - use apply_chat_template then encode like in training
        system_prompt_text = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}],
            tokenize=False
        )
        system_tokens = self.tokenizer.encode(system_prompt_text)
        
        # 2. User input with speech tokens
        # Build speech placeholder string
        sp_start_token = self.tokenizer.convert_ids_to_tokens(self.speech_start_id)
        sp_pad_token = self.tokenizer.convert_ids_to_tokens(self.speech_pad_id)
        sp_end_token = self.tokenizer.convert_ids_to_tokens(self.speech_end_id)
        
        # User suffix with audio duration info
        show_keys = ['Start time', 'End time', 'Speaker ID', 'Content']
        if context_info and context_info.strip():
            user_suffix = f"This is a {audio_duration:.2f} seconds audio, with extra info: {context_info.strip()}\n\nPlease transcribe it with these keys: " + ", ".join(show_keys)
        else:
            user_suffix = f"This is a {audio_duration:.2f} seconds audio, please transcribe it with these keys: " + ", ".join(show_keys)
        
        user_input_string = ''.join(
            [sp_start_token] + [sp_pad_token] * vae_tok_len + [sp_end_token]
        ) + '\n' + user_suffix
        
        user_tokens = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_input_string}],
            tokenize=True
        )
        
        # Combine tokens
        full_tokens = system_tokens + user_tokens
        
        # Create acoustic input mask
        acoustic_input_mask = [1 if token == self.speech_pad_id else 0 for token in full_tokens]
        
        return {
            "input_ids": full_tokens,
            "acoustic_input_mask": acoustic_input_mask,
            "speech": audio_array,
            "vae_tok_len": vae_tok_len,
        }
    
    def _batch_encode(
        self,
        encodings: List[Dict[str, Any]],
        padding: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
        return_tensors: Optional[str] = None,
    ) -> BatchEncoding:
        """
        Combine multiple encodings into a batch.
        
        Args:
            encodings: List of encoded samples
            padding: Whether to pad sequences
            max_length: Maximum sequence length
            truncation: Whether to truncate
            return_tensors: Output format
            
        Returns:
            BatchEncoding with batched data
        """
        # Extract components
        input_ids_list = [enc["input_ids"] for enc in encodings]
        acoustic_masks_list = [enc["acoustic_input_mask"] for enc in encodings]
        speech_list = [enc["speech"] for enc in encodings]
        vae_tok_lens = [enc["vae_tok_len"] for enc in encodings]
        
        # Determine max length for padding
        if padding:
            if max_length is not None:
                target_length = max_length
            else:
                target_length = max(len(ids) for ids in input_ids_list)
            
            # Pad sequences
            padded_input_ids = []
            padded_acoustic_masks = []
            attention_masks = []
            
            for input_ids, acoustic_mask in zip(input_ids_list, acoustic_masks_list):
                # Truncate if needed
                if truncation and len(input_ids) > target_length:
                    input_ids = input_ids[:target_length]
                    acoustic_mask = acoustic_mask[:target_length]
                
                # Pad sequences to left (for autoregressive generation)
                padding_length = target_length - len(input_ids)
                padded_ids = [self.pad_id] * padding_length + input_ids
                padded_acoustic = [0] * padding_length + acoustic_mask
                attention_mask = [0] * padding_length + [1] * len(input_ids)
                
                padded_input_ids.append(padded_ids)
                padded_acoustic_masks.append(padded_acoustic)
                attention_masks.append(attention_mask)
            
            input_ids_list = padded_input_ids
            acoustic_masks_list = padded_acoustic_masks
        else:
            attention_masks = [[1] * len(ids) for ids in input_ids_list]
        
        # Process speech tensors - raw audio is 1D, so we keep it as is
        max_speech_length = max(len(s) for s in speech_list)
        padded_speeches = np.zeros((len(speech_list), max_speech_length), dtype=np.float32)
        speech_masks = np.zeros((len(speech_list), max(vae_tok_lens)), dtype=bool)
        
        for i, (speech, vae_len) in enumerate(zip(speech_list, vae_tok_lens)):
            padded_speeches[i, :len(speech)] = speech
            speech_masks[i, :vae_len] = True
        
        # Create batch encoding
        batch_encoding = BatchEncoding()
        
        if return_tensors == "pt":
            batch_encoding["input_ids"] = torch.tensor(input_ids_list, dtype=torch.long)
            batch_encoding["attention_mask"] = torch.tensor(attention_masks, dtype=torch.long)
            batch_encoding["acoustic_input_mask"] = torch.tensor(acoustic_masks_list, dtype=torch.bool)
            batch_encoding["speech_tensors"] = torch.tensor(padded_speeches, dtype=torch.float32)
            batch_encoding["speech_masks"] = torch.tensor(speech_masks, dtype=torch.bool)
            # Note: vae_tok_seqlens and speech_type are not included as they are not model inputs
        else:
            batch_encoding["input_ids"] = input_ids_list if len(input_ids_list) > 1 else input_ids_list[0]
            batch_encoding["attention_mask"] = attention_masks if len(attention_masks) > 1 else attention_masks[0]
            batch_encoding["acoustic_input_mask"] = acoustic_masks_list if len(acoustic_masks_list) > 1 else acoustic_masks_list[0]
            batch_encoding["speech_tensors"] = padded_speeches if len(padded_speeches) > 1 else padded_speeches[0]
            batch_encoding["speech_masks"] = speech_masks if len(speech_masks) > 1 else speech_masks[0]
        
        return batch_encoding
    
    def batch_decode(self, *args, **kwargs):
        """
        Decode batch of token IDs to text.
        Forwards to tokenizer's batch_decode method.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)
    
    def decode(self, *args, **kwargs):
        """
        Decode token IDs to text.
        Forwards to tokenizer's decode method.
        """
        return self.tokenizer.decode(*args, **kwargs)
    
    def post_process_transcription(self, text: str) -> List[Dict[str, Any]]:
        """
        Post-process the generated transcription text to extract structured data.
        
        Args:
            text: Generated text from the model
            
        Returns:
            List of dictionaries with transcription segments
        """
        try:
            # Try to parse as JSON
            if "```json" in text:
                # Extract JSON from markdown code block
                json_start = text.find("```json") + 7
                json_end = text.find("```", json_start)
                json_str = text[json_start:json_end].strip()
            else:
                # Try to find JSON array or object
                json_start = text.find("[")
                if json_start == -1:
                    json_start = text.find("{")
                if json_start != -1:
                    # Find matching closing bracket
                    bracket_count = 0
                    json_end = json_start
                    for i in range(json_start, len(text)):
                        if text[i] in "[{":
                            bracket_count += 1
                        elif text[i] in "]}":
                            bracket_count -= 1
                            if bracket_count == 0:
                                json_end = i + 1
                                break
                    json_str = text[json_start:json_end]
                else:
                    json_str = text
            
            # Parse JSON
            result = json.loads(json_str)
            
            # Ensure it's a list
            if isinstance(result, dict):
                result = [result]
            
            # Validate and clean up the result
            cleaned_result = []
            for item in result:
                if isinstance(item, dict):
                    cleaned_item = {}
                    # Map keys to expected format
                    key_mapping = {
                        "Start time": "start_time",
                        "Start": "start_time",
                        "End time": "end_time",
                        "End": "end_time",
                        "Speaker ID": "speaker_id",
                        "Speaker": "speaker_id",
                        "Content": "text",
                    }
                    for key, mapped_key in key_mapping.items():
                        if key in item:
                            cleaned_item[mapped_key] = item[key]
                    
                    if cleaned_item:
                        cleaned_result.append(cleaned_item)
            
            return cleaned_result
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from transcription: {e}")
            logger.debug(f"Raw text: {text}")
            return []
        except Exception as e:
            logger.warning(f"Error post-processing transcription: {e}")
            return []
    
    @property
    def model_input_names(self):
        """Return the list of inputs accepted by the model."""
        return ["input_ids", "attention_mask", "acoustic_input_mask", "speech_tensors", "speech_masks"]

__all__ = ["VibeVoiceASRProcessor"]
