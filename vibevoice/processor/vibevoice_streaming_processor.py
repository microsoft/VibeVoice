import math
import warnings
from typing import List, Optional, Union, Dict, Any, Tuple
import os
import re

import numpy as np
import torch

from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from transformers.utils import TensorType, logging
from .vibevoice_tokenizer_processor import AudioNormalizer

logger = logging.get_logger(__name__)


class VibeVoiceStreamingProcessor:
    r"""
    Constructs a VibeVoice Streaming processor which wraps a VibeVoice tokenizer and audio processor into a single processor.

    Args:
        tokenizer (`VibeVoiceTextTokenizer` or `VibeVoiceTextTokenizerFast`):
            The tokenizer for text processing.
        audio_processor (`VibeVoiceTokenizerProcessor`):
            The audio processor for speech processing.
        speech_tok_compress_ratio (`int`, *optional*, defaults to 3200):
            The compression ratio for speech tokenization.
        db_normalize (`bool`, *optional*, defaults to True):
            Whether to apply decibel normalization to audio inputs.
    """

    def __init__(self, tokenizer=None, audio_processor=None, speech_tok_compress_ratio=3200, db_normalize=True, **kwargs):
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor
        self.speech_tok_compress_ratio = speech_tok_compress_ratio
        self.db_normalize = db_normalize
        self.audio_normalizer = AudioNormalizer() if db_normalize else None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Instantiate a VibeVoiceStreamingProcessor from a pretrained VibeVoice Streaming processor.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:
                - a string, the *model id* of a pretrained model
                - a path to a *directory* containing processor config

        Returns:
            [`VibeVoiceStreamingProcessor`]: The processor object instantiated from pretrained model.
        """
        import os
        import json
        from transformers.utils import cached_file
        from .vibevoice_tokenizer_processor import VibeVoiceTokenizerProcessor
        from vibevoice.modular.modular_vibevoice_text_tokenizer import (
            VibeVoiceTextTokenizer, 
            VibeVoiceTextTokenizerFast
        )
        
        # Try to load from local path first, then from HF hub
        config_path = os.path.join(pretrained_model_name_or_path, "preprocessor_config.json")
        config = None
        
        if os.path.exists(config_path):
            # Local path exists
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            # Try to load from HF hub
            try:
                config_file = cached_file(
                    pretrained_model_name_or_path,
                    "preprocessor_config.json",
                    **kwargs
                )
                with open(config_file, 'r') as f:
                    config = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load preprocessor_config.json from {pretrained_model_name_or_path}: {e}")
                logger.warning("Using default configuration")
                config = {
                    "speech_tok_compress_ratio": 3200,
                    "db_normalize": True,
                }
        
        # Extract main processor parameters
        speech_tok_compress_ratio = config.get("speech_tok_compress_ratio", 3200)
        db_normalize = config.get("db_normalize", True)
        
        # Load tokenizer - try from model path first, then fallback to Qwen        
        language_model_pretrained_name = config.get("language_model_pretrained_name", None) or kwargs.pop("language_model_pretrained_name", "Qwen/Qwen2.5-1.5B")
        logger.info(f"Loading tokenizer from {language_model_pretrained_name}")
        if 'qwen' in language_model_pretrained_name.lower():
            tokenizer = VibeVoiceTextTokenizerFast.from_pretrained(
                language_model_pretrained_name,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported tokenizer type for {language_model_pretrained_name}. Supported types: Qwen, Llama, Gemma.")
        
        # Load audio processor
        if "audio_processor" in config:
            # Create audio processor from config
            audio_config = config["audio_processor"]
            audio_processor = VibeVoiceTokenizerProcessor(
                sampling_rate=audio_config.get("sampling_rate", 24000),
                normalize_audio=audio_config.get("normalize_audio", True),
                target_dB_FS=audio_config.get("target_dB_FS", -25),
                eps=audio_config.get("eps", 1e-6),
            )
        else:
            # Create default audio processor
            audio_processor = VibeVoiceTokenizerProcessor()
        
        # Create and return the processor
        return cls(
            tokenizer=tokenizer,
            audio_processor=audio_processor,
            speech_tok_compress_ratio=speech_tok_compress_ratio,
            db_normalize=db_normalize,
        )
    
    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        """
        Save a processor to a directory, so that it can be re-loaded using the
        [`~VibeVoiceStreamingProcessor.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the processor will be saved.
        """
        import os
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save processor configuration
        processor_config = {
            "processor_class": "VibeVoiceStreamingProcessor",
            "speech_tok_compress_ratio": self.speech_tok_compress_ratio,
            "db_normalize": self.db_normalize,
            "audio_processor": {
                "feature_extractor_type": "VibeVoiceTokenizerProcessor",
                "sampling_rate": getattr(self.audio_processor, 'sampling_rate', 24000),
                "normalize_audio": getattr(self.audio_processor, 'normalize_audio', True),
                "target_dB_FS": getattr(self.audio_processor, 'target_dB_FS', -25),
                "eps": getattr(self.audio_processor, 'eps', 1e-6),
            }
        }
        
        config_path = os.path.join(save_directory, "preprocessor_config.json")
        with open(config_path, 'w') as f:
            json.dump(processor_config, f, indent=2)
        
        logger.info(f"Processor configuration saved in {config_path}")
    
    def __call__(self) -> BatchEncoding:
        """
        Note:
            This method is intentionally not implemented in the streaming processor.
            Use `process_input_with_cached_prompt` for streaming use cases.
        """
        raise NotImplementedError(
            "VibeVoiceStreamingProcessor.__call__ is not implemented. "
            "Use process_input_with_cached_prompt for streaming inputs."
        )

    def process_input_with_cached_prompt(
        self,
        text: Optional[str] = None,
        cached_prompt: Optional[Dict[str, Any]] = None,
        padding: Union[bool, str, PaddingStrategy] = True,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_attention_mask: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        """
        Main method to process one text script based on cached prompt. The function currently only supports single examples.

        Args:
            text (`str`):
                The input text to process.
            cached_prompt (`Dict[str, Any]`, *optional*):
                The cached prompt to use for processing. It contains the kv cache of the voice prompt.
            padding (`bool`, `str` or `PaddingStrategy`, defaults to `True`):
                Whether to pad sequences to the same length
            truncation (`bool`, `str` or `TruncationStrategy`, defaults to `False`):
                Whether to truncate sequences
            max_length (`int`, *optional*):
                Maximum length of the returned sequences
            return_tensors (`str` or `TensorType`, *optional*):
                If set, will return tensors of a particular framework
            return_attention_mask (`bool`, defaults to `True`):
                Whether to return the attention mask

        Returns:
            `BatchEncoding`: A BatchEncoding with the following fields:
                - **input_ids** -- List of token id sequences or tensor
                - **attention_mask** -- List of attention masks or tensor
                - **tts_lm_input_ids** -- List of token id sequences or tensor used for TTS LM
                - **tts_lm_attention_mask** -- List of attention masks or tensor used for TTS LM
                - **tts_text_ids** -- List of token id sequences or tensor for TTS text input
                - **speech_tensors** -- Padded speech inputs (if voice_samples provided)
                - **speech_masks** -- Speech masks (if voice_samples provided)
                - **speech_input_mask** -- Boolean masks indicating speech token positions
        """
        # Only support single example
        texts = [text]
        cached_prompts = [cached_prompt]
        is_batched = False
        
        # Process each input
        all_encodings = []
        for text_input, cached_prompt_input in zip(texts, cached_prompts):
            script_tokens = self.tokenizer.encode(text_input.strip() + "\n", add_special_tokens=False)
            input_id_length = cached_prompt_input['lm']['last_hidden_state'].size(1)
            tts_lm_input_id_length = cached_prompt_input['tts_lm']['last_hidden_state'].size(1)

            # psudo input ids and masks
            input_ids = [self.tokenizer.pad_id] * input_id_length
            tts_lm_input_ids = [self.tokenizer.pad_id] * tts_lm_input_id_length
            speech_input_mask = [False] * tts_lm_input_id_length

            encoding = {
                        "input_ids": input_ids,
                        "tts_lm_input_ids": tts_lm_input_ids,
                        "tts_text_ids": script_tokens,
                        "speech_inputs": None,
                        "speech_input_mask": speech_input_mask,
                    }
            all_encodings.append(encoding)
            
        # Combine batch
        batch_encoding = self._batch_encode(
            all_encodings,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            return_attention_mask=return_attention_mask,
        )
        
        return batch_encoding
    
    def _batch_encode(
        self,
        encodings: List[Dict[str, Any]],
        padding: Union[bool, str, PaddingStrategy] = True,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_attention_mask: bool = True,
    ) -> BatchEncoding:
        """Combine multiple encodings into a batch with padding."""
        # Extract input_ids and create attention_mask
        input_ids_list = [enc["input_ids"] for enc in encodings]
        tts_lm_input_ids_list = [enc["tts_lm_input_ids"] for enc in encodings]
        tts_text_ids_list = [enc["tts_text_ids"] for enc in encodings]
        speech_input_masks_list = [enc["speech_input_mask"] for enc in encodings]
        
        attention_masks = [[1] * len(ids) for ids in input_ids_list] if return_attention_mask else None
        tts_lm_attention_masks = [[1] * len(ids) for ids in tts_lm_input_ids_list] if return_attention_mask else None
            
        # Process speech inputs
        all_speech_inputs = []
        has_speech = False
        for enc in encodings:
            if enc["speech_inputs"] is not None:
                all_speech_inputs.extend(enc["speech_inputs"])
                has_speech = True
                
        # Prepare batch encoding
        batch_encoding = BatchEncoding()
        
        # Handle tensor conversion
        if return_tensors is not None:
            batch_encoding["input_ids"] = torch.tensor(input_ids_list, dtype=torch.long)
            batch_encoding["tts_lm_input_ids"] = torch.tensor(tts_lm_input_ids_list, dtype=torch.long)
            batch_encoding["tts_text_ids"] = torch.tensor(tts_text_ids_list, dtype=torch.long)

            if return_attention_mask and attention_masks is not None:
                batch_encoding["attention_mask"] = torch.tensor(attention_masks, dtype=torch.long)
                batch_encoding["tts_lm_attention_mask"] = torch.tensor(tts_lm_attention_masks, dtype=torch.long)
            
            batch_encoding["speech_input_mask"] = torch.tensor(speech_input_masks_list, dtype=torch.bool)
        else:
            batch_encoding["input_ids"] = input_ids_list
            batch_encoding["tts_lm_input_ids"] = tts_lm_input_ids_list
            batch_encoding["tts_text_ids"] = tts_text_ids_list
            if return_attention_mask and attention_masks is not None:
                batch_encoding["attention_mask"] = attention_masks
                batch_encoding["tts_lm_attention_mask"] = tts_lm_attention_masks
            batch_encoding["speech_input_mask"] = speech_input_masks_list
            
        # Process speech tensors if present
        if has_speech:
            speech_dict = self.prepare_speech_inputs(
                all_speech_inputs,
                return_tensors=return_tensors,
            )
            batch_encoding["speech_tensors"] = speech_dict["padded_speeches"]
            batch_encoding["speech_masks"] = speech_dict["speech_masks"]
        else:
            batch_encoding["speech_tensors"] = None
            batch_encoding["speech_masks"] = None
            
        return batch_encoding

    def prepare_speech_inputs(
        self,
        speech_inputs: List[np.ndarray],
        return_tensors: Optional[Union[str, TensorType]] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Any]:
        """
        Prepare speech inputs for model consumption.
        
        Args:
            speech_inputs: List of speech arrays
            return_tensors: Output tensor type
            device: Device to place tensors on
            dtype: Data type for tensors
            
        Returns:
            Dictionary with padded_speeches and speech_masks
        """
        if not speech_inputs:
            return {"padded_speeches": None, "speech_masks": None}
        
        # Calculate sequence lengths
        vae_tok_seqlens = [math.ceil(s.shape[0] / self.speech_tok_compress_ratio) for s in speech_inputs]
        # vae_tok_seqlens = [math.ceil(s.shape[0] / self.speech_tok_compress_ratio) if s.ndim == 1 else s.shape[0] for s in speech_inputs]
        max_speech_length = max(s.shape[0] for s in speech_inputs)
        
        # Pad speeches
        if speech_inputs[0].ndim == 1:
            padded_speeches = np.full((len(speech_inputs), max_speech_length), fill_value=0, dtype=np.float32)
        else:
            padded_speeches = np.full((len(speech_inputs), max_speech_length, speech_inputs[0].shape[-1]), fill_value=0, dtype=np.float32)
        speech_masks = np.zeros((len(speech_inputs), max(vae_tok_seqlens)), dtype=np.bool_)
        
        for i, (speech, vae_tok_length) in enumerate(zip(speech_inputs, vae_tok_seqlens)):
            padded_speeches[i, :len(speech)] = speech
            speech_masks[i, :vae_tok_length] = True
        
        result = {
            "padded_speeches": padded_speeches,
            "speech_masks": speech_masks,
        }
        
        # Convert to tensors if requested
        if return_tensors == "pt":
            result["padded_speeches"] = torch.tensor(padded_speeches, device=device, dtype=dtype or torch.float32)
            result["speech_masks"] = torch.tensor(speech_masks, device=device, dtype=torch.bool)
            
        return result

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to VibeVoiceTextTokenizer's [`~PreTrainedTokenizer.batch_decode`].
        Please refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to VibeVoiceTextTokenizer's [`~PreTrainedTokenizer.decode`].
        Please refer to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        """
        Return the list of inputs accepted by the model.
        """
        tokenizer_input_names = self.tokenizer.model_input_names
        audio_processor_input_names = self.audio_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + audio_processor_input_names + ["speech_inputs", "speech_input_mask"]))

    def save_audio(self, 
        audio: Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]],
        output_path: str = "output.wav",
        sampling_rate: Optional[int] = None,
        normalize: bool = False,
        batch_prefix: str = "audio_",
    ) -> str:
        """
        Save audio data to a file.
        Args:
            audio (Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]]):
                The audio data to save. Can be a single tensor/array or a list of them.
            output_path (str, optional): Path to save the audio file. Defaults to "output.wav".
            sampling_rate (int, optional): Sampling rate for the audio. If None, uses the processor's default.
            normalize (bool, optional): Whether to normalize the audio before saving. Defaults to False.
            batch_prefix (str, optional): Prefix for batch audio files. Defaults to "audio_".
        Returns:
            str: The path to the saved audio file.
        """
        return self.audio_processor.save_audio(audio, output_path=output_path, sampling_rate=sampling_rate, normalize=normalize, batch_prefix=batch_prefix)
    
__all__ = [
    "VibeVoiceStreamingProcessor",
]