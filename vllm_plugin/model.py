"""
VibeVoice vLLM Plugin Model - Native Multimodal Integration

This module implements the VibeVoice ASR model with full vLLM multimodal registry
integration for speech-to-text inference.
"""

from typing import List, Optional, Tuple, Union, Dict, Any, Iterable, Mapping, Sequence
import os
import torch
import torch.nn as nn
import numpy as np
import base64


# ============================================================================
# Audio Loading: FFmpeg-based AudioMediaIO
# ============================================================================
# VibeVoice uses FFmpeg for audio decoding to ensure consistent behavior
# across different audio formats (MP3, WAV, FLAC, etc.).


from vibevoice.processor.audio_utils import load_audio_use_ffmpeg, load_audio_bytes_use_ffmpeg, AudioNormalizer


def _ffmpeg_load_bytes(data: bytes) -> tuple[np.ndarray, int]:
    """Load audio bytes using FFmpeg via stdin-pipe decoding.
    
    Returns:
        Tuple of (audio_waveform, sample_rate). Sample rate is always 24000.
    """
    audio, sr = load_audio_bytes_use_ffmpeg(data, resample=True, target_sr=24000)
    normalizer = AudioNormalizer()
    audio = normalizer(audio)
    return audio, sr

def _ffmpeg_load_file(filepath) -> tuple[np.ndarray, int]:
    """Load audio file using FFmpeg.
    
    Returns:
        Tuple of (audio_waveform, sample_rate). Sample rate is always 24000.
    """
    audio, sr = load_audio_use_ffmpeg(str(filepath), resample=True, target_sr=24000)
    normalizer = AudioNormalizer()
    audio = normalizer(audio)
    return audio, sr

# Register FFmpeg-based audio loader
import vllm.multimodal.audio as _vllm_audio_module
_OriginalAudioMediaIO = _vllm_audio_module.AudioMediaIO

class _PatchedAudioMediaIO(_OriginalAudioMediaIO):
    """AudioMediaIO implementation using FFmpeg for audio decoding."""
    
    def load_bytes(self, data: bytes) -> tuple[np.ndarray, int]:
        return _ffmpeg_load_bytes(data)
    
    def load_base64(self, media_type: str, data: str) -> tuple[np.ndarray, int]:
        return _ffmpeg_load_bytes(base64.b64decode(data))
    
    def load_file(self, filepath) -> tuple[np.ndarray, int]:
        return _ffmpeg_load_file(filepath)

# Replace globally
_vllm_audio_module.AudioMediaIO = _PatchedAudioMediaIO

# Also patch in utils module where it's imported
import vllm.multimodal.utils as _vllm_utils_module
_vllm_utils_module.AudioMediaIO = _PatchedAudioMediaIO

# ============================================================================

from transformers import BatchFeature
from transformers.models.whisper import WhisperFeatureExtractor
from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.parse import MultiModalDataParser
from vllm.sequence import IntermediateTensors
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP, MultiModalEmbeddings
from vllm.model_executor.models.utils import (
    init_vllm_registered_model,
    maybe_prefix,
    AutoWeightsLoader,
    WeightsMapper,
)
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs

# Import VibeVoice components
from vibevoice.modular.modular_vibevoice_tokenizer import (
    VibeVoiceAcousticTokenizerModel,
    VibeVoiceSemanticTokenizerModel,
    VibeVoiceTokenizerStreamingCache,
    VibeVoiceTokenizerEncoderOutput,
)
from vibevoice.modular.configuration_vibevoice import (
    VibeVoiceAcousticTokenizerConfig,
    VibeVoiceSemanticTokenizerConfig,
)


class SpeechConnector(nn.Module):
    """Projects speech features to language model hidden dimension.
    
    Architecture: fc1 -> RMSNorm -> fc2 (no activation function)
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.norm = LlamaRMSNorm(output_dim, eps=1e-6)
        self.fc2 = nn.Linear(output_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.norm(x)
        x = self.fc2(x)
        return x


class LlamaRMSNorm(nn.Module):
    """RMSNorm layer used in SpeechConnector."""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class VibeVoiceAudioEncoder(nn.Module):
    """
    VibeVoice Audio Encoder module.
    
    Encapsulates Acoustic/Semantic VAE Tokenizers and projection Connectors.
    Converts raw audio waveforms into embeddings compatible with the language model.
    
    Features:
        - Streaming support for long audio (>60s by default)
        - Configurable dtype for numerical precision
        - Supports both sampling and deterministic (mean) modes
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        import sys 

        def get_cfg(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        self.acoustic_vae_dim = get_cfg(config, "acoustic_vae_dim", 64)
        self.semantic_vae_dim = get_cfg(config, "semantic_vae_dim", 128)

        decoder_config = get_cfg(config, "decoder_config")
        text_config = get_cfg(config, "text_config")
        
        target_hidden_size = None
        
        if decoder_config is not None:
            target_hidden_size = get_cfg(decoder_config, "hidden_size")

        if target_hidden_size is None and text_config is not None:
            target_hidden_size = get_cfg(text_config, "hidden_size")

        if target_hidden_size is None:
            target_hidden_size = get_cfg(config, "hidden_size")

        if target_hidden_size is None:
            print("[VibeVoice] WARN: Could not find hidden_size in config! Defaulting to 3584 (7B).", file=sys.stderr)
            self.hidden_size = 3584
        else:
            self.hidden_size = target_hidden_size

        ac_cfg = get_cfg(config, "acoustic_tokenizer_config")
        sc_cfg = get_cfg(config, "semantic_tokenizer_config")
        
        if ac_cfg is None or sc_cfg is None:
            raise ValueError("Missing acoustic/semantic tokenizer config in model config")

        # Handle both dict and already-constructed config objects
        if isinstance(ac_cfg, VibeVoiceAcousticTokenizerConfig):
            acoustic_config = ac_cfg
        elif isinstance(ac_cfg, dict):
            acoustic_config = VibeVoiceAcousticTokenizerConfig(**ac_cfg)
        else:
            raise TypeError(f"acoustic_tokenizer_config has unexpected type: {type(ac_cfg)}")
        
        if isinstance(sc_cfg, VibeVoiceSemanticTokenizerConfig):
            semantic_config = sc_cfg
        elif isinstance(sc_cfg, dict):
            semantic_config = VibeVoiceSemanticTokenizerConfig(**sc_cfg)
        else:
            raise TypeError(f"semantic_tokenizer_config has unexpected type: {type(sc_cfg)}")
        
        # Tokenizers use float32 for numerical precision
        self.acoustic_tokenizer = VibeVoiceAcousticTokenizerModel(acoustic_config)
        self.semantic_tokenizer = VibeVoiceSemanticTokenizerModel(semantic_config)
        
        # Get audio encoder dtype from config (defaults to float32 for precision)
        root_torch_dtype = get_cfg(config, "torch_dtype", None)
        if root_torch_dtype is not None:
            if isinstance(root_torch_dtype, str):
                self._audio_encoder_dtype = getattr(torch, root_torch_dtype)
            else:
                self._audio_encoder_dtype = root_torch_dtype
        else:
            self._audio_encoder_dtype = torch.float32
        
        self.acoustic_connector = SpeechConnector(self.acoustic_vae_dim, self.hidden_size)
        self.semantic_connector = SpeechConnector(self.semantic_vae_dim, self.hidden_size)
        
        self.compress_ratio = get_cfg(config, "speech_tok_compress_ratio", 3200)
        
        # Streaming controls
        self.sample_rate = get_cfg(config, "target_sample_rate", 24000)

        # Default to True (per requirement): segment + cache inside one forward call.
        self.enable_streaming = get_cfg(config, "enable_streaming", True)
        self.streaming_segment_duration = get_cfg(config, "streaming_segment_duration", 60.0)

        # Control whether to use sample() or .mean for acoustic tokens
        # Default: use sample() for training-consistent behavior
        # Set VIBEVOICE_USE_MEAN=1 for deterministic output
        use_mean_env = os.getenv("VIBEVOICE_USE_MEAN", "").strip().lower()
        self.use_sample = use_mean_env not in ("1", "true", "yes")
        
        # Language model dtype (set by VibeVoiceForCausalLM.__init__)
        # This is the dtype that audio embeddings will be converted to before
        # being passed to the language model. Defaults to bfloat16.
        self._lm_dtype: torch.dtype = torch.bfloat16

    def _ensure_audio_encoder_dtype(self):
        """Ensure all audio encoder components use the correct dtype from config.
        
        vLLM may convert weights to a different dtype (e.g., bfloat16) during loading.
        This method converts audio encoder components back to the config-specified dtype
        (typically float32) for numerical precision during audio encoding.
        """
        import sys
        target_dtype = self._audio_encoder_dtype
        
        # Check and convert acoustic_tokenizer
        try:
            acoustic_dtype = next(self.acoustic_tokenizer.parameters()).dtype
            if acoustic_dtype != target_dtype:
                self.acoustic_tokenizer = self.acoustic_tokenizer.to(dtype=target_dtype)
                print(f"[VibeVoice] Converted acoustic_tokenizer to {target_dtype} (was {acoustic_dtype})", file=sys.stderr)
        except StopIteration:
            pass
        
        # Check and convert semantic_tokenizer
        try:
            semantic_dtype = next(self.semantic_tokenizer.parameters()).dtype
            if semantic_dtype != target_dtype:
                self.semantic_tokenizer = self.semantic_tokenizer.to(dtype=target_dtype)
                print(f"[VibeVoice] Converted semantic_tokenizer to {target_dtype} (was {semantic_dtype})", file=sys.stderr)
        except StopIteration:
            pass
        
        # Check and convert acoustic_connector
        try:
            ac_conn_dtype = next(self.acoustic_connector.parameters()).dtype
            if ac_conn_dtype != target_dtype:
                self.acoustic_connector = self.acoustic_connector.to(dtype=target_dtype)
                print(f"[VibeVoice] Converted acoustic_connector to {target_dtype} (was {ac_conn_dtype})", file=sys.stderr)
        except StopIteration:
            pass
        
        # Check and convert semantic_connector
        try:
            sc_conn_dtype = next(self.semantic_connector.parameters()).dtype
            if sc_conn_dtype != target_dtype:
                self.semantic_connector = self.semantic_connector.to(dtype=target_dtype)
                print(f"[VibeVoice] Converted semantic_connector to {target_dtype} (was {sc_conn_dtype})", file=sys.stderr)
        except StopIteration:
            pass

    def forward(
        self,
        audio: torch.Tensor,
        *,
        use_streaming: bool = True,
        segment_duration_s: Optional[float] = None,
        use_sample: Optional[bool] = None,
    ) -> torch.Tensor:
        """Encode audio with optional streaming for long clips.
        
        Args:
            audio: Input audio tensor [B, T] or [T]
            use_streaming: Whether to enable segmented encoding for long audio
            segment_duration_s: Segment length in seconds (defaults to 60s)
            use_sample: If True, use sampling for acoustic tokens; if False, use mean
                       Defaults to self.use_sample (controlled by VIBEVOICE_USE_MEAN env var)
        
        Returns:
            Audio embeddings tensor compatible with the language model
        """
        # Ensure audio encoder components use correct dtype
        self._ensure_audio_encoder_dtype()
        
        # Audio input should match the audio encoder dtype
        audio = audio.to(dtype=self._audio_encoder_dtype)
        
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        # Resolve streaming options
        segment_duration = segment_duration_s or self.streaming_segment_duration
        sample_rate = self.sample_rate
        total_samples = audio.shape[-1]
        segment_samples = int(segment_duration * sample_rate)
        
        use_streaming = use_streaming and self.enable_streaming and total_samples > segment_samples
        
        # Resolve use_sample flag
        if use_sample is None:
            use_sample = self.use_sample

        # Keep encoding in inference mode to avoid autograd build-up
        with torch.no_grad():
            if not use_streaming:
                acoustic_input = audio.unsqueeze(1)
                acoustic_out = self.acoustic_tokenizer.encode(acoustic_input)
                # Use sample() or .mean based on use_sample flag
                if use_sample:
                    acoustic_tokens = acoustic_out.sample(
                        dist_type=self.acoustic_tokenizer.std_dist_type
                    )[0]
                else:
                    acoustic_tokens = acoustic_out.mean
                
                # Connector is now float32, no conversion needed
                acoustic_embeds = self.acoustic_connector(acoustic_tokens)

                semantic_out = self.semantic_tokenizer.encode(acoustic_input)
                # Semantic always uses .mean for consistency
                semantic_tokens = semantic_out.mean
                # Connector is now float32, no conversion needed
                semantic_embeds = self.semantic_connector(semantic_tokens)
            else:
                # ==========================================
                # Streaming path (Retained for future use)
                # ==========================================
                acoustic_cache = VibeVoiceTokenizerStreamingCache()
                semantic_cache = VibeVoiceTokenizerStreamingCache()
                acoustic_mean_segments = []
                semantic_mean_segments = []
                batch_size = audio.shape[0]
                sample_indices = torch.arange(batch_size, device=audio.device)

                def _iter_segments(total_length: int, segment_length: int):
                    for start in range(0, total_length, segment_length):
                        end = min(start + segment_length, total_length)
                        if end > start:
                            yield start, end

                segments = list(_iter_segments(total_samples, segment_samples))
                num_segments = len(segments)
                for seg_idx, (start, end) in enumerate(segments):
                    chunk = audio[:, start:end].contiguous()
                    if chunk.numel() == 0:
                        continue

                    # Check if this is the final segment
                    is_final = (seg_idx == num_segments - 1)

                    # --- Acoustic Encode ---
                    acoustic_enc_out = self.acoustic_tokenizer.encode(
                        chunk.unsqueeze(1),
                        cache=acoustic_cache,
                        sample_indices=sample_indices,
                        use_cache=True,
                        is_final_chunk=is_final,
                    )
                    acoustic_mean_segments.append(acoustic_enc_out.mean)

                    # --- Semantic Encode ---
                    semantic_enc_out = self.semantic_tokenizer.encode(
                        chunk.unsqueeze(1),
                        cache=semantic_cache,
                        sample_indices=sample_indices,
                        use_cache=True,
                        is_final_chunk=is_final,
                    )
                    semantic_mean_segments.append(semantic_enc_out.mean)

                # Concatenate sequence outputs (Acoustic)
                if len(acoustic_mean_segments) == 0:
                    acoustic_mean_full = torch.zeros(
                        (batch_size, 0, self.acoustic_vae_dim), 
                        device=audio.device, 
                        dtype=self._audio_encoder_dtype  # Use config dtype
                    )
                else:
                    acoustic_mean_full = torch.cat(acoustic_mean_segments, dim=1).contiguous()
                
                # Get acoustic tokens based on use_sample flag
                acoustic_enc_full = VibeVoiceTokenizerEncoderOutput(
                    mean=acoustic_mean_full,
                    std=self.acoustic_tokenizer.fix_std,
                )
                if use_sample:
                    acoustic_tokens = acoustic_enc_full.sample(
                        dist_type=self.acoustic_tokenizer.std_dist_type
                    )[0]
                else:
                    acoustic_tokens = acoustic_enc_full.mean
                # Connector uses same dtype as tokenizer
                acoustic_embeds = self.acoustic_connector(acoustic_tokens)

                # Concatenate sequence outputs (Semantic)
                if len(semantic_mean_segments) == 0:
                    semantic_tokens = torch.zeros(
                        (batch_size, 0, self.semantic_vae_dim), 
                        device=audio.device, 
                        dtype=self._audio_encoder_dtype  # Use config dtype
                    )
                else:
                    semantic_tokens = torch.cat(semantic_mean_segments, dim=1).contiguous()
                # Connector uses same dtype as tokenizer
                semantic_embeds = self.semantic_connector(semantic_tokens)

        # Combine acoustic and semantic embeddings
        combined_embeds = acoustic_embeds + semantic_embeds
        
        # Convert to language model dtype for compatibility
        # Audio encoder uses config.torch_dtype (typically float32) for numerical precision,
        # but LM expects the dtype specified by vLLM's --dtype flag (e.g., bfloat16, float16)
        combined_embeds = combined_embeds.to(dtype=self._lm_dtype)
        
        return combined_embeds

# ============================================================================
# vLLM Multimodal Processing Infrastructure
# ============================================================================

class VibeVoiceProcessingInfo(BaseProcessingInfo):
    """Processing info for VibeVoice multimodal model."""
    
    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_feature_extractor(self, **kwargs) -> WhisperFeatureExtractor:
        """
        Get a WhisperFeatureExtractor for vLLM profiling compatibility.
        
        IMPORTANT: This is NOT used in actual inference!
        VibeVoice uses its own acoustic/semantic VAE tokenizers operating on
        raw 24kHz waveforms, NOT Whisper mel spectrograms.
        
        This feature extractor exists only to satisfy vLLM's multimodal
        profiling infrastructure which may query audio parameters like
        sampling_rate and chunk_length for memory estimation.
        """
        # Read config from preprocessor_config.json if available
        import json
        import os
        model_path = self.ctx.model_config.model
        preprocessor_path = os.path.join(model_path, "preprocessor_config.json")
        
        # Default values: keep a coherent (sr, hop) pair.
        # VibeVoice runs at 24kHz in this repo (see demo/asr_transcribe_file.py).
        config = {
            "sampling_rate": 24000,
            "feature_size": 128,
            # 10ms hop at 24kHz
            "hop_length": 240,
            "chunk_length": 30,
            "n_fft": 400,
            "padding_value": 0.0,
        }
        
        # Try to load from config file
        if os.path.exists(preprocessor_path):
            try:
                with open(preprocessor_path, "r") as f:
                    file_config = json.load(f)
                    config.update({k: file_config[k] for k in config.keys() if k in file_config})
            except Exception:
                pass  # Use defaults
        
        return WhisperFeatureExtractor(
            feature_size=config["feature_size"],
            sampling_rate=config["sampling_rate"],
            hop_length=config["hop_length"],
            chunk_length=config["chunk_length"],
            n_fft=config["n_fft"],
            padding_value=config["padding_value"],
        )

    def get_audio_token_info(self) -> dict:
        """
        Get audio special tokens and their IDs.
        
        Returns dict with:
            audio_token, audio_bos_token, audio_eos_token,
            audio_token_id, audio_bos_id, audio_eos_id
        """
        tokenizer = self.get_tokenizer()
        vocab = tokenizer.get_vocab()
        
        # VibeVoice special tokens
        tokens = {
            "audio_token": "<|AUDIO|>",
            "audio_bos_token": "<|audio_bos|>",
            "audio_eos_token": "<|audio_eos|>",
        }
        
        # Get IDs
        tokens["audio_token_id"] = vocab.get(tokens["audio_token"])
        tokens["audio_bos_id"] = vocab.get(tokens["audio_bos_token"])
        tokens["audio_eos_id"] = vocab.get(tokens["audio_eos_token"])
        
        return tokens

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        """Return the maximum number of audio tokens per item.

        This tells vLLM's scheduler the upper bound so that
        ``encoder_compute_budget`` is large enough for any audio length
        the model can handle, preventing the silent scheduling deadlock
        described in docs/max_num_batched_tokens_issue.md.

        Formula: audio_tokens = ceil(audio_samples / compress_ratio) + 3
        where +3 accounts for speech_start, speech_end, and newline tokens.
        The max audio samples is bounded by seq_len (the model's context
        window cannot hold more tokens than that).
        """
        hf_config = self.get_hf_config()

        def _cfg(key: str, default):
            if isinstance(hf_config, dict):
                return hf_config.get(key, default)
            return getattr(hf_config, key, default)

        compress_ratio = int(_cfg("speech_tok_compress_ratio", 3200))
        sample_rate = int(_cfg("target_sample_rate", 24000))

        # Upper bound: 61-minute audio at 24 kHz
        max_audio_samples = 61 * 60 * sample_rate  # 88,464,000
        max_audio_tokens = int(np.ceil(max_audio_samples / compress_ratio)) + 3

        # Cannot exceed the model's context window
        max_audio_tokens = min(max_audio_tokens, seq_len)

        return {"audio": max_audio_tokens}


class VibeVoiceDummyInputsBuilder(BaseDummyInputsBuilder[VibeVoiceProcessingInfo]):
    """
    Build dummy inputs for multimodal profiling.
    
    vLLM uses dummy inputs to:
    1. Measure peak GPU activation memory → determine KV cache capacity
    2. Warm up CUDA graphs
    
    The dummy audio length must be consistent with ``get_mm_max_tokens_per_item``
    so that the memory estimate covers the worst-case (longest audio) scenario.
    """
    
    def _get_max_audio_samples(self, seq_len: int) -> int:
        """Compute maximum audio samples consistent with ``get_mm_max_tokens_per_item``.
        
        Uses the same formula: max_tokens = min(ceil(61min * sr / ratio) + 3, seq_len),
        then converts back to samples.
        """
        hf_config = self.info.get_hf_config()

        def _cfg(key: str, default):
            if isinstance(hf_config, dict):
                return hf_config.get(key, default)
            return getattr(hf_config, key, default)

        compress_ratio = int(_cfg("speech_tok_compress_ratio", 3200))
        sample_rate = int(_cfg("target_sample_rate", 24000))

        # Upper bound: 61-minute audio at 24 kHz
        max_hour_samples = 61 * 60 * sample_rate  # 88,464,000
        max_tokens_from_audio = int(np.ceil(max_hour_samples / compress_ratio)) + 3
        # Cannot exceed model context window
        max_tokens = min(max_tokens_from_audio, seq_len)
        # Convert tokens back to samples
        return max_tokens * compress_ratio

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        if num_audios <= 0:
            return ""
        
        token_info = self.info.get_audio_token_info()
        audio_token = token_info["audio_token"]
        return audio_token * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Generate dummy audio data for profiling.
        
        The audio length is derived from ``seq_len`` so that profiling
        accurately measures memory for the longest audio the model can handle.
        Supports ``AudioDummyOptions.length`` override for faster startup.
        """
        num_audios = mm_counts.get("audio", 0)
        max_audio_len = self._get_max_audio_samples(seq_len)

        audio_overrides = mm_options.get("audio") if mm_options else None

        return {
            "audio": self._get_dummy_audios(
                length=max_audio_len,
                num_audios=num_audios,
                overrides=audio_overrides,
            )
        }

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, Any] | None = None,
    ) -> ProcessorInputs:
        """Build ProcessorInputs for dummy profiling."""
        return ProcessorInputs(
            prompt=self.get_dummy_text(mm_counts),
            mm_data=self.get_dummy_mm_data(seq_len, mm_counts, mm_options),
        )


def _vibevoice_field_config(hf_inputs: Mapping[str, torch.Tensor]):
    """Map HF processor output keys to audio modality.
    
    Returns a config dict that tells vLLM how to batch multimodal data.
    """
    # Always define the config for all fields we use
    # Even if the field isn't in hf_inputs, vLLM needs to know how to batch it
    config = {
        # These are our custom fields for VibeVoice
        "raw_audio": MultiModalFieldConfig.batched("audio"),
        "raw_audio_lengths": MultiModalFieldConfig.batched("audio"),
        "salt": MultiModalFieldConfig.batched("audio"),
    }
    
    # Add optional Whisper features if present
    if "input_features" in hf_inputs:
        config["input_features"] = MultiModalFieldConfig.batched("audio")
    if "feature_attention_mask" in hf_inputs:
        config["feature_attention_mask"] = MultiModalFieldConfig.batched("audio")
    
    return config


class VibeVoiceMultiModalProcessor(BaseMultiModalProcessor[VibeVoiceProcessingInfo]):
    """
    Multimodal processor for VibeVoice.
    
    Handles the conversion of raw audio inputs to model-ready features,
    and manages the prompt token replacement for audio placeholders.
    """
    
    def _get_data_parser(self) -> MultiModalDataParser:
        """Create a data parser with the correct target sample rate (24kHz)."""
        # VibeVoice requires 24kHz, not 16kHz (Whisper default)
        target_sr = 24000
        return MultiModalDataParser(target_sr=target_sr)
    
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """
        Process prompt and audio for vLLM multimodal pipeline.
        
        We intentionally do NOT run a HF processor that would pre-expand
        `<|AUDIO|>` inside this method. Instead we:
        1) Tokenize the prompt as-is (so `<|AUDIO|>` stays a single token)
        2) Store raw audio tensors for `embed_multimodal` to encode later
        3) Let vLLM call `_get_prompt_updates` to expand the single `<|AUDIO|>`
           into the full ASR format: [speech_start] + N*[speech_pad] + [speech_end] + [\\n]
        """
        # Handle the case where 'audios' key is used (transformers deprecation)
        mm_data = dict(mm_data)  # Make a mutable copy
        audios = mm_data.pop("audios", None)
        if audios is not None and "audio" not in mm_data:
            mm_data["audio"] = audios
        
        # Text-only input handling
        if not mm_data.get("audio"):
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")
        
        # Get raw audio data
        raw_audio_list = mm_data.get("audio")
        if isinstance(raw_audio_list, np.ndarray):
            raw_audio_list = [raw_audio_list]
        elif not isinstance(raw_audio_list, list):
            raw_audio_list = list(raw_audio_list)
        
        # Tokenize prompt directly to preserve <|AUDIO|> as a single token
        # vLLM will expand it via _get_prompt_updates
        tokenizer = self.info.get_tokenizer()
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
        
        # Create result with input_ids
        result = BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")
        
        # Add raw audio tensors for VibeVoice encoder
        # Stack into a single tensor for vLLM's batched field config
        max_len = max(len(a) for a in raw_audio_list)
        raw_audio_tensors = []
        audio_lengths = []
        for audio in raw_audio_list:
            audio_len = len(audio)
            audio_lengths.append(audio_len)
            if audio_len < max_len:
                audio = np.pad(audio, (0, max_len - audio_len), mode='constant')
            raw_audio_tensors.append(torch.from_numpy(audio).float())
        
        # Stack into [num_audios, max_len] tensor
        stacked_audio = torch.stack(raw_audio_tensors, dim=0)  # Shape: [num_audios, max_len]
        result["raw_audio"] = stacked_audio
        # Convert lengths to tensor as well
        result["raw_audio_lengths"] = torch.tensor(audio_lengths, dtype=torch.long)
        
        # Add a random salt to ensure unique hash and bypass cache
        import uuid
        # Use a random integer for salt
        salt_val = hash(str(uuid.uuid4())) % 100000
        result["salt"] = torch.tensor([salt_val], dtype=torch.long).expand(len(raw_audio_list))
        
        return result

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        """Return whether the HF processor applies prompt updates.
        
        Returns False because we handle token expansion via _get_prompt_updates.
        """
        return False

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        """Configure which HF output fields map to which modality."""
        return _vibevoice_field_config(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        """
        Define how to replace the audio placeholder in the prompt.

        vLLM's OpenAI multimodal parsing inserts the model placeholder string
        from `get_placeholder_str` (here: `<|AUDIO|>`) into the conversation.
        We expand that single token into N repeated `<|AUDIO|>` tokens, where N
        is derived from waveform length and `speech_tok_compress_ratio`.
        """
        token_info = self.info.get_audio_token_info()
        audio_token = token_info["audio_token"]
        audio_token_id = token_info["audio_token_id"]
        audio_bos_id = token_info.get("audio_bos_id")
        audio_eos_id = token_info.get("audio_eos_id")

        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        def _tok_id(name: str) -> int | None:
            return vocab.get(name)

        # Look up speech token IDs from vocabulary
        # These tokens mark the start/end of audio embeddings in the prompt
        speech_start_id = (
            _tok_id("<|object_ref_start|>")
            or getattr(tokenizer, "speech_start_id", None)
            or _tok_id("<|speech_start|>")
        )
        speech_end_id = (
            _tok_id("<|object_ref_end|>")
            or getattr(tokenizer, "speech_end_id", None)
            or _tok_id("<|speech_end|>")
        )
        speech_pad_id = (
            _tok_id("<|box_start|>")
            or getattr(tokenizer, "speech_pad_id", None)
            or _tok_id("<|speech_pad|>")
        )

        if audio_token_id is None:
            return []
        
        # Get raw audio lengths (in samples, after resampling to 24kHz) from our stored data
        out_mm_data = out_mm_kwargs.get_data()
        raw_audio_lengths = out_mm_data.get("raw_audio_lengths", [])

        # Fetch defaults from model config when available (falls back to 3200)
        hf_config = self.info.get_hf_config()
        if isinstance(hf_config, dict):
            compress_ratio = int(hf_config.get("speech_tok_compress_ratio", 3200))
        else:
            compress_ratio = int(getattr(hf_config, "speech_tok_compress_ratio", 3200))

        def _to_int_len(x) -> int:
            if x is None:
                return 0
            if isinstance(x, torch.Tensor):
                # Accept 0-dim or 1-dim scalar-like tensors
                if x.numel() == 1:
                    return int(x.item())
                # If a full tensor is passed accidentally, fall back to its length
                return int(x.shape[0])
            return int(x)
        
        def get_replacement(item_idx: int):
            if raw_audio_lengths and item_idx < len(raw_audio_lengths):
                audio_len = _to_int_len(raw_audio_lengths[item_idx])
                num_features = max(1, int(np.ceil(audio_len / compress_ratio)))
            else:
                # Fallback: estimate for 30 second audio at 24kHz
                num_features = int(np.ceil(30 * 24000 / compress_ratio))
            
            if num_features == 0:
                raise ValueError(
                    f"Audio at index {item_idx} is too short to be represented"
                )
            
            # Build replacement token sequence:
            #   <|speech_start|> + N * <|speech_pad|> + <|speech_end|> + \n
            # The newline is important for correct prompt structure.
            newline_id = 198  # '\n' token
            if speech_start_id is not None and speech_pad_id is not None and speech_end_id is not None:
                embed_id = int(speech_pad_id)
                replacement_ids = [int(speech_start_id)] + [embed_id] * num_features + [int(speech_end_id), newline_id]
            # Fallback: add audio BOS/EOS boundaries around repeated <|AUDIO|>.
            elif audio_bos_id is not None and audio_eos_id is not None:
                embed_id = int(audio_token_id)
                replacement_ids = [int(audio_bos_id)] + [embed_id] * num_features + [int(audio_eos_id)]
            else:
                embed_id = int(audio_token_id)
                replacement_ids = [embed_id] * num_features

            return PromptUpdateDetails.select_token_id(
                replacement_ids,
                embed_token_id=int(embed_id),
            )
        
        return [
            PromptReplacement(
                modality="audio",
                # Keep string placeholder matching for maximum vLLM compatibility.
                target=audio_token,
                replacement=get_replacement,
            )
        ]


# ============================================================================
# Main Model Class
# ============================================================================

@MULTIMODAL_REGISTRY.register_processor(
    VibeVoiceMultiModalProcessor,
    info=VibeVoiceProcessingInfo,
    dummy_inputs=VibeVoiceDummyInputsBuilder,
)
class VibeVoiceForCausalLM(nn.Module, SupportsMultiModal, SupportsPP):
    """
    VibeVoice ASR model with native vLLM multimodal integration.
    
    This model combines VibeVoice acoustic/semantic tokenizers for audio encoding
    with a causal language model for text generation.
    """
    
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        """Return the placeholder string format for a given modality.
        
        Returns "<|AUDIO|>" which vLLM inserts into the conversation prompt.
        This single placeholder is later expanded by `_get_prompt_updates` into:
            [speech_start_id] + [speech_pad_id] * N + [speech_end_id] + [newline_id]
        where N = ceil(audio_samples / compress_ratio).
        """
        if modality.startswith("audio"):
            return "<|AUDIO|>"
        raise ValueError(f"Unsupported modality: {modality}")
    
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        
        self.audio_encoder = VibeVoiceAudioEncoder(config)
        
        # Pass decoder_config to the language model initialization
        decoder_config = getattr(config, "decoder_config", config)
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=decoder_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen2ForCausalLM"],
        )
        
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        # Set the language model dtype for audio encoder output conversion
        # This should match vLLM's --dtype flag (e.g., bfloat16, float16, float32)
        # Audio encoder internal computation stays in fp32 for numerical precision,
        # but output is converted to LM dtype for compatibility
        lm_dtype = vllm_config.model_config.dtype
        if lm_dtype is not None:
            self.audio_encoder._lm_dtype = lm_dtype
        
        # Ensure audio encoder uses correct dtype (typically fp32 for precision)
        try:
            self.audio_encoder._ensure_audio_encoder_dtype()
        except Exception:
            pass

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)


    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        """
        Extract audio embeddings using VibeVoice's acoustic/semantic tokenizers.
        
        Called by vLLM to get audio embeddings that replace audio placeholder tokens.
        
        Returns:
            Tuple of embedding tensors, one per audio input.
        """
        # Get raw audio data (stored by our processor)
        raw_audio = kwargs.get("raw_audio")
        raw_audio_lengths = kwargs.get("raw_audio_lengths")
        
        # Handle no audio input - this happens during memory profiling
        if raw_audio is None:
            return []
        
        # Handle empty audio list
        if isinstance(raw_audio, (list, tuple)) and len(raw_audio) == 0:
            return []
        
        # Flatten raw_audio_lengths if it's nested
        def flatten_lengths(lengths):
            """Flatten nested lists/tensors of lengths to a single list."""
            if lengths is None:
                return []
            
            result = []
            if isinstance(lengths, torch.Tensor):
                lengths = lengths.tolist()
            
            if isinstance(lengths, (list, tuple)):
                for item in lengths:
                    if isinstance(item, (list, tuple)):
                        result.extend(flatten_lengths(item))
                    elif isinstance(item, torch.Tensor):
                        if item.dim() == 0:
                            result.append(item.item())
                        else:
                            result.extend(item.tolist())
                    else:
                        result.append(item)
            else:
                result.append(lengths)
            return result
        
        raw_audio_lengths = flatten_lengths(raw_audio_lengths)

        # Streaming controls. Enabled by default; can be overridden per-call.
        use_streaming_flag = bool(
            kwargs.get(
                "use_streaming",
                getattr(self.audio_encoder, "enable_streaming", True),
            )
        )
        streaming_segment_duration = kwargs.get(
            "streaming_segment_duration",
            getattr(self.audio_encoder, "streaming_segment_duration", 60.0),
        )
        
        # Process each audio through the VibeVoice encoder
        embeddings = []
        
        # Get model device for tensor placement.
        # dtype is NOT set here — audio_encoder.forward() handles it internally:
        #   input: converted to fp32 (self._audio_encoder_dtype)
        #   output: converted to bfloat16 (self._lm_dtype)
        try:
            device = next(self.audio_encoder.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Handle both stacked tensor and list of tensors
        # vLLM batches as: [batch_size, 1, seq_len] or [batch_size, seq_len]
        if isinstance(raw_audio, torch.Tensor):
            if raw_audio.dim() == 3:
                # Shape: [batch_size, 1, seq_len] - squeeze the middle dimension
                num_audios = raw_audio.shape[0]
                audio_list = [raw_audio[i].squeeze(0) for i in range(num_audios)]
            elif raw_audio.dim() == 2:
                # Shape: [batch_size, seq_len]
                num_audios = raw_audio.shape[0]
                audio_list = [raw_audio[i] for i in range(num_audios)]
            else:
                # Single 1D tensor
                audio_list = [raw_audio]
        elif isinstance(raw_audio, (list, tuple)):
            audio_list = list(raw_audio)
        else:
            # Single tensor
            audio_list = [raw_audio]
        
        for i, audio_tensor in enumerate(audio_list):
            try:
                if isinstance(audio_tensor, list):
                    audio_tensor = torch.stack(audio_tensor)
                
                # Ensure tensor
                if not isinstance(audio_tensor, torch.Tensor):
                    audio_tensor = torch.tensor(audio_tensor)
                
                # Only place on correct device; audio_encoder.forward() handles dtype
                audio_tensor = audio_tensor.to(device=device)
                
                # Get actual length if available, otherwise use full length
                if raw_audio_lengths and i < len(raw_audio_lengths):
                    actual_len = int(raw_audio_lengths[i])
                    if actual_len > 0 and actual_len <= audio_tensor.shape[-1]:
                        # Truncate from the last dimension (sequence length)
                        audio_tensor = audio_tensor[..., :actual_len]
                
                # Skip if audio is too short (< 1 frame)
                if audio_tensor.numel() < 160:  # Minimum ~1ms at 24kHz
                    continue
                
                # Encode audio through VibeVoice encoder
                audio_embeds = self.audio_encoder(
                    audio_tensor,
                    use_streaming=use_streaming_flag,
                    segment_duration_s=streaming_segment_duration,
                )
                
                # audio_embeds shape: [1, seq_len, hidden_size]
                # We need to return it as a single embedding tensor per audio
                final_embed = audio_embeds.squeeze(0)
                embeddings.append(final_embed)
                
            except Exception as e:
                # Log error but don't crash - this helps debug profiling issues
                print(f"[VibeVoice] Error encoding audio {i}: {e}")
                import traceback
                traceback.print_exc()
                # Return empty embedding to avoid crash
                continue
        
        return tuple(embeddings)

    def get_input_embeddings(self) -> torch.nn.Module:
        """Return the text embedding layer (embed_tokens).
        
        vLLM uses this to get the embedding module for converting token IDs 
        to embeddings during decode phase.
        
        Returns:
            The embed_tokens module from the language model
        """
        # Get embed_tokens from the language model
        if hasattr(self.language_model, 'model') and hasattr(self.language_model.model, 'embed_tokens'):
            return self.language_model.model.embed_tokens
        elif hasattr(self.language_model, 'embed_tokens'):
            return self.language_model.embed_tokens
        else:
            # Try to get from inner model
            inner = self.language_model
            if hasattr(inner, 'language_model'):
                inner = inner.language_model
            if hasattr(inner, 'model') and hasattr(inner.model, 'embed_tokens'):
                return inner.model.embed_tokens
        
        raise AttributeError("Cannot find embed_tokens layer")
    
    def embed_input_ids(
        self, 
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        is_multimodal: Optional[torch.Tensor] = None,
        **kwargs,  # Accept any additional kwargs for compatibility
    ) -> torch.Tensor:
        """Apply token embeddings to input_ids and merge with multimodal embeddings.
        
        This is the preferred method in vLLM V1 for converting token IDs
        to embeddings and merging multimodal (audio) embeddings.
        
        Args:
            input_ids: Tensor of token IDs to embed
            multimodal_embeddings: Pre-computed multimodal embeddings (audio).
                                   Can be a Tensor or a List of Tensors (vLLM standard).
            is_multimodal: Boolean mask indicating which positions are multimodal
            **kwargs: Additional arguments for compatibility
            
        Returns:
            Tensor of embeddings with multimodal content merged in
        """
        from vllm.model_executor.models.utils import _merge_multimodal_embeddings
        
        # Get text embeddings
        embed_tokens = self.get_input_embeddings()
        inputs_embeds = embed_tokens(input_ids)
        
        # Merge multimodal embeddings if provided
        if multimodal_embeddings is not None and is_multimodal is not None:
            # Use vLLM's standard merge function which handles List[Tensor] correctly
            inputs_embeds = _merge_multimodal_embeddings(
                inputs_embeds,
                multimodal_embeddings,
                is_multimodal,
            )
        
        return inputs_embeds

    def get_language_model(self) -> torch.nn.Module:
        """Return the language model backbone."""
        return self.language_model

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        """Load model weights from checkpoint.
        
        The checkpoint has weights named like:
        - lm_head.weight                    -> language_model.lm_head.weight
        - model.language_model.layers.X...  -> language_model.model.layers.X...
        - model.acoustic_tokenizer...       -> audio_encoder.acoustic_tokenizer...
        - model.semantic_tokenizer...       -> audio_encoder.semantic_tokenizer...
        - model.acoustic_connector...       -> audio_encoder.acoustic_connector...
        - model.semantic_connector...       -> audio_encoder.semantic_connector...
        
        Let vLLM handle all dtype conversions according to --dtype flag.
        """
        # Map weight prefixes for VibeVoice
        # The checkpoint uses "model." prefix, we need to remap it
        mapper = WeightsMapper(
            orig_to_new_prefix={
                # Audio encoder components: model.X -> audio_encoder.X
                "model.acoustic_tokenizer.": "audio_encoder.acoustic_tokenizer.",
                "model.semantic_tokenizer.": "audio_encoder.semantic_tokenizer.",
                "model.acoustic_connector.": "audio_encoder.acoustic_connector.",
                "model.semantic_connector.": "audio_encoder.semantic_connector.",
                # Language model: model.language_model.X -> language_model.model.X
                "model.language_model.": "language_model.model.",
                # LM head: lm_head.X -> language_model.lm_head.X
                "lm_head.": "language_model.lm_head.",
            }
        )
        
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=mapper)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        """
        Forward pass for VibeVoice ASR model.
        
        Handles embedding computation and language model forward pass.
        Uses inputs_embeds if provided (from vLLM multimodal merge), 
        otherwise computes embeddings from input_ids.
        
        Args:
            input_ids: Token IDs. May be None when inputs_embeds is provided.
            positions: Position indices for the input tokens.
            intermediate_tensors: Intermediate tensors for pipeline parallelism.
            inputs_embeds: Pre-computed embeddings (from multimodal merge or decode).
        """
        # PRIORITY: Use inputs_embeds if provided (from vLLM multimodal merge or decode)
        # Only compute from input_ids if inputs_embeds is not available
        if inputs_embeds is None and input_ids is not None:
            # Compute embeddings from input_ids
            inputs_embeds = self.get_input_embeddings()(input_ids)
        
        # If we have intermediate tensors (pipeline parallelism), don't use inputs_embeds
        if intermediate_tensors is not None:
            inputs_embeds = None
        
        # Get the inner model - handle both wrapped and direct language models
        language_model = self.language_model
        if hasattr(language_model, "language_model"):
            language_model = language_model.language_model
        
        # Call the language model's model (Qwen2Model)
        # vLLM V1 passes kv_caches and attn_metadata via context, not arguments
        # IMPORTANT: Pass input_ids=None when using inputs_embeds to avoid double embedding
        hidden_states = language_model.model(
            input_ids=None,  # Always None when we have inputs_embeds
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds
        )
        return hidden_states


# Alias for training checkpoint compatibility
VibeVoiceForASRTraining = VibeVoiceForCausalLM
