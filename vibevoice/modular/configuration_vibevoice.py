""" VibeVoice_AcousticTokenizer model configuration"""

from typing import Dict, List, Optional, Tuple

import torch
from transformers.configuration_utils import PretrainedConfig 
from transformers.utils import logging

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

logger = logging.get_logger(__name__)


def _convert_dtype_to_string(config_dict: dict) -> dict:
    """
    Convert torch.dtype objects to their string representation for JSON serialization.
    
    This fixes the "Object of type dtype is not JSON serializable" error that occurs
    when transformers tries to log/serialize the config with torch_dtype as a torch.dtype object.
    
    See: https://github.com/microsoft/VibeVoice/issues/199
    """
    if "torch_dtype" in config_dict and config_dict["torch_dtype"] is not None:
        dtype = config_dict["torch_dtype"]
        if isinstance(dtype, torch.dtype):
            # Convert torch.dtype to string (e.g., torch.bfloat16 -> "bfloat16")
            config_dict["torch_dtype"] = str(dtype).replace("torch.", "")
    return config_dict


class VibeVoiceAcousticTokenizerConfig(PretrainedConfig):
    model_type = "vibevoice_acoustic_tokenizer"

    def __init__(
        self,
        channels: int = 1,
        corpus_normalize: float = 0.0,
        causal: bool = True,
        vae_dim: int = 64,
        fix_std: float = 0.5,
        std_dist_type: str = 'gaussian',
        # common 
        mixer_layer: str = 'depthwise_conv',
        conv_norm: str = 'none',
        pad_mode: str = 'constant',
        disable_last_norm: bool = True,
        layernorm: str = 'RMSNorm',
        layernorm_eps: float = 1e-5,
        layernorm_elementwise_affine: bool = True,
        conv_bias: bool = True,
        layer_scale_init_value: float = 1e-6,
        weight_init_value: float = 1e-2,
        # encoder specific
        encoder_n_filters: int = 32,
        encoder_ratios: Optional[List[int]] = [8,5,5,4,2,2],
        encoder_depths: str = "3-3-3-3-3-3-8",
        # decoder specific
        decoder_n_filters: int = 32,
        decoder_ratios: Optional[List[int]] = None, # if None, same as encoder
        decoder_depths: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.channels = channels
        self.corpus_normalize = corpus_normalize
        self.causal = causal
        self.vae_dim = vae_dim
        self.fix_std = fix_std
        self.std_dist_type = std_dist_type
        
        # common parameters
        self.conv_norm = conv_norm
        self.pad_mode = pad_mode
        self.layernorm_eps = layernorm_eps
        self.disable_last_norm = disable_last_norm
        self.layernorm = layernorm
        self.layernorm_elementwise_affine = layernorm_elementwise_affine
        self.conv_bias = conv_bias
        self.layer_scale_init_value = layer_scale_init_value
        self.weight_init_value = weight_init_value
        self.mixer_layer = mixer_layer

        # encoder specific parameters
        self.encoder_n_filters = encoder_n_filters
        self.encoder_ratios = encoder_ratios
        self.encoder_depths = encoder_depths
        
        # decoder specific parameters
        self.decoder_ratios = decoder_ratios if decoder_ratios is not None else encoder_ratios
        self.decoder_n_filters = decoder_n_filters
        self.decoder_depths = decoder_depths


class VibeVoiceSemanticTokenizerConfig(PretrainedConfig):
    model_type = "vibevoice_semantic_tokenizer"
    
    def __init__(
        self,
        channels: int = 1,
        corpus_normalize: float = 0.0,
        causal: bool = True,
        vae_dim: int = 64,
        fix_std: float = 0,
        std_dist_type: str = 'none',
        # common 
        mixer_layer: str = 'depthwise_conv',
        conv_norm: str = 'none',
        pad_mode: str = 'constant',
        disable_last_norm: bool = True,
        layernorm: str = 'RMSNorm',
        layernorm_eps: float = 1e-5,
        layernorm_elementwise_affine: bool = True,
        conv_bias: bool = True,
        layer_scale_init_value: float = 1e-6,
        weight_init_value: float = 1e-2,
        # encoder specific
        encoder_n_filters: int = 32,
        encoder_ratios: Optional[List[int]] = [8,5,5,4,2,2],
        encoder_depths: str = "3-3-3-3-3-3-8",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.channels = channels
        self.corpus_normalize = corpus_normalize
        self.causal = causal
        self.vae_dim = vae_dim
        self.fix_std = fix_std
        self.std_dist_type = std_dist_type
        
        # common parameters
        self.conv_norm = conv_norm
        self.pad_mode = pad_mode
        self.layernorm_eps = layernorm_eps
        self.disable_last_norm = disable_last_norm
        self.layernorm = layernorm
        self.layernorm_elementwise_affine = layernorm_elementwise_affine
        self.conv_bias = conv_bias
        self.layer_scale_init_value = layer_scale_init_value
        self.weight_init_value = weight_init_value
        self.mixer_layer = mixer_layer

        # encoder specific parameters
        self.encoder_n_filters = encoder_n_filters
        self.encoder_ratios = encoder_ratios
        self.encoder_depths = encoder_depths
        

class VibeVoiceDiffusionHeadConfig(PretrainedConfig):
    model_type = "vibevoice_diffusion_head"

    def __init__(
        self,
        hidden_size=768,
        head_layers=4,
        head_ffn_ratio=3.0,
        rms_norm_eps=1e-5,
        latent_size=64,
        speech_vae_dim=None,
        prediction_type="v_prediction",
        diffusion_type="ddpm",
        ddpm_num_steps=1000,
        ddpm_num_inference_steps=20,
        ddpm_beta_schedule="cosine",
        ddpm_batch_mul=4,
        **kwargs
    ):
        self.hidden_size = hidden_size
        self.head_layers = head_layers
        self.head_ffn_ratio = head_ffn_ratio
        self.rms_norm_eps = rms_norm_eps
        self.latent_size = latent_size
        self.speech_vae_dim = speech_vae_dim
        self.prediction_type = prediction_type
        self.diffusion_type = diffusion_type
        self.ddpm_num_steps = ddpm_num_steps
        self.ddpm_num_inference_steps = ddpm_num_inference_steps
        self.ddpm_beta_schedule = ddpm_beta_schedule
        self.ddpm_batch_mul = ddpm_batch_mul
        
        super().__init__(**kwargs)

class VibeVoiceConfig(PretrainedConfig):
    model_type = "vibevoice"
    is_composition = True
    sub_configs = {
        "acoustic_tokenizer_config": VibeVoiceAcousticTokenizerConfig, 
        "semantic_tokenizer_config": VibeVoiceSemanticTokenizerConfig,
        "decoder_config": Qwen2Config,
        "diffusion_head_config": VibeVoiceDiffusionHeadConfig,
    }
    # keys_to_ignore_at_inference = ["past_key_values"]
    # Default tensor parallel plan for base model `Qwen2`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    
    def __init__(
        self,
        acoustic_tokenizer_config=None,
        semantic_tokenizer_config=None,
        decoder_config=None,
        diffusion_head_config=None,
        **kwargs
    ):

        # kwargs["_attn_implementation"] = "flash_attention_2"
        kwargs["_attn_implementation_autoset"] = False 

        if acoustic_tokenizer_config is None:
            self.acoustic_tokenizer_config = self.sub_configs["acoustic_tokenizer_config"]()
        elif isinstance(acoustic_tokenizer_config, dict):
            acoustic_tokenizer_config["model_type"] = "vibevoice_acoustic_tokenizer"
            self.acoustic_tokenizer_config = self.sub_configs["acoustic_tokenizer_config"](**acoustic_tokenizer_config)
        elif isinstance(acoustic_tokenizer_config, VibeVoiceAcousticTokenizerConfig):
            # If an instance of the config class is provided
            self.acoustic_tokenizer_config = acoustic_tokenizer_config

        if semantic_tokenizer_config is None:
            self.semantic_tokenizer_config = self.sub_configs["semantic_tokenizer_config"]()
        elif isinstance(semantic_tokenizer_config, dict):
            semantic_tokenizer_config["model_type"] = "vibevoice_semantic_tokenizer"
            self.semantic_tokenizer_config = self.sub_configs["semantic_tokenizer_config"](**semantic_tokenizer_config)
        elif isinstance(semantic_tokenizer_config, VibeVoiceSemanticTokenizerConfig):
            # If an instance of the config class is provided
            self.semantic_tokenizer_config = semantic_tokenizer_config

        if decoder_config is None:
            self.decoder_config = self.sub_configs["decoder_config"]()
        elif isinstance(decoder_config, dict):
            # If a dictionary is provided, instantiate the config class with it
            # self.decoder_config = self.sub_configs["decoder_config"](**decoder_config)
            if decoder_config.get("model_type", '') == "qwen2":
                self.decoder_config = Qwen2Config(**decoder_config)
            else:
                raise ValueError(f"Unsupported decoder model type: {decoder_config.get('model_type', '')}")
        elif isinstance(decoder_config, (Qwen2Config,)):
            # If an instance of the config class is provided
            self.decoder_config = decoder_config

        if diffusion_head_config is None:
            self.diffusion_head_config = self.sub_configs["diffusion_head_config"]()
        elif isinstance(diffusion_head_config, dict):
            diffusion_head_config["model_type"] = "vibevoice_diffusion_head"
            self.diffusion_head_config = self.sub_configs["diffusion_head_config"](**diffusion_head_config)
        elif isinstance(diffusion_head_config, VibeVoiceDiffusionHeadConfig):
            # If an instance of the config class is provided
            self.diffusion_head_config = diffusion_head_config

        # other parameters
        self.acoustic_vae_dim = getattr(self.acoustic_tokenizer_config, 'vae_dim', 64)
        self.semantic_vae_dim = getattr(self.semantic_tokenizer_config, 'vae_dim', 128)

        super().__init__(**kwargs)

    def get_text_config(self, decoder=False):
        """
        Returns the text config for this model.
        
        vLLM uses this method to get the text configuration from multimodal models.
        This allows vLLM to correctly determine hidden_size, num_attention_heads,
        and other properties needed for memory profiling and model execution.
        
        For VibeVoice, the "text config" is the decoder_config (Qwen2Config).
        
        Args:
            decoder: If True, return the decoder config (for encoder-decoder models).
                    For VibeVoice, this is always the decoder_config.
        
        Returns:
            The decoder configuration (Qwen2Config) which contains hidden_size, etc.
        """
        return self.decoder_config

    def to_dict(self):
        """
        Override to_dict to handle torch.dtype serialization.
        
        Fixes: https://github.com/microsoft/VibeVoice/issues/199
        """
        output = super().to_dict()
        return _convert_dtype_to_string(output)

class VibeVoiceASRConfig(PretrainedConfig):
    model_type = "vibevoice"
    is_composition = True
    sub_configs = {
        "acoustic_tokenizer_config": VibeVoiceAcousticTokenizerConfig, 
        "semantic_tokenizer_config": VibeVoiceSemanticTokenizerConfig,
        "decoder_config": Qwen2Config,
    }
    # keys_to_ignore_at_inference = ["past_key_values"]
    # Default tensor parallel plan for base model `Qwen2`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    
    def __init__(
        self,
        acoustic_tokenizer_config=None,
        semantic_tokenizer_config=None,
        decoder_config=None,
        **kwargs
    ):

        # kwargs["_attn_implementation"] = "flash_attention_2"
        kwargs["_attn_implementation_autoset"] = False 

        if acoustic_tokenizer_config is None:
            self.acoustic_tokenizer_config = self.sub_configs["acoustic_tokenizer_config"]()
        elif isinstance(acoustic_tokenizer_config, dict):
            acoustic_tokenizer_config["model_type"] = "vibevoice_acoustic_tokenizer"
            self.acoustic_tokenizer_config = self.sub_configs["acoustic_tokenizer_config"](**acoustic_tokenizer_config)
        elif isinstance(acoustic_tokenizer_config, VibeVoiceAcousticTokenizerConfig):
            # If an instance of the config class is provided
            self.acoustic_tokenizer_config = acoustic_tokenizer_config

        if semantic_tokenizer_config is None:
            self.semantic_tokenizer_config = self.sub_configs["semantic_tokenizer_config"]()
        elif isinstance(semantic_tokenizer_config, dict):
            semantic_tokenizer_config["model_type"] = "vibevoice_semantic_tokenizer"
            self.semantic_tokenizer_config = self.sub_configs["semantic_tokenizer_config"](**semantic_tokenizer_config)
        elif isinstance(semantic_tokenizer_config, VibeVoiceSemanticTokenizerConfig):
            # If an instance of the config class is provided
            self.semantic_tokenizer_config = semantic_tokenizer_config

        if decoder_config is None:
            self.decoder_config = self.sub_configs["decoder_config"]()
        elif isinstance(decoder_config, dict):
            # If a dictionary is provided, instantiate the config class with it
            # self.decoder_config = self.sub_configs["decoder_config"](**decoder_config)
            if decoder_config.get("model_type", '') == "qwen2":
                self.decoder_config = Qwen2Config(**decoder_config)
            else:
                raise ValueError(f"Unsupported decoder model type: {decoder_config.get('model_type', '')}")
        elif isinstance(decoder_config, Qwen2Config):
            # If an instance of the config class is provided
            self.decoder_config = decoder_config

        # other parameters
        self.acoustic_vae_dim = getattr(self.acoustic_tokenizer_config, 'vae_dim', 64)
        self.semantic_vae_dim = getattr(self.semantic_tokenizer_config, 'vae_dim', 128)

        super().__init__(**kwargs)

    def to_dict(self):
        """
        Override to_dict to handle torch.dtype serialization.
        
        Fixes: https://github.com/microsoft/VibeVoice/issues/199
        """
        output = super().to_dict()
        return _convert_dtype_to_string(output)

    def get_text_config(self, decoder: bool = False):
        """Return the text (decoder) config for generation."""
        return self.decoder_config

    @property
    def vocab_size(self):
        """Return vocab_size from decoder config for generation compatibility."""
        return self.decoder_config.vocab_size

    @property
    def num_attention_heads(self):
        """Return num_attention_heads from decoder config for Ulysses SP compatibility."""
        return self.decoder_config.num_attention_heads
    
    @property
    def num_key_value_heads(self):
        """Return num_key_value_heads from decoder config for Ulysses SP compatibility."""
        return self.decoder_config.num_key_value_heads
    
    @property
    def hidden_size(self):
        """Return hidden_size from decoder config for model compatibility."""
        return self.decoder_config.hidden_size
    
    @property
    def num_hidden_layers(self):
        """Return num_hidden_layers from decoder config for Ulysses SP compatibility."""
        return self.decoder_config.num_hidden_layers
    
    @property
    def head_dim(self):
        """Return head_dim from decoder config for Ulysses SP compatibility."""
        return getattr(self.decoder_config, 'head_dim', self.hidden_size // self.num_attention_heads)
    
__all__ = [
    "VibeVoiceAcousticTokenizerConfig", 
    "VibeVoiceSemanticTokenizerConfig", 
    "VibeVoiceDiffusionHeadConfig", 
    "VibeVoiceConfig",
    "VibeVoiceASRConfig"
]