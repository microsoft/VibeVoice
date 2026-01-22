""" VibeVoice Streaming model configuration"""

from transformers.configuration_utils import PretrainedConfig 
from transformers.utils import logging

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from .configuration_vibevoice import VibeVoiceAcousticTokenizerConfig, VibeVoiceDiffusionHeadConfig

logger = logging.get_logger(__name__)


class VibeVoiceStreamingConfig(PretrainedConfig):
    model_type = "vibevoice_streaming"
    is_composition = True
    sub_configs = {
        "acoustic_tokenizer_config": VibeVoiceAcousticTokenizerConfig,
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
        decoder_config=None,
        diffusion_head_config=None,
        tts_backbone_num_hidden_layers=20,
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
        # The decoder of the model is divided into two components. The lower Transformer layers are only used for encoding text, while the upper Transformer layers are used for encoding text and generating speech. `tts_backbone_num_hidden_layers` indicates the number of upper layers used for TTS.
        self.tts_backbone_num_hidden_layers = tts_backbone_num_hidden_layers

        super().__init__(**kwargs)

    def get_text_config(self, decoder: bool = False):
        """Return the text (decoder) config for generation."""
        return self.decoder_config

    def _get_decoder_attr(self, name: str, fallback_names=None, default=None):
        fallback_names = fallback_names or []
        if hasattr(self.decoder_config, name):
            return getattr(self.decoder_config, name)
        for fallback in fallback_names:
            if hasattr(self.decoder_config, fallback):
                return getattr(self.decoder_config, fallback)
        return default

    @property
    def vocab_size(self):
        """Return vocab_size from decoder config for generation compatibility."""
        return self._get_decoder_attr("vocab_size")

    @property
    def num_attention_heads(self):
        """Return num_attention_heads from decoder config for generation compatibility."""
        return self._get_decoder_attr("num_attention_heads")

    @property
    def num_key_value_heads(self):
        """Return num_key_value_heads from decoder config for generation compatibility."""
        return self._get_decoder_attr("num_key_value_heads")

    @property
    def hidden_size(self):
        """Return hidden_size from decoder config for generation compatibility."""
        return self._get_decoder_attr("hidden_size")

    @property
    def num_hidden_layers(self):
        """Return num_hidden_layers from decoder config for generation compatibility."""
        return self._get_decoder_attr(
            "num_hidden_layers",
            fallback_names=["n_layer", "num_layers"],
            default=self.tts_backbone_num_hidden_layers,
        )

    @property
    def head_dim(self):
        """Return head_dim from decoder config for generation compatibility."""
        head_dim = self._get_decoder_attr("head_dim")
        if head_dim is not None:
            return head_dim
        if self.hidden_size is None or self.num_attention_heads in (None, 0):
            return None
        return self.hidden_size // self.num_attention_heads

    @property
    def tie_word_embeddings(self):
        """Return tie_word_embeddings from decoder config when available."""
        if hasattr(self, "_tie_word_embeddings"):
            return self._tie_word_embeddings
        return bool(self._get_decoder_attr("tie_word_embeddings", default=False))

    @tie_word_embeddings.setter
    def tie_word_embeddings(self, value):
        self._tie_word_embeddings = bool(value)

__all__ = [
    "VibeVoiceStreamingConfig"
]
