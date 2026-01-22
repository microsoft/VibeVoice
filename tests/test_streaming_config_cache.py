import pytest

import inspect

from transformers.cache_utils import DynamicCache
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from vibevoice.modular.configuration_vibevoice_streaming import VibeVoiceStreamingConfig


def test_streaming_config_exposes_generation_attrs():
    config = VibeVoiceStreamingConfig(decoder_config=Qwen2Config())

    assert isinstance(config.num_hidden_layers, int)
    assert config.num_hidden_layers > 0
    assert isinstance(config.num_attention_heads, int)
    assert config.num_attention_heads > 0
    assert isinstance(config.hidden_size, int)
    assert config.hidden_size > 0


def test_dynamic_cache_init_with_streaming_config():
    config = VibeVoiceStreamingConfig(decoder_config=Qwen2Config())
    sig = inspect.signature(DynamicCache.__init__)
    accepts_config = any(
        param.name in ("config", "decoder_config")
        for param in list(sig.parameters.values())[1:]
    )

    if accepts_config:
        cache = DynamicCache(config)
        assert cache is not None
    else:
        assert config.num_hidden_layers > 0
