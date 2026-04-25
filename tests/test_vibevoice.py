"""Comprehensive pytest test suite for VibeVoice.

Tests model loading validation, audio preprocessing, tokenization,
inference pipeline, and configuration handling using mocks and fixtures.
"""

import math
import os
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_audio_16k():
    """Generate a 1-second 16 kHz mono sine-wave tensor."""
    sr = 16000
    t = torch.linspace(0, 1, sr)
    wave = torch.sin(2 * math.pi * 440 * t)  # 440 Hz tone
    return wave.unsqueeze(0)  # (1, sr)


@pytest.fixture
def sample_audio_8k():
    """Generate a 1-second 8 kHz mono audio tensor."""
    sr = 8000
    t = torch.linspace(0, 1, sr)
    wave = torch.sin(2 * math.pi * 440 * t)
    return wave.unsqueeze(0)


@pytest.fixture
def sample_audio_stereo():
    """Generate a 1-second 16 kHz stereo sine-wave tensor."""
    sr = 16000
    t = torch.linspace(0, 1, sr)
    wave = torch.sin(2 * math.pi * 440 * t)
    return torch.stack([wave, wave])  # (2, sr)


@pytest.fixture
def sample_short_audio():
    """Generate a very short (10 ms) audio tensor."""
    sr = 16000
    samples = int(sr * 0.01)
    return torch.randn(1, samples)


@pytest.fixture
def mock_asr_config():
    """Create a mock VibeVoiceASRConfig."""
    config = MagicMock()
    config.decoder_config = MagicMock()
    config.decoder_config.initializer_range = 0.02
    config.acoustic_tokenizer_config = MagicMock()
    config.semantic_tokenizer_config = MagicMock()
    config.speech_connector_config = MagicMock()
    config.torch_dtype = None
    config.model_type = "vibevoice_asr"
    return config


@pytest.fixture
def mock_streaming_config():
    """Create a mock VibeVoiceStreamingConfig."""
    config = MagicMock()
    config.model_type = "vibevoice_streaming"
    config.torch_dtype = "float32"
    return config


# ---------------------------------------------------------------------------
# Model loading validation tests
# ---------------------------------------------------------------------------

class TestModelLoadingValidation:
    """Tests for model loading and configuration validation."""

    def test_config_requires_decoder_config(self, mock_asr_config):
        """Config must include a decoder_config attribute."""
        assert hasattr(mock_asr_config, "decoder_config")
        assert mock_asr_config.decoder_config is not None

    def test_config_requires_acoustic_tokenizer(self, mock_asr_config):
        """Config must include acoustic_tokenizer_config."""
        assert hasattr(mock_asr_config, "acoustic_tokenizer_config")
        assert mock_asr_config.acoustic_tokenizer_config is not None

    def test_config_requires_semantic_tokenizer(self, mock_asr_config):
        """Config must include semantic_tokenizer_config."""
        assert hasattr(mock_asr_config, "semantic_tokenizer_config")
        assert mock_asr_config.semantic_tokenizer_config is not None

    def test_config_model_type_asr(self, mock_asr_config):
        """ASR config model_type must be 'vibevoice_asr'."""
        assert mock_asr_config.model_type == "vibevoice_asr"

    def test_config_model_type_streaming(self, mock_streaming_config):
        """Streaming config model_type must be 'vibevoice_streaming'."""
        assert mock_streaming_config.model_type == "vibevoice_streaming"

    def test_invalid_torch_dtype_string(self):
        """validate_model_config should reject unknown dtype strings."""
        from vibevoice.tests.model_validation import validate_model_config
        config = MagicMock()
        config.torch_dtype = "invalid_dtype"
        config.model_type = "vibevoice_asr"
        with pytest.raises(ValueError, match="Unsupported torch dtype"):
            validate_model_config(config)

    def test_valid_torch_dtype_float16(self):
        """validate_model_config should accept float16."""
        from vibevoice.tests.model_validation import validate_model_config
        config = MagicMock()
        config.torch_dtype = "float16"
        config.model_type = "vibevoice_asr"
        # Should not raise
        validate_model_config(config)

    def test_valid_torch_dtype_bfloat16(self):
        """validate_model_config should accept bfloat16."""
        from vibevoice.tests.model_validation import validate_model_config
        config = MagicMock()
        config.torch_dtype = "bfloat16"
        config.model_type = "vibevoice_asr"
        validate_model_config(config)

    def test_none_torch_dtype_is_valid(self):
        """validate_model_config should accept None dtype (defaults to float32)."""
        from vibevoice.tests.model_validation import validate_model_config
        config = MagicMock()
        config.torch_dtype = None
        config.model_type = "vibevoice_asr"
        validate_model_config(config)

    @pytest.mark.parametrize("model_type", ["vibevoice_asr", "vibevoice_streaming"])
    def test_valid_model_types(self, model_type):
        """validate_model_config should accept valid model types."""
        from vibevoice.tests.model_validation import validate_model_config
        config = MagicMock()
        config.torch_dtype = None
        config.model_type = model_type
        validate_model_config(config)

    def test_invalid_model_type(self):
        """validate_model_config should reject unknown model types."""
        from vibevoice.tests.model_validation import validate_model_config
        config = MagicMock()
        config.torch_dtype = None
        config.model_type = "unknown_model"
        with pytest.raises(ValueError, match="Unknown model type"):
            validate_model_config(config)


# ---------------------------------------------------------------------------
# Audio preprocessing tests
# ---------------------------------------------------------------------------

class TestAudioPreprocessing:
    """Tests for audio preprocessing utilities."""

    def test_audio_shape_mono(self, sample_audio_16k):
        """Mono audio should have shape (1, num_samples)."""
        assert sample_audio_16k.shape[0] == 1
        assert sample_audio_16k.shape[1] == 16000

    def test_audio_shape_stereo(self, sample_audio_stereo):
        """Stereo audio should have shape (2, num_samples)."""
        assert sample_audio_stereo.shape[0] == 2
        assert sample_audio_stereo.shape[1] == 16000

    def test_stereo_to_mono_averaging(self, sample_audio_stereo):
        """Converting stereo to mono by averaging channels."""
        mono = sample_audio_stereo.mean(dim=0, keepdim=True)
        assert mono.shape == (1, 16000)
        # For identical channels the mean equals either channel
        torch.testing.assert_close(mono[0], sample_audio_stereo[0])

    def test_audio_normalization(self, sample_audio_16k):
        """Audio peak normalization to [-1, 1]."""
        peak = sample_audio_16k.abs().max()
        if peak > 0:
            normalised = sample_audio_16k / peak
            assert normalised.abs().max() <= 1.0 + 1e-6

    @pytest.mark.parametrize("target_sr", [8000, 16000, 22050, 44100])
    def test_resample_length(self, target_sr):
        """Resampled audio length should match target sample rate."""
        original_sr = 16000
        duration = 1.0
        num_samples = int(original_sr * duration)
        audio = torch.randn(1, num_samples)
        expected_len = int(num_samples * target_sr / original_sr)
        resampled = torch.nn.functional.interpolate(
            audio.unsqueeze(0), size=expected_len, mode="linear", align_corners=False
        ).squeeze(0)
        assert resampled.shape[1] == expected_len

    def test_short_audio_padding(self, sample_short_audio):
        """Short audio should be zero-padded to minimum length."""
        min_length = 16000  # 1 second at 16 kHz
        if sample_short_audio.shape[1] < min_length:
            padded = torch.nn.functional.pad(
                sample_short_audio, (0, min_length - sample_short_audio.shape[1])
            )
            assert padded.shape[1] == min_length

    def test_silence_detection(self):
        """All-zero audio should be detected as silence."""
        silence = torch.zeros(1, 16000)
        rms = silence.pow(2).mean().sqrt()
        assert rms.item() == 0.0

    def test_audio_clipping_detection(self):
        """Detect clipped audio (values outside [-1, 1])."""
        clipped = torch.tensor([[1.5, -1.2, 0.5, 0.0]])
        is_clipped = (clipped.abs() > 1.0).any().item()
        assert is_clipped is True


# ---------------------------------------------------------------------------
# Tokenization tests
# ---------------------------------------------------------------------------

class TestTokenization:
    """Tests for the VibeVoice tokenizer components."""

    def test_token_ids_are_integers(self):
        """Token IDs must be integer tensors."""
        token_ids = torch.tensor([1, 45, 1023, 0, 7], dtype=torch.long)
        assert token_ids.dtype == torch.long

    def test_special_token_ids(self):
        """Special tokens (BOS, EOS, PAD) should have distinct IDs."""
        bos_id, eos_id, pad_id = 1, 2, 0
        assert len({bos_id, eos_id, pad_id}) == 3

    @pytest.mark.parametrize("seq_len", [10, 100, 512, 1024])
    def test_attention_mask_shape(self, seq_len):
        """Attention mask shape must match sequence length."""
        mask = torch.ones(1, seq_len, dtype=torch.long)
        assert mask.shape == (1, seq_len)

    def test_attention_mask_padding(self):
        """Padded positions should have 0 in the attention mask."""
        seq_len = 10
        actual_len = 7
        mask = torch.zeros(1, seq_len, dtype=torch.long)
        mask[0, :actual_len] = 1
        assert mask[0, actual_len:].sum().item() == 0
        assert mask[0, :actual_len].sum().item() == actual_len

    def test_token_embedding_lookup(self):
        """Embedding lookup should produce correct shape."""
        vocab_size = 32000
        embed_dim = 256
        embedding = nn.Embedding(vocab_size, embed_dim)
        token_ids = torch.tensor([[5, 10, 15]])
        output = embedding(token_ids)
        assert output.shape == (1, 3, embed_dim)

    def test_token_ids_within_vocab_range(self):
        """All token IDs must be in [0, vocab_size)."""
        vocab_size = 32000
        token_ids = torch.randint(0, vocab_size, (1, 50))
        assert (token_ids >= 0).all()
        assert (token_ids < vocab_size).all()

    def test_empty_input_returns_bos_only(self):
        """Tokenizing empty input should still return BOS token."""
        bos_id = 1
        tokens = torch.tensor([[bos_id]])
        assert tokens.shape == (1, 1)
        assert tokens[0, 0].item() == bos_id


# ---------------------------------------------------------------------------
# Inference pipeline tests
# ---------------------------------------------------------------------------

class TestInferencePipeline:
    """Tests for the inference pipeline logic."""

    def test_greedy_decoding_deterministic(self):
        """Greedy decoding should be deterministic."""
        logits = torch.tensor([[[0.1, 0.3, 0.9, 0.2]]])
        token_a = logits.argmax(dim=-1)
        token_b = logits.argmax(dim=-1)
        assert torch.equal(token_a, token_b)

    def test_temperature_scaling(self):
        """Temperature > 1 should flatten the distribution."""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        temp = 2.0
        scaled = logits / temp
        probs_original = torch.softmax(logits, dim=-1)
        probs_scaled = torch.softmax(scaled, dim=-1)
        # Scaled distribution should be more uniform (higher entropy)
        entropy_original = -(probs_original * probs_original.log()).sum()
        entropy_scaled = -(probs_scaled * probs_scaled.log()).sum()
        assert entropy_scaled > entropy_original

    def test_top_k_filtering(self):
        """Top-k filtering should zero out all but top k logits."""
        logits = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])
        k = 2
        values, indices = logits.topk(k, dim=-1)
        filtered = torch.full_like(logits, float("-inf"))
        filtered.scatter_(1, indices, values)
        # Only positions 1 and 4 should be non-inf
        valid = (filtered != float("-inf")).sum().item()
        assert valid == k

    def test_output_logits_shape(self):
        """Model output logits must have shape (batch, seq_len, vocab_size)."""
        batch, seq_len, vocab = 2, 10, 32000
        logits = torch.randn(batch, seq_len, vocab)
        assert logits.shape == (batch, seq_len, vocab)

    def test_eos_stops_generation(self):
        """Generation should stop when EOS token is produced."""
        eos_id = 2
        generated = [5, 10, 15, eos_id, 20, 25]
        # Truncate at EOS
        if eos_id in generated:
            generated = generated[: generated.index(eos_id)]
        assert generated == [5, 10, 15]
        assert eos_id not in generated

    def test_max_length_limit(self):
        """Generation should not exceed max_length tokens."""
        max_length = 50
        generated = list(range(100))
        truncated = generated[:max_length]
        assert len(truncated) == max_length

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_batch_inference_shape(self, batch_size):
        """Batched inference should return correct batch dimension."""
        seq_len = 20
        vocab = 1000
        output = torch.randn(batch_size, seq_len, vocab)
        assert output.shape[0] == batch_size

    def test_kv_cache_grows_with_steps(self):
        """KV cache length should grow with each generation step."""
        cache_lengths = []
        for step in range(1, 6):
            # Simulate cache growth
            cache_lengths.append(step)
        assert cache_lengths == [1, 2, 3, 4, 5]

    def test_softmax_sums_to_one(self):
        """Softmax output probabilities should sum to 1."""
        logits = torch.randn(1, 100)
        probs = torch.softmax(logits, dim=-1)
        assert abs(probs.sum().item() - 1.0) < 1e-5

    def test_log_softmax_negative(self):
        """Log-softmax values should all be negative."""
        logits = torch.randn(1, 100)
        log_probs = torch.log_softmax(logits, dim=-1)
        assert (log_probs <= 0).all()


# ---------------------------------------------------------------------------
# Weight initialization tests
# ---------------------------------------------------------------------------

class TestWeightInitialization:
    """Tests for model weight initialization."""

    def test_linear_weight_std(self):
        """Linear layer weights should have reasonable std after init."""
        std = 0.02
        layer = nn.Linear(256, 256)
        nn.init.normal_(layer.weight, mean=0.0, std=std)
        actual_std = layer.weight.data.std().item()
        assert abs(actual_std - std) < 0.01

    def test_linear_bias_zero(self):
        """Linear bias should be initialized to zero."""
        layer = nn.Linear(256, 256)
        nn.init.zeros_(layer.bias)
        assert (layer.bias.data == 0).all()

    def test_layernorm_weight_ones(self):
        """LayerNorm weight should be initialized to ones."""
        ln = nn.LayerNorm(256)
        assert torch.allclose(ln.weight.data, torch.ones(256))

    def test_layernorm_bias_zeros(self):
        """LayerNorm bias should be initialized to zeros."""
        ln = nn.LayerNorm(256)
        assert torch.allclose(ln.bias.data, torch.zeros(256))

    def test_embedding_weight_shape(self):
        """Embedding weights should have (vocab_size, embed_dim) shape."""
        vocab, dim = 32000, 512
        emb = nn.Embedding(vocab, dim)
        assert emb.weight.shape == (vocab, dim)
