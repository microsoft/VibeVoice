"""Exercise projection slicing through the real ASR forward and GenerationMixin."""
import pytest
import torch
from transformers import Qwen2Config

from vibevoice.modular.configuration_vibevoice import (
    VibeVoiceASRConfig, VibeVoiceAcousticTokenizerConfig, VibeVoiceSemanticTokenizerConfig,
)
from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration


@pytest.fixture
def model():
    torch.manual_seed(0)
    encoder = dict(encoder_n_filters=2, encoder_ratios=[2], encoder_depths="1-1", vae_dim=4)
    config = VibeVoiceASRConfig(
        acoustic_tokenizer_config=VibeVoiceAcousticTokenizerConfig(**encoder, decoder_n_filters=2),
        semantic_tokenizer_config=VibeVoiceSemanticTokenizerConfig(**encoder),
        decoder_config=Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                                  num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                                  pad_token_id=0, eos_token_id=None),
    )
    return VibeVoiceASRForConditionalGeneration(config).eval()


@pytest.mark.parametrize("keep", [1, 3, torch.tensor([0, 2])])
@pytest.mark.parametrize("return_dict", [True, False])
def test_only_requested_positions_reach_vocabulary_projection(model, keep, return_dict):
    ids = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
    shapes = []
    handle = model.lm_head.register_forward_pre_hook(lambda module, args: shapes.append(args[0].shape))
    with torch.inference_mode():
        full = model(ids, use_cache=False).logits
        output = model(ids, use_cache=False, logits_to_keep=keep, return_dict=return_dict)
    handle.remove()
    selected = output.logits if return_dict else output[0]
    indices = slice(-keep, None) if isinstance(keep, int) else keep
    torch.testing.assert_close(selected, full[:, indices, :])
    assert shapes[-1][1] == selected.shape[1] == (keep if isinstance(keep, int) else len(keep))


def test_generate_automatically_requests_last_position_and_preserves_ids(model):
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    shapes = []
    handle = model.lm_head.register_forward_pre_hook(lambda module, args: shapes.append(args[0].shape))
    with torch.inference_mode():
        optimized = model.generate(ids, max_new_tokens=4, do_sample=False, pad_token_id=0)
        assert all(shape[1] == 1 for shape in shapes)
        reference = model.generate(ids, max_new_tokens=4, do_sample=False, pad_token_id=0, logits_to_keep=0)
    handle.remove()
    torch.testing.assert_close(optimized, reference, rtol=0, atol=0)


def test_labels_keep_full_shifted_loss_and_gradients(model):
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    reference = model(ids, labels=ids, use_cache=False)
    reference.loss.backward()
    gradient = model.lm_head.weight.grad.clone()
    model.zero_grad(set_to_none=True)
    selected = model(ids, labels=ids, use_cache=False, logits_to_keep=1)
    selected.loss.backward()
    assert selected.logits.shape == reference.logits.shape
    torch.testing.assert_close(selected.loss, reference.loss, rtol=0, atol=0)
    torch.testing.assert_close(model.lm_head.weight.grad, gradient, rtol=0, atol=0)
