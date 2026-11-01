"""Storage bounds and numerical compatibility for the shared streaming cache."""
import pytest
import torch

from vibevoice.modular.modular_vibevoice_tokenizer import (
    SConv1d,
    SConvTranspose1d,
    VibeVoiceTokenizerStreamingCache,
)


def test_tail_cache_does_not_retain_full_chunk_or_other_samples():
    chunk = torch.randn(2, 8, 10000, requires_grad=True)
    indices = torch.tensor([7, 3])
    cache = VibeVoiceTokenizerStreamingCache()
    cache.set("conv", indices, chunk[:, :, -4:])
    for row, sample in enumerate(indices.tolist()):
        state = cache.cache[("conv", sample)]
        torch.testing.assert_close(state, chunk[row, :, -4:])
        assert not state.requires_grad
        assert state.untyped_storage().nbytes() == state.numel() * state.element_size()
    assert cache.cache[("conv", 7)].data_ptr() != cache.cache[("conv", 3)].data_ptr()
    cache.clear("conv", sample_indices=torch.tensor([7]))
    assert ("conv", 7) not in cache.cache
    assert cache.cache[("conv", 3)].untyped_storage().nbytes() == 8 * 4 * 4


class ViewCache(VibeVoiceTokenizerStreamingCache):
    """Pre-fix storage behavior, used only as a numerical reference."""

    def set(self, layer_id, sample_indices, states):
        for row, sample in enumerate(sample_indices.tolist()):
            self.cache[(layer_id, sample)] = states[row].detach()


@pytest.mark.parametrize("kind", ["encoder", "decoder"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_streaming_convolutions_match_previous_cache_values(kind, batch_size):
    torch.manual_seed(0)
    if kind == "encoder":
        layer = SConv1d(2, 3, kernel_size=8, stride=4, causal=True, pad_mode="constant")
    else:
        layer = SConvTranspose1d(2, 3, kernel_size=8, stride=4, causal=True)
    cache, reference = VibeVoiceTokenizerStreamingCache(), ViewCache()
    indices = torch.tensor([7, 3][:batch_size])
    with torch.inference_mode():
        for step, length in enumerate([64, 48, 19]):
            # Exercise sample ordering as well as a shorter final chunk.
            indices = indices.flip(0)
            chunk = torch.randn(batch_size, 2, length)
            kwargs = dict(sample_indices=indices, use_cache=True)
            if kind == "encoder":
                kwargs["is_final_chunk"] = step == 2
            expected = layer(chunk, cache=reference, **kwargs)
            actual = layer(chunk, cache=cache, **kwargs)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(cache.get(layer.layer_id, indices),
                                       reference.get(layer.layer_id, indices), rtol=0, atol=0)


def test_cache_padding_reset_and_selective_clear():
    cache = VibeVoiceTokenizerStreamingCache()
    cache.set("conv", torch.tensor([3]), torch.tensor([[[1., 2.]]]))
    cache.set("conv", torch.tensor([7]), torch.tensor([[[3., 4., 5.]]]))
    torch.testing.assert_close(cache.get("conv", torch.tensor([7, 3])),
                               torch.tensor([[[3., 4., 5.]], [[0., 1., 2.]]]))
    cache.set_to_zero(torch.tensor([3]))
    torch.testing.assert_close(cache.get("conv", torch.tensor([3])), torch.zeros(1, 1, 2))
    cache.clear("conv", torch.tensor([7]))
    assert cache.get("conv", torch.tensor([7, 3])) is None
    assert cache.get("conv", torch.tensor([3])) is not None
