"""CPU checks for recovery bounds, preserved evidence, and speech-feature reuse."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vibevoice.modular.asr_repetition import (
    _ExcludeEdges, _find_edge, generate_with_repetition_recovery,
)
from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration


class CharacterTokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids if not skip_special_tokens or i != 0)


class ScriptedModel:
    """Emit fixed candidates through the actual stopping criteria."""
    def __init__(self, candidates):
        self.candidates = iter(candidates)
        self.encode_speech = Mock(return_value=torch.ones(1, 4))
        self.features = []

    def generate(self, input_ids, speech_features, stopping_criteria, **kwargs):
        self.features.append(speech_features)
        assert "speech_tensors" not in kwargs
        assert "past_key_values" not in kwargs
        assert kwargs["do_sample"] is False and kwargs["num_beams"] == 1
        for token in next(self.candidates)[:kwargs["max_new_tokens"]]:
            input_ids = torch.cat([input_ids, torch.tensor([[token]])], dim=1)
            if any(stop(input_ids, None).item() for stop in stopping_criteria) or token == 0:
                break
        return input_ids


def recover(model, *, max_retries=2, max_period=32, **changes):
    inputs = {"input_ids": torch.tensor([[99]]), "speech_tensors": torch.ones(1, 8),
              "acoustic_input_mask": torch.tensor([[True]])}
    inputs.update(changes)
    return generate_with_repetition_recovery(
        model, CharacterTokenizer(), inputs,
        max_new_tokens=256, eos_token_id=0, pad_token_id=0,
        max_retries=max_retries, max_period=max_period,
    )


@pytest.mark.parametrize("period, attempts", [(8, 1), (32, 2)])
def test_configured_period_controls_nine_token_loop_recovery(period, attempts):
    prefix = list(map(ord, '"Content":"'))
    cycle = list(map(ord, "abcdefghi"))
    expected = (prefix, ord("a")) if period >= 9 else None
    assert _find_edge(prefix + cycle * 3, CharacterTokenizer(), period) == expected
    model = ScriptedModel([prefix + cycle * 8 + [0], list(map(ord, "finished")) + [0]])
    _, result = recover(model, max_period=period)
    assert len(result["attempts"]) == attempts
    assert result["max_period"] == period


def test_edge_exclusion_is_local_to_the_exact_generated_prefix():
    processor = _ExcludeEdges(2, [([3, 4], 5), ([3, 4], 6)])
    scores = torch.arange(8, dtype=torch.float32).unsqueeze(0)
    changed = processor(torch.tensor([[90, 91, 3, 4]]), scores.clone())
    assert torch.isneginf(changed[0, 5:7]).all()
    assert torch.equal(changed[0, :5], scores[0, :5])
    assert changed[0, 7] == scores[0, 7]
    for history in ([3], [3, 4, 7], [4, 3]):
        assert torch.equal(processor(torch.tensor([[90, 91] + history]), scores.clone()), scores)


@pytest.mark.parametrize("phrases, retries, status", [
    (["abcdefghij "] * 2, 2, "no_supported_fork"),
    (["abcdefghij ", "klmnopqrst ", "uvwxyzabcd "], 2, "retry_limit_reached"),
    (["abcdefghij "], 0, "retry_limit_reached"),
    (["abcdefghij ", "klmnopqrst "], 1, "retry_limit_reached"),
    (["abcdefghij ", "klmnopqrst ", "uvwxyzabcd ", "efghijklmn "], 3, "retry_limit_reached"),
])
def test_bounded_retries_preserve_original_output_and_features(phrases, retries, status):
    candidates = [list(map(ord, phrase * 8)) + [0] for phrase in phrases]
    model = ScriptedModel(candidates)
    _, result = recover(model, max_retries=retries)
    assert result["status"] == status
    assert result["max_retries"] == retries
    assert len(result["attempts"]) == len(candidates)
    assert result["attempts"][0]["output_ids"] == candidates[0]
    for attempt, candidate in zip(result["attempts"][1:], candidates[1:]):
        assert len(attempt["output_ids"]) < len(candidate)
        assert attempt["repeat_detected"] and not attempt["ended_with_eos"]
    assert all(f is model.features[0] for f in model.features)
    model.encode_speech.assert_called_once()


@pytest.mark.parametrize("ids, status, eos, limit", [
    (list(map(ord, "quiet real speech")) + [0], "no_repetition_detected", True, False),
    (list(map(ord, "one two two three four")) + [0], "no_repetition_detected", True, False),
    (list(range(300, 600)), "output_limit_reached", False, True),
    (list(map(ord, "partial speech")), "stopped_without_eos", False, False),
])
def test_nonloop_output_and_termination_status(ids, status, eos, limit):
    sequences, result = recover(ScriptedModel([ids]))
    assert sequences[0, 1:].tolist() == ids[:256]
    assert len(result["attempts"]) == 1
    assert result["excluded_edges"] == []
    assert result["status"] == status
    assert result["attempts"][0]["ended_with_eos"] is eos
    assert result["attempts"][0]["hit_token_limit"] is limit


@pytest.mark.parametrize("options, error", [
    ({"input_ids": torch.ones(2, 1, dtype=torch.long)}, "one audio"),
    ({"max_retries": -1}, "nonnegative"), ({"max_period": 0}, "positive"),
])
def test_invalid_recovery_options_are_rejected_before_encoding(options, error):
    model = ScriptedModel([])
    with pytest.raises(ValueError, match=error):
        recover(model, **options)
    model.encode_speech.assert_not_called()


def test_cached_speech_features_are_only_forwarded_at_prefill():
    features = torch.ones(1, 4)
    prepare = VibeVoiceASRForConditionalGeneration.prepare_inputs_for_generation
    first = prepare(None, torch.tensor([[1, 2]]), cache_position=torch.tensor([0, 1]),
                    speech_features=features, acoustic_input_mask=torch.tensor([[True, False]]))
    assert first["speech_features"] is features
    cache = SimpleNamespace(get_seq_length=lambda: 2)
    following = prepare(None, torch.tensor([[1, 2, 3]]), past_key_values=cache,
                        cache_position=torch.tensor([2]), speech_features=features)
    assert following.get("speech_features") is None
    assert following["input_ids"].tolist() == [[3]]


@pytest.mark.parametrize("feature_shape, mask", [
    ((1, 4), [True, False]), ((1, 1), [True, False]),
    ((4,), [True, False]), ((1, 4), [True, True]),
])
def test_forward_uses_cached_features_without_resampling(feature_shape, mask):
    embeddings = torch.nn.Embedding(5, 4)
    language_model = Mock(return_value=SimpleNamespace(
        last_hidden_state=torch.zeros(1, 2, 4), past_key_values=None,
        hidden_states=None, attentions=None,
    ))
    model = SimpleNamespace(
        config=SimpleNamespace(output_attentions=False, output_hidden_states=False,
                               use_return_dict=True, use_cache=True),
        get_input_embeddings=lambda: embeddings,
        encode_speech=Mock(side_effect=AssertionError("unexpected resampling")),
        model=language_model, lm_head=torch.nn.Linear(4, 5),
    )
    features = torch.full(feature_shape, 7.0)
    inputs = {"input_ids": torch.tensor([[1, 2]]), "speech_features": features,
              "acoustic_input_mask": torch.tensor([mask])}
    if feature_shape != (sum(mask), 4):
        with pytest.raises(ValueError, match="shape"):
            VibeVoiceASRForConditionalGeneration.forward(model, **inputs)
        language_model.assert_not_called()
        return
    VibeVoiceASRForConditionalGeneration.forward(model, **inputs)
    inserted = language_model.call_args.kwargs["inputs_embeds"]
    assert torch.equal(inserted[0, 0], features[0])
    assert torch.equal(inserted[0, 1], embeddings.weight[2])
    model.encode_speech.assert_not_called()


def test_real_generate_applies_the_exclusion_and_leaves_the_loop():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    class TinyASR(Qwen2ForCausalLM):
        def forward(self, input_ids=None, speech_features=None, acoustic_input_mask=None, **kwargs):
            output = super().forward(input_ids=input_ids, **kwargs)
            # Deterministic scores: repeat 'a' unless excluded; 'b' leads to EOS.
            output.logits.fill_(-10)
            output.logits[..., ord("a")] = 3
            output.logits[..., ord("b")] = 2
            output.logits[..., 0] = 4 if input_ids[0, -1] == ord("b") else 1
            return output

    model = TinyASR(Qwen2Config(
        vocab_size=128, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=1, num_key_value_heads=1,
    )).eval()
    model.encode_speech = Mock(return_value=torch.ones(1, 4))
    _, result = recover(model)
    original, retry = result["attempts"]
    assert original["output_ids"] == [ord("a")] * 256
    assert original["hit_token_limit"] and original["repeat_detected"]
    assert retry["output_ids"] == [ord("b"), 0]
    assert result["status"] == "no_repetition_detected"
    model.encode_speech.assert_called_once()


@pytest.mark.parametrize("enabled", [False, True])
def test_file_demo_exposes_attempts_only_when_enabled(enabled):
    from demo.vibevoice_asr_inference_from_file import VibeVoiceASRBatchInference

    pipeline = VibeVoiceASRBatchInference.__new__(VibeVoiceASRBatchInference)
    tokenizer = CharacterTokenizer()
    tokenizer.eos_token_id = 0
    pipeline.processor = Mock(tokenizer=tokenizer, pad_id=0)
    pipeline.processor.return_value = {
        "input_ids": torch.tensor([[99]]), "speech_tensors": torch.ones(1, 8),
        "acoustic_input_mask": torch.tensor([[True]]), "attention_mask": torch.ones(1, 1),
    }
    pipeline.processor.decode.return_value = "quiet speech"
    pipeline.processor.post_process_transcription.return_value = []
    pipeline.device = "cpu"
    ids = list(map(ord, "quiet speech")) + [0]
    pipeline.model = ScriptedModel([ids]) if enabled else Mock()
    if not enabled:
        pipeline.model.generate.return_value = torch.tensor([[99] + ids])
    results = pipeline.transcribe_with_batching(
        ["unused.wav"], batch_size=1, do_sample=False, recover_repetition=enabled,
        recovery_max_retries=0, recovery_max_period=9,
    )
    assert results[0]["raw_text"] == "quiet speech"
    if enabled:
        assert results[0]["recovery"]["attempts"][0]["output_ids"] == ids
        assert results[0]["recovery"]["max_retries"] == 0
        assert results[0]["recovery"]["max_period"] == 9
    else:
        assert "recovery" not in results[0]
        assert "speech_features" not in pipeline.model.generate.call_args.kwargs
        pipeline.model.encode_speech.assert_not_called()


@pytest.mark.parametrize("options", [
    {"batch_size": 2, "do_sample": False},
    {"batch_size": 1, "do_sample": True},
    {"batch_size": 1, "do_sample": False, "num_beams": 2},
])
def test_file_demo_rejects_unsupported_recovery_modes(options):
    from demo.vibevoice_asr_inference_from_file import VibeVoiceASRBatchInference

    pipeline = VibeVoiceASRBatchInference.__new__(VibeVoiceASRBatchInference)
    with pytest.raises(ValueError, match="greedy"):
        pipeline.transcribe_with_batching(["unused.wav"], recover_repetition=True, **options)
