"""Opt-in, bounded ASR branch recovery. Loop detection is not an accuracy check."""

import torch
from transformers import LogitsProcessor, StoppingCriteria


def _has_repetition(text):
    # Match the existing recovery client's three-repeat, 500-character heuristic.
    if len(text) < 30:
        return False
    window = text[-500:]
    for size in range(10, len(window) // 3 + 1):
        if window[-size:] * 3 == window[-3 * size:]:
            return True
    words = window.split()
    for size in range(2, 6):
        if len(words) >= 3 * size and words[-size:] * 3 == words[-3 * size:]:
            return True
    return False


def _find_edge(ids, tokenizer, max_period=32):
    """Locate the first lexical token of a periodic suffix."""
    candidates = []
    for period in range(1, min(max_period, len(ids) // 3) + 1):
        if ids[-period:] * 3 != ids[-3 * period:]:
            continue
        start = len(ids) - 3 * period
        while start > 0 and ids[start - 1] == ids[start - 1 + period]:
            start -= 1
        for index in range(start, start + period):
            if any(c.isalpha() for c in tokenizer.decode([ids[index]])):
                candidates.append((start, period, index))
                break
    if not candidates:
        return None
    _, _, index = min(candidates)
    return ids[:index], ids[index]


class _ExcludeEdges(LogitsProcessor):
    def __init__(self, prompt_length, edges):
        self.prompt_length = prompt_length
        self.edges = edges

    def __call__(self, input_ids, scores):
        for prefix, token in self.edges:
            if input_ids.shape[1] != self.prompt_length + len(prefix):
                continue
            if input_ids[0, self.prompt_length:].tolist() == prefix:
                scores[0, token] = -float("inf")
        return scores


class _ObserveRepeat(StoppingCriteria):
    def __init__(self, prompt_length, tokenizer, stop):
        self.prompt_length = prompt_length
        self.tokenizer = tokenizer
        self.stop = stop
        self.first_repeat = None

    def __call__(self, input_ids, scores, **kwargs):
        if self.first_repeat is None:
            ids = input_ids[0, self.prompt_length:].tolist()
            if _has_repetition(self.tokenizer.decode(ids, skip_special_tokens=True)):
                self.first_repeat = ids
        return torch.tensor(
            [self.stop and self.first_repeat is not None], device=input_ids.device
        )


@torch.no_grad()
def generate_with_repetition_recovery(
    model, tokenizer, inputs, *, max_new_tokens, eos_token_id, pad_token_id,
    max_retries=2, max_period=32,
):
    """Generate one greedy ASR sequence with bounded loop-triggered retries.

    Encode audio once and reuse those exact features. Each retry regenerates the
    prompt with a fresh KV cache, banning only previously rejected prefix/token
    edges. The original decode runs to EOS or its token limit; retries stop at
    their first detected loop. All attempts are retained, including failures.

    Returns (sequences, recovery), where recovery is JSON-serializable metadata.
    A non-repeating candidate may still be an inaccurate transcription.
    """
    if max_retries < 0 or max_period < 1:
        raise ValueError("max_retries must be nonnegative and max_period must be positive.")
    if inputs["input_ids"].shape[0] != 1:
        raise ValueError("Repetition recovery supports one audio input at a time.")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")
    if inputs.get("past_key_values") is not None:
        raise ValueError("Repetition recovery requires a fresh prompt.")
    if inputs.get("acoustic_input_mask") is None:
        raise ValueError("Repetition recovery requires acoustic_input_mask.")
    inputs = dict(inputs)
    if inputs.get("speech_features") is None:
        inputs["speech_features"] = model.encode_speech(
            speech_tensors=inputs["speech_tensors"],
            speech_masks=inputs.get("speech_masks"),
            speech_semantic_tensors=inputs.get("speech_semantic_tensors"),
        )
    elif inputs.get("speech_tensors") is not None:
        raise ValueError("Provide speech_tensors or speech_features, not both.")
    for name in ("speech_tensors", "speech_masks", "speech_semantic_tensors"):
        inputs.pop(name, None)

    prompt_length = inputs["input_ids"].shape[1]
    edges = []
    attempts = []
    for attempt in range(max_retries + 1):
        observer = _ObserveRepeat(prompt_length, tokenizer, stop=attempt > 0)
        sequences = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            num_return_sequences=1,
            repetition_penalty=1.0,
            use_cache=True,
            return_dict_in_generate=False,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            stopping_criteria=[observer],
            logits_processor=[_ExcludeEdges(prompt_length, edges)],
        )
        ids = sequences[0, prompt_length:].tolist()
        ended_with_eos = bool(ids and ids[-1] == eos_token_id)
        attempts.append({
            "output_ids": ids,
            "raw_text": tokenizer.decode(ids, skip_special_tokens=True),
            "ended_with_eos": ended_with_eos,
            "hit_token_limit": len(ids) == max_new_tokens and not ended_with_eos,
            "repeat_detected": observer.first_repeat is not None,
        })
        if observer.first_repeat is None:
            if ended_with_eos:
                status = "no_repetition_detected"
            else:
                status = "output_limit_reached" if attempts[-1]["hit_token_limit"] else "stopped_without_eos"
            break
        if attempt == max_retries:
            status = "retry_limit_reached"
            break
        edge = _find_edge(observer.first_repeat, tokenizer, max_period)
        if edge is None or edge in edges:
            status = "no_supported_fork"
            break
        edges.append(edge)

    return sequences, {
        "status": status, "attempts": attempts, "excluded_edges": edges,
        "max_retries": max_retries, "max_period": max_period,
    }
