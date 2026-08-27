"""The streaming ASR sequence format, apart from the server that drives it.

The model was trained on a single interleaved sequence

    Prompt + [<sp_start> AudioWindow <sp_end> Text <|text_chunk_end|>] * N

so a client's job is to cut audio into the windows training used and rebuild
that sequence chunk by chunk. This module does that, plus the inverse: turning
the per-chunk text back into readable segments. Three processes need one or
both -- the vLLM engine, the streaming server, and the browser demo -- and a
copy of the chunk arithmetic in each would be free to drift.

Nothing here imports vLLM or torch at module level, so the demo can reach it by
bare import (``vllm_plugin/`` on sys.path, ``import asr_streaming``) without
executing ``vllm_plugin/__init__.py`` and its model registration.
"""
import json
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

# The model signals end-of-chunk with this token rather than EOS. It is the
# first free slot after Qwen2.5's vocabulary and was trained, not synthesised.
# The tokenizer reads it from the checkpoint rather than defining it, so a
# checkpoint whose files disagree with this literal resolves to something else
# -- or to nothing, in which case every chunk runs to max_tokens.
TEXT_CHUNK_END_ID = 151665

# Plain text, no chat template, no BOS. Not the processor's SYSTEM_PROMPT +
# <|im_start|> template, which serves the non-streaming single-shot path only.
_PROMPT_HEAD = (
    "You are a helpful assistant that transcribes audio input into text output. "
    "Please transcribe the following audios streamingly with these keys: "
    "speaker, content"
)


@dataclass(frozen=True)
class ChunkGeometry:
    """How a checkpoint cuts audio: read from it, never guessed.

    A window carries ``chunk_frames`` of new audio plus ``lookahead_frames``
    that the next window repeats. The released checkpoints differ here -- 15
    frames (2.0s) and 22 frames (2.9333s), 4 frames of lookahead either way --
    and serving one at the other's geometry does not raise, it just transcribes
    worse. So it is read from ``preprocessor_config.json``, which every
    streaming checkpoint records it in.
    """

    sample_rate: int
    frame_samples: int
    chunk_frames: int
    lookahead_frames: int

    @property
    def window_frames(self) -> int:
        return self.chunk_frames + self.lookahead_frames

    @property
    def chunk_samples(self) -> int:
        return self.chunk_frames * self.frame_samples

    @property
    def window_samples(self) -> int:
        return self.window_frames * self.frame_samples

    @property
    def chunk_seconds(self) -> float:
        """The advance, which segment times and the demo's capture use.

        Chunk k is fed audio ``[k*advance, k*advance + window)`` and the text it
        returns covers ``[k*advance, (k+1)*advance)`` -- the trailing lookahead
        is audio whose transcript arrives with the next chunk.
        """
        return self.chunk_samples / self.sample_rate

    @classmethod
    def from_pretrained(cls, model_path: str) -> "ChunkGeometry":
        """Load the geometry from a local checkpoint directory or a HF repo id."""
        name = "preprocessor_config.json"
        if os.path.isdir(model_path):
            config_path = os.path.join(model_path, name)
        else:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(model_path, name)

        with open(config_path) as f:
            cfg = json.load(f)

        missing = [k for k in ("chunk_frames", "lookahead_frames") if k not in cfg]
        if missing:
            raise ValueError(
                f"{config_path} has no {', '.join(missing)}, so {model_path} is "
                "not a streaming checkpoint. Serve it with start_server.py instead."
            )
        return cls(
            sample_rate=cfg["target_sample_rate"],
            frame_samples=cfg["speech_tok_compress_ratio"],
            chunk_frames=cfg["chunk_frames"],
            lookahead_frames=cfg["lookahead_frames"],
        )

    def describe(self) -> str:
        return (f"chunk {self.chunk_frames} frames ({self.chunk_seconds:.4f}s), "
                f"lookahead {self.lookahead_frames} frames, "
                f"window {self.window_samples} samples @ {self.sample_rate} Hz")


def build_prompt(context_info: Optional[str] = None) -> str:
    """Training-format prompt, with the optional hotword/context suffix."""
    if context_info:
        return f"{_PROMPT_HEAD} and extra info: {context_info}\n"
    return f"{_PROMPT_HEAD}\n"


def split_windows(audio, geometry: ChunkGeometry, pad_last: bool = True) -> List[np.ndarray]:
    """Cut audio into overlapping windows, matching ``split_then_encode``.

    ``pad_last`` zero-pads the final window to a full window. That is not
    cosmetic: training split already-encoded features, so its last chunk had
    ``floor(remainder/frame)`` frames, while encoding a short segment rounds
    *up*. Without the pad the final window would carry one extra frame and no
    lookahead -- off-distribution exactly where the sequence ends.
    """
    audio = np.asarray(audio, dtype=np.float32)
    total = audio.shape[0]
    window_samples = geometry.window_samples

    windows = []
    start = 0
    while start < total:
        seg = audio[start:min(start + window_samples, total)]
        if seg.shape[0]:
            if pad_last and seg.shape[0] < window_samples:
                seg = np.pad(seg, (0, window_samples - seg.shape[0]))
            windows.append(seg)
        # Advance by the text boundary, not by the window: a window that already
        # reaches the end of the audio still only transcribes its first chunk,
        # so the trailing lookahead needs a window of its own.
        start = min(start + geometry.chunk_samples, total)
    return windows


def load_audio_bytes(audio_bytes: bytes, target_sr: int) -> np.ndarray:
    """Decode in-memory audio to the mono float32 the windows are cut from.

    The suffix is a placeholder: ffmpeg sniffs the container from the bytes, so
    an mp3 or a webm opus blob written to a ``.wav`` name decodes fine.

    Decodes at the native rate and resamples with librosa rather than letting
    ffmpeg do both, which is what the streaming checkpoints were evaluated with.
    The two resamplers differ by ~1% RMS, and near-tied greedy argmaxes turn
    that into different words.
    """
    from vibevoice.processor.audio_utils import load_audio_use_ffmpeg

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        audio, sr = load_audio_use_ffmpeg(tmp.name, resample=False)

    if sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio


# --- Chunks back into readable segments -------------------------------------
# The model emits one block per chunk, each re-stating the speaker, each free to
# end mid-sentence. Shown verbatim that is thirty fragments a minute. The job
# below merges them into paragraphs that end where sentences do, identically for
# every caller -- a rule that differs between the demo and the server is a bug
# nobody notices until the two are compared side by side.

_CHUNK_SPEAKER_RE = re.compile(r"Speaker\s+(\d+)\s*:")

# Chunks merge until the segment is at least this long, then close at the next
# sentence-ending punctuation. The cap stops a stretch with no punctuation from
# growing one card until it swallows the whole clip; kept at twice the minimum,
# so punctuation and not the cap is what usually ends a card.
SEGMENT_MIN_S = 10.0
SEGMENT_MAX_S = 20.0

# Two dialects, because the rule differs. A CJK stop mark is unambiguous. An
# ASCII period is not -- requiring whitespace or end-of-text after it keeps
# "3.5" and "Mr. Smith" from splitting a segment in half.
_SENTENCE_END_RE = re.compile(
    r"[。！？…]+[”’」』）】》]*"
    r"|[.!?]+[\"'’”)\]]*(?=\s|$)"
)

# Backreference, so only a repeat of the *same* tag collapses: [Noise][Noise]
# folds, [Noise][Laughter] does not. Tags become adjacent only when blocks are
# merged into one card, which is why this runs here and not per block.
_REPEAT_TAG_RE = re.compile(r"(\[[^\[\]]*\])(?:\s*\1)+")

# A card is one paragraph, so the line breaks the model emits inside a block
# fold to single spaces -- except between two CJK characters, where a space is
# not a word separator and just leaves a visible gap.
_WS_RUN_RE = re.compile(r"\s+")
_CJK = r"　-〿぀-ヿ一-鿿＀-･"
_CJK_SPACE_RE = re.compile(rf"(?<=[{_CJK}]) (?=[{_CJK}])")


def tidy_body(text: str) -> str:
    """Fold a merged card's text back into one paragraph."""
    return _CJK_SPACE_RE.sub("", _WS_RUN_RE.sub(" ", text)).strip()


def chunk_segments(chunk_texts: List[str],
                   geometry: ChunkGeometry,
                   duration: Optional[float] = None) -> List[Dict]:
    """Merge the per-chunk stream into readable, sentence-aligned segments.

    Times stay arithmetic rather than inferred: chunk k *is* the k-th window, so
    every boundary is a multiple of ``geometry.chunk_seconds``. Two things split
    a chunk internally -- a speaker change, and a sentence end past the
    SEGMENT_MIN_S mark. In both cases the words go to the right card while the
    timestamp stays on the chunk edge.

    The speaker carries forward because the model only re-states it when it
    changes, so a chunk with no label belongs to the previous speaker.

    ``duration`` only ever clamps the tail. While a clip is still streaming its
    real duration is unknown, and the chunk count is what defines the timeline.
    """
    advance = geometry.chunk_seconds
    segments: List[Dict] = []
    cur_start = 0.0
    cur_text = ""
    speaker = None

    def flush(end: float) -> None:
        nonlocal cur_start, cur_text
        body = tidy_body(_REPEAT_TAG_RE.sub(r"\1", cur_text))
        # Silence really does come back empty; those chunks fold into whichever
        # segment is open instead of becoming a card of their own.
        if body:
            if end > cur_start:
                segments.append({
                    "Start": round(cur_start, 2),
                    "End": round(end, 2),
                    "Speaker": speaker if speaker is not None else "?",
                    "Content": body,
                })
            elif segments:
                # No timeline left to give this text -- the remainder after a
                # sentence split in the final chunk. Dropping it is how words go
                # missing off the end of a transcript, so it joins the card it
                # was split from instead.
                prev = segments[-1]
                prev["Content"] = tidy_body(
                    _REPEAT_TAG_RE.sub(r"\1", prev["Content"] + " " + body))
        cur_start = end
        cur_text = ""

    for k, text in enumerate(chunk_texts):
        start = k * advance
        end = start + advance
        if duration and duration > start:
            end = min(end, duration)

        # split() on a single-group pattern yields [pre, id, text, id, text...],
        # which is what lets a mid-chunk speaker change break the segment.
        pieces = _CHUNK_SPEAKER_RE.split(text)
        added = pieces[0]
        # Two people can easily both talk inside one chunk. Closing the card at
        # ``start`` would put the boundary where the previous card already
        # ended, and a zero-width card carries no text -- so the cut is placed
        # proportionally through the chunk, by how much of its text is consumed.
        chunk_chars = max(1, sum(len(p) for p in pieces))
        consumed = len(pieces[0])
        for i in range(1, len(pieces), 2):
            if speaker is not None and pieces[i] != speaker:
                cur_text += added
                flush(max(start + (end - start) * (consumed / chunk_chars), cur_start))
                added = ""
            speaker = pieces[i]
            added += pieces[i + 1]
            consumed += len(pieces[i]) + len(pieces[i + 1])
        cur_text += added

        span = end - cur_start
        if span >= SEGMENT_MIN_S:
            # Only a stop mark inside *this* chunk is usable: one further back
            # would strand the text between it and here in the wrong card.
            last = None
            for last in _SENTENCE_END_RE.finditer(added):
                pass
            if last is not None:
                carry = added[last.end():]
                cur_text = cur_text[:len(cur_text) - len(carry)] if carry else cur_text
                flush(end)
                cur_text = carry
                continue
        if span >= SEGMENT_MAX_S:
            flush(end)

    total = len(chunk_texts) * advance
    flush(min(total, duration) if duration and duration > cur_start else total)
    return segments


def segments_to_text(segments: List[Dict]) -> str:
    """Render segments as the plain transcript, one line per segment."""
    return "\n".join(
        f"Speaker {seg.get('Speaker', '?')}: {seg.get('Content', '')}"
        for seg in segments
    )


def _srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_srt(segments: List[Dict]) -> str:
    """SRT, driven by the same segments as everything else so they agree."""
    return "\n".join(
        f"{i}\n{_srt_time(seg['Start'])} --> {_srt_time(seg['End'])}\n"
        f"Speaker {seg.get('Speaker', '?')}: {seg.get('Content', '')}\n"
        for i, seg in enumerate(segments, 1)
    )
