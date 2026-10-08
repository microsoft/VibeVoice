#!/usr/bin/env python3
"""Regression test for issue #415: a decode loop repeats the same clause across
chunks, and chunk_segments used to render every copy verbatim.

Torch-free: imports asr_streaming directly (numpy only) the same way the demo
does, so it runs without weights, a GPU, or a vLLM install. Run with:

    python3 vllm_plugin/tests/test_asr_streaming_repeat.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asr_streaming as a  # noqa: E402

# Released streaming geometry: 15-frame chunk, 4-frame lookahead at 24 kHz.
GEOMETRY = a.ChunkGeometry(
    sample_rate=24000, frame_samples=3200, chunk_frames=15, lookahead_frames=4)

CLAUSE = "the cat sat on the mat. "
LOOP_CHUNKS = 8            # a stuck model emits the same clause every chunk
EMPHASIS = "No. No. "      # two repeats: real speech, must survive


def _content(chunk_texts):
    segs = a.chunk_segments(chunk_texts, GEOMETRY)
    return " ".join(seg["Content"] for seg in segs)


def test_loop_collapses():
    body = _content([f"Speaker 0: {CLAUSE}"] * LOOP_CHUNKS)
    hits = body.count("the cat sat on the mat")
    print(f"[loop] {LOOP_CHUNKS} looped chunks -> clause appears {hits}x: {body!r}")
    assert hits == 1, f"decode loop still repeats the clause {hits}x (issue #415)"


def test_emphasis_survives():
    # A short genuine repeat is below the loop threshold and stays untouched.
    body = _content([f"Speaker 0: {EMPHASIS}"])
    hits = body.count("No")
    print(f"[emphasis] genuine double repeat -> 'No' appears {hits}x: {body!r}")
    assert hits == 2, f"conservative threshold ate real repetition: {body!r}"


if __name__ == "__main__":
    test_loop_collapses()
    test_emphasis_survives()
    print("PASS")
