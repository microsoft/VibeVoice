"""
vibevoice_asr_chunked_inference.py
------------------------------------
Chunked inference for VibeVoice-ASR on GPUs with limited VRAM (e.g. RTX 4090, 24 GB).

Why chunking?
  VibeVoice-ASR encodes audio at 7.5 Hz. On a 24 GB GPU with default `sdpa` attention
  the peak VRAM ceiling is ~30 minutes. For longer audio, split into chunks and
  transcribe each independently, then concatenate the results.

Caveat:
  Each chunk receives independent speaker diarization, so speaker IDs (SPEAKER_00,
  SPEAKER_01, …) are NOT globally consistent across chunks. If you need a single
  consistent speaker mapping, post-process the outputs to re-align speaker labels.

Usage:
  python demo/vibevoice_asr_chunked_inference.py \\
      --model_path microsoft/VibeVoice-ASR \\
      --audio_file long_audio.m4a \\
      --chunk_minutes 25 \\
      --device cuda

  # Optionally persist raw per-chunk JSON:
  python demo/vibevoice_asr_chunked_inference.py \\
      --model_path microsoft/VibeVoice-ASR \\
      --audio_file long_audio.m4a \\
      --chunk_minutes 25 \\
      --device cuda \\
      --save_chunks_json chunks_output.json

See docs/vibevoice-asr.md § "Hardware Requirements & VRAM Guide" for more detail.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Audio utilities (pure stdlib + ffmpeg — no extra Python deps beyond the
# packages already required by VibeVoice itself)
# ---------------------------------------------------------------------------

def get_audio_duration_seconds(audio_path: str) -> float:
    """Return the duration of an audio file in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as exc:
        raise RuntimeError(
            f"ffprobe failed on '{audio_path}'. Make sure ffmpeg is installed "
            f"(apt install ffmpeg). Original error: {exc}"
        ) from exc


def extract_audio_chunk(
    audio_path: str,
    start_sec: float,
    duration_sec: float,
    output_path: str,
) -> None:
    """Extract a time slice from an audio file using ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-t", str(duration_sec),
        "-i", audio_path,
        "-ac", "1",          # mono
        "-ar", "16000",      # 16 kHz — expected by VibeVoice-ASR
        "-vn",               # no video
        output_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)


# ---------------------------------------------------------------------------
# Model loading — reuse the same helper the existing demo script uses
# ---------------------------------------------------------------------------

def load_model_and_processor(model_path: str, device: str, attn_impl: str):
    """Load VibeVoice-ASR model and processor."""
    try:
        from vibevoice.model import VibeVoiceASRModel          # noqa: F401
        from vibevoice.processor import VibeVoiceASRProcessor  # noqa: F401
    except ImportError:
        pass  # fall through to transformers-based loading

    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    import torch

    print(f"Loading model from '{model_path}' …")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if "cuda" in device else torch.float32,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    print("Model loaded.")
    return model, processor


def transcribe_chunk(
    audio_path: str,
    model,
    processor,
    device: str,
    hotwords: str | None = None,
) -> dict:
    """Run VibeVoice-ASR inference on a single audio file and return result dict."""
    import torch
    import torchaudio

    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)

    inputs = processor(
        waveform.squeeze(0).numpy(),
        sampling_rate=16000,
        return_tensors="pt",
        hotwords=hotwords,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs)

    result = processor.decode(outputs[0], skip_special_tokens=False)
    return result if isinstance(result, dict) else {"text": str(result)}


# ---------------------------------------------------------------------------
# Timestamp shifting
# ---------------------------------------------------------------------------

def shift_timestamps(result: dict, offset_seconds: float) -> dict:
    """
    Add `offset_seconds` to every timestamp in a VibeVoice-ASR result dict.

    VibeVoice-ASR returns a list of segments, each looking like:
        {"speaker": "SPEAKER_00", "start": 0.5, "end": 3.2, "text": "Hello"}

    If the model returns a plain string (no structured output), this is a no-op.
    """
    if "segments" not in result:
        return result

    shifted = dict(result)
    shifted["segments"] = [
        {
            **seg,
            "start": round(seg["start"] + offset_seconds, 3),
            "end":   round(seg["end"]   + offset_seconds, 3),
        }
        for seg in result["segments"]
    ]
    return shifted


def merge_results(chunk_results: list[dict]) -> dict:
    """Concatenate per-chunk results into a single result dict."""
    all_segments = []
    all_text_parts = []

    for res in chunk_results:
        if "segments" in res:
            all_segments.extend(res["segments"])
        if "text" in res:
            all_text_parts.append(res["text"].strip())

    merged: dict = {}
    if all_segments:
        merged["segments"] = all_segments
    if all_text_parts:
        merged["text"] = " ".join(all_text_parts)

    return merged


def format_transcript(merged: dict) -> str:
    """Pretty-print the merged transcription to a human-readable string."""
    lines = []
    if "segments" in merged:
        for seg in merged["segments"]:
            start = seg.get("start", "?")
            end   = seg.get("end",   "?")
            spk   = seg.get("speaker", "SPEAKER_??")
            text  = seg.get("text", "").strip()
            lines.append(f"[{start:.2f}s → {end:.2f}s]  {spk}: {text}")
    elif "text" in merged:
        lines.append(merged["text"])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Chunked inference for VibeVoice-ASR — for GPUs with limited VRAM."
    )
    parser.add_argument(
        "--model_path", required=True,
        help="HuggingFace model ID or local path (e.g. microsoft/VibeVoice-ASR).",
    )
    parser.add_argument(
        "--audio_file", required=True,
        help="Path to the input audio file (any format supported by ffmpeg).",
    )
    parser.add_argument(
        "--chunk_minutes", type=float, default=25.0,
        help=(
            "Maximum duration (minutes) of each chunk. "
            "Default: 25 min (safe for RTX 4090 / 24 GB with sdpa). "
            "Use 50+ min if you have flash_attention_2 installed."
        ),
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Device string: 'cuda', 'cuda:0', 'cpu', etc.",
    )
    parser.add_argument(
        "--attn_implementation", default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
        help=(
            "Attention backend. Use 'flash_attention_2' with --chunk_minutes 55 "
            "to process 60-min audio on a 24 GB GPU without chunking."
        ),
    )
    parser.add_argument(
        "--hotwords", default=None,
        help="Optional hotwords / context string passed to the model.",
    )
    parser.add_argument(
        "--save_chunks_json", default=None,
        help="If set, save the raw per-chunk results to this JSON file.",
    )
    args = parser.parse_args()

    audio_path = args.audio_file
    if not os.path.exists(audio_path):
        sys.exit(f"ERROR: audio file not found: {audio_path}")

    # ---- Determine chunk boundaries ----------------------------------------
    total_seconds = get_audio_duration_seconds(audio_path)
    chunk_seconds = args.chunk_minutes * 60.0
    n_chunks = math.ceil(total_seconds / chunk_seconds)

    print(
        f"\nAudio duration : {total_seconds / 60:.1f} min  ({total_seconds:.1f} s)"
    )
    print(f"Chunk size     : {args.chunk_minutes:.1f} min  ({chunk_seconds:.0f} s)")
    print(f"Number of chunks: {n_chunks}\n")

    if n_chunks == 1:
        print(
            "INFO: Audio fits in a single chunk. Consider using the standard "
            "vibevoice_asr_inference_from_file.py script instead."
        )

    # ---- Load model once -----------------------------------------------------
    model, processor = load_model_and_processor(
        args.model_path, args.device, args.attn_implementation
    )

    # ---- Transcribe each chunk -----------------------------------------------
    chunk_results = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(n_chunks):
            start_sec    = i * chunk_seconds
            duration_sec = min(chunk_seconds, total_seconds - start_sec)
            chunk_path   = os.path.join(tmpdir, f"chunk_{i:04d}.wav")

            print(
                f"[{i+1}/{n_chunks}] Extracting chunk: "
                f"{start_sec/60:.1f} min → {(start_sec + duration_sec)/60:.1f} min …"
            )
            extract_audio_chunk(audio_path, start_sec, duration_sec, chunk_path)

            print(f"[{i+1}/{n_chunks}] Transcribing …")
            raw_result = transcribe_chunk(
                chunk_path, model, processor, args.device, args.hotwords
            )
            shifted = shift_timestamps(raw_result, offset_seconds=start_sec)
            chunk_results.append(shifted)
            print(f"[{i+1}/{n_chunks}] Done.\n")

    # ---- Merge & print -------------------------------------------------------
    merged = merge_results(chunk_results)
    transcript = format_transcript(merged)

    print("=" * 70)
    print("FINAL TRANSCRIPT")
    print("=" * 70)
    print(transcript)
    print("=" * 70)

    # ---- Optionally persist per-chunk JSON -----------------------------------
    if args.save_chunks_json:
        out_path = Path(args.save_chunks_json)
        out_path.write_text(
            json.dumps(
                {"chunks": chunk_results, "merged": merged},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nPer-chunk JSON saved to: {out_path}")

    print(
        "\nNOTE: Speaker IDs (SPEAKER_00, SPEAKER_01, …) are independent per chunk "
        "and are NOT globally consistent. Post-process to re-align if needed."
    )


if __name__ == "__main__":
    main()
