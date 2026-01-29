#!/usr/bin/env python3
"""
Demo script for processing markdown documents with chunked TTS using VibeVoice-Realtime model.
"""

import argparse
import copy
import glob
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
from vibevoice.processor.chunking import MarkdownChunker, MarkdownChunk


class VoiceMapper:
    """Maps speaker names to voice file paths"""

    def __init__(self):
        self.setup_voice_presets()

    def setup_voice_presets(self):
        """Setup voice presets by scanning voices directory."""
        voices_dir = os.path.join(os.path.dirname(__file__), "voices/streaming_model")

        # Check if voices directory exists
        if not os.path.exists(voices_dir):
            print(f"Error: Voices directory not found at {voices_dir}")
            print("Please provide voice files or download them to demo/voices/streaming_model/")
            self.voice_presets = {}
            self.available_voices = {}
            return

        # Scan for all VOICE files in voices directory
        self.voice_presets = {}

        # Get all .pt files in voices directory
        pt_files = glob.glob(os.path.join(voices_dir, "**", "*.pt"), recursive=True)

        # Create dictionary with filename (without extension) as key
        for pt_file in pt_files:
            # key: filename without extension
            name = os.path.splitext(os.path.basename(pt_file))[0].lower()
            full_path = os.path.abspath(pt_file)
            self.voice_presets[name] = full_path

        # Sort voice presets alphabetically by name for better UI
        self.voice_presets = dict(sorted(self.voice_presets.items()))

        # Filter out voices that don't exist (this is now redundant but kept for safety)
        self.available_voices = {
            name: path for name, path in self.voice_presets.items()
            if os.path.exists(path)
        }

        print(f"Found {len(self.available_voices)} voice files in {voices_dir}")
        print(f"Available voices: {', '.join(self.available_voices.keys())}")

    def get_voice_path(self, speaker_name: str) -> str:
        """Get voice file path for a given speaker name"""
        # Check if any voices are available
        if not self.voice_presets:
            raise ValueError(
                f"No voice presets available. "
                f"Please provide voice files in demo/voices/streaming_model/ "
                f"or use --voice-path to specify a voice file directly."
            )

        # First try exact match
        speaker_name = speaker_name.lower()
        if speaker_name in self.voice_presets:
            return self.voice_presets[speaker_name]

        # Try partial matching (case insensitive)
        matched_path = None
        for preset_name, path in self.voice_presets.items():
            if preset_name.lower() in speaker_name or speaker_name in preset_name.lower():
                if matched_path is not None:
                    raise ValueError(
                        f"Multiple voice presets match the speaker name '{speaker_name}', "
                        f"please make the speaker name more specific."
                    )
                matched_path = path
        if matched_path is not None:
            return matched_path

        # Default to first voice if no match found
        default_voice = list(self.voice_presets.values())[0]
        print(
            f"Warning: No voice preset found for '{speaker_name}', "
            f"using default voice: {default_voice}"
        )
        return default_voice


class ChunkedTTSProcessorRealtime:
    """Process markdown documents in chunks for TTS synthesis using Realtime model."""

    def __init__(
        self,
        processor: VibeVoiceStreamingProcessor,
        model: VibeVoiceStreamingForConditionalGenerationInference,
        voice_cache: Dict,
        chunk_depth: int = 1,
        strip_markdown: bool = True,
        include_heading: bool = True,
        cfg_scale: float = 1.25,
    ):
        self.processor = processor
        self.model = model
        self.voice_cache = voice_cache
        self.chunker = MarkdownChunker(chunk_depth=chunk_depth, strip_markdown=strip_markdown)
        self.include_heading = include_heading
        self.cfg_scale = cfg_scale

    def process_markdown_file(
        self,
        markdown_file: str,
        output_file: Optional[str] = None,
        device: Optional[str] = None,
        pause_duration_ms: int = 500,
        verbose: bool = True,
    ) -> np.ndarray:
        """Process a markdown file and return merged audio."""
        with open(markdown_file, "r", encoding="utf-8") as handle:
            markdown_text = handle.read()

        chunks = self.chunker.chunk(markdown_text)

        if verbose:
            print(f"Processing {len(chunks)} chunks from {markdown_file}")
            print(self.chunker.get_chunk_summary(chunks))

        audio_segments: List[np.ndarray] = []
        for index, chunk in enumerate(chunks, 1):
            if verbose:
                print(f"Synthesizing chunk {index}/{len(chunks)}: {chunk.heading}")

            audio = self._synthesize_chunk(
                chunk=chunk,
                device=device,
            )
            audio_segments.append(audio)

            if verbose:
                duration = len(audio) / self._sample_rate() if len(audio) else 0.0
                print(f"Generated {duration:.2f}s of audio")

        merged_audio = self._merge_audio_segments(
            audio_segments, pause_duration_ms=pause_duration_ms
        )

        if output_file:
            self.processor.save_audio(
                merged_audio, output_path=output_file, sampling_rate=self._sample_rate()
            )
            if verbose:
                total_duration = (
                    len(merged_audio) / self._sample_rate() if len(merged_audio) else 0.0
                )
                print(f"Saved {total_duration:.2f}s to {output_file}")

        return merged_audio

    def _synthesize_chunk(
        self,
        chunk: MarkdownChunk,
        device: Optional[str],
    ) -> np.ndarray:
        if not chunk.content.strip() and not (self.include_heading and chunk.heading):
            return np.zeros(0, dtype=np.float32)

        text_parts = []
        if self.include_heading and chunk.heading:
            text_parts.append(chunk.heading)
        if chunk.content.strip():
            text_parts.append(chunk.content.strip())
        content = "\n".join(text_parts)

        # Prepare inputs for model using cached voice prompt
        inputs = self.processor.process_input_with_cached_prompt(
            text=content,
            cached_prompt=copy.deepcopy(self.voice_cache),
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Move tensors to target device
        for key, value in inputs.items():
            if torch.is_tensor(value):
                inputs[key] = value.to(device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=self.cfg_scale,
                tokenizer=self.processor.tokenizer,
                generation_config={"do_sample": False},
                verbose=False,
                all_prefilled_outputs=copy.deepcopy(self.voice_cache),
            )

        return self._extract_audio_from_outputs(outputs)

    def _extract_audio_from_outputs(self, outputs) -> np.ndarray:
        if outputs is None:
            raise ValueError("Model returned no outputs")

        if hasattr(outputs, "speech_outputs") and outputs.speech_outputs:
            return self._to_numpy_audio(outputs.speech_outputs[0])

        if isinstance(outputs, dict) and "speech_outputs" in outputs and outputs["speech_outputs"]:
            return self._to_numpy_audio(outputs["speech_outputs"][0])

        if hasattr(outputs, "audio"):
            return self._to_numpy_audio(outputs.audio)

        if isinstance(outputs, torch.Tensor):
            return self._to_numpy_audio(outputs)

        raise ValueError(f"Unexpected output format: {type(outputs)}")

    def _merge_audio_segments(
        self, segments: List[np.ndarray], pause_duration_ms: int = 500
    ) -> np.ndarray:
        if not segments:
            return np.zeros(0, dtype=np.float32)

        pause_samples = int(self._sample_rate() * pause_duration_ms / 1000)
        silence = np.zeros(pause_samples, dtype=np.float32)

        merged: List[np.ndarray] = []
        for index, segment in enumerate(segments):
            merged.append(self._to_numpy_audio(segment))
            if index < len(segments) - 1 and pause_samples > 0:
                merged.append(silence)

        if not merged:
            return np.zeros(0, dtype=np.float32)

        return np.concatenate(merged)

    def _to_numpy_audio(self, audio: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(audio, torch.Tensor):
            # Convert to float32 first to handle BFloat16 tensors
            audio = audio.float().detach().cpu().numpy()
        audio = np.asarray(audio).squeeze()
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        return audio

    def _sample_rate(self) -> int:
        return int(getattr(self.processor.audio_processor, "sampling_rate", 24000))


def resolve_device(device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate TTS audio from markdown documents using heading-based chunking with VibeVoice-Realtime model."
    )
    parser.add_argument("--markdown", "-m", required=True, help="Path to markdown (.md) file")
    parser.add_argument(
        "--speaker",
        "-s",
        required=True,
        help="Speaker name (e.g., Carter, Davis, Emma, Frank, Grace, Mike)",
    )
    parser.add_argument("--output", "-o", default="output.wav", help="Output audio file path")
    parser.add_argument("--depth", "-d", type=int, default=1, help="Heading depth to chunk on")
    parser.add_argument(
        "--model",
        default="microsoft/VibeVoice-Realtime-0.5B",
        help="Model name or path",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Device",
    )
    parser.add_argument("--pause-ms", type=int, default=500, help="Pause duration between chunks")
    parser.add_argument(
        "--include-heading", action="store_true", help="Speak headings in each chunk"
    )
    parser.add_argument(
        "--no-strip", action="store_true", help="Keep markdown formatting in content"
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=1.25,
        help="CFG (Classifier-Free Guidance) scale for generation (default: 1.5)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress progress output"
    )

    args = parser.parse_args()

    md_path = Path(args.markdown)
    if not md_path.exists():
        print(f"Markdown file not found: {args.markdown}")
        sys.exit(1)

    device = resolve_device(args.device)

    # Initialize voice mapper
    voice_mapper = VoiceMapper()

    # Get voice path for the specified speaker
    voice_path = voice_mapper.get_voice_path(args.speaker)

    if not args.quiet:
        print(f"Loading model: {args.model}")
        print(f"Device: {device}")
        print(f"Using voice preset for {args.speaker}: {voice_path}")

    # Decide dtype & attention implementation
    if device == "mps":
        load_dtype = torch.float32  # MPS requires float32
        attn_impl_primary = "sdpa"  # flash_attention_2 not supported on MPS
    elif device == "cuda":
        load_dtype = torch.bfloat16
        attn_impl_primary = "flash_attention_2"
    else:  # cpu
        load_dtype = torch.float32
        attn_impl_primary = "sdpa"

    if not args.quiet:
        print(f"Using torch_dtype: {load_dtype}, attn_implementation: {attn_impl_primary}")

    # Load processor
    processor = VibeVoiceStreamingProcessor.from_pretrained(args.model)

    # Load model with device-specific logic
    try:
        if device == "mps":
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                args.model,
                torch_dtype=load_dtype,
                attn_implementation=attn_impl_primary,
                device_map=None,  # load then move
            )
            model.to("mps")
        elif device == "cuda":
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                args.model,
                torch_dtype=load_dtype,
                device_map="cuda",
                attn_implementation=attn_impl_primary,
            )
        else:  # cpu
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                args.model,
                torch_dtype=load_dtype,
                device_map="cpu",
                attn_implementation=attn_impl_primary,
            )
    except Exception as e:
        if attn_impl_primary == "flash_attention_2":
            print(
                f"[ERROR] : {type(e).__name__}: {e}\n"
                f"Error loading model. Trying to use SDPA. "
                f"However, note that only flash_attention_2 has been fully tested, "
                f"and using SDPA may result in lower audio quality."
            )
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                args.model,
                torch_dtype=load_dtype,
                device_map=(args.device if args.device in ("cuda", "cpu") else None),
                attn_implementation="sdpa",
            )
            if device == "mps":
                model.to("mps")
        else:
            raise e

    model.eval()
    model.set_ddpm_inference_steps(num_steps=10)

    # Load voice cache
    voice_cache = torch.load(voice_path, map_location=device, weights_only=False)

    # Create chunked processor
    chunked_processor = ChunkedTTSProcessorRealtime(
        processor=processor,
        model=model,
        voice_cache=voice_cache,
        chunk_depth=args.depth,
        strip_markdown=not args.no_strip,
        include_heading=args.include_heading,
        cfg_scale=args.cfg_scale,
    )

    try:
        chunked_processor.process_markdown_file(
            markdown_file=str(md_path),
            output_file=args.output,
            device=device,
            pause_duration_ms=args.pause_ms,
            verbose=not args.quiet,
        )
    except Exception as exc:
        print(f"Error during processing: {exc}")
        raise


if __name__ == "__main__":
    main()
