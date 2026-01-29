from __future__ import annotations

from typing import List, Optional, Union, Dict, Any

import numpy as np
import torch

from .markdown_chunker import MarkdownChunker, MarkdownChunk


class ChunkedTTSProcessor:
    """Process markdown documents in chunks for TTS synthesis."""

    def __init__(
        self,
        processor,
        model,
        chunk_depth: int = 1,
        strip_markdown: bool = True,
        include_heading: bool = True,
        speaker_id: int = 0,
    ):
        self.processor = processor
        self.model = model
        self.chunker = MarkdownChunker(chunk_depth=chunk_depth, strip_markdown=strip_markdown)
        self.include_heading = include_heading
        self.speaker_id = speaker_id

    def process_markdown_file(
        self,
        markdown_file: str,
        voice_sample: Union[str, np.ndarray],
        output_file: Optional[str] = None,
        device: Optional[str] = None,
        pause_duration_ms: int = 500,
        verbose: bool = True,
        generation_kwargs: Optional[Dict[str, Any]] = None,
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
                voice_sample=voice_sample,
                device=device,
                generation_kwargs=generation_kwargs,
            )
            audio_segments.append(audio)

            if verbose:
                duration = len(audio) / self._sample_rate() if len(audio) else 0.0
                print(f"Generated {duration:.2f}s of audio")

        merged_audio = self._merge_audio_segments(audio_segments, pause_duration_ms=pause_duration_ms)

        if output_file:
            self.processor.save_audio(merged_audio, output_path=output_file, sampling_rate=self._sample_rate())
            if verbose:
                total_duration = len(merged_audio) / self._sample_rate() if len(merged_audio) else 0.0
                print(f"Saved {total_duration:.2f}s to {output_file}")

        return merged_audio

    def _synthesize_chunk(
        self,
        chunk: MarkdownChunk,
        voice_sample: Union[str, np.ndarray],
        device: Optional[str],
        generation_kwargs: Optional[Dict[str, Any]],
    ) -> np.ndarray:
        if not chunk.content.strip() and not (self.include_heading and chunk.heading):
            return np.zeros(0, dtype=np.float32)

        text_parts = []
        if self.include_heading and chunk.heading:
            text_parts.append(chunk.heading)
        if chunk.content.strip():
            text_parts.append(chunk.content.strip())
        content = "\n".join(text_parts)

        script = f"Speaker {self.speaker_id}: {content}"
        inputs = self.processor(
            text=script,
            voice_samples=[voice_sample],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(device)

        gen_kwargs = {"do_sample": False}
        if generation_kwargs:
            gen_kwargs.update(generation_kwargs)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        return self._extract_audio_from_outputs(outputs)

    def _extract_audio_from_outputs(self, outputs: Any) -> np.ndarray:
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

    def _merge_audio_segments(self, segments: List[np.ndarray], pause_duration_ms: int = 500) -> np.ndarray:
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
            audio = audio.detach().cpu().numpy()
        audio = np.asarray(audio).squeeze()
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        return audio

    def _sample_rate(self) -> int:
        return int(getattr(self.processor.audio_processor, "sampling_rate", 24000))
