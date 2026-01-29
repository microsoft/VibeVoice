import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

from backend.routes import tts


class DummyNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class DummyTorch:
    def load(self, *args, **kwargs):
        return {"voice": "cache"}

    def is_tensor(self, value):
        return False

    def no_grad(self):
        return DummyNoGrad()


class DummyModel:
    def generate(self, **kwargs):
        return {"speech_outputs": [np.zeros(3, dtype=np.float32)]}


class DummyProcessor:
    def __init__(self):
        self.texts = []
        self.saved_audio = []
        self.audio_processor = SimpleNamespace(sampling_rate=10)
        self.tokenizer = object()

    def process_input_with_cached_prompt(self, text, cached_prompt, **kwargs):
        self.texts.append(text)
        return {"input_ids": [1, 2, 3]}

    def save_audio(self, audio, output_path, sampling_rate):
        self.saved_audio.append((audio, output_path, sampling_rate))
        Path(output_path).write_bytes(b"RIFF")


def _setup_tts(monkeypatch, tmp_path):
    dummy_processor = DummyProcessor()
    dummy_model = DummyModel()
    dummy_torch = DummyTorch()

    def fake_load_model_and_processor(model_name, device):
        return dummy_model, dummy_processor, "cpu", dummy_torch

    monkeypatch.setattr(tts, "_load_model_and_processor", fake_load_model_and_processor)
    monkeypatch.setattr(
        tts,
        "_extract_audio_from_outputs",
        lambda outputs, torch_module: np.arange(3, dtype=np.float32),
    )
    voice_file = tmp_path / "voice.pt"
    voice_file.write_bytes(b"voice")
    monkeypatch.setattr(tts, "_find_voice_file", lambda voice_id: voice_file)

    tts._VOICE_CACHE.clear()
    tts._MODEL_CACHE.clear()
    tts._PROCESSOR_CACHE.clear()
    tts._MODEL_LOCKS.clear()
    # Use monkeypatch to temporarily override audio_dir so it is restored after the test
    monkeypatch.setattr(tts.settings, "audio_dir", tmp_path / "audio", raising=False)

    return dummy_processor


def test_convert_strips_markdown_and_excludes_heading(monkeypatch, tmp_path):
    processor = _setup_tts(monkeypatch, tmp_path)
    payload = tts.TTSRequest(
        content="# Heading\n\nThis is **bold** and `code`.",
        voice_id="demo-voice",
        filename="doc.md",
        chunk_depth=1,
        pause_ms=0,
        include_heading=False,
        strip_markdown=True,
        device="cpu",
        iterations=1,
    )

    response = asyncio.run(tts.convert_to_speech(payload))
    assert response.success is True
    assert len(processor.texts) == 1
    text = processor.texts[0]
    assert "Heading" not in text
    assert "**" not in text
    assert "`" not in text
    assert "bold" in text


def test_convert_includes_heading_and_pause(monkeypatch, tmp_path):
    processor = _setup_tts(monkeypatch, tmp_path)
    payload = tts.TTSRequest(
        content="# One\n\nFirst.\n\n# Two\n\nSecond.",
        voice_id="demo-voice",
        filename="doc.md",
        chunk_depth=1,
        pause_ms=1000,
        include_heading=True,
        strip_markdown=False,
        device="cpu",
        iterations=1,
    )

    response = asyncio.run(tts.convert_to_speech(payload))
    assert response.success is True
    assert len(processor.texts) == 2
    assert processor.texts[0].startswith("One")
    assert "First." in processor.texts[0]
    assert processor.texts[1].startswith("Two")
    assert "Second." in processor.texts[1]

    audio, _, sampling_rate = processor.saved_audio[0]
    assert sampling_rate == 10
    # The fake `_extract_audio_from_outputs` returns 3 samples per chunk (np.arange(3)).
    # With `include_heading=True` the input produces 2 chunks, and we configured `pause_ms=1000`.
    # The pause between chunks is translated to `pause_samples = sampling_rate * (pause_ms / 1000)`.
    # Total expected length = (samples_per_chunk * num_chunks) + (pause_samples * (num_chunks - 1)).
    samples_per_chunk = len(np.arange(3, dtype=np.float32))
    num_chunks = 2
    pause_samples = int(payload.pause_ms / 1000 * sampling_rate)
    expected_length = samples_per_chunk * num_chunks + pause_samples * (num_chunks - 1)
    assert len(audio) == expected_length


def test_convert_preserves_markdown_when_strip_disabled(monkeypatch, tmp_path):
    processor = _setup_tts(monkeypatch, tmp_path)
    payload = tts.TTSRequest(
        content="# Heading\n\nThis is **bold** and `code`.",
        voice_id="demo-voice",
        filename="doc.md",
        chunk_depth=1,
        pause_ms=0,
        include_heading=False,
        strip_markdown=False,
        device="cpu",
        iterations=1,
    )

    response = asyncio.run(tts.convert_to_speech(payload))
    assert response.success is True
    assert len(processor.texts) == 1
    text = processor.texts[0]
    assert "**bold**" in text
    assert "`code`" in text


def test_ttsrequest_validates_numeric_fields(monkeypatch, tmp_path):
    from pydantic import ValidationError
    # Ensure a voice and processor are available for runtime checks
    _setup_tts(monkeypatch, tmp_path)
    # chunk_depth must be >= 1
    with pytest.raises(ValidationError):
        tts.TTSRequest(
            content="# Test",
            voice_id="demo-voice",
            chunk_depth=0,
            pause_ms=0,
            iterations=1,
        )

    # pause_ms must be >= 0
    with pytest.raises(ValidationError):
        tts.TTSRequest(
            content="# Test",
            voice_id="demo-voice",
            chunk_depth=1,
            pause_ms=-10,
            iterations=1,
        )

    # iterations greater than server max should be rejected at runtime
    payload = tts.TTSRequest(
        content="# Test",
        voice_id="demo-voice",
        chunk_depth=1,
        pause_ms=0,
        iterations=20,
    )
    with pytest.raises(tts.HTTPException) as excinfo:
        asyncio.run(tts.convert_to_speech(payload))
    assert excinfo.value.status_code == 400
    assert f"Max iterations per request is {tts.settings.max_iterations_per_request}" in str(excinfo.value.detail)
