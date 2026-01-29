import os
from pathlib import Path
from fastapi.testclient import TestClient
import pytest

from backend.main import app
from backend.config import settings

client = TestClient(app)


def _voice_available(voice_id: str) -> bool:
    # Repo root derived reliably from this test file's location
    repo_root = Path(__file__).resolve().parents[3]
    demo_voices_dir = repo_root / "demo" / "voices" / "streaming_model"
    for voices_dir in (settings.voices_dir, demo_voices_dir):
        # Skip if voices_dir is unset
        if not voices_dir:
            continue
        # Coerce string-backed configs to Path so both str and Path work
        voices_path = Path(voices_dir)
        # Skip if the path does not exist
        if not voices_path.exists():
            continue
        # Use a single glob-based test to find matching .pt files by stem
        if any(path.stem == voice_id for path in voices_path.glob("*.pt")):
            return True
    return False


def _skip_tts_if_unavailable(voice_id: str) -> None:
    if os.getenv("VIBEVOICE_RUN_TTS") != "1":
        pytest.skip("Set VIBEVOICE_RUN_TTS=1 to run TTS integration tests.")
    if not _voice_available(voice_id):
        pytest.skip(f"Voice '{voice_id}' not available for integration test.")


def test_health_check():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'healthy'}


def test_config_endpoint():
    response = client.get('/config')
    assert response.status_code == 200
    data = response.json()
    assert 'default_model' in data
    assert 'default_device' in data
    assert 'sample_rate' in data
    assert 'default_pause_ms' in data
    assert 'default_chunk_depth' in data


def test_list_voices_returns_list():
    response = client.get('/voices')
    assert response.status_code == 200
    data = response.json()
    assert 'voices' in data
    assert isinstance(data['voices'], list)


def test_tts_route_registered():
    paths = [r.path for r in app.routes]
    assert any(p.startswith('/api/tts') for p in paths), f"TTS routes missing, available routes: {paths}"


def test_tts_convert_requires_content():
    response = client.post('/api/tts/convert', json={
        'content': '   ',
        'voice_id': 'test-voice',
    })
    assert response.status_code == 400
    assert response.json()['detail'] == 'Content is empty'


def test_tts_convert_requires_voice_id():
    response = client.post('/api/tts/convert', json={
        'content': '# Hello',
        'voice_id': '   ',
    })
    assert response.status_code == 400
    assert response.json()['detail'] == 'voice_id is required'


def test_tts_convert_success():
    _skip_tts_if_unavailable("demo-voice")
    response = client.post('/api/tts/convert', json={
        'content': '# Title\n\nHello world',
        'voice_id': 'demo-voice',
        'filename': 'demo.md',
        'chunk_depth': 2,
        'pause_ms': 300,
        'include_heading': True,
        'strip_markdown': True,
        'device': 'auto',
    })
    assert response.status_code == 200, f"Unexpected status: {response.status_code}; routes={[r.path for r in app.routes]}; body={response.text}"
    data = response.json()
    assert data['success'] is True
    assert data['audio_url']
    assert data['duration'] is not None


def test_export_rejects_invalid_format():
    # With strict Literal typing for `format`, invalid values are rejected by Pydantic and FastAPI
    response = client.post('/api/export/download', json={
        'audio_url': '/static/audio/output.wav',
        'format': 'flac',
    })
    assert response.status_code == 422
    # The validation errors should reference the `format` field
    assert 'format' in str(response.json()['detail'])


def test_export_sanitizes_filename():
    # Filenames containing path traversal segments should be rejected by validation
    response = client.post('/api/export/download', json={
        'audio_url': '/static/audio/output.wav',
        'format': 'wav',
        'filename': '../evil.wav'
    })
    # Pydantic validation should reject traversal-containing filenames
    assert response.status_code == 422
    assert 'filename' in str(response.json().get('detail', ''))

def test_export_rejects_traversal():
    response = client.post('/api/export/download', json={
        'audio_url': '/static/../secret.wav',
        'format': 'wav',
    })
    assert response.status_code == 400
    assert response.json()['detail'] == 'Invalid audio URL'


def test_export_allows_static_audio():
    response = client.post('/api/export/download', json={
        'audio_url': '/static/audio/output.wav',
        'format': 'wav',
    })
    assert response.status_code == 200
    data = response.json()
    assert data['success'] is True
    assert data['download_url'] == '/static/audio/output.wav'


def test_export_rejects_static_directory():
    response = client.post('/api/export/download', json={
        'audio_url': '/static',
        'format': 'wav',
    })
    assert response.status_code == 400
    assert response.json()['detail'] == 'Invalid audio URL'


def test_tts_convert_respects_options():
    """Ensure API accepts optional conversion parameters and returns success.

    Note: This test verifies the API accepts these options and responds successfully; it does not assert
    on the audio content or specific behavioral effects of the options.
    """
    _skip_tts_if_unavailable("demo-voice")
    # Verify API accepts parameters and returns successful conversion when using chunking/heading/strip options
    response = client.post('/api/tts/convert', json={
        'content': '# Heading\n\nThis is a test with **bold** and `code`.',
        'voice_id': 'demo-voice',
        'filename': 'options.md',
        'chunk_depth': 1,
        'pause_ms': 100,
        'include_heading': False,
        'strip_markdown': False,
        'device': 'auto',
    })
    assert response.status_code == 200, f"Unexpected status: {response.status_code}; routes={[r.path for r in app.routes]}; body={response.text}"
    data = response.json()
    assert data['success'] is True
    assert data['audio_url']
    assert data['duration'] is not None
