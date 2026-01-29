from fastapi.testclient import TestClient
from backend.main import app


def test_download_rejects_percent_encoded_traversal():
    client = TestClient(app)
    payload = {
        "audio_url": "http://localhost/static/audio/%2e%2e/secret.txt",
        "filename": "out.wav",
        "format": "wav",
    }
    resp = client.post('/api/export/download', json=payload)
    assert resp.status_code == 400
    assert resp.json().get('detail') == 'Invalid audio URL'


def test_download_accepts_normal_static_path():
    client = TestClient(app)
    payload = {
        "audio_url": "http://localhost/static/audio/out.wav",
        "filename": "out.wav",
        "format": "wav",
    }
    resp = client.post('/api/export/download', json=payload)
    assert resp.status_code == 200
    assert resp.json().get('success') is True
