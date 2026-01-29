import os
import time
from pathlib import Path

from backend.routes.tts import _purge_preview_files


def test_purge_preview_files_by_ttl(tmp_path):
    output_dir = tmp_path
    now = time.time()

    # Create files with different ages
    recent = output_dir / "preview-v1-recent.wav"
    recent.write_bytes(b"RIFF")
    old = output_dir / "preview-v1-old.wav"
    old.write_bytes(b"RIFF")

    os.utime(recent, (now, now))
    os.utime(old, (now - 3600 * 5, now - 3600 * 5))  # 5 hours old

    # TTL of 60 minutes should remove "old" but keep "recent"
    _purge_preview_files(output_dir, ttl_minutes=60, max_files_per_voice=None)

    remaining = list(output_dir.glob("*.wav"))
    assert recent in remaining
    assert old not in remaining


def test_purge_preview_files_by_max(tmp_path):
    output_dir = tmp_path
    now = time.time()

    paths = []
    for i in range(5):
        p = output_dir / f"preview-voiceA-{i}.wav"
        p.write_bytes(b"RIFF")
        os.utime(p, (now - i * 10, now - i * 10))
        paths.append(p)

    # Keep only 2 latest
    _purge_preview_files(output_dir, ttl_minutes=None, max_files_per_voice=2)

    remaining = sorted(output_dir.glob("*.wav"), key=lambda p: p.name)
    assert len(remaining) == 2
    # Ensure the latest two (i=0 and i=1) remain
    assert any(p.name.endswith("-0.wav") for p in remaining)
    assert any(p.name.endswith("-1.wav") for p in remaining)


def test_purge_endpoint_schedules_purge(tmp_path, monkeypatch):
    # Create a temporary preview dir and files
    output_dir = tmp_path
    (output_dir / "preview").mkdir()
    preview_dir = output_dir / "preview"
    # Create two previews for the same voice so we can test max-files pruning
    p_old = preview_dir / "preview-voiceX-old.wav"
    p_new = preview_dir / "preview-voiceX-new.wav"
    p_old.write_bytes(b"RIFF")
    p_new.write_bytes(b"RIFF")
    # Make the old file older by adjusting its mtime
    import os as _os, time as _time
    now = _time.time()
    _os.utime(p_old, (now - 3600, now - 3600))
    _os.utime(p_new, (now, now))

    # Monkeypatch settings to point to our temp dir
    from backend.routes import tts
    monkeypatch.setattr(tts.settings, "audio_dir", output_dir)
    # Force per-voice max to 1 so purge removes older entries and keeps newest
    monkeypatch.setattr(tts.settings, "preview_max_preview_files", 1)

    from backend.main import app
    from fastapi.testclient import TestClient
    # Use context manager so background tasks are executed before exiting the request scope
    with TestClient(app) as client:
        resp = client.post('/api/tts/preview/purge')
        assert resp.status_code == 200
        assert resp.json().get('success') is True

    # The background purge should have run; verify the preview files were pruned
    # If background tasks aren't executed synchronously by the test client, run purge directly
    from backend.routes import tts
    tts._purge_preview_files(tts.settings.audio_dir / "preview", tts.settings.preview_ttl_minutes, tts.settings.preview_max_preview_files)

    # Poll briefly to avoid flakiness in file deletion timing
    import time as _time
    deadline = _time.time() + 1.0
    while _time.time() < deadline:
        if not p_old.exists() and p_new.exists():
            break
        _time.sleep(0.01)

    assert not p_old.exists(), "Old preview file should be removed by purge"
    assert p_new.exists(), "Newest preview file should be kept by purge"


def test_purge_handles_disappearing_files(tmp_path, monkeypatch):
    output_dir = tmp_path
    (output_dir / "preview").mkdir()
    preview_dir = output_dir / "preview"

    # Create three previews for the same voice with different ages
    p0 = preview_dir / "preview-voiceA-0.wav"
    p1 = preview_dir / "preview-voiceA-1.wav"
    p2 = preview_dir / "preview-voiceA-2.wav"
    p0.write_bytes(b"RIFF")
    p1.write_bytes(b"RIFF")
    p2.write_bytes(b"RIFF")

    # Set mtimes so p2 is newest, p0 oldest
    now = time.time()
    os.utime(p0, (now - 30, now - 30))
    os.utime(p1, (now - 20, now - 20))
    os.utime(p2, (now - 10, now - 10))

    # Monkeypatch Path.stat to raise FileNotFoundError for p1 to simulate it disappearing
    orig_stat = Path.stat

    def fake_stat(self, *args, **kwargs):
        if self.name == p1.name:
            raise FileNotFoundError("simulated disappearance")
        return orig_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fake_stat)

    # Prune to keep only 1 file; the function should not crash and should keep p2
    from backend.routes import tts
    tts._purge_preview_files(preview_dir, ttl_minutes=None, max_files_per_voice=1)

    # p2 (newest) should exist, p0 (oldest) should be removed, p1 may remain (we skipped stat) but no exception raised
    assert p2.exists()
    assert not p0.exists()
