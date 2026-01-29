import gc
import time
import pytest

from backend.routes import tts


def _wait_for_key_removal(mapping, key, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        gc.collect()
        if key not in mapping:
            return True
        time.sleep(0.05)
    return False


def test_model_lock_pruning():
    # Ensure a clean starting point
    tts._MODEL_LOCKS.clear()

    cache_key = ("__test_model__", "cpu")

    # Create a lock via the helper and confirm it's present
    lock = tts._get_or_create_model_lock(cache_key)
    assert cache_key in tts._MODEL_LOCKS

    # Remove the strong reference and wait for weak-value dict to drop it
    del lock
    assert _wait_for_key_removal(tts._MODEL_LOCKS, cache_key), "Model lock was not pruned from weak map"


def test_voice_lock_pruning():
    # Ensure a clean starting point
    tts._VOICE_LOCKS.clear()

    voice_id = "__test_voice__"

    lock = tts._get_or_create_voice_lock(voice_id)
    assert voice_id in tts._VOICE_LOCKS

    del lock
    assert _wait_for_key_removal(tts._VOICE_LOCKS, voice_id), "Voice lock was not pruned from weak map"
