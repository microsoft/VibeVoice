import importlib

import pytest
from fastapi.testclient import TestClient
import backend.config as config


@pytest.fixture
def cors_client(monkeypatch, request):
    """Fixture that sets config.settings.debug, reloads backend.main and yields a TestClient.

    Restores the original config.settings.debug value and reloads backend.main on teardown to
    avoid leaking mutated global state between tests.
    """
    original_debug = config.settings.debug
    # Apply requested debug flag
    monkeypatch.setattr(config.settings, "debug", request.param)

    import backend.main as main
    importlib.reload(main)

    client = TestClient(main.app)

    try:
        yield client
    finally:
        # Restore original debug flag and reload backend.main to reset middleware/config
        monkeypatch.setattr(config.settings, "debug", original_debug)
        importlib.reload(main)


@pytest.mark.parametrize("cors_client", [True], indirect=True)
def test_cors_regex_enabled(cors_client):
    # Origin on a non-standard localhost port should be accepted when regex is enabled
    resp = cors_client.get("/", headers={"Origin": "http://localhost:4000"})
    assert resp.status_code == 200
    assert "access-control-allow-origin" in resp.headers


@pytest.mark.parametrize("cors_client", [False], indirect=True)
def test_cors_regex_disabled(cors_client):
    resp = cors_client.get("/", headers={"Origin": "http://localhost:4000"})
    assert resp.status_code == 200
    # When regex is disabled the 4000 port origin should not be allowed
    assert "access-control-allow-origin" not in resp.headers
