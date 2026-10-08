"""Exercise the real HTTP routes without loading vLLM or a GPU checkpoint."""
import base64
import importlib.util
from pathlib import Path
import socket
import sys
import types
from unittest.mock import Mock

from fastapi.testclient import TestClient
import pytest
import requests


@pytest.fixture
def server(monkeypatch):
    # The package initializer registers GPU models. These tests need only the
    # actual transport and sequence modules, with their normal dependencies.
    root = Path(__file__).resolve().parents[1] / "vllm_plugin"
    package = types.ModuleType("vllm_plugin")
    package.__path__ = [str(root)]
    monkeypatch.setitem(sys.modules, "vllm_plugin", package)
    for name in ("asr_streaming", "asr_streaming_server"):
        fullname = f"vllm_plugin.{name}"
        spec = importlib.util.spec_from_file_location(fullname, root / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, fullname, module)
        spec.loader.exec_module(module)
    return module


@pytest.fixture
def client(server, monkeypatch):
    geometry = server.ChunkGeometry(24000, 3200, 15, 4)
    monkeypatch.setattr(server.ChunkGeometry, "from_pretrained",
                        lambda _: geometry)
    app = server.create_app(server.ServerConfig(model="unused-test-model"))
    app.state.engine = types.SimpleNamespace(geometry=geometry)
    # Do not enter the lifespan: the tests must never load a model.
    client = TestClient(app)
    yield client
    client.close()


@pytest.fixture
def outbound(monkeypatch):
    # Assert both that responses are safe and that no request/DNS lookup was
    # attempted. The tests never contact an internal or external destination.
    request = Mock(side_effect=AssertionError("unexpected outbound HTTP request"))
    resolve = Mock(side_effect=AssertionError("unexpected DNS lookup"))
    connect = Mock(side_effect=AssertionError("unexpected outbound connection"))
    monkeypatch.setattr(requests.sessions.Session, "request", request)
    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    monkeypatch.setattr(socket.socket, "connect", connect)
    yield
    request.assert_not_called()
    resolve.assert_not_called()
    connect.assert_not_called()


ROUTES = ("/v1/transcribe", "/v1/transcribe_batch", "/v1/chat/completions")


def payload_for(route, *, url=None, encoded=None, stream=False):
    if route == "/v1/chat/completions":
        if encoded is not None:
            url = f"data:audio/wav;base64,{encoded}"
        return {"stream": stream, "messages": [{
            "role": "user", "content": [
                {"type": "audio_url", "audio_url": {"url": url}},
            ],
        }]}
    item = {"audio_url": url} if url is not None else {"audio_base64": encoded}
    return {"audios": [item]} if route.endswith("_batch") else item


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("url", [
    "http://127.0.0.1:8001/audio.wav",
    "http://localhost/audio.wav",
    "http://10.0.0.1/audio.wav",
    "http://172.16.0.1/audio.wav",
    "http://192.168.1.1/audio.wav",
    "http://169.254.169.254/audio.wav",
    "http://[::1]/audio.wav",
    "http://[::ffff:127.0.0.1]/audio.wav",
    "http://2130706433/audio.wav",
    "https://example.com/audio.wav",
    "https://example.com/redirect?url=http://127.0.0.1/",
    "https://user:password@example.com/audio.wav",
    "//example.com/audio.wav",
    "ftp://example.com/audio.wav",
    "file:///tmp/audio.wav",
])
def test_disallowed_urls_never_access_network(client, outbound, inference,
                                            route, url):
    response = client.post(route, json=payload_for(route, url=url))
    assert response.status_code == 400
    assert url not in response.text
    inference[1].assert_not_called()


@pytest.fixture
def inference(server, monkeypatch):
    raw = b"test audio bytes"
    decoder = Mock(return_value=server.np.zeros(24000, dtype=server.np.float32))
    monkeypatch.setattr(server, "load_audio_bytes", decoder)

    class Session:
        async def transcribe(self, audio):
            yield "Speaker 1: hello"

        async def push(self, window):
            return "Speaker 1: hello"

    monkeypatch.setattr(server, "_new_session", lambda *_: Session())
    return raw, decoder


@pytest.mark.parametrize("route", ROUTES)
def test_inline_audio_reaches_inference(client, outbound, inference, route):
    raw, decoder = inference
    encoded = base64.b64encode(raw).decode("ascii")
    response = client.post(route, json=payload_for(route, encoded=encoded))
    assert response.status_code == 200
    assert "hello" in response.text
    decoder.assert_called_once_with(raw, 24000)


def test_inline_audio_chat_stream(client, outbound, inference):
    raw, decoder = inference
    encoded = base64.b64encode(raw).decode("ascii")
    response = client.post("/v1/chat/completions", json=payload_for(
        "/v1/chat/completions", encoded=encoded, stream=True))
    assert response.status_code == 200
    assert "hello" in response.text
    assert "data: [DONE]" in response.text
    decoder.assert_called_once_with(raw, 24000)


def test_remote_url_rejected_before_chat_stream(client, outbound, inference):
    response = client.post("/v1/chat/completions", json=payload_for(
        "/v1/chat/completions", url="https://example.com/audio.wav", stream=True))
    assert response.status_code == 400
    assert response.headers["content-type"] == "application/json"
    inference[1].assert_not_called()


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("encoded", ["", "%%%", "YQ", "YQ==!", "\u2603"])
def test_invalid_base64_returns_client_error(client, outbound, inference,
                                           route, encoded):
    response = client.post(route, json=payload_for(route, encoded=encoded))
    assert response.status_code == 400
    inference[1].assert_not_called()


@pytest.mark.parametrize("url", [
    None, 7, {},
    "data:audio/wav;base64",
    "data:audio/wav,YXVkaW8=",
    "data:audio/wav;base64;extra,YXVkaW8=",
    "data:audio/wav;base64,http://127.0.0.1/audio.wav",
])
def test_invalid_data_uri_returns_client_error(client, outbound, inference, url):
    response = client.post("/v1/chat/completions", json=payload_for(
        "/v1/chat/completions", url=url))
    assert response.status_code == 400
    inference[1].assert_not_called()


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("size", [4, 5, 7])
def test_inline_audio_size_limit(client, outbound, server, inference,
                                 monkeypatch, route, size):
    monkeypatch.setattr(server, "MAX_AUDIO_BYTES", 4)
    raw = b"a" * size
    encoded = base64.b64encode(raw).decode("ascii")
    response = client.post(route, json=payload_for(route, encoded=encoded))
    if size == 4:
        assert response.status_code == 200
        inference[1].assert_called_once_with(raw, 24000)
    else:
        assert response.status_code == 413
        inference[1].assert_not_called()


def test_batch_accepts_multiple_inline_files(client, outbound, inference):
    raw, decoder = inference
    encoded = base64.b64encode(raw).decode("ascii")
    response = client.post("/v1/transcribe_batch", json={"audios": [
        {"audio_base64": encoded}, {"audio_base64": encoded},
    ]})
    assert response.status_code == 200
    assert len(response.json()["results"]) == 2
    assert decoder.call_count == 2


@pytest.mark.parametrize("route", ["/v1/transcribe", "/v1/transcribe_batch"])
def test_extra_url_cannot_trigger_download_with_inline_audio(
        client, outbound, inference, route):
    raw, decoder = inference
    item = {"audio_base64": base64.b64encode(raw).decode("ascii"),
            "audio_url": "http://127.0.0.1/audio.wav"}
    payload = {"audios": [item]} if route.endswith("_batch") else item
    response = client.post(route, json=payload)
    assert response.status_code == 200
    decoder.assert_called_once_with(raw, 24000)


def test_openapi_only_advertises_inline_audio(client):
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    for name in ("TranscribeRequest", "BatchAudioItem"):
        assert "audio_base64" in schemas[name]["properties"]
        assert "audio_url" not in schemas[name]["properties"]


def test_websocket_audio_still_transcribes(client, outbound, server, inference):
    with client.websocket_connect("/v1/stream") as ws:
        ws.send_json({})
        ws.send_bytes(server.np.zeros(24000, dtype="<f4").tobytes())
        ws.send_text("end")
        assert "hello" in ws.receive_json()["text"]
        assert ws.receive_json()["done"] is True
