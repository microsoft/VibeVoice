"""FastAPI server for VibeVoice streaming ASR on vLLM V1.

The model transcribes one chunk at a time off a growing interleaved sequence,
so a request here is a *session*, not a single generate call. Three layers,
bottom-up: the engine holder, the per-chunk session that drives it, and the
transport that exposes it. The sequence format itself lives next door in
``asr_streaming.py``, where it reads as a record of the training run rather
than as this server's internals.

This serves the same model as ``start_server.py``, and the choice between them
is about who holds the session. A stock ``vllm serve`` speaks request/response
only, so a client there has to re-send every audio window on every chunk.
Holding the engine in-process lets a session keep its windows engine-side
behind stable uuids -- flat payload per turn -- and lets the server offer a
WebSocket, which is what a live microphone needs.

Endpoints:
  GET  /health, /healthz, /v1/models, /v1/config
  POST /v1/transcribe        whole clip, one response, chunk texts included
  POST /v1/transcribe_batch  several clips concurrently
  POST /v1/chat/completions  OpenAI dialect, the one both backends share
  WS   /v1/stream            true streaming: send audio frames, get text back

Run:
    python3 -m vllm_plugin.asr_streaming_server --model /path/to/checkpoint
"""
import argparse
import asyncio
import base64
import json
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlparse

import numpy as np
import requests
import uvicorn
from fastapi import (FastAPI, HTTPException, Request, WebSocket,
                     WebSocketDisconnect)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from vllm_plugin.asr_streaming import (ChunkGeometry, TEXT_CHUNK_END_ID,
                                       build_prompt, chunk_segments,
                                       load_audio_bytes, segments_to_srt,
                                       split_windows)

logger = logging.getLogger("vibevoice.streaming")

MAX_AUDIO_BYTES = 256 * 1024 * 1024
MAX_BATCH_ITEMS = 64


@dataclass
class ServerConfig:
    """Everything the server needs that is not read from the checkpoint."""

    model: str
    served_model_name: str = "vibevoice"
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 16384
    tensor_parallel_size: int = 1
    # One window per streaming chunk, so this caps session length, not audio
    # size: 512 windows is about 17 minutes at a 2.0s chunk, proportionally
    # longer for a checkpoint with a larger one.
    max_audio_windows: int = 512
    # Sized to hold every window of every in-flight session. An LRU eviction
    # here is not a slowdown but an error: a request whose history was evicted
    # arrives with None for that window and the engine raises "Cache miss".
    mm_processor_cache_gb: float = 16.0
    enforce_eager: bool = False
    enable_prefix_caching: bool = True
    resend_all_audio: bool = False


# --- engine ---------------------------------------------------------------
# The audio never leaves the request: it rides along as multimodal data and the
# plugin's registered processor VAE-encodes it inside the engine. Three things
# follow, and a streaming session depends on all three:
#
#   * automatic prefix caching, which is where all of streaming's speedup lives
#     -- chunk k re-reads chunks 0..k-1 as a prefix instead of recomputing them;
#   * the scheduler's encoder-skip, which drops the VAE call for any window
#     already covered by the KV cache;
#   * continuous batching across sessions, rather than one session at a time.

class VibeVoiceAsyncEngine:
    """Owns a single ``AsyncLLM``; all sessions share it."""

    def __init__(self, config: ServerConfig, geometry: ChunkGeometry):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        self.config = config
        self.geometry = geometry

        engine_args = AsyncEngineArgs(
            model=config.model,
            # The checkpoint's root config says float32; loading it as such
            # would more than double the weight footprint for no benefit.
            dtype=config.dtype,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            tensor_parallel_size=config.tensor_parallel_size,
            trust_remote_code=True,
            # Non-negotiable for streaming: without it every chunk re-prefills
            # the whole session and the cost goes quadratic. Exposed only so the
            # APC-on/APC-off token equality check has something to turn off.
            enable_prefix_caching=config.enable_prefix_caching,
            limit_mm_per_prompt={"audio": config.max_audio_windows},
            mm_processor_cache_gb=config.mm_processor_cache_gb,
            enforce_eager=config.enforce_eager,
        )
        self.engine = AsyncLLM.from_engine_args(engine_args)

        # Loaded by concrete class, not AutoTokenizer: the checkpoint's
        # tokenizer_config.json names VibeVoiceASRTextTokenizerFast, which is
        # not importable from transformers. The session layer needs this class
        # specifically anyway -- text_chunk_end_id lives on it.
        from vibevoice.modular.modular_vibevoice_text_tokenizer import (
            VibeVoiceASRTextTokenizerFast)

        self.tokenizer = VibeVoiceASRTextTokenizerFast.from_pretrained(config.model)

        # Nothing downstream notices this on its own: the weights still load,
        # start-up is clean, the first chunk transcribes correctly -- and then
        # stop_token_ids never matches what the model emits and every later
        # chunk comes back empty with the request still returning 200.
        if self.tokenizer.text_chunk_end_id != TEXT_CHUNK_END_ID:
            raise RuntimeError(
                f"tokenizer layout mismatch: <|text_chunk_end|> resolved to "
                f"{self.tokenizer.text_chunk_end_id}, expected {TEXT_CHUNK_END_ID}.\n"
                f"  model path: {config.model}\n"
                f"  id {TEXT_CHUNK_END_ID} there is "
                f"{self.tokenizer.convert_ids_to_tokens(TEXT_CHUNK_END_ID)!r}\n"
                "Regenerate the tokenizer with "
                "vllm_plugin/tools/generate_tokenizer_files.py.")

    async def generate(
        self,
        prompt_token_ids: Sequence[int],
        audio_items: Sequence[Optional[Any]],
        audio_uuids: Sequence[str],
        sampling_params: Any,
        request_id: str,
    ):
        """Run one chunk turn and return the final ``RequestOutput``.

        ``audio_items`` carries ``None`` for every window the engine has already
        processed; ``audio_uuids`` must stay byte-identical across turns for
        those, since that identifier is what keeps the prefix-cache block hashes
        stable.
        """
        from vllm.inputs import TokensPrompt

        if len(audio_items) != len(audio_uuids):
            raise ValueError(
                f"audio items ({len(audio_items)}) and uuids "
                f"({len(audio_uuids)}) must line up one-to-one")

        prompt = TokensPrompt(
            prompt_token_ids=list(prompt_token_ids),
            multi_modal_data={"audio": list(audio_items)},
            multi_modal_uuids={"audio": list(audio_uuids)},
        )

        final = None
        async for out in self.engine.generate(prompt, sampling_params, request_id):
            final = out
        if final is None:
            raise RuntimeError(f"engine produced no output for {request_id}")
        return final

    async def shutdown(self):
        self.engine.shutdown()


# --- session --------------------------------------------------------------
# The reference implementation (``modeling_vibevoice_asr.py::streaming_generate``)
# walks the interleaved sequence with a hand-held KV cache. vLLM has no API for
# "keep my KV cache and append", so the equivalent here is one request per chunk
# carrying the whole sequence so far, with prefix caching making the repeated
# prefix free. That inverts the usual advice -- resending the prompt is normally
# waste -- but it is what lets the engine batch many sessions continuously.
#
# Two invariants this layer exists to protect:
#
#   * The token sequence must be byte-identical to the HF path. No newline after
#     <sp_end>; <|text_chunk_end|> appended after *every* chunk including one cut
#     short by EOS.
#   * The accumulated prefix is the *pre-expansion* one, holding a single
#     <|AUDIO|> per window. The plugin's processor expands each into
#     [sp_start] + N*[sp_pad] + [sp_end]; feeding an expanded sequence back in
#     would leave no placeholder to match.

_RESIDUAL_SPECIALS = (
    "<|text_chunk_end|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|speech_start|>",
    "<|speech_end|>",
    "<|speech_pad|>",
)


def _strip_specials(text: str) -> str:
    for tok in _RESIDUAL_SPECIALS:
        text = text.replace(tok, "")
    return text


class StreamingSession:
    """One ASR conversation. Not thread-safe; drive it from a single task."""

    def __init__(
        self,
        engine: VibeVoiceAsyncEngine,
        session_id: str,
        *,
        context_info: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
    ):
        from vllm import SamplingParams

        self.engine = engine
        self.config = engine.config
        self.geometry = engine.geometry
        self.session_id = session_id

        tokenizer = engine.tokenizer
        self.audio_token_id = tokenizer.convert_tokens_to_ids("<|AUDIO|>")
        self.text_chunk_end_id = tokenizer.text_chunk_end_id
        if self.audio_token_id is None or self.audio_token_id < 0:
            raise RuntimeError("tokenizer has no <|AUDIO|> placeholder")

        self._tokens = list(
            tokenizer.encode(build_prompt(context_info), add_special_tokens=False))
        self._windows: List[np.ndarray] = []
        self._turn = 0
        self.max_new_tokens = max_new_tokens
        # Per-chunk generated token ids, kept for the HF token-level comparison.
        # Text equality is not enough: two token sequences can detokenize to the
        # same string, and a divergence that starts invisible compounds.
        self.chunk_token_ids: List[List[int]] = []

        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            # Caveat above 1.0: the reference penalizes only tokens generated
            # within the current chunk, while vLLM penalizes prompt tokens too
            # -- here, the entire session history.
            repetition_penalty=repetition_penalty,
            # The model signals end-of-chunk with this token rather than EOS. It
            # is excluded from the output and re-appended to the prefix by hand,
            # so it lands in the sequence even on an EOS stop.
            stop_token_ids=[self.text_chunk_end_id],
            detokenize=True,
        )

    @property
    def num_windows(self) -> int:
        return len(self._windows)

    async def push(self, window) -> str:
        """Transcribe one window of audio and advance the session."""
        window = np.asarray(window, dtype=np.float32).reshape(-1)
        self._windows.append(window)

        tokens = self._tokens + [self.audio_token_id]
        k = len(self._windows) - 1

        # Two separate ceilings, checked here rather than left to the engine,
        # which reports either as an opaque mid-session request failure. The
        # window cap is fixed per session while the context fills at a rate that
        # depends on how much text each chunk produces, so which one is reached
        # first varies with the audio.
        if len(self._windows) > self.config.max_audio_windows:
            minutes = (self.config.max_audio_windows
                       * self.geometry.chunk_seconds / 60)
            raise ValueError(
                f"session exceeded {self.config.max_audio_windows} audio "
                f"windows (~{minutes:.0f} minutes at this checkpoint's chunk). "
                f"Raise --max-audio-windows, or start a new session.")
        # Against what the engine sees, not what is accumulated here: each
        # <|AUDIO|> is one token in the pre-expansion prefix and
        # window_frames + 2 once the processor expands it. Counting the prefix
        # as-is is short by window_frames + 1 per window -- 20 at the released
        # geometry, enough that the engine hits its own limit hundreds of
        # windows earlier and reports the opaque failure this check replaces.
        expanded = len(tokens) + len(self._windows) * (
            self.geometry.window_frames + 1)
        if expanded + self.max_new_tokens > self.config.max_model_len:
            raise ValueError(
                f"session outgrew the {self.config.max_model_len}-token context "
                f"after {k} windows ({expanded} tokens once the audio windows "
                f"are expanded). Raise --max-model-len, or start a new session.")

        # Only the new window carries data. The rest are None + a stable uuid,
        # which keeps the per-turn payload flat instead of growing with the
        # session, and keeps the prefix-cache block hashes stable across turns.
        uuids = [f"{self.session_id}-{i}" for i in range(k + 1)]
        if self.config.resend_all_audio:
            audio: List[Optional[np.ndarray]] = list(self._windows)
        else:
            audio = [None] * k + [window]

        self._turn += 1
        req_id = f"{self.session_id}-t{self._turn}"

        try:
            out = await self.engine.generate(
                tokens, audio, uuids, self.sampling_params, req_id)
        except ValueError as exc:
            if "data is not provided" not in str(exc):
                raise
            # The multimodal processor cache evicted a window we claimed was
            # already there. Recoverable only by re-sending every window; if
            # this fires at all, --mm-processor-cache-gb is undersized.
            logger.warning("session %s lost a cached window, resending all %d",
                           self.session_id, len(self._windows))
            out = await self.engine.generate(
                tokens, list(self._windows), uuids, self.sampling_params,
                req_id + "-full")

        gen = list(out.outputs[0].token_ids)
        # Defensive: vLLM drops the stop token, but a stop on EOS or on the
        # length cap leaves whatever the model actually emitted.
        while gen and gen[-1] == self.text_chunk_end_id:
            gen.pop()

        self._tokens = tokens + gen + [self.text_chunk_end_id]
        self.chunk_token_ids.append(gen)
        return _strip_specials(out.outputs[0].text)

    async def transcribe(self, audio):
        """Run a whole clip through the session, yielding text per chunk."""
        for window in split_windows(audio, self.geometry):
            yield await self.push(window)


# --- transport ------------------------------------------------------------

def _fetch_url(url: str, timeout: int) -> bytes:
    """Fetch audio over http(s), refusing other schemes and oversized bodies."""
    if urlparse(url).scheme not in ("http", "https"):
        raise HTTPException(status_code=400,
                            detail="audio_url must be http or https")
    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()
        body = resp.raw.read(MAX_AUDIO_BYTES + 1, decode_content=True)
    except requests.RequestException as exc:
        raise HTTPException(status_code=400,
                            detail=f"Failed to fetch audio_url: {exc}") from None
    if len(body) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413,
                            detail=f"audio_url body exceeds {MAX_AUDIO_BYTES} bytes")
    return body


class TranscribeRequest(BaseModel):
    audio_base64: Optional[str] = Field(None, description="Base64 audio bytes.")
    audio_url: Optional[str] = Field(None, description="Audio URL if no base64.")
    context_info: Optional[str] = Field(None, description="Hotwords / context.")
    max_tokens: int = Field(256, ge=1, le=2048)
    # Greedy is the reference decoding config; 2.0 is where the demo's slider
    # ends. Rejecting the top of the slider with a 422 kills the session
    # mid-transcription, and clamping silently would make it do nothing.
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.0, ge=0.5, le=2.0)

    def audio_bytes(self) -> bytes:
        if self.audio_base64:
            return base64.b64decode(self.audio_base64)
        if self.audio_url:
            return _fetch_url(self.audio_url, timeout=10)
        raise HTTPException(status_code=400,
                            detail="audio_base64 or audio_url is required")


class TranscribeResponse(BaseModel):
    text: str
    chunk_texts: List[str]
    total_chunks: int
    audio_duration: float
    generate_time: float
    # Elapsed over duration, the orientation the tests and demos print: below 1
    # is faster than real time. The inverse reads the same but ranks backwards.
    rtf: float


class BatchAudioItem(BaseModel):
    audio_base64: Optional[str] = None
    audio_url: Optional[str] = None


class BatchTranscribeRequest(BaseModel):
    audios: List[BatchAudioItem] = Field(..., min_length=1,
                                         max_length=MAX_BATCH_ITEMS)
    context_info: Optional[str] = None
    max_tokens: int = Field(256, ge=1, le=2048)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.0, ge=0.5, le=2.0)


class BatchTranscribeResponse(BaseModel):
    results: List[TranscribeResponse]
    total_time: float


def _new_session(engine: VibeVoiceAsyncEngine,
                 req: TranscribeRequest) -> StreamingSession:
    return StreamingSession(
        engine,
        f"s-{uuid.uuid4().hex[:16]}",
        context_info=req.context_info,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
    )


async def _run_clip(engine: VibeVoiceAsyncEngine,
                    req: TranscribeRequest) -> TranscribeResponse:
    # Both of these block, and both go to a thread for the same reason: a URL
    # fetch waits on the network and ffmpeg shells out, either of which would
    # otherwise stall the event loop for every other in-flight session.
    raw = await asyncio.to_thread(req.audio_bytes)
    sample_rate = engine.geometry.sample_rate
    audio = await asyncio.to_thread(load_audio_bytes, raw, sample_rate)

    session = _new_session(engine, req)
    t0 = time.time()
    chunk_texts = [text async for text in session.transcribe(audio)]
    gen_t = time.time() - t0

    duration = len(audio) / sample_rate
    return TranscribeResponse(
        text="".join(chunk_texts),
        chunk_texts=chunk_texts,
        total_chunks=len(chunk_texts),
        audio_duration=duration,
        generate_time=gen_t,
        rtf=(gen_t / duration) if duration > 0 else 0.0,
    )


# --- OpenAI dialect -------------------------------------------------------
# The session endpoints above are this server's own. Chat-completions is the one
# dialect it shares with ``start_server.py``, and sharing it is the point: same
# served model name, so an existing client can be moved between the two backends
# without being rewritten.
#
# The transcript leaves as a JSON array of segments because that is what the
# dialect already returned -- the non-streaming checkpoint emits segments with
# timestamps directly. A streaming checkpoint emits plain per-chunk blocks, so
# the array is assembled on this side rather than the reply format silently
# changing with the checkpoint.

# Two callers write hotwords into the text part of the user message, and they
# word it differently: the demo puts them below a "context information" heading,
# tests/test_api.py inline after "with extra info:". Everything outside the
# match is the caller's own boilerplate, and feeding that to build_prompt()
# would replace the training prompt with something the model never saw.
_CONTEXT_RE = re.compile(
    r"context information[^\n:]*:[ \t]*\n(.+)|with extra info:[ \t]*([^\n]+)",
    re.IGNORECASE | re.DOTALL)


def _audio_from_messages(messages: list) -> bytes:
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") != "audio_url":
                continue
            url = (part.get("audio_url") or {}).get("url", "")
            if url.startswith("data:"):
                return base64.b64decode(url.split(",", 1)[1])
            return _fetch_url(url, timeout=30)
    raise HTTPException(status_code=400, detail="No audio_url part in messages")


def _context_from_messages(messages: list) -> Optional[str]:
    """Pull the hotword block out of the text parts, if there is one.

    Text without the heading is ignored rather than passed through -- an
    arbitrary instruction is far more likely to derail the prompt than to act
    as context.
    """
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") != "text":
                continue
            m = _CONTEXT_RE.search(part.get("text") or "")
            if m:
                found = (m.group(1) or m.group(2) or "").strip()
                if found:
                    return found
    return None


def _sse(cid: str, model: str, delta: dict, finish: Optional[str] = None) -> str:
    return "data: " + json.dumps({
        "id": cid, "object": "chat.completion.chunk", "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }) + "\n\n"


def _segment_delta(seg: dict, index: int) -> str:
    """One element of the JSON array, with the comma that joins it to the last.

    Sent per segment rather than in one piece at the end so cards appear while
    the clip is still being transcribed; a partial array is a valid intermediate
    state for a client that recovers whole objects as they arrive.
    """
    return ("," if index else "") + json.dumps(seg, ensure_ascii=False)


def _field(body: dict, name: str, default):
    """Read an optional body field without treating 0 or 0.0 as absent."""
    value = body.get(name)
    return default if value is None else type(default)(value)


def create_app(config: ServerConfig) -> FastAPI:
    """Build the app around one engine. The engine starts with the app."""
    geometry = ChunkGeometry.from_pretrained(config.model)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("loading %s -- %s", config.model, geometry.describe())
        app.state.engine = VibeVoiceAsyncEngine(config, geometry)
        app.state.geometry = geometry
        logger.info("ready: %s served as %r", config.model,
                    config.served_model_name)
        yield
        await app.state.engine.shutdown()

    app = FastAPI(title="VibeVoice vLLM streaming ASR", version="2.0",
                  lifespan=lifespan)

    @app.get("/healthz")
    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"object": "list",
                "data": [{"id": config.served_model_name, "object": "model",
                          "owned_by": "vibevoice"}]}

    @app.get("/v1/config")
    async def get_config():
        """The geometry a client has to cut audio at, read from the checkpoint.

        The demo captures at ``sample_rate`` and lays its cards out on
        ``chunk_seconds``, so both are served rather than hard-coded there --
        that is what lets one page drive either released checkpoint.
        """
        return {
            "model": config.served_model_name,
            "sample_rate": geometry.sample_rate,
            "chunk_seconds": geometry.chunk_seconds,
            "chunk_frames": geometry.chunk_frames,
            "lookahead_frames": geometry.lookahead_frames,
            "window_samples": geometry.window_samples,
            "max_audio_windows": config.max_audio_windows,
        }

    @app.post("/v1/transcribe", response_model=TranscribeResponse)
    async def transcribe(req: TranscribeRequest):
        try:
            return await _run_clip(app.state.engine, req)
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("transcribe failed")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/transcribe_batch", response_model=BatchTranscribeResponse)
    async def transcribe_batch(req: BatchTranscribeRequest):
        """Transcribe several clips. Concurrent -- the engine does the batching."""
        subs = [
            TranscribeRequest(
                audio_base64=item.audio_base64,
                audio_url=item.audio_url,
                context_info=req.context_info,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                repetition_penalty=req.repetition_penalty,
            )
            for item in req.audios
        ]
        try:
            t0 = time.time()
            results = await asyncio.gather(
                *(_run_clip(app.state.engine, s) for s in subs))
            return BatchTranscribeResponse(results=results,
                                           total_time=time.time() - t0)
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("batch transcribe failed")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        try:
            body = await request.json()
            messages = body.get("messages") or []
            req = TranscribeRequest(
                max_tokens=min(_field(body, "max_tokens", 256), 2048),
                temperature=_field(body, "temperature", 0.0),
                top_p=_field(body, "top_p", 1.0),
                repetition_penalty=_field(body, "repetition_penalty", 1.0),
                context_info=_context_from_messages(messages),
            )
        except (AttributeError, TypeError, ValueError) as exc:
            # This dialect reads the raw body, so none of the automatic 422s
            # FastAPI gives the typed endpoints apply here. Without this, a
            # malformed body or an out-of-range temperature returns 500, which
            # reads as a server fault for what is a client mistake.
            raise HTTPException(status_code=400, detail=str(exc))
        # To a thread for the same reason as _run_clip: an audio_url part is
        # fetched over the network, and a stall here stalls every other session.
        audio_bytes = await asyncio.to_thread(_audio_from_messages, messages)
        cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        sample_rate = geometry.sample_rate

        if not body.get("stream"):
            audio = await asyncio.to_thread(load_audio_bytes, audio_bytes,
                                            sample_rate)
            session = _new_session(app.state.engine, req)
            chunks = [t async for t in session.transcribe(audio)]
            segments = chunk_segments(chunks, geometry, len(audio) / sample_rate)
            return {
                "id": cid, "object": "chat.completion",
                "model": config.served_model_name,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant",
                                         "content": json.dumps(
                                             segments, ensure_ascii=False)}}],
            }

        async def events():
            model_name = config.served_model_name
            try:
                audio = await asyncio.to_thread(load_audio_bytes, audio_bytes,
                                                sample_rate)
                session = _new_session(app.state.engine, req)
                yield _sse(cid, model_name, {"role": "assistant", "content": "["})
                chunks, sent = [], 0
                async for chunk_text in session.transcribe(audio):
                    chunks.append(chunk_text)
                    # Every segment but the last: the trailing one is still open
                    # and a later chunk can extend it or move where it ends.
                    for seg in chunk_segments(chunks, geometry)[sent:-1]:
                        yield _sse(cid, model_name,
                                   {"content": _segment_delta(seg, sent)})
                        sent += 1
                # The real duration is known now, so the tail can be clamped.
                for seg in chunk_segments(chunks, geometry,
                                          len(audio) / sample_rate)[sent:]:
                    yield _sse(cid, model_name,
                               {"content": _segment_delta(seg, sent)})
                    sent += 1
                yield _sse(cid, model_name, {"content": "]"})
                yield _sse(cid, model_name, {}, finish="stop")
            except Exception as e:
                logger.exception("chat stream failed")
                # The stream already carries a 200, so an error can only be
                # reported inside it. It lands outside the array, which leaves
                # whatever segments already arrived renderable.
                yield _sse(cid, model_name, {"content": f"\n[error] {e}"},
                           finish="stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    @app.websocket("/v1/stream")
    async def stream(ws: WebSocket):
        """Live session: push PCM, receive one transcript per chunk of audio.

        Protocol -- first message is JSON config (all keys optional), then binary
        frames of little-endian float32 mono PCM at the checkpoint's sample rate.
        A text message ``"end"`` flushes the tail. Each chunk comes back as
        ``{"chunk": idx, "text": ..., "segments": [...]}``, and the final message
        adds ``done``, ``total_chunks`` and ``srt``.

        The buffer keeps the lookahead overlap rather than draining fully: the
        window is longer than the advance.
        """
        await ws.accept()
        try:
            cfg = await ws.receive_json()
            req = TranscribeRequest(**{
                k: v for k, v in cfg.items()
                if k in ("context_info", "max_tokens", "temperature", "top_p",
                         "repetition_penalty")
            })
            session = _new_session(app.state.engine, req)

            window_samples = geometry.window_samples
            chunk_samples = geometry.chunk_samples
            buf = np.zeros(0, dtype=np.float32)
            texts: List[str] = []

            async def emit(window) -> None:
                texts.append(await session.push(window))
                payload: Dict[str, Any] = {
                    "chunk": len(texts) - 1,
                    "text": texts[-1],
                    "segments": chunk_segments(texts, geometry),
                }
                await ws.send_json(payload)

            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    return
                if msg.get("bytes") is not None:
                    # "<f4" not np.float32: the browser sends little-endian and
                    # the native dtype would reinterpret it on a big-endian host.
                    buf = np.concatenate(
                        [buf, np.frombuffer(msg["bytes"], dtype="<f4")])
                    while len(buf) >= window_samples:
                        await emit(buf[:window_samples])
                        buf = buf[chunk_samples:]
                elif msg.get("text") == "end":
                    # Same tail rule as the offline path: pad the remainder out
                    # to a full window instead of encoding a short one, which
                    # would emit one extra VAE frame and no lookahead.
                    for window in split_windows(buf, geometry):
                        await emit(window)
                    segments = chunk_segments(texts, geometry)
                    await ws.send_json({"done": True,
                                        "total_chunks": len(texts),
                                        "text": "".join(texts),
                                        "segments": segments,
                                        "srt": segments_to_srt(segments)})
                    return
        except WebSocketDisconnect:
            return
        except Exception as e:
            logger.exception("stream session failed")
            try:
                await ws.send_json({"error": str(e)})
            except Exception:
                pass

    return app


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VibeVoice streaming ASR server (in-process vLLM engine)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Serve a streaming checkpoint
  python3 -m vllm_plugin.asr_streaming_server --model /path/to/checkpoint

  # Two GPUs, longer sessions
  python3 -m vllm_plugin.asr_streaming_server --model /path/to/checkpoint \\
      --tensor-parallel-size 2 --max-model-len 32768

The chunk and lookahead are read from the checkpoint's preprocessor_config.json,
so a checkpoint always runs at the geometry it was trained on.
        """)
    parser.add_argument("--model", "-m", required=True,
                        help="Path or HF repo id of a streaming checkpoint")
    parser.add_argument("--served-model-name", default="vibevoice",
                        help="Name reported by /v1/models (default: vibevoice)")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Address to bind (default: 0.0.0.0)")
    parser.add_argument("--port", "-p", type=int, default=8000,
                        help="Port to serve on (default: 8000)")
    parser.add_argument("--dtype", default="bfloat16",
                        help="Weight dtype (default: bfloat16)")
    parser.add_argument("--tensor-parallel-size", "--tp", dest="tensor_parallel_size",
                        type=int, default=1,
                        help="Number of GPUs per replica (default: 1)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                        help="Fraction of GPU memory to use (default: 0.85)")
    parser.add_argument("--max-model-len", type=int, default=16384,
                        help="Context length per session (default: 16384)")
    parser.add_argument("--max-audio-windows", type=int, default=512,
                        help="Audio windows per session (default: 512)")
    parser.add_argument("--mm-processor-cache-gb", type=float, default=16.0,
                        help="Multimodal cache, must hold every in-flight "
                             "session's windows (default: 16)")
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Skip CUDA graph capture (default: False)")
    parser.add_argument("--no-enable-prefix-caching", dest="enable_prefix_caching",
                        action="store_false",
                        help="Disable prefix caching; diagnostic only, streaming "
                             "goes quadratic without it (default: enabled)")
    parser.add_argument("--resend-all-audio", action="store_true",
                        help="Send every window as data each turn instead of "
                             "uuid placeholders; diagnostic only (default: False)")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = ServerConfig(
        model=args.model,
        served_model_name=args.served_model_name,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        max_audio_windows=args.max_audio_windows,
        mm_processor_cache_gb=args.mm_processor_cache_gb,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=args.enable_prefix_caching,
        resend_all_audio=args.resend_all_audio,
    )
    uvicorn.run(create_app(config), host=args.host, port=args.port,
                log_level="info", workers=1)


if __name__ == "__main__":
    main()
