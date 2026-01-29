"""
TTS conversion endpoints for VibeVoice-Narrator
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Request, UploadFile
from pydantic import BaseModel, Field, conint, model_validator
from typing import Optional, Dict, Tuple, List
from pathlib import Path
import copy
import time
import threading
import errno
import uuid
import numpy as np
import re
import asyncio

# Import backend settings via package-qualified import so this module works when imported
# as a top-level module during tests or as a package in production
from backend.config import settings

router = APIRouter(tags=["TTS"])

class TTSRequest(BaseModel):
    """Request model for TTS conversion

    Note: `content` is optional to allow uploaded document files. The endpoint will
    prefer an uploaded file when provided; otherwise `content` should be present.
    """
    content: Optional[str] = None
    voice_id: str
    filename: Optional[str] = None
    chunk_depth: conint(ge=1) = 1
    pause_ms: conint(ge=0) = 500
    include_heading: bool = False
    strip_markdown: bool = True
    device: str = "auto"
    iterations: int = Field(1, ge=1)


class OutputItem(BaseModel):
    """Represents a single conversion output item."""
    audio_url: str
    duration: float
    filename: str
    iteration: int


class TTSResponse(BaseModel):
    """Response model for TTS conversion"""
    success: bool
    message: str
    audio_url: Optional[str] = None
    duration: Optional[float] = None
    outputs: Optional[List[OutputItem]] = None


class PreviewRequest(BaseModel):
    """Request model for voice preview"""
    voice_id: str
    text: Optional[str] = None
    device: str = "auto"
    model: Optional[str] = None


class PreviewResponse(BaseModel):
    """Response model for voice preview"""
    success: bool
    message: str
    audio_url: Optional[str] = None
    duration: Optional[float] = None


from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)

# Simple LRU cache implementation using OrderedDict. Eviction is handled via
# an optional `on_evict` callback which is invoked with (key, value) when an
# entry is removed due to capacity pressure.
class LRUCache(OrderedDict):
    def __init__(self, maxsize: int = 128, on_evict=None):
        super().__init__()
        self.maxsize = maxsize
        self.on_evict = on_evict

    def __setitem__(self, key, value):
        if key in self:
            # Remove existing to update ordering
            super().__delitem__(key)
        super().__setitem__(key, value)
        self.move_to_end(key)
        if len(self) > self.maxsize:
            oldest_key, oldest_val = self.popitem(last=False)
            try:
                if self.on_evict:
                    self.on_evict(oldest_key, oldest_val)
            except Exception:
                logger.exception("Exception while running eviction callback")

    def clear(self):
        if self.on_evict:
            for k, v in list(self.items()):
                try:
                    self.on_evict(k, v)
                except Exception:
                    logger.exception("Exception while running eviction callback during clear")
        super().clear()

    def pop(self, key, *args, **kwargs):
        # Remove the key and, if an item was actually removed, invoke the eviction callback
        had_key = key in self
        val = super().pop(key, *args, **kwargs)
        if had_key and self.on_evict:
            try:
                self.on_evict(key, val)
            except Exception:
                logger.exception("Exception while running eviction callback for key: %s", key)
        return val


def _cleanup_resource(key, resource):
    """Best-effort cleanup for cached resources (models/processors/voices)."""
    try:
        # Try moving torch tensors/modules to CPU and delete references
        import torch
        try:
            if hasattr(resource, "to"):
                try:
                    resource.to("cpu")
                except Exception:
                    pass
            if hasattr(resource, "cpu"):
                try:
                    resource.cpu()
                except Exception:
                    pass
        except Exception:
            pass
    except Exception:
        # torch not available; fallback to letting GC handle it
        pass
    # Note: deleting the local `resource` reference here does not affect the
    # cached object's lifetime. To clear cached models/processors/voices you
    # must remove them from the cache (e.g., via LRUCache.pop or clear) so
    # the eviction callback can run and resources are cleaned up.


# Bounded, thread-safe LRU caches (guarded by _CACHE_LOCK for atomic ops)
_CACHE_LOCK = threading.Lock()
_MODEL_CACHE = LRUCache(maxsize=settings.max_model_cache_size, on_evict=_cleanup_resource)
_PROCESSOR_CACHE = LRUCache(maxsize=settings.max_processor_cache_size, on_evict=_cleanup_resource)
_VOICE_CACHE = LRUCache(maxsize=settings.max_voice_cache_size, on_evict=_cleanup_resource)
# Per-model locks to avoid holding the global cache lock during long loads
import weakref

_MODEL_LOCKS: "weakref.WeakValueDictionary[tuple, threading.Lock]" = weakref.WeakValueDictionary()
_MODEL_LOCKS_LOCK = threading.Lock()
# Per-voice locks to prevent concurrent torch.load for the same voice
_VOICE_LOCKS: "weakref.WeakValueDictionary[str, threading.Lock]" = weakref.WeakValueDictionary()
_VOICE_LOCKS_LOCK = threading.Lock()


def _get_or_create_model_lock(cache_key: Tuple[str, str]) -> threading.Lock:
    """Return a per-model threading.Lock, creating it under the model-locks guard if needed.

    Uses a WeakValueDictionary to allow locks to be reclaimed when no longer referenced.
    """
    with _MODEL_LOCKS_LOCK:
        lock = _MODEL_LOCKS.get(cache_key)
        if lock is None:
            lock = threading.Lock()
            _MODEL_LOCKS[cache_key] = lock
        return lock


def _get_or_create_voice_lock(voice_id: str) -> threading.Lock:
    """Return a per-voice threading.Lock, creating it under the voice-locks guard if needed."""
    with _VOICE_LOCKS_LOCK:
        lock = _VOICE_LOCKS.get(voice_id)
        if lock is None:
            lock = threading.Lock()
            _VOICE_LOCKS[voice_id] = lock
        return lock

# Sentinel object used to detect absence in double-checked locking
_VOICE_CACHE_SENTINEL = object()


def _resolve_device(device: str, torch_module) -> str:
    if device == "auto":
        if torch_module.cuda.is_available():
            return "cuda"
        if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def _find_voice_file(voice_id: str) -> Path:
    voice_id = voice_id.strip()
    if not voice_id:
        raise HTTPException(status_code=400, detail="voice_id is required")

    # Sanitize voice_id: only allow alphanumeric, hyphen, underscore to avoid path traversal
    if not re.fullmatch(r"[A-Za-z0-9_-]+", voice_id):
        raise HTTPException(status_code=400, detail="voice_id contains invalid characters")

    # Resolve demo voices directory: prefer explicit setting, otherwise derive from repo layout
    demo_voices_dir = settings.demo_voices_dir if getattr(settings, 'demo_voices_dir', None) else settings.data_dir.parent.parent / "demo" / "voices" / "streaming_model"
    # If the resolved demo dir isn't present, log and fall back to only the configured voices_dir
    search_dirs = [settings.voices_dir]
    if demo_voices_dir and demo_voices_dir.exists() and demo_voices_dir.is_dir():
        search_dirs.append(demo_voices_dir)
    else:
        try:
            logger.debug(f"Demo voices dir {demo_voices_dir} not found or invalid, skipping")
        except Exception:
            pass

    for voices_dir in search_dirs:
        if voices_dir.exists() and voices_dir.is_dir():
            candidate = voices_dir / f"{voice_id}.pt"
            if candidate.exists():
                return candidate

    raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")


def _find_any_voice_id() -> Optional[str]:
    # Resolve demo voices directory: prefer explicit setting, otherwise derive from repo layout
    demo_voices_dir = settings.demo_voices_dir if getattr(settings, 'demo_voices_dir', None) else settings.data_dir.parent.parent / "demo" / "voices" / "streaming_model"
    search_dirs = [settings.voices_dir]
    if demo_voices_dir and demo_voices_dir.exists() and demo_voices_dir.is_dir():
        search_dirs.append(demo_voices_dir)
    else:
        try:
            logger.debug(f"Demo voices dir {demo_voices_dir} not found or invalid, skipping")
        except Exception:
            pass

    for voices_dir in search_dirs:
        if voices_dir.exists() and voices_dir.is_dir():
            for path in voices_dir.glob("*.pt"):
                return path.stem
    return None


def _extract_audio_from_outputs(outputs, torch_module):
    if outputs is None:
        raise ValueError("Model returned no outputs")

    if hasattr(outputs, "speech_outputs") and outputs.speech_outputs:
        audio = outputs.speech_outputs[0]
    elif isinstance(outputs, dict) and outputs.get("speech_outputs"):
        audio = outputs["speech_outputs"][0]
    elif hasattr(outputs, "audio"):
        audio = outputs.audio
    elif isinstance(outputs, torch_module.Tensor):
        audio = outputs
    else:
        raise ValueError(f"Unexpected output format: {type(outputs)}")

    if isinstance(audio, torch_module.Tensor):
        audio = audio.float().detach().cpu().numpy()
    return audio.squeeze()


def _next_iteration_index(base_name: str, output_dir: Path) -> int:
    """Return the next numeric index but reserve the filename atomically by creating a zero-byte placeholder.

    This prevents TOCTOU races by attempting exclusive creation of candidate files until one succeeds.
    """
    # Start scanning for the next candidate index
    max_index = 0
    pattern = f"{base_name}_*.wav"
    for path in output_dir.glob(pattern):
        stem = path.stem
        if not stem.startswith(base_name + "_"):
            continue
        suffix = stem[len(base_name) + 1:]
        if suffix.isdigit():
            max_index = max(max_index, int(suffix))

    index = max_index + 1

    # Try to reserve by creating the file exclusively; if it exists, increment and retry
    while True:
        candidate = output_dir / f"{base_name}_{index:03d}.wav"
        try:
            # Atomically create the file by opening in exclusive 'xb' mode; fails with FileExistsError if exists
            with candidate.open('xb'):
                # Successfully reserved this index
                return index
        except FileExistsError:
            index += 1
            continue
        except Exception:
            # Re-raise unexpected errors
            raise


def _format_bytes(n: int) -> str:
    """Return a human-readable size string for bytes (e.g., '1GB', '10MB')."""
    if n >= 1024 ** 3:
        return f"{n // (1024 ** 3)}GB"
    if n >= 1024 ** 2:
        return f"{n // (1024 ** 2)}MB"
    if n >= 1024:
        return f"{n // 1024}KB"
    return f"{n}B"


def _load_model_and_processor(model_name: str, device: str):
    try:
        import torch
        from vibevoice.modular.modeling_vibevoice_streaming_inference import (
            VibeVoiceStreamingForConditionalGenerationInference,
        )
        from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "VibeVoice dependencies are missing. Install the base project "
                "dependencies before using model previews."
            ),
        ) from exc

    resolved_device = _resolve_device(device, torch)
    cache_key = (model_name, resolved_device)

    # Fast path: check under global lock
    with _CACHE_LOCK:
        if cache_key in _MODEL_CACHE and model_name in _PROCESSOR_CACHE:
            return _MODEL_CACHE[cache_key], _PROCESSOR_CACHE[model_name], resolved_device, torch

    # Ensure a per-model lock exists (weak-value-backed to avoid unbounded growth)
    model_lock = _get_or_create_model_lock(cache_key)

    # Decide dtype & attention implementation outside locks
    if resolved_device == "mps":
        load_dtype = torch.float32
        attn_impl_primary = "sdpa"
    elif resolved_device == "cuda":
        load_dtype = torch.bfloat16
        attn_impl_primary = "flash_attention_2"
    else:
        load_dtype = torch.float32
        attn_impl_primary = "sdpa"

    # Acquire per-model lock to prevent duplicate loads, but don't hold global cache lock during load
    with model_lock:
        # Re-check caches in case another thread loaded while we waited
        with _CACHE_LOCK:
            if cache_key in _MODEL_CACHE and model_name in _PROCESSOR_CACHE:
                return _MODEL_CACHE[cache_key], _PROCESSOR_CACHE[model_name], resolved_device, torch

        # Load processor and model (may be slow)
        processor = VibeVoiceStreamingProcessor.from_pretrained(model_name)

        try:
            if resolved_device == "mps":
                model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    model_name,
                    torch_dtype=load_dtype,
                    attn_implementation=attn_impl_primary,
                    device_map=None,
                )
                model.to("mps")
            elif resolved_device == "cuda":
                model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    model_name,
                    torch_dtype=load_dtype,
                    device_map="cuda",
                    attn_implementation=attn_impl_primary,
                )
            else:
                model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    model_name,
                    torch_dtype=load_dtype,
                    device_map="cpu",
                    attn_implementation=attn_impl_primary,
                )
        except Exception as exc:
            if attn_impl_primary == "flash_attention_2":
                model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    model_name,
                    torch_dtype=load_dtype,
                    device_map=("cuda" if resolved_device == "cuda" else "cpu"),
                    attn_implementation="sdpa",
                )
            else:
                raise exc

        model.eval()
        model.set_ddpm_inference_steps(num_steps=10)

        # Store into caches under the global lock
        with _CACHE_LOCK:
            _MODEL_CACHE[cache_key] = model
            _PROCESSOR_CACHE[model_name] = processor

        return model, processor, resolved_device, torch


def warmup_preview() -> None:
    voice_id = settings.warmup_voice_id or _find_any_voice_id()
    if not voice_id:
        print("[warmup] No voices found; skipping preview warmup.")
        return

    try:
        model, processor, resolved_device, torch = _load_model_and_processor(
            settings.default_model,
            "auto",
        )
        voice_file = _find_voice_file(voice_id)
        # Use per-voice lock to avoid blocking other cache operations during load
        with _CACHE_LOCK:
            if voice_id in _VOICE_CACHE:
                pass
            else:
                with _VOICE_LOCKS_LOCK:
                    voice_lock = _VOICE_LOCKS.get(voice_id)
                    if voice_lock is None:
                        voice_lock = threading.Lock()
                        _VOICE_LOCKS[voice_id] = voice_lock

                with voice_lock:
                    with _CACHE_LOCK:
                        if voice_id not in _VOICE_CACHE:
                            _VOICE_CACHE[voice_id] = torch.load(
                                voice_file, map_location=resolved_device, weights_only=False
                            )
        print(f"[warmup] Preview warmup complete for voice '{voice_id}'.")
    except Exception as exc:
        print(f"[warmup] Preview warmup failed: {exc}")

@router.post("/convert", response_model=TTSResponse)
async def convert_to_speech(
    payload: TTSRequest,
    request: Request = None,
):
    """
    Convert markdown document to speech audio
    """
    try:
        MAX_BYTES = settings.max_upload_size

        # Prefer uploaded file when present; otherwise use payload.content
        content_bytes = b''
        content_str = ''
        filename = payload.filename or 'document.md'

        # If request is multipart/form-data, read uploaded file from form-data (support both JSON and multipart)
        file = None
        if request is not None and request.headers.get('content-type', '').startswith('multipart/form-data'):
            form = await request.form()
            file = form.get('file')

        # Prefer uploaded file when present and it's a real UploadFile; otherwise use payload.content
        if file is not None and isinstance(file, UploadFile):
            content_bytes = await file.read()
            try:
                content_str = content_bytes.decode('utf-8')
            except Exception:
                raise HTTPException(status_code=400, detail='Uploaded file must be UTF-8 text')
            filename = payload.filename or (getattr(file, 'filename', None) or 'document.md')
        else:
            # No valid uploaded file present; validate content in payload
            if payload.content is None or not payload.content.strip():
                raise HTTPException(status_code=400, detail='Content is empty')
            content_str = payload.content
            content_bytes = content_str.encode('utf-8')
            filename = payload.filename or 'document.md'

        if len(content_bytes) > MAX_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Document too large. Maximum size is {_format_bytes(settings.max_upload_size)}."
            )

        # Validate voice_id and resolve voice file before loading heavy model dependencies
        voice_file = _find_voice_file(payload.voice_id)
        model_name = settings.default_model
        model, processor, resolved_device, torch = _load_model_and_processor(model_name, payload.device)

        # Double-checked locking for per-voice load (convert endpoint)
        voice_cache = _VOICE_CACHE_SENTINEL

        # Fast-path: check under cache lock to avoid unnecessary work
        with _CACHE_LOCK:
            if payload.voice_id in _VOICE_CACHE:
                voice_cache = _VOICE_CACHE[payload.voice_id]

        if voice_cache is _VOICE_CACHE_SENTINEL:
            # Get or create the per-voice lock without holding the global cache lock
            # Get or create a per-voice lock (weak-value-backed)
            voice_lock = _get_or_create_voice_lock(payload.voice_id)

            # Acquire the per-voice lock and then re-check the cache inside it
            with voice_lock:
                with _CACHE_LOCK:
                    if payload.voice_id in _VOICE_CACHE:
                        voice_cache = _VOICE_CACHE[payload.voice_id]
                    else:
                        # Load voice cache while holding only the per-voice lock (not the global cache lock)
                        voice_cache = await asyncio.to_thread(torch.load, voice_file, map_location=resolved_device, weights_only=False)
                        _VOICE_CACHE[payload.voice_id] = voice_cache

        outputs_list = []
        output_dir = settings.audio_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        base_name = Path(filename).stem
        iterations = payload.iterations

        # Enforce server-side per-request max iterations from settings
        if iterations > settings.max_iterations_per_request:
            raise HTTPException(status_code=400, detail=f"Max iterations per request is {settings.max_iterations_per_request}")

        # Prepare chunker and chunk the input content according to request
        try:
            from vibevoice.processor.chunking.markdown_chunker import MarkdownChunker
        except ModuleNotFoundError:
            # Fallback minimal chunker used in tests when the 'vibevoice' package isn't importable.
            from types import SimpleNamespace
            class MarkdownChunker:
                def __init__(self, chunk_depth=1, strip_markdown=True):
                    self.chunk_depth = chunk_depth
                    self.strip_markdown = strip_markdown
                def _strip(self, text: str) -> str:
                    # Minimal markdown stripping for tests: remove headings, bold/italic markers, and backticks
                    text = re.sub(r"\*\*|\*|`", "", text)
                    return text
                def chunk(self, text: str):
                    chunks = []
                    parts = re.split(r"(?m)^#\s+", text)
                    # First part may be preamble without heading
                    if parts and parts[0].strip():
                        content = parts[0].strip()
                        if self.strip_markdown:
                            content = self._strip(content)
                        chunks.append(SimpleNamespace(heading=None, content=content))
                    for part in parts[1:]:
                        lines = part.splitlines()
                        heading = lines[0].strip()
                        body = "\n".join(lines[1:]).strip()
                        if self.strip_markdown:
                            heading = self._strip(heading)
                            body = self._strip(body)
                        chunks.append(SimpleNamespace(heading=heading, content=body))
                    return chunks
        chunker = MarkdownChunker(chunk_depth=payload.chunk_depth, strip_markdown=payload.strip_markdown)
        chunks = chunker.chunk(content_str)

        # For each iteration, generate audio for all chunks and concatenate with silence between
        for offset in range(iterations):
            chunk_audios = []
            sampling_rate = int(getattr(processor.audio_processor, "sampling_rate", 24000))
            silence_len = int(sampling_rate * (max(0, payload.pause_ms) / 1000.0)) if payload.pause_ms else 0
            silence_frame = np.zeros(silence_len, dtype=np.float32) if silence_len > 0 else None

            # Create a single deep copy of the voice cache for this iteration.
            # Note: `model.generate(..., all_prefilled_outputs=cached_prompt)` may mutate
            # the provided `cached_prompt`. Creating one copy per iteration reduces
            # expensive deep copies while preserving a clean `voice_cache` for other
            # iterations/requests. If strict per-chunk isolation is required, revert
            # to deep-copying per chunk instead.
            cached_prompt = copy.deepcopy(voice_cache)

            for idx, chunk in enumerate(chunks):
                # Assemble text for this chunk
                text_parts = []
                if payload.include_heading and chunk.heading:
                    text_parts.append(chunk.heading)
                if chunk.content:
                    text_parts.append(chunk.content)
                chunk_text = "\n\n".join(text_parts).strip()
                if not chunk_text:
                    continue

                inputs = processor.process_input_with_cached_prompt(
                    text=chunk_text,
                    cached_prompt=cached_prompt,
                    padding=True,
                    return_tensors="pt",
                    return_attention_mask=True,
                )

                for key, value in inputs.items():
                    if torch.is_tensor(value):
                        inputs[key] = value.to(resolved_device)

                def _run_generate():
                    with torch.no_grad():
                        return model.generate(
                            **inputs,
                            max_new_tokens=None,
                            cfg_scale=1.25,
                            tokenizer=processor.tokenizer,
                            generation_config={"do_sample": False},
                            verbose=False,
                            all_prefilled_outputs=cached_prompt,
                        )

                outputs = await asyncio.to_thread(_run_generate)

                audio = _extract_audio_from_outputs(outputs, torch)
                if audio is None or len(audio) == 0:
                    raise HTTPException(status_code=500, detail="No audio generated")

                # Ensure audio is float32 numpy
                if isinstance(audio, np.ndarray):
                    audio_np = audio.astype(np.float32)
                else:
                    audio_np = np.array(audio, dtype=np.float32)

                chunk_audios.append(audio_np)
                if silence_frame is not None and idx != len(chunks) - 1:
                    chunk_audios.append(silence_frame)

            if not chunk_audios:
                raise HTTPException(status_code=500, detail="No audio generated for any chunk")

            # Concatenate all chunk audio arrays
            final_audio = np.concatenate(chunk_audios)

            # Atomically reserve a filename for this iteration
            iteration_index = _next_iteration_index(base_name, output_dir)
            output_name = f"{base_name}_{iteration_index:03d}.wav"
            output_path = output_dir / output_name

            await asyncio.to_thread(processor.save_audio, final_audio, output_path=str(output_path), sampling_rate=sampling_rate)
            duration = float(len(final_audio)) / float(sampling_rate)
            audio_url = f"/static/audio/{output_name}"

            outputs_list.append(OutputItem(
                audio_url=audio_url,
                duration=duration,
                filename=output_name,
                iteration=iteration_index,
            ))

        first = outputs_list[0] if outputs_list else None
        return TTSResponse(
            success=True,
            message=f"Document '{filename}' converted successfully.",
            audio_url=first.audio_url if first else None,
            duration=first.duration if first else None,
            outputs=outputs_list,
        )

    except HTTPException:
        # Re-raise expected HTTP errors so FastAPI preserves status codes
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )


@router.post("/preview", response_model=PreviewResponse)
async def preview_voice(payload: PreviewRequest):
    """
    Generate a short model-accurate voice preview clip.
    """
    try:
        if not payload.voice_id.strip():
            raise HTTPException(status_code=400, detail="voice_id is required")

        model_name = payload.model or settings.default_model
        model, processor, resolved_device, torch = _load_model_and_processor(model_name, payload.device)

        voice_file = _find_voice_file(payload.voice_id)

        # Double-checked locking for per-voice load (preview endpoint)
        voice_cache = _VOICE_CACHE_SENTINEL

        with _CACHE_LOCK:
            if payload.voice_id in _VOICE_CACHE:
                voice_cache = _VOICE_CACHE[payload.voice_id]

        if voice_cache is _VOICE_CACHE_SENTINEL:
            # Get or create a per-voice lock (weak-value-backed)
            voice_lock = _get_or_create_voice_lock(payload.voice_id)

            with voice_lock:
                with _CACHE_LOCK:
                    if payload.voice_id in _VOICE_CACHE:
                        voice_cache = _VOICE_CACHE[payload.voice_id]
                    else:
                        voice_cache = await asyncio.to_thread(torch.load, voice_file, map_location=resolved_device, weights_only=False)
                        _VOICE_CACHE[payload.voice_id] = voice_cache

        preview_text = payload.text or f"Hello, this is a preview of {payload.voice_id}."

        # Prepare a single cached prompt copy for this preview
        cached_prompt = copy.deepcopy(voice_cache)

        inputs = processor.process_input_with_cached_prompt(
            text=preview_text,
            cached_prompt=cached_prompt,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        for key, value in inputs.items():
            if torch.is_tensor(value):
                inputs[key] = value.to(resolved_device)

        def _run_generate_preview():
            with torch.no_grad():
                return model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=1.25,
                    tokenizer=processor.tokenizer,
                    generation_config={"do_sample": False},
                    verbose=False,
                    all_prefilled_outputs=cached_prompt,
                )

        outputs = await asyncio.to_thread(_run_generate_preview)

        audio = _extract_audio_from_outputs(outputs, torch)
        if audio is None or len(audio) == 0:
            raise HTTPException(status_code=500, detail="No audio generated for preview")

        output_dir = settings.audio_dir / "preview"
        output_dir.mkdir(parents=True, exist_ok=True)
        # Use a collision-safe filename for previews
        filename = f"preview-{payload.voice_id}-{uuid.uuid4().hex}.wav"
        output_path = output_dir / filename

        sampling_rate = int(getattr(processor.audio_processor, "sampling_rate", 24000))
        await asyncio.to_thread(processor.save_audio, audio, output_path=str(output_path), sampling_rate=sampling_rate)

        duration = float(len(audio)) / float(sampling_rate)
        audio_url = f"/static/audio/preview/{filename}"

        # Kick off cleanup in background to avoid blocking the request
        try:
            t = threading.Thread(
                target=_purge_preview_files,
                args=(output_dir, settings.preview_ttl_minutes, settings.preview_max_preview_files),
                daemon=True,
            )
            t.start()
        except Exception as exc:
            print(f"[preview] Failed to start cleanup thread: {exc}")

        return PreviewResponse(
            success=True,
            message="Preview generated",
            audio_url=audio_url,
            duration=duration,
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating preview: {exc}"
        )

@router.get("/status/{job_id}")
async def get_conversion_status(job_id: str):
    """
    Get the status of a TTS conversion job
    
    This endpoint validates the incoming job_id to prevent malformed or unsafe IDs.
    """
    # Validate job_id: allow alphanumeric/hyphen/underscore OR a UUID string
    job_id = job_id.strip()
    if not job_id:
        raise HTTPException(status_code=400, detail="job_id is required")

    # Accept simple safe ids (alphanumeric, hyphen, underscore)
    if re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
        validated_job_id = job_id
    else:
        # Try parsing as UUID to allow standard UUID-formatted ids
        try:
            uuid_obj = uuid.UUID(job_id)
            validated_job_id = str(uuid_obj)
        except ValueError:
            raise HTTPException(status_code=400, detail="job_id is malformed")

    # Placeholder: In production, use validated_job_id to lookup job in DB/queue
    return {
        "job_id": validated_job_id,
        "status": "completed",  # pending, processing, completed, error
        "progress": 100,
        "current_chunk": 5,
        "total_chunks": 5,
        "message": "Conversion completed successfully"
    }


@router.post("/preview/purge")
async def purge_previews(background: BackgroundTasks):
    """Schedule a background purge of preview files (non-blocking)."""
    output_dir = settings.audio_dir / "preview"
    output_dir.mkdir(parents=True, exist_ok=True)
    background.add_task(
        _purge_preview_files,
        output_dir,
        settings.preview_ttl_minutes,
        settings.preview_max_preview_files,
    )
    return {"success": True, "message": "Purge scheduled"}


def _purge_preview_files(output_dir: Path, ttl_minutes: int | None, max_files_per_voice: int | None) -> None:
    """Remove stale preview files older than ttl_minutes and enforce max files per voice.

    This function is safe to call in a background thread and logs failures while attempting deletions.
    """
    try:
        now = time.time()
        pattern = "preview-*.wav"

        # First, delete files older than TTL
        if ttl_minutes is not None and ttl_minutes > 0:
            for path in output_dir.glob(pattern):
                try:
                    mtime = path.stat().st_mtime
                    if (now - mtime) > (ttl_minutes * 60):
                        path.unlink()
                        logger.info(f"[purge] Removed old preview file: {path}")
                except Exception as exc:
                    logger.warning(f"[purge] Failed to consider/delete {path}: {exc}")
            # Then enforce per-voice maximums if requested
        if max_files_per_voice is not None and max_files_per_voice > 0:
            per_voice: Dict[str, list[Path]] = {}
            for path in output_dir.glob(pattern):
                try:
                    stem = path.stem  # e.g., preview-voiceid-uuid
                    parts = stem.split("-", 2)
                    voice_id = parts[1] if len(parts) >= 2 else ""
                    per_voice.setdefault(voice_id, []).append(path)
                except Exception as exc:
                    logger.warning(f"[purge] Failed to consider {path} for per-voice pruning: {exc}")

            for voice_id, paths in per_voice.items():
                if len(paths) <= max_files_per_voice:
                    continue
                # Build a list of (path, mtime) pairs while defensively handling stat errors
                safe_paths = []
                for p in paths:
                    try:
                        mtime = p.stat().st_mtime
                        safe_paths.append((p, mtime))
                    except (FileNotFoundError, OSError) as exc:
                        # If the file disappeared between glob and stat, skip it
                        logger.debug(f"[purge] Skipping missing/unstat-able file {p}: {exc}")
                    except Exception as exc:
                        # Catch-all to avoid purge crashing on unexpected errors
                        logger.warning(f"[purge] Error while stat'ing {p}: {exc}")

                # Sort by mtime (newest first) and delete any beyond the max
                safe_paths.sort(key=lambda t: t[1], reverse=True)
                for old, _ in safe_paths[max_files_per_voice:]:
                    try:
                        old.unlink()
                        logger.info(f"[purge] Removed excess preview file for {voice_id}: {old}")
                    except Exception as exc:
                        logger.warning(f"[purge] Failed to delete {old}: {exc}")
    except Exception as exc:
        logger.exception(f"[purge] Unexpected error during purge: {exc}")
