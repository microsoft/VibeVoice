"""
Speech-to-Speech Pipeline with WebSocket Streaming
===================================================

Unified pipeline integrating:
- Silero VAD (MIT)
- faster-whisper ASR (MIT)
- Qwen2.5-1.5B LLM (Apache 2.0)
- VibeVoice TTS (MIT)

Target: <800ms end-to-end latency for real-time conversation.
"""

import logging
import asyncio
import time
import json
import os
import sys
from typing import Optional, Dict, Any, AsyncGenerator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import threading
from queue import Queue, Empty

import numpy as np
import torch

# Add parent directory to path for VibeVoice imports
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketState
import uvicorn

logger = logging.getLogger(__name__)


class PipelineState(Enum):
    """Pipeline processing states"""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING_VAD = "processing_vad"
    TRANSCRIBING = "transcribing"
    GENERATING = "generating"
    SYNTHESIZING = "synthesizing"
    SPEAKING = "speaking"


@dataclass
class PipelineConfig:
    """Configuration for S2S Pipeline"""
    # Component settings
    asr_model: str = "small.en"  # faster-whisper built-in model, fast and accurate
    llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    tts_model: str = "microsoft/VibeVoice-Realtime-0.5B"
    
    # Device settings
    device: str = "cuda"
    asr_compute_type: str = "int8"
    llm_dtype: str = "bfloat16"
    
    # Audio settings
    input_sample_rate: int = 16000
    output_sample_rate: int = 24000
    
    # VAD settings
    vad_threshold: float = 0.5
    min_speech_ms: int = 250
    min_silence_ms: int = 300
    
    # LLM settings
    max_llm_tokens: int = 64
    llm_temperature: float = 0.7
    
    # TTS settings
    tts_voice: str = "en-Carter_man"
    tts_cfg_scale: float = 1.5
    tts_inference_steps: int = 5
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8005


@dataclass
class PipelineMetrics:
    """Metrics for pipeline performance"""
    vad_latency_ms: float = 0.0
    asr_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    tts_first_chunk_ms: float = 0.0
    total_latency_ms: float = 0.0
    tokens_generated: int = 0
    audio_duration_ms: float = 0.0


class S2SPipeline:
    """
    Speech-to-Speech Pipeline with streaming support.
    
    Processes audio through:
    1. VAD - Voice Activity Detection
    2. ASR - Speech Recognition
    3. LLM - Response Generation
    4. TTS - Speech Synthesis
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.state = PipelineState.IDLE
        self._initialized = False
        
        # Components (lazy loaded)
        self._vad = None
        self._asr = None
        self._llm = None
        self._tts_service = None
        
        # Metrics
        self.last_metrics = PipelineMetrics()
        self.total_requests = 0
        
        logger.info("S2S Pipeline created")
    
    async def initialize(self) -> None:
        """Initialize all pipeline components"""
        if self._initialized:
            return
        
        logger.info("Initializing S2S Pipeline components...")
        start_time = time.time()
        
        # Initialize VAD
        logger.info("Loading VAD...")
        from .vad_module import StreamingVAD, VADConfig
        vad_config = VADConfig(
            threshold=self.config.vad_threshold,
            min_speech_duration_ms=self.config.min_speech_ms,
            min_silence_duration_ms=self.config.min_silence_ms
        )
        self._vad = StreamingVAD(vad_config)
        self._vad.initialize()
        
        # Initialize ASR
        logger.info("Loading ASR...")
        from .asr_module import StreamingASR, ASRConfig
        asr_config = ASRConfig(
            model_size=self.config.asr_model,
            device=self.config.device,
            compute_type=self.config.asr_compute_type
        )
        self._asr = StreamingASR(asr_config)
        self._asr.initialize()
        
        # Initialize LLM
        logger.info("Loading LLM...")
        from .llm_module import StreamingLLM, LLMConfig
        llm_config = LLMConfig(
            model_name=self.config.llm_model,
            device=self.config.device,
            dtype=self.config.llm_dtype,
            max_new_tokens=self.config.max_llm_tokens,
            temperature=self.config.llm_temperature
        )
        self._llm = StreamingLLM(llm_config)
        self._llm.initialize()
        
        # Initialize TTS (connect to VibeVoice server)
        logger.info("Initializing TTS connection...")
        self._tts_service = TTSClient(
            model_path=self.config.tts_model,
            device=self.config.device,
            voice=self.config.tts_voice,
            cfg_scale=self.config.tts_cfg_scale,
            inference_steps=self.config.tts_inference_steps
        )
        await self._tts_service.initialize()
        
        init_time = (time.time() - start_time) * 1000
        self._initialized = True
        logger.info(f"S2S Pipeline initialized in {init_time:.0f}ms")
    
    async def process_audio(
        self,
        audio_bytes: bytes,
        sample_rate: int = 16000
    ) -> AsyncGenerator[bytes, None]:
        """
        Process audio input and stream audio response.
        
        Args:
            audio_bytes: Raw PCM16 audio bytes from client
            sample_rate: Audio sample rate
            
        Yields:
            PCM16 audio chunks for playback
        """
        if not self._initialized:
            await self.initialize()
        
        pipeline_start = time.time()
        self.state = PipelineState.PROCESSING_VAD
        
        # Step 1: VAD - Extract speech segments
        vad_start = time.time()
        speech_segments = self._vad.process_audio(audio_bytes, sample_rate)
        vad_time = (time.time() - vad_start) * 1000
        
        if not speech_segments:
            self.state = PipelineState.IDLE
            return
        
        # Process each speech segment
        for speech_audio in speech_segments:
            self.state = PipelineState.TRANSCRIBING
            
            # Step 2: ASR - Transcribe speech
            asr_start = time.time()
            transcription = self._asr.transcribe_segment(speech_audio, sample_rate)
            asr_time = (time.time() - asr_start) * 1000
            
            if not transcription.text.strip():
                continue
            
            logger.info(f"User: {transcription.text}")
            self.state = PipelineState.GENERATING
            
            # Step 3: LLM - Generate response
            llm_start = time.time()
            llm_response = self._llm.respond(transcription.text)
            llm_time = (time.time() - llm_start) * 1000
            
            logger.info(f"Assistant: {llm_response.text}")
            self.state = PipelineState.SYNTHESIZING
            
            # Step 4: TTS - Synthesize speech
            tts_start = time.time()
            first_chunk = True
            
            async for audio_chunk in self._tts_service.synthesize_streaming(llm_response.text):
                if first_chunk:
                    tts_first_chunk = (time.time() - tts_start) * 1000
                    total_time = (time.time() - pipeline_start) * 1000
                    
                    # Update metrics
                    self.last_metrics = PipelineMetrics(
                        vad_latency_ms=vad_time,
                        asr_latency_ms=asr_time,
                        llm_latency_ms=llm_time,
                        tts_first_chunk_ms=tts_first_chunk,
                        total_latency_ms=total_time,
                        tokens_generated=llm_response.tokens_generated
                    )
                    
                    logger.info(f"E2E Latency: {total_time:.0f}ms "
                               f"(VAD:{vad_time:.0f} ASR:{asr_time:.0f} "
                               f"LLM:{llm_time:.0f} TTS:{tts_first_chunk:.0f})")
                    
                    first_chunk = False
                
                self.state = PipelineState.SPEAKING
                yield audio_chunk
        
        self.total_requests += 1
        self.state = PipelineState.IDLE
    
    async def process_speech_segment(
        self,
        speech_audio: np.ndarray,
        sample_rate: int = 16000
    ) -> AsyncGenerator[bytes, None]:
        """
        Process a complete speech segment (from VAD).
        
        Args:
            speech_audio: Speech audio numpy array
            sample_rate: Audio sample rate
            
        Yields:
            PCM16 audio chunks
        """
        if not self._initialized:
            await self.initialize()
        
        pipeline_start = time.time()
        
        # ASR
        self.state = PipelineState.TRANSCRIBING
        asr_start = time.time()
        transcription = self._asr.transcribe_segment(speech_audio, sample_rate)
        asr_time = (time.time() - asr_start) * 1000
        
        if not transcription.text.strip():
            self.state = PipelineState.IDLE
            return
        
        logger.info(f"[ASR] '{transcription.text}' ({asr_time:.0f}ms)")
        
        # LLM
        self.state = PipelineState.GENERATING
        llm_start = time.time()
        llm_response = self._llm.respond(transcription.text)
        llm_time = (time.time() - llm_start) * 1000
        
        logger.info(f"[LLM] '{llm_response.text}' ({llm_time:.0f}ms)")
        
        # TTS
        self.state = PipelineState.SYNTHESIZING
        tts_start = time.time()
        first_chunk = True
        
        async for audio_chunk in self._tts_service.synthesize_streaming(llm_response.text):
            if first_chunk:
                tts_first = (time.time() - tts_start) * 1000
                total = (time.time() - pipeline_start) * 1000
                
                self.last_metrics = PipelineMetrics(
                    asr_latency_ms=asr_time,
                    llm_latency_ms=llm_time,
                    tts_first_chunk_ms=tts_first,
                    total_latency_ms=total,
                    tokens_generated=llm_response.tokens_generated
                )
                
                logger.info(f"[E2E] {total:.0f}ms (ASR:{asr_time:.0f} LLM:{llm_time:.0f} TTS:{tts_first:.0f})")
                first_chunk = False
            
            self.state = PipelineState.SPEAKING
            yield audio_chunk
        
        self.total_requests += 1
        self.state = PipelineState.IDLE
    
    def reset(self) -> None:
        """Reset pipeline state"""
        self.state = PipelineState.IDLE
        if self._vad:
            self._vad.reset()
        if self._llm:
            self._llm.reset()
    
    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status"""
        return {
            "state": self.state.value,
            "initialized": self._initialized,
            "total_requests": self.total_requests,
            "last_metrics": {
                "vad_ms": self.last_metrics.vad_latency_ms,
                "asr_ms": self.last_metrics.asr_latency_ms,
                "llm_ms": self.last_metrics.llm_latency_ms,
                "tts_first_chunk_ms": self.last_metrics.tts_first_chunk_ms,
                "total_ms": self.last_metrics.total_latency_ms,
            }
        }


class TTSClient:
    """
    TTS Client that connects to VibeVoice TTS service.
    Can run as embedded service or connect to external.
    """
    
    def __init__(
        self,
        model_path: str = "microsoft/VibeVoice-Realtime-0.5B",
        device: str = "cuda",
        voice: str = "en-Carter_man",
        cfg_scale: float = 1.5,
        inference_steps: int = 5
    ):
        self.model_path = model_path
        self.device = device
        self.voice = voice
        self.cfg_scale = cfg_scale
        self.inference_steps = inference_steps
        self._service = None
        self._initialized = False
        self.sample_rate = 24000
    
    async def initialize(self) -> None:
        """Initialize TTS service"""
        if self._initialized:
            return
        
        try:
            # Try to import VibeVoice components
            from vibevoice.modular.modeling_vibevoice_streaming_inference import (
                VibeVoiceStreamingForConditionalGenerationInference,
            )
            from vibevoice.processor.vibevoice_streaming_processor import (
                VibeVoiceStreamingProcessor,
            )
            
            logger.info(f"Loading VibeVoice TTS: {self.model_path}")
            
            # Determine dtype and attention
            if self.device == "cuda":
                load_dtype = torch.bfloat16
                attn_impl = "flash_attention_2"
            else:
                load_dtype = torch.float32
                attn_impl = "sdpa"
            
            # Load processor
            self._processor = VibeVoiceStreamingProcessor.from_pretrained(self.model_path)
            
            # Load model
            try:
                self._model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map=self.device if self.device == "cuda" else None,
                    attn_implementation=attn_impl,
                )
            except Exception:
                logger.warning("Flash attention not available, using SDPA")
                self._model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map=self.device if self.device == "cuda" else None,
                    attn_implementation="sdpa",
                )
            
            self._model.eval()
            
            # Configure scheduler
            self._model.model.noise_scheduler = self._model.model.noise_scheduler.from_config(
                self._model.model.noise_scheduler.config,
                algorithm_type="sde-dpmsolver++",
                beta_schedule="squaredcos_cap_v2",
            )
            self._model.set_ddpm_inference_steps(num_steps=self.inference_steps)
            
            # Load voice preset
            self._load_voice_preset()
            
            self._initialized = True
            logger.info("VibeVoice TTS initialized")
            
        except ImportError as e:
            logger.error(f"VibeVoice not available: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize TTS: {e}")
            raise
    
    def _load_voice_preset(self) -> None:
        """Load voice preset"""
        voices_dir = REPO_ROOT / "demo" / "voices" / "streaming_model"
        
        # Find voice file
        voice_file = voices_dir / f"{self.voice}.pt"
        if not voice_file.exists():
            # Try to find any voice
            voice_files = list(voices_dir.glob("*.pt"))
            if voice_files:
                voice_file = voice_files[0]
                self.voice = voice_file.stem
                logger.warning(f"Voice not found, using: {self.voice}")
            else:
                raise FileNotFoundError(f"No voice presets found in {voices_dir}")
        
        device = torch.device(self.device if self.device != "cpu" else "cpu")
        self._voice_preset = torch.load(voice_file, map_location=device, weights_only=False)
        logger.info(f"Loaded voice preset: {self.voice}")
    
    async def synthesize_streaming(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Synthesize speech from text with streaming.
        
        Args:
            text: Text to synthesize
            
        Yields:
            PCM16 audio bytes
        """
        if not self._initialized:
            await self.initialize()
        
        if not text.strip():
            return
        
        import copy
        from vibevoice.modular.streamer import AudioStreamer
        
        # Prepare inputs
        text = text.replace("'", "'")
        processor_kwargs = {
            "text": text.strip(),
            "cached_prompt": self._voice_preset,
            "padding": True,
            "return_tensors": "pt",
            "return_attention_mask": True,
        }
        
        processed = self._processor.process_input_with_cached_prompt(**processor_kwargs)
        device = torch.device(self.device if self.device != "cpu" else "cpu")
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in processed.items()
        }
        
        # Create streamer
        audio_streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
        stop_event = threading.Event()
        errors = []
        
        # Run generation in thread
        def run_generation():
            try:
                self._model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=self.cfg_scale,
                    tokenizer=self._processor.tokenizer,
                    generation_config={
                        "do_sample": False,
                        "temperature": 1.0,
                        "top_p": 1.0,
                    },
                    audio_streamer=audio_streamer,
                    stop_check_fn=stop_event.is_set,
                    verbose=False,
                    refresh_negative=True,
                    all_prefilled_outputs=copy.deepcopy(self._voice_preset),
                )
            except Exception as e:
                errors.append(e)
                audio_streamer.end()
        
        thread = threading.Thread(target=run_generation, daemon=True)
        thread.start()
        
        # Stream audio chunks
        try:
            stream = audio_streamer.get_stream(0)
            for audio_chunk in stream:
                if torch.is_tensor(audio_chunk):
                    audio_chunk = audio_chunk.detach().cpu().to(torch.float32).numpy()
                else:
                    audio_chunk = np.asarray(audio_chunk, dtype=np.float32)
                
                if audio_chunk.ndim > 1:
                    audio_chunk = audio_chunk.reshape(-1)
                
                # Normalize
                peak = np.max(np.abs(audio_chunk)) if audio_chunk.size else 0.0
                if peak > 1.0:
                    audio_chunk = audio_chunk / peak
                
                # Convert to PCM16 bytes
                audio_chunk = np.clip(audio_chunk, -1.0, 1.0)
                pcm16 = (audio_chunk * 32767.0).astype(np.int16)
                
                yield pcm16.tobytes()
                await asyncio.sleep(0)  # Yield control
                
        finally:
            stop_event.set()
            audio_streamer.end()
            thread.join(timeout=1.0)
            
            if errors:
                logger.error(f"TTS error: {errors[0]}")


# =============================================================================
# FastAPI Application
# =============================================================================

def create_app(config: Optional[PipelineConfig] = None) -> FastAPI:
    """Create FastAPI application"""
    
    app = FastAPI(
        title="VibeVoice Speech-to-Speech",
        description="Real-time Speech-to-Speech with <800ms latency",
        version="1.0.0"
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Pipeline instance
    pipeline_config = config or PipelineConfig()
    pipeline = S2SPipeline(pipeline_config)
    app.state.pipeline = pipeline
    app.state.config = pipeline_config
    
    @app.on_event("startup")
    async def startup():
        logger.info("Starting S2S Pipeline server...")
        await pipeline.initialize()
        logger.info(f"Server ready on port {pipeline_config.port}")
    
    @app.get("/")
    async def index():
        """Serve frontend"""
        frontend_path = Path(__file__).parent / "frontend" / "index.html"
        if frontend_path.exists():
            return FileResponse(frontend_path)
        return JSONResponse({"message": "S2S Pipeline API", "status": "running"})
    
    @app.get("/health")
    async def health():
        """Health check"""
        return {
            "status": "ok",
            "pipeline_state": pipeline.state.value,
            "initialized": pipeline._initialized
        }
    
    @app.get("/status")
    async def status():
        """Get pipeline status"""
        return pipeline.get_status()
    
    @app.get("/config")
    async def get_config():
        """Get pipeline configuration"""
        return {
            "asr_model": pipeline_config.asr_model,
            "llm_model": pipeline_config.llm_model,
            "tts_model": pipeline_config.tts_model,
            "input_sample_rate": pipeline_config.input_sample_rate,
            "output_sample_rate": pipeline_config.output_sample_rate,
        }
    
    @app.websocket("/stream")
    async def websocket_stream(ws: WebSocket):
        """
        WebSocket endpoint for real-time S2S.
        
        Client sends: Binary PCM16 audio at 16kHz
        Server sends: Binary PCM16 audio at 24kHz + JSON status messages
        """
        await ws.accept()
        logger.info("Client connected to S2S stream")
        
        try:
            # Send ready message
            await ws.send_json({
                "type": "status",
                "state": "ready",
                "input_sample_rate": pipeline_config.input_sample_rate,
                "output_sample_rate": pipeline_config.output_sample_rate
            })
            
            # Audio buffer for VAD
            audio_buffer = []
            
            while True:
                try:
                    # Receive audio data
                    data = await ws.receive()
                    
                    if "bytes" in data:
                        # Binary audio data
                        audio_bytes = data["bytes"]
                        
                        # Convert to numpy
                        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                        
                        # Process through VAD
                        speech_segments = pipeline._vad.process_audio(audio_bytes, pipeline_config.input_sample_rate)
                        
                        # Process each speech segment
                        for speech_audio in speech_segments:
                            # Send transcribing status
                            await ws.send_json({
                                "type": "status",
                                "state": "processing"
                            })
                            
                            # Process through pipeline
                            async for audio_chunk in pipeline.process_speech_segment(
                                speech_audio,
                                pipeline_config.input_sample_rate
                            ):
                                # Send audio chunk
                                await ws.send_bytes(audio_chunk)
                            
                            # Send metrics
                            await ws.send_json({
                                "type": "metrics",
                                **pipeline.last_metrics.__dict__
                            })
                            
                            # Send done status
                            await ws.send_json({
                                "type": "status",
                                "state": "ready"
                            })
                    
                    elif "text" in data:
                        # JSON message
                        msg = json.loads(data["text"])
                        
                        if msg.get("type") == "reset":
                            pipeline.reset()
                            await ws.send_json({"type": "status", "state": "reset"})
                        
                        elif msg.get("type") == "text":
                            # Direct text input (skip ASR)
                            text = msg.get("text", "")
                            if text:
                                await ws.send_json({"type": "status", "state": "processing"})
                                
                                # Generate response
                                llm_start = time.time()
                                response = pipeline._llm.respond(text)
                                llm_time = (time.time() - llm_start) * 1000
                                
                                # Synthesize
                                tts_start = time.time()
                                first_chunk = True
                                async for audio_chunk in pipeline._tts_service.synthesize_streaming(response.text):
                                    if first_chunk:
                                        tts_first = (time.time() - tts_start) * 1000
                                        await ws.send_json({
                                            "type": "transcript",
                                            "user": text,
                                            "assistant": response.text,
                                            "llm_ms": llm_time,
                                            "tts_first_ms": tts_first
                                        })
                                        first_chunk = False
                                    
                                    await ws.send_bytes(audio_chunk)
                                
                                await ws.send_json({"type": "status", "state": "ready"})
                
                except WebSocketDisconnect:
                    logger.info("Client disconnected")
                    break
                except RuntimeError as e:
                    # Handle "Cannot call receive once disconnect received" gracefully
                    if "disconnect" in str(e).lower():
                        break
                    logger.error(f"WebSocket runtime error: {e}")
                    break
                except Exception as e:
                    logger.error(f"WebSocket error: {e}")
                    # Only try to send error if connection is still open
                    try:
                        if ws.client_state.name == "CONNECTED":
                            await ws.send_json({"type": "error", "message": str(e)})
                    except Exception:
                        pass  # Connection already closed, ignore
                    break
        
        finally:
            pipeline.reset()
            logger.info("Client disconnected from S2S stream")
    
    return app


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run the S2S pipeline server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="VibeVoice S2S Pipeline Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8005)
    parser.add_argument("--asr-model", type=str, default="distil-whisper/distil-small.en")
    parser.add_argument("--llm-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--tts-model", type=str, default="microsoft/VibeVoice-Realtime-0.5B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-level", type=str, default="INFO")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create config
    config = PipelineConfig(
        host=args.host,
        port=args.port,
        asr_model=args.asr_model,
        llm_model=args.llm_model,
        tts_model=args.tts_model,
        device=args.device
    )
    
    # Create and run app
    app = create_app(config)
    
    logger.info(f"Starting S2S server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
