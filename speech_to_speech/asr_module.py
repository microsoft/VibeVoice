"""
ASR Module using faster-whisper for Speech-to-Speech Pipeline
==============================================================

MIT License - Commercially free, no attribution required.
Target Latency: <200ms for streaming transcription.

Features:
- Streaming ASR with chunked audio processing
- INT8 quantization for faster inference
- Configurable model sizes (tiny, base, small, medium)
- Language detection and forced language support
"""

import logging
import numpy as np
from typing import Optional, List, Tuple, Iterator, Generator
from dataclasses import dataclass
from enum import Enum
import threading
import queue
import time

logger = logging.getLogger(__name__)


class WhisperModel(Enum):
    """Available Whisper model sizes"""
    TINY = "tiny"
    TINY_EN = "tiny.en"
    BASE = "base"
    BASE_EN = "base.en"
    SMALL = "small"
    SMALL_EN = "small.en"
    DISTIL_SMALL_EN = "distil-whisper/distil-small.en"
    DISTIL_MEDIUM_EN = "distil-whisper/distil-medium.en"
    MEDIUM = "medium"
    MEDIUM_EN = "medium.en"
    LARGE_V2 = "large-v2"
    LARGE_V3 = "large-v3"
    LARGE_V3_TURBO = "large-v3-turbo"


@dataclass
class ASRConfig:
    """Configuration for faster-whisper ASR"""
    # Model settings
    model_size: str = "distil-whisper/distil-small.en"  # Best for <200ms latency
    device: str = "cuda"  # cuda, cpu, or auto
    compute_type: str = "int8"  # int8, int8_float16, float16, float32
    
    # Inference settings
    beam_size: int = 1  # Use greedy decoding for speed
    best_of: int = 1
    patience: float = 1.0
    length_penalty: float = 1.0
    
    # Language settings
    language: Optional[str] = "en"  # None for auto-detect
    task: str = "transcribe"  # transcribe or translate
    
    # VAD settings (faster-whisper has built-in VAD)
    vad_filter: bool = False  # Disable since we use Silero VAD
    
    # Streaming settings
    chunk_length_s: float = 0.4  # 400ms chunks for streaming
    
    # Temperature for sampling (0 = greedy)
    temperature: float = 0.0
    
    # Suppress tokens
    suppress_blank: bool = True
    suppress_tokens: Optional[List[int]] = None
    
    # Word timestamps
    word_timestamps: bool = False  # Disable for speed


@dataclass
class TranscriptionResult:
    """Result from ASR transcription"""
    text: str
    language: str
    language_probability: float
    duration_ms: float
    is_final: bool = True
    segments: Optional[List[dict]] = None


class FasterWhisperASR:
    """
    faster-whisper ASR with streaming support.
    
    Usage:
        asr = FasterWhisperASR(ASRConfig())
        asr.load_model()
        
        # Transcribe audio segment
        result = asr.transcribe(audio_segment)
        print(result.text)
    """
    
    def __init__(self, config: Optional[ASRConfig] = None):
        self.config = config or ASRConfig()
        self.model = None
        self._model_loaded = False
        self._lock = threading.Lock()
        
        logger.info(f"ASR initialized: model={self.config.model_size}, "
                   f"device={self.config.device}, compute_type={self.config.compute_type}")
    
    def load_model(self) -> None:
        """Load the faster-whisper model"""
        try:
            from faster_whisper import WhisperModel
            
            logger.info(f"Loading faster-whisper model: {self.config.model_size}")
            start_time = time.time()
            
            # Determine device
            device = self.config.device
            if device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load model with optimizations
            self.model = WhisperModel(
                self.config.model_size,
                device=device,
                compute_type=self.config.compute_type,
                download_root=None,  # Use default cache
                local_files_only=False,
            )
            
            load_time = (time.time() - start_time) * 1000
            self._model_loaded = True
            logger.info(f"faster-whisper model loaded in {load_time:.0f}ms")
            
        except ImportError:
            raise ImportError("faster-whisper not installed. Run: pip install faster-whisper")
        except Exception as e:
            logger.error(f"Failed to load faster-whisper: {e}")
            raise RuntimeError(f"ASR model loading failed: {e}")
    
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None
    ) -> TranscriptionResult:
        """
        Transcribe an audio segment.
        
        Args:
            audio: Audio samples (float32, -1.0 to 1.0)
            sample_rate: Audio sample rate (will resample to 16kHz if different)
            language: Override language (None uses config default)
            
        Returns:
            TranscriptionResult with transcribed text
        """
        if not self._model_loaded:
            raise RuntimeError("ASR model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        # Ensure correct format
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Normalize if needed
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val
        elif max_val < 0.01:
            # Audio too quiet
            return TranscriptionResult(
                text="",
                language=language or self.config.language or "en",
                language_probability=1.0,
                duration_ms=0,
                is_final=True
            )
        
        # Resample if needed
        if sample_rate != 16000:
            audio = self._resample(audio, sample_rate, 16000)
        
        # Transcribe with thread safety
        with self._lock:
            segments, info = self.model.transcribe(
                audio,
                language=language or self.config.language,
                task=self.config.task,
                beam_size=self.config.beam_size,
                best_of=self.config.best_of,
                patience=self.config.patience,
                length_penalty=self.config.length_penalty,
                temperature=self.config.temperature,
                vad_filter=self.config.vad_filter,
                word_timestamps=self.config.word_timestamps,
                suppress_blank=self.config.suppress_blank,
            )
            
            # Collect all segments
            text_parts = []
            segment_list = []
            for segment in segments:
                text_parts.append(segment.text.strip())
                segment_list.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip()
                })
        
        transcription_time = (time.time() - start_time) * 1000
        full_text = " ".join(text_parts).strip()
        
        logger.debug(f"ASR transcription: '{full_text[:50]}...' in {transcription_time:.0f}ms")
        
        return TranscriptionResult(
            text=full_text,
            language=info.language,
            language_probability=info.language_probability,
            duration_ms=transcription_time,
            is_final=True,
            segments=segment_list if segment_list else None
        )
    
    def transcribe_streaming(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000
    ) -> Generator[str, None, None]:
        """
        Stream transcription results as they become available.
        
        Args:
            audio: Audio samples
            sample_rate: Audio sample rate
            
        Yields:
            Partial transcription strings
        """
        if not self._model_loaded:
            raise RuntimeError("ASR model not loaded. Call load_model() first.")
        
        # Ensure correct format
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Resample if needed
        if sample_rate != 16000:
            audio = self._resample(audio, sample_rate, 16000)
        
        with self._lock:
            segments, info = self.model.transcribe(
                audio,
                language=self.config.language,
                task=self.config.task,
                beam_size=self.config.beam_size,
                temperature=self.config.temperature,
                vad_filter=self.config.vad_filter,
            )
            
            # Yield each segment as it's decoded
            for segment in segments:
                yield segment.text.strip()
    
    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate"""
        try:
            import scipy.signal
            
            duration = len(audio) / orig_sr
            target_length = int(duration * target_sr)
            resampled = scipy.signal.resample(audio, target_length)
            return resampled.astype(np.float32)
            
        except ImportError:
            # Fallback to simple resampling
            ratio = target_sr / orig_sr
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length).astype(int)
            return audio[indices]
    
    def detect_language(self, audio: np.ndarray, sample_rate: int = 16000) -> Tuple[str, float]:
        """
        Detect the language of an audio segment.
        
        Args:
            audio: Audio samples
            sample_rate: Audio sample rate
            
        Returns:
            Tuple of (language_code, probability)
        """
        if not self._model_loaded:
            raise RuntimeError("ASR model not loaded. Call load_model() first.")
        
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        if sample_rate != 16000:
            audio = self._resample(audio, sample_rate, 16000)
        
        with self._lock:
            # Use first 30 seconds max for detection
            max_samples = 30 * 16000
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            
            _, info = self.model.transcribe(
                audio,
                language=None,  # Auto-detect
                task="transcribe",
                beam_size=1,
            )
            
            return info.language, info.language_probability


class StreamingASR:
    """
    High-level streaming ASR with buffering support.
    
    Designed for integration with the S2S pipeline.
    """
    
    def __init__(self, config: Optional[ASRConfig] = None):
        self.asr = FasterWhisperASR(config)
        self.audio_buffer = []
        self.min_audio_length_s = 0.3  # Minimum audio for transcription
        
    def initialize(self) -> None:
        """Initialize the ASR model"""
        self.asr.load_model()
        
    def transcribe_segment(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000
    ) -> TranscriptionResult:
        """
        Transcribe a complete speech segment.
        
        Args:
            audio: Audio samples from VAD
            sample_rate: Audio sample rate
            
        Returns:
            Transcription result
        """
        # Check minimum length
        duration_s = len(audio) / sample_rate
        if duration_s < self.min_audio_length_s:
            return TranscriptionResult(
                text="",
                language="en",
                language_probability=1.0,
                duration_ms=0,
                is_final=True
            )
        
        return self.asr.transcribe(audio, sample_rate)
    
    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        sample_rate: int = 16000
    ) -> TranscriptionResult:
        """
        Transcribe raw audio bytes.
        
        Args:
            audio_bytes: Raw PCM16 audio bytes
            sample_rate: Audio sample rate
            
        Returns:
            Transcription result
        """
        # Convert bytes to numpy array
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return self.transcribe_segment(audio, sample_rate)
    
    def reset(self) -> None:
        """Reset ASR state"""
        self.audio_buffer = []


# Convenience function
def create_asr(
    model_size: str = "distil-whisper/distil-small.en",
    device: str = "cuda",
    compute_type: str = "int8"
) -> StreamingASR:
    """Create a configured streaming ASR instance"""
    config = ASRConfig(
        model_size=model_size,
        device=device,
        compute_type=compute_type
    )
    asr = StreamingASR(config)
    asr.initialize()
    return asr
