"""
Silero VAD Module for Speech-to-Speech Pipeline
================================================

MIT License - Commercially free, no attribution required.
Latency: <1ms per audio chunk on CPU.

Features:
- Voice activity detection with streaming support
- Configurable speech/silence thresholds
- Audio buffering with speech segment extraction
"""

import logging
import numpy as np
from typing import Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque
import torch

logger = logging.getLogger(__name__)


@dataclass
class VADConfig:
    """Configuration for Silero VAD"""
    # Model settings
    sample_rate: int = 16000
    
    # Detection thresholds
    threshold: float = 0.5  # Speech probability threshold
    min_speech_duration_ms: int = 250  # Minimum speech segment duration
    min_silence_duration_ms: int = 300  # Silence duration to end speech
    
    # Audio chunk settings
    window_size_samples: int = 512  # 32ms at 16kHz (must be 256, 512, or 1024)
    
    # Buffer settings
    speech_pad_ms: int = 30  # Padding before/after speech
    max_speech_duration_s: float = 30.0  # Max speech segment length
    
    # Callback settings
    on_speech_start: Optional[Callable] = None
    on_speech_end: Optional[Callable[[np.ndarray], None]] = None


@dataclass
class VADState:
    """Internal state for VAD processing"""
    triggered: bool = False
    temp_end: int = 0
    current_sample: int = 0
    speech_start_sample: int = 0
    speech_buffer: List[np.ndarray] = field(default_factory=list)


class SileroVAD:
    """
    Silero VAD wrapper for real-time speech detection.
    
    Usage:
        vad = SileroVAD(VADConfig())
        vad.load_model()
        
        # Process audio chunks
        for chunk in audio_stream:
            speech_segments = vad.process_chunk(chunk)
            for segment in speech_segments:
                # Process complete speech segment
                transcribe(segment)
    """
    
    def __init__(self, config: Optional[VADConfig] = None):
        self.config = config or VADConfig()
        self.model = None
        self.state = VADState()
        self._model_loaded = False
        
        # Pre-calculate sample counts
        self.min_speech_samples = int(
            self.config.min_speech_duration_ms * self.config.sample_rate / 1000
        )
        self.min_silence_samples = int(
            self.config.min_silence_duration_ms * self.config.sample_rate / 1000
        )
        self.speech_pad_samples = int(
            self.config.speech_pad_ms * self.config.sample_rate / 1000
        )
        self.max_speech_samples = int(
            self.config.max_speech_duration_s * self.config.sample_rate
        )
        
        logger.info(f"VAD initialized: threshold={self.config.threshold}, "
                   f"min_speech={self.config.min_speech_duration_ms}ms, "
                   f"min_silence={self.config.min_silence_duration_ms}ms")
    
    def load_model(self, use_onnx: bool = True) -> None:
        """Load Silero VAD model"""
        try:
            # Load using torch.hub (official method)
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=use_onnx
            )
            
            self.model = model
            self._get_speech_timestamps = utils[0]
            self._save_audio = utils[1] 
            self._read_audio = utils[2]
            self._VADIterator = utils[3]
            self._collect_chunks = utils[4]
            
            self._model_loaded = True
            logger.info("Silero VAD model loaded successfully (ONNX={})".format(use_onnx))
            
        except Exception as e:
            logger.error(f"Failed to load Silero VAD: {e}")
            raise RuntimeError(f"VAD model loading failed: {e}")
    
    def reset_state(self) -> None:
        """Reset VAD state for new conversation"""
        self.state = VADState()
        if self.model is not None and hasattr(self.model, 'reset_states'):
            self.model.reset_states()
        logger.debug("VAD state reset")
    
    def process_chunk(self, audio_chunk: np.ndarray) -> List[np.ndarray]:
        """
        Process an audio chunk and return any completed speech segments.
        
        Args:
            audio_chunk: Audio samples (float32, -1.0 to 1.0, 16kHz mono)
            
        Returns:
            List of completed speech segments (may be empty)
        """
        if not self._model_loaded:
            raise RuntimeError("VAD model not loaded. Call load_model() first.")
        
        # Ensure correct format
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
        
        # Normalize if needed
        max_val = np.abs(audio_chunk).max()
        if max_val > 1.0:
            audio_chunk = audio_chunk / max_val
        
        completed_segments = []
        
        # Process in windows
        window_size = self.config.window_size_samples
        for i in range(0, len(audio_chunk), window_size):
            window = audio_chunk[i:i + window_size]
            
            # Pad if necessary
            if len(window) < window_size:
                window = np.pad(window, (0, window_size - len(window)))
            
            # Get speech probability
            speech_prob = self._get_speech_probability(window)
            
            # Update state based on probability
            segment = self._update_state(window, speech_prob)
            if segment is not None:
                completed_segments.append(segment)
        
        return completed_segments
    
    def _get_speech_probability(self, audio_window: np.ndarray) -> float:
        """Get speech probability for an audio window"""
        try:
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio_window)
            
            # Run model inference
            with torch.no_grad():
                speech_prob = self.model(audio_tensor, self.config.sample_rate).item()
            
            return speech_prob
            
        except Exception as e:
            logger.error(f"VAD inference error: {e}")
            return 0.0
    
    def _update_state(self, audio_window: np.ndarray, speech_prob: float) -> Optional[np.ndarray]:
        """Update VAD state and return completed speech segment if any"""
        completed_segment = None
        
        if speech_prob >= self.config.threshold:
            # Speech detected
            if not self.state.triggered:
                # Speech start
                self.state.triggered = True
                self.state.speech_start_sample = self.state.current_sample
                self.state.speech_buffer = []
                
                if self.config.on_speech_start:
                    self.config.on_speech_start()
                
                logger.debug(f"Speech started at sample {self.state.current_sample}")
            
            # Add to buffer
            self.state.speech_buffer.append(audio_window.copy())
            self.state.temp_end = 0
            
            # Check max duration
            total_samples = sum(len(b) for b in self.state.speech_buffer)
            if total_samples >= self.max_speech_samples:
                completed_segment = self._finalize_speech_segment()
                
        else:
            # Silence detected
            if self.state.triggered:
                # Continue buffering during silence padding
                self.state.speech_buffer.append(audio_window.copy())
                
                if self.state.temp_end == 0:
                    self.state.temp_end = self.state.current_sample
                
                # Check if silence duration exceeded
                silence_duration = self.state.current_sample - self.state.temp_end
                if silence_duration >= self.min_silence_samples:
                    # Check minimum speech duration
                    total_samples = sum(len(b) for b in self.state.speech_buffer)
                    if total_samples >= self.min_speech_samples:
                        completed_segment = self._finalize_speech_segment()
                    else:
                        # Too short, discard
                        self.state.triggered = False
                        self.state.speech_buffer = []
                        logger.debug("Speech segment too short, discarded")
        
        self.state.current_sample += len(audio_window)
        return completed_segment
    
    def _finalize_speech_segment(self) -> np.ndarray:
        """Finalize and return the current speech segment"""
        # Concatenate all buffered audio
        speech_segment = np.concatenate(self.state.speech_buffer)
        
        # Reset state
        self.state.triggered = False
        self.state.speech_buffer = []
        self.state.temp_end = 0
        
        duration_ms = len(speech_segment) / self.config.sample_rate * 1000
        logger.debug(f"Speech segment finalized: {duration_ms:.0f}ms")
        
        # Callback if set
        if self.config.on_speech_end:
            self.config.on_speech_end(speech_segment)
        
        return speech_segment
    
    def force_end_speech(self) -> Optional[np.ndarray]:
        """Force end current speech segment (e.g., on disconnect)"""
        if self.state.triggered and self.state.speech_buffer:
            total_samples = sum(len(b) for b in self.state.speech_buffer)
            if total_samples >= self.min_speech_samples:
                return self._finalize_speech_segment()
        
        self.reset_state()
        return None
    
    @property
    def is_speech_active(self) -> bool:
        """Check if speech is currently being detected"""
        return self.state.triggered
    
    @property
    def current_speech_duration_ms(self) -> float:
        """Get duration of current speech segment in milliseconds"""
        if not self.state.triggered:
            return 0.0
        total_samples = sum(len(b) for b in self.state.speech_buffer)
        return total_samples / self.config.sample_rate * 1000


class StreamingVAD:
    """
    High-level streaming VAD with async support.
    
    Designed for WebSocket integration with automatic
    speech segment detection and buffering.
    """
    
    def __init__(self, config: Optional[VADConfig] = None):
        self.vad = SileroVAD(config)
        self.audio_buffer = deque(maxlen=100)  # Rolling buffer for context
        
    def initialize(self) -> None:
        """Initialize the VAD model"""
        self.vad.load_model(use_onnx=True)
        
    def process_audio(self, audio_bytes: bytes, sample_rate: int = 16000) -> List[np.ndarray]:
        """
        Process raw audio bytes and return speech segments.
        
        Args:
            audio_bytes: Raw PCM16 audio bytes
            sample_rate: Audio sample rate (should be 16000)
            
        Returns:
            List of speech segments as numpy arrays
        """
        # Convert bytes to numpy array
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Resample if needed
        if sample_rate != self.vad.config.sample_rate:
            # Simple resampling (for production, use scipy.signal.resample)
            ratio = self.vad.config.sample_rate / sample_rate
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length).astype(int)
            audio = audio[indices]
        
        # Process through VAD
        return self.vad.process_chunk(audio)
    
    def reset(self) -> None:
        """Reset VAD state"""
        self.vad.reset_state()
        self.audio_buffer.clear()
        
    def finalize(self) -> Optional[np.ndarray]:
        """Finalize any remaining speech"""
        return self.vad.force_end_speech()


# Convenience function
def create_vad(
    threshold: float = 0.5,
    min_speech_ms: int = 250,
    min_silence_ms: int = 300
) -> StreamingVAD:
    """Create a configured streaming VAD instance"""
    config = VADConfig(
        threshold=threshold,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms
    )
    vad = StreamingVAD(config)
    vad.initialize()
    return vad
