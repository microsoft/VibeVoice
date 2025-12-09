"""
VibeVoice - Real-time Streaming Text-to-Speech

A high-quality, low-latency text-to-speech system with streaming support.

High-Level API (Recommended):
    - VibeVoiceStreamingTTS: Easy-to-use TTS with streaming
    - AudioPlayer: Audio playback with speaker selection

Low-Level API (Advanced):
    - VibeVoiceStreamingForConditionalGenerationInference: Core TTS model
    - VibeVoiceStreamingProcessor: Text and audio processor
    - AudioStreamer: Low-level audio streaming

Quick Start:
    >>> from vibevoice import VibeVoiceStreamingTTS, AudioPlayer
    >>>
    >>> # Initialize TTS
    >>> tts = VibeVoiceStreamingTTS(
    ...     model_path="microsoft/VibeVoice-Realtime-0.5B",
    ...     device="cuda"
    ... )
    >>>
    >>> # Generate and play
    >>> player = AudioPlayer()
    >>> def text_gen():
    ...     for word in ["Hello", "world"]:
    ...         yield word
    >>>
    >>> audio_stream = tts.text_to_speech_streaming(text_gen())
    >>> player.play_stream(audio_stream, realtime=True)
"""

# High-level API
from .inference import (
    VibeVoiceStreamingTTS,
    AudioPlayer,
    synthesize_speech,
    list_default_voices
)

# Low-level API
from .modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference
)
from .processor.vibevoice_streaming_processor import (
    VibeVoiceStreamingProcessor
)
from .modular.streamer import (
    AudioStreamer,
    AsyncAudioStreamer
)

__all__ = [
    # High-level API
    'VibeVoiceStreamingTTS',
    'AudioPlayer',
    'synthesize_speech',
    'list_default_voices',
    # Low-level API
    'VibeVoiceStreamingForConditionalGenerationInference',
    'VibeVoiceStreamingProcessor',
    'AudioStreamer',
    'AsyncAudioStreamer',
]

__version__ = '0.0.1'
