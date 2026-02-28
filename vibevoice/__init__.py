# vibevoice/__init__.py

# High-level API
from .inference import (
    VibeVoiceStreamingTTS,
    AudioPlayer,
    synthesize_speech,
    list_default_voices
)

# Low-level API
from vibevoice.modular import (
    VibeVoiceStreamingForConditionalGenerationInference,
    VibeVoiceStreamingConfig,
)
from vibevoice.processor import (
    VibeVoiceStreamingProcessor,
    VibeVoiceTokenizerProcessor,
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
    "VibeVoiceStreamingForConditionalGenerationInference",
    "VibeVoiceStreamingConfig",
    "VibeVoiceStreamingProcessor",
    "VibeVoiceTokenizerProcessor",
    'AudioStreamer',
    'AsyncAudioStreamer',
]

__version__ = '0.0.1'