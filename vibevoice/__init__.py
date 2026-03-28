# vibevoice/__init__.py
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("vibevoice")
except PackageNotFoundError:
    __version__ = "unknown"

from vibevoice.modular import (
    VibeVoiceStreamingForConditionalGenerationInference,
    VibeVoiceStreamingConfig,
    VibeVoiceConfig,
    VibeVoiceASRConfig,
    VibeVoiceForConditionalGeneration,
    VibeVoiceASRForConditionalGeneration,
)
from vibevoice.processor import (
    VibeVoiceProcessor,
    VibeVoiceStreamingProcessor,
    VibeVoiceTokenizerProcessor,
)

__all__ = [
    "__version__",
    "VibeVoiceStreamingForConditionalGenerationInference",
    "VibeVoiceStreamingConfig",
    "VibeVoiceConfig",
    "VibeVoiceASRConfig",
    "VibeVoiceForConditionalGeneration",
    "VibeVoiceASRForConditionalGeneration",
    "VibeVoiceProcessor",
    "VibeVoiceStreamingProcessor",
    "VibeVoiceTokenizerProcessor",
]