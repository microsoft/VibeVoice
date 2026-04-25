"""Model configuration validation utilities for VibeVoice.

Provides validation functions that check model configuration objects
before loading to catch common misconfiguration issues early.
"""

import torch

_VALID_MODEL_TYPES = frozenset(["vibevoice_asr", "vibevoice_streaming"])

_VALID_TORCH_DTYPES = frozenset([
    "float16",
    "float32",
    "bfloat16",
])


def validate_model_config(config) -> None:
    """Validate a VibeVoice model configuration.

    Raises :class:`ValueError` when the configuration contains invalid
    values that would cause runtime failures during model loading.

    Parameters
    ----------
    config:
        A configuration object with ``model_type`` and ``torch_dtype``
        attributes (typically a ``PretrainedConfig`` subclass).
    """
    model_type = getattr(config, "model_type", None)
    if model_type not in _VALID_MODEL_TYPES:
        raise ValueError(
            f"Unknown model type: {model_type!r}. "
            f"Expected one of {sorted(_VALID_MODEL_TYPES)}"
        )

    dtype = getattr(config, "torch_dtype", None)
    if dtype is not None:
        if isinstance(dtype, str):
            if dtype not in _VALID_TORCH_DTYPES:
                raise ValueError(
                    f"Unsupported torch dtype: {dtype!r}. "
                    f"Expected one of {sorted(_VALID_TORCH_DTYPES)} or None"
                )
            # Verify that the string maps to a real torch dtype
            if not hasattr(torch, dtype):
                raise ValueError(
                    f"Unsupported torch dtype: {dtype!r} is not a valid torch attribute"
                )
        elif not isinstance(dtype, torch.dtype):
            raise ValueError(
                f"Unsupported torch dtype: {dtype!r}. "
                f"Expected a string, torch.dtype, or None"
            )
