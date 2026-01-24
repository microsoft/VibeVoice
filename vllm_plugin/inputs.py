"""Audio input mapper for vLLM multimodal pipeline.

This module handles audio data loading and preprocessing for VibeVoice ASR inference.
It converts various audio input formats (path, bytes, numpy array) into tensors
that can be processed by the VibeVoice model.
"""
import torch
import numpy as np
from typing import Union, List
from vllm.multimodal.inputs import MultiModalInputs
from vibevoice.processor.audio_utils import load_audio_use_ffmpeg, load_audio_bytes_use_ffmpeg, AudioNormalizer


def load_audio(audio_path: str, target_sr: int = 24000) -> np.ndarray:
    """Load and normalize audio from file path.
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate (default 24kHz for VibeVoice)
        
    Returns:
        Normalized audio waveform as numpy array
    """
    # Load with FFmpeg (handles various formats)
    audio, sr = load_audio_use_ffmpeg(audio_path, resample=True, target_sr=target_sr)
    
    # Normalize audio
    normalizer = AudioNormalizer()
    audio = normalizer(audio)
    
    return audio


def vibevoice_audio_input_mapper(ctx, data: Union[str, bytes, np.ndarray, List[str]]) -> MultiModalInputs:
    """Map audio input data to vLLM MultiModalInputs format.
    
    This function is registered as the input mapper for VibeVoice audio processing.
    It handles multiple input formats and converts them to normalized tensors.
    
    Args:
        ctx: vLLM context (unused)
        data: Audio data in one of these formats:
            - str: Path to audio file
            - bytes: Raw audio bytes (any format FFmpeg supports)
            - np.ndarray: Pre-loaded audio waveform
            - List[str]: List of audio paths (only first is used)
            
    Returns:
        MultiModalInputs containing:
            - audio: Audio tensor (float32)
            - audio_length: Length of audio in samples
    """
    # Handle list input (take first item)
    if isinstance(data, list):
        data = data[0]

    audio_waveform = None
    
    if isinstance(data, str):
        # Load from file path
        audio_waveform = load_audio(data)
        
    elif isinstance(data, bytes):
        # Decode bytes directly via ffmpeg stdin pipe to avoid temp-file IO
        audio_waveform, _sr = load_audio_bytes_use_ffmpeg(data, resample=True, target_sr=24000)
        normalizer = AudioNormalizer()
        audio_waveform = normalizer(audio_waveform)
                
    elif isinstance(data, np.ndarray):
        # Already loaded numpy array
        audio_waveform = data
    else:
        raise ValueError(f"Unsupported audio data type: {type(data)}")
        
    # Convert to tensor
    audio_tensor = torch.from_numpy(audio_waveform).float()
    audio_length = audio_tensor.shape[0]
    
    return MultiModalInputs({
        "audio": audio_tensor,
        "audio_length": audio_length
    })
