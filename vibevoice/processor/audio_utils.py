import os
import threading

import numpy as np
from subprocess import run
from typing import List, Optional, Union, Dict, Any

COMMON_AUDIO_EXTS = [
    '.mp3', '.MP3', '.Mp3',  # All case variations of mp3
    '.m4a', 
    '.mp4', '.MP4',
    '.wav', '.WAV',
    '.m4v',
    '.aac',
    '.ogg',
    '.mov', '.MOV',
    '.opus',
    '.m4b',
    '.flac',
    '.wma', '.WMA',
    '.rm', '.3gp', '.mpeg', '.flv', '.webm', '.mp2', '.aif', '.aiff', '.oga', '.ogv', '.mpga', '.m3u8', '.amr'
]

def load_audio_use_ffmpeg(file: str, resample: bool = False, target_sr: int = 24000):
    """
    Open an audio file and read as mono waveform, optionally resampling.
    Returns both the audio data and the original sample rate.

    Parameters
    ----------
    file: str
        The audio file to open
    resample: bool
        Whether to resample the audio
    target_sr: int
        The target sample rate if resampling is requested

    Returns
    -------
    A tuple containing:
    - A NumPy array with the audio waveform in float32 dtype
    - The original sample rate of the audio file
    """
    if not resample:
        # First, get the original sample rate
        cmd_probe = [
            "ffprobe",
            "-v", "quiet",
            "-show_entries", "stream=sample_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file
        ]
        
        original_sr = int(run(cmd_probe, capture_output=True, check=True).stdout.decode().strip())
    else:
        original_sr = None

    # Now load the audio
    sr_to_use = target_sr if resample else original_sr
    
    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr_to_use),
        "-",
    ]
    
    out = _run_ffmpeg(cmd).stdout
    audio_data = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
    
    return audio_data, sr_to_use


def _get_ffmpeg_max_concurrency() -> int:
    """Get the maximum FFmpeg concurrency from environment variable."""
    v = os.getenv("VIBEVOICE_FFMPEG_MAX_CONCURRENCY", "")
    try:
        n = int(v) if v.strip() else 0
    except Exception:
        n = 0
    # 0/negative means no explicit limit.
    return n


_FFMPEG_MAX_CONCURRENCY = _get_ffmpeg_max_concurrency()
_FFMPEG_SEM = threading.Semaphore(_FFMPEG_MAX_CONCURRENCY) if _FFMPEG_MAX_CONCURRENCY > 0 else None


def _run_ffmpeg(cmd: list, *, stdin_bytes: bytes = None):
    """Run ffmpeg with optional global concurrency limiting.

    This is important for vLLM multi-request concurrency: spawning too many
    ffmpeg processes can saturate CPU/IO and cause request failures/timeouts.
    """
    if _FFMPEG_SEM is None:
        return run(cmd, capture_output=True, check=True, input=stdin_bytes)
    with _FFMPEG_SEM:
        return run(cmd, capture_output=True, check=True, input=stdin_bytes)


def load_audio_bytes_use_ffmpeg(data: bytes, *, resample: bool = False, target_sr: int = 24000):
    """Decode audio bytes via ffmpeg stdin pipe.

    Compared to writing bytes to a temp file, this avoids filesystem IO and
    reduces contention under high request concurrency.
    
    Parameters
    ----------
    data: bytes
        The audio data bytes
    resample: bool
        Whether to resample the audio (must be True)
    target_sr: int
        The target sample rate if resampling is requested

    Returns
    -------
    A tuple containing:
    - A NumPy array with the audio waveform in float32 dtype
    - The sample rate
    """
    if not resample:
        # For stdin bytes, we don't have a cheap/robust way to probe original sr.
        # Keep behavior explicit.
        raise ValueError("load_audio_bytes_use_ffmpeg requires resample=True")

    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-threads", "0",
        "-i", "pipe:0",
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(target_sr),
        "-",
    ]
    out = _run_ffmpeg(cmd, stdin_bytes=data).stdout
    audio_data = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
    return audio_data, target_sr


class AudioNormalizer:
    """
    Audio normalization class for VibeVoice tokenizer.
    
    This class provides audio normalization to ensure consistent input levels
    for the VibeVoice tokenizer while maintaining audio quality.
    """
    
    def __init__(self, target_dB_FS: float = -25, eps: float = 1e-6):
        """
        Initialize the audio normalizer.
        
        Args:
            target_dB_FS (float): Target dB FS level for the audio. Default: -25
            eps (float): Small value to avoid division by zero. Default: 1e-6
        """
        self.target_dB_FS = target_dB_FS
        self.eps = eps
    
    def tailor_dB_FS(self, audio: np.ndarray) -> tuple:
        """
        Adjust the audio to the target dB FS level.
        
        Args:
            audio (np.ndarray): Input audio signal
            
        Returns:
            tuple: (normalized_audio, rms, scalar)
        """
        rms = np.sqrt(np.mean(audio**2))
        scalar = 10 ** (self.target_dB_FS / 20) / (rms + self.eps)
        normalized_audio = audio * scalar
        return normalized_audio, rms, scalar
    
    def avoid_clipping(self, audio: np.ndarray, scalar: Optional[float] = None) -> tuple:
        """
        Avoid clipping by scaling down if necessary.
        
        Args:
            audio (np.ndarray): Input audio signal
            scalar (float, optional): Explicit scaling factor
            
        Returns:
            tuple: (normalized_audio, scalar)
        """
        if scalar is None:
            max_val = np.max(np.abs(audio))
            if max_val > 1.0:
                scalar = max_val + self.eps
            else:
                scalar = 1.0
        
        return audio / scalar, scalar
    
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize the audio by adjusting to target dB FS and avoiding clipping.
        
        Args:
            audio (np.ndarray): Input audio signal
            
        Returns:
            np.ndarray: Normalized audio signal
        """
        # First adjust to target dB FS
        audio, _, _ = self.tailor_dB_FS(audio)
        # Then avoid clipping
        audio, _ = self.avoid_clipping(audio)
        return audio