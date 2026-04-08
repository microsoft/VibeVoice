#!/usr/bin/env python
"""
VibeVoice ASR Gradio Demo

This demo uses the vLLM API server instead of loading the model directly.
Supports concurrent requests (non-blocking) and streaming output.

Usage:
    python gradio_asr_demo_api.py --api_url http://localhost:8000
"""
import os
import sys
import io
import json
import time
import base64
import asyncio
import tempfile
import argparse
import threading
import subprocess
import traceback
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
import numpy as np
import soundfile as sf
import gradio as gr
from typing import AsyncGenerator

# Try to import pydub for MP3 conversion
try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False
    print("⚠️ Warning: pydub not available, falling back to WAV format")

# Common audio extensions supported
COMMON_AUDIO_EXTS = {
    '.wav', '.mp3', '.flac', '.ogg', '.opus', '.m4a', '.aac',
    '.wma', '.aiff', '.aif'
}

# Common video extensions supported
COMMON_VIDEO_EXTS = {
    '.mp4', '.webm', '.mov', '.avi', '.mkv', '.flv', '.wmv',
    '.m4v', '.mpeg', '.mpg', '.3gp', '.ts'
}

# Default max video size in MB
DEFAULT_MAX_VIDEO_SIZE_MB = 50

# Default directory to save uploaded files
DEFAULT_UPLOAD_SAVE_DIR = "local/from_custom"

# Custom temporary directory
CUSTOM_TEMP_DIR = os.environ.get("VIBEVOICE_TEMP_DIR", "/tmp/vibevoice_demo")
os.makedirs(CUSTOM_TEMP_DIR, exist_ok=True)

# ============================================================================
# Cloudflared Tunnel Support
# ============================================================================
CLOUDFLARED_PATH = os.path.expanduser("~/.local/bin/cloudflared")

def download_cloudflared():
    """Download cloudflared binary if not exists"""
    if os.path.exists(CLOUDFLARED_PATH):
        return True
    
    print("📥 Downloading cloudflared...")
    os.makedirs(os.path.dirname(CLOUDFLARED_PATH), exist_ok=True)
    
    download_url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    try:
        subprocess.run(
            ["wget", "-q", download_url, "-O", CLOUDFLARED_PATH],
            check=True, timeout=120
        )
        os.chmod(CLOUDFLARED_PATH, 0o755)
        print("✅ cloudflared downloaded successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to download cloudflared: {e}")
        return False

def start_cloudflared_tunnel(port: int):
    """Start cloudflared tunnel and return the process"""
    if not download_cloudflared():
        print("❌ Cannot start cloudflared tunnel")
        return None
    
    print(f"🌐 Starting cloudflared tunnel for port {port}...")
    
    process = subprocess.Popen(
        [CLOUDFLARED_PATH, "tunnel", "--url", f"http://localhost:{port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Read output in background to find the URL
    def read_output():
        for line in process.stdout:
            print(f"[cloudflared] {line.strip()}")
    
    thread = threading.Thread(target=read_output, daemon=True)
    thread.start()
    
    # Give it a moment to start
    time.sleep(3)
    
    return process


# ============================================================================
# Audio Utilities
# ============================================================================

def _guess_mime_type(path: str) -> str:
    """Guess MIME type from file extension."""
    ext = os.path.splitext(path)[1].lower()
    mime_map = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".mp4": "video/mp4",
        ".m4v": "video/mp4",
        ".mov": "video/mp4",
        ".webm": "video/webm",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".opus": "audio/ogg",
        ".aac": "audio/aac",
    }
    return mime_map.get(ext, "application/octet-stream")


def _get_duration_seconds_ffprobe(path: str) -> float:
    """Get audio duration using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8").strip()
        return float(out)
    except Exception:
        # Fallback: try soundfile
        try:
            info = sf.info(path)
            return info.duration
        except Exception:
            return 0.0


def load_audio_ffmpeg(path: str, target_sr: int = None) -> Tuple[np.ndarray, int]:
    """Load audio file using ffmpeg for better format support."""
    try:
        # Use soundfile first (faster for supported formats)
        audio_data, sr = sf.read(path, dtype='float32')
        
        # Debug: log audio info
        print(f"[DEBUG] soundfile loaded: shape={audio_data.shape}, sr={sr}, dtype={audio_data.dtype}")
        print(f"[DEBUG] audio range: min={audio_data.min():.6f}, max={audio_data.max():.6f}")
        
        # Convert to mono if multi-channel
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)  # Convert to mono
            print(f"[DEBUG] Converted to mono: shape={audio_data.shape}")
        
        # Ensure data is in [-1, 1] range (soundfile should do this, but verify)
        max_val = max(abs(audio_data.max()), abs(audio_data.min()))
        if max_val > 10.0:
            # Likely int16 or other integer format, normalize it
            audio_data = audio_data / max_val
            print(f"[DEBUG] Normalized audio (int format detected), original max_val={max_val}")
        elif max_val > 1.0:
            # Float format with slight overflow, just clip it
            audio_data = np.clip(audio_data, -1.0, 1.0)
            print(f"[DEBUG] Clipped audio (slight overflow), original max_val={max_val}")
        
        # Check for silent audio
        if audio_data.max() == 0 and audio_data.min() == 0:
            print(f"[WARNING] Audio appears to be completely silent!")
        
        return audio_data, sr
    except Exception as e:
        print(f"[DEBUG] soundfile failed: {e}, trying ffmpeg...")
    
    # Fallback to ffmpeg
    try:
        target_sr = target_sr or 16000
        cmd = [
            "ffmpeg", "-i", path,
            "-f", "f32le", "-acodec", "pcm_f32le",
            "-ac", "1", "-ar", str(target_sr),
            "-"
        ]
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        audio_bytes, _ = process.communicate()
        audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
        
        print(f"[DEBUG] ffmpeg loaded: shape={audio_data.shape}, sr={target_sr}")
        print(f"[DEBUG] audio range: min={audio_data.min():.6f}, max={audio_data.max():.6f}")
        
        # Check for silent audio
        if len(audio_data) == 0:
            raise RuntimeError("ffmpeg returned empty audio data")
        if audio_data.max() == 0 and audio_data.min() == 0:
            print(f"[WARNING] Audio appears to be completely silent!")
        
        return audio_data, target_sr
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")


def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB."""
    try:
        return os.path.getsize(file_path) / (1024 * 1024)
    except Exception:
        return 0.0


def is_video_file(file_path: str) -> bool:
    """Check if the file is a video file based on extension."""
    ext = os.path.splitext(file_path)[1].lower()
    return ext in COMMON_VIDEO_EXTS


def format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def segments_to_srt(segments: List[Dict]) -> str:
    """
    Convert ASR segments to SRT subtitle format.
    
    Args:
        segments: List of segment dictionaries with Start, End, Content keys
    
    Returns:
        SRT formatted string
    """
    srt_lines = []
    
    for i, seg in enumerate(segments, 1):
        start = seg.get('Start', seg.get('start', seg.get('Start time', 0)))
        end = seg.get('End', seg.get('end', seg.get('End time', 0)))
        content = seg.get('Content', seg.get('content', seg.get('text', '')))
        speaker = seg.get('Speaker', seg.get('speaker', seg.get('Speaker ID', None)))
        
        if start is None or end is None:
            continue
        
        start_time = format_srt_time(float(start))
        end_time = format_srt_time(float(end))
        
        # Add speaker prefix if available
        text = f"[Speaker {speaker}] {content}" if speaker is not None else content
        
        srt_lines.append(f"{i}")
        srt_lines.append(f"{start_time} --> {end_time}")
        srt_lines.append(text)
        srt_lines.append("")  # Empty line between entries
    
    return "\n".join(srt_lines)


def segments_to_vtt(segments: List[Dict]) -> str:
    """
    Convert ASR segments to WebVTT subtitle format (for HTML5 video).
    
    Args:
        segments: List of segment dictionaries with Start, End, Content keys
    
    Returns:
        WebVTT formatted string
    """
    vtt_lines = ["WEBVTT", ""]
    
    for i, seg in enumerate(segments, 1):
        start = seg.get('Start', seg.get('start', seg.get('Start time', 0)))
        end = seg.get('End', seg.get('end', seg.get('End time', 0)))
        content = seg.get('Content', seg.get('content', seg.get('text', '')))
        speaker = seg.get('Speaker', seg.get('speaker', seg.get('Speaker ID', None)))
        
        if start is None or end is None:
            continue
        
        # WebVTT uses HH:MM:SS.mmm format (dot instead of comma)
        start_time = format_srt_time(float(start)).replace(',', '.')
        end_time = format_srt_time(float(end)).replace(',', '.')
        
        # Add speaker prefix if available
        text = f"[Speaker {speaker}] {content}" if speaker is not None else content
        
        vtt_lines.append(f"{i}")
        vtt_lines.append(f"{start_time} --> {end_time}")
        vtt_lines.append(text)
        vtt_lines.append("")  # Empty line between entries
    
    return "\n".join(vtt_lines)


# Audio formats that need conversion (browsers and some APIs may not support them directly)
AUDIO_FORMATS_NEED_CONVERSION = {'.opus', '.ogg', '.flac', '.aiff', '.aif', '.wma'}

# Audio formats that can be used directly
AUDIO_FORMATS_DIRECT = {'.wav', '.mp3', '.m4a', '.aac'}


def convert_audio_to_mp3(
    audio_path: str,
    output_path: Optional[str] = None,
    sample_rate: int = 16000,
    bitrate: str = "128k"
) -> Tuple[Optional[str], Optional[str]]:
    """
    Convert audio file to MP3 format using ffmpeg.
    
    This is useful for converting formats like opus, ogg, flac that may not be
    well-supported by browsers or some APIs.
    
    Args:
        audio_path: Path to the input audio file
        output_path: Optional output path. If None, creates a temp file.
        sample_rate: Target sample rate
        bitrate: Audio bitrate (e.g., '128k')
    
    Returns:
        Tuple of (mp3_path, error_message)
        - If successful: (mp3_path, None)
        - If failed: (None, error_message)
    """
    try:
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3", dir=CUSTOM_TEMP_DIR)
            temp_file.close()
            output_path = temp_file.name
        
        cmd = [
            "ffmpeg", "-y",  # Overwrite output file
            "-i", audio_path,
            "-acodec", "libmp3lame",  # MP3 codec
            "-ar", str(sample_rate),  # Sample rate
            "-ac", "1",  # Mono
            "-b:a", bitrate,  # Audio bitrate
            output_path
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300  # 5 minutes timeout
        )
        
        if result.returncode != 0:
            error_msg = result.stderr.decode('utf-8', errors='ignore')
            if os.path.exists(output_path):
                os.unlink(output_path)
            return None, f"ffmpeg error: {error_msg[:500]}"
        
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            return None, "Failed to convert audio: output file is empty"
        
        return output_path, None
        
    except subprocess.TimeoutExpired:
        return None, "Audio conversion timed out (>5 minutes)"
    except Exception as e:
        return None, f"Error converting audio: {str(e)}"


def extract_audio_from_video(
    video_path: str,
    output_path: Optional[str] = None,
    sample_rate: int = 16000,
    output_format: str = "mp3",
    bitrate: str = "128k"
) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract audio from video file using ffmpeg.
    
    Args:
        video_path: Path to the video file
        output_path: Optional output path for extracted audio. If None, creates a temp file.
        sample_rate: Target sample rate for extracted audio
        output_format: Output audio format ('mp3' or 'wav')
        bitrate: Audio bitrate for mp3 (e.g., '128k')
    
    Returns:
        Tuple of (audio_path, error_message)
        - If successful: (audio_path, None)
        - If failed: (None, error_message)
    """
    try:
        if output_path is None:
            suffix = f".{output_format}"
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=CUSTOM_TEMP_DIR)
            temp_file.close()
            output_path = temp_file.name
        
        # Use ffmpeg to extract audio
        if output_format == "mp3":
            cmd = [
                "ffmpeg", "-y",  # Overwrite output file
                "-i", video_path,
                "-vn",  # No video
                "-acodec", "libmp3lame",  # MP3 codec
                "-ar", str(sample_rate),  # Sample rate
                "-ac", "1",  # Mono
                "-b:a", bitrate,  # Audio bitrate
                output_path
            ]
        else:
            cmd = [
                "ffmpeg", "-y",  # Overwrite output file
                "-i", video_path,
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # PCM 16-bit
                "-ar", str(sample_rate),  # Sample rate
                "-ac", "1",  # Mono
                output_path
            ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300  # 5 minutes timeout
        )
        
        if result.returncode != 0:
            error_msg = result.stderr.decode('utf-8', errors='ignore')
            # Clean up temp file on error
            if os.path.exists(output_path):
                os.unlink(output_path)
            return None, f"ffmpeg error: {error_msg[:500]}"
        
        # Verify the output file exists and has content
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            return None, "Failed to extract audio: output file is empty"
        
        return output_path, None
        
    except subprocess.TimeoutExpired:
        return None, "Audio extraction timed out (>5 minutes)"
    except Exception as e:
        return None, f"Error extracting audio: {str(e)}"


def convert_video_to_mp4(
    video_path: str,
    output_path: Optional[str] = None,
    height: int = 480,
    crf: int = 28,
    fps: int = 30
) -> Tuple[Optional[str], Optional[str]]:
    """
    Convert video to MP4 format with compression (480p by default).
    
    Args:
        video_path: Path to the input video file (e.g., WebM)
        output_path: Optional output path. If None, creates a temp file.
        height: Target video height (width auto-scaled to maintain aspect ratio)
        crf: Constant Rate Factor for compression (18-28 recommended, higher = smaller file)
        fps: Target frame rate (default 30fps)
    
    Returns:
        Tuple of (mp4_path, error_message)
        - If successful: (mp4_path, None)
        - If failed: (None, error_message)
    """
    try:
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir=CUSTOM_TEMP_DIR)
            temp_file.close()
            output_path = temp_file.name
        
        # Use ffmpeg to convert to MP4 with H.264 codec
        # Scale to target height while maintaining aspect ratio (-2 ensures even dimensions)
        cmd = [
            "ffmpeg", "-y",  # Overwrite output file
            "-i", video_path,
            "-vf", f"scale=-2:{height},fps={fps}",  # Scale to 480p height + set fps
            "-c:v", "libx264",  # H.264 video codec
            "-preset", "fast",  # Encoding speed/compression tradeoff
            "-crf", str(crf),  # Quality (lower = better, 18-28 typical)
            "-c:a", "aac",  # AAC audio codec
            "-b:a", "128k",  # Audio bitrate
            "-movflags", "+faststart",  # Enable streaming
            output_path
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=600  # 10 minutes timeout
        )
        
        if result.returncode != 0:
            error_msg = result.stderr.decode('utf-8', errors='ignore')
            # Clean up temp file on error
            if os.path.exists(output_path):
                os.unlink(output_path)
            return None, f"ffmpeg error: {error_msg[:500]}"
        
        # Verify the output file exists and has content
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            return None, "Failed to convert video: output file is empty"
        
        return output_path, None
        
    except subprocess.TimeoutExpired:
        return None, "Video conversion timed out (>10 minutes)"
    except Exception as e:
        return None, f"Error converting video: {str(e)}"


def parse_time_to_seconds(val: Optional[str]) -> Optional[float]:
    """Parse seconds or hh:mm:ss to float seconds."""
    if val is None:
        return None
    val = val.strip()
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        pass
    if ":" in val:
        parts = val.split(":")
        if not all(p.strip().replace(".", "", 1).isdigit() for p in parts):
            return None
        parts = [float(p) for p in parts]
        if len(parts) == 3:
            h, m, s = parts
        elif len(parts) == 2:
            h = 0
            m, s = parts
        else:
            return None
        return h * 3600 + m * 60 + s
    return None


def slice_audio_to_temp(
    audio_path: str,
    start_sec: Optional[float],
    end_sec: Optional[float]
) -> Tuple[Optional[str], Optional[str]]:
    """Slice audio to [start_sec, end_sec) and write to a temp WAV file."""
    try:
        audio_data, sample_rate = load_audio_ffmpeg(audio_path)
        n_samples = len(audio_data)
        full_duration = n_samples / float(sample_rate)
        start = 0.0 if start_sec is None else max(0.0, start_sec)
        end = full_duration if end_sec is None else min(full_duration, end_sec)
        if end <= start:
            return None, f"Invalid time range: start={start}, end={end}"
        start_idx = int(start * sample_rate)
        end_idx = int(end * sample_rate)
        segment = audio_data[start_idx:end_idx]
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=CUSTOM_TEMP_DIR)
        temp_file.close()
        # Use 32767.0 instead of 32768.0 to avoid potential overflow
        segment_int16 = (segment * 32767.0).astype(np.int16)
        sf.write(temp_file.name, segment_int16, sample_rate, subtype='PCM_16')
        return temp_file.name, None
    except Exception as e:
        return None, f"Error slicing audio: {e}"


def clip_and_encode_audio(
    audio_data: np.ndarray,
    sr: int,
    start_time: float,
    end_time: float,
    segment_idx: int,
    use_mp3: bool = True,
    target_sr: int = 16000,
    mp3_bitrate: str = "32k"
) -> Tuple[int, Optional[str], Optional[str]]:
    """Clip audio segment and encode to base64."""
    try:
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        start_sample = max(0, start_sample)
        end_sample = min(len(audio_data), end_sample)
        
        if start_sample >= end_sample:
            return segment_idx, None, f"Invalid segment range: {start_time}-{end_time}"
        
        segment_data = audio_data[start_sample:end_sample]
        
        # Resample if needed
        if sr != target_sr and target_sr < sr:
            import scipy.signal
            num_samples = int(len(segment_data) * target_sr / sr)
            segment_data = scipy.signal.resample(segment_data, num_samples)
            sr = target_sr
        
        # Use 32767.0 instead of 32768.0 to avoid potential overflow
        segment_data_int16 = (segment_data * 32767.0).astype(np.int16)
        
        # Convert to MP3 if pydub is available
        if use_mp3 and HAS_PYDUB:
            try:
                wav_buffer = io.BytesIO()
                sf.write(wav_buffer, segment_data_int16, sr, format='WAV', subtype='PCM_16')
                wav_buffer.seek(0)
                audio_segment = AudioSegment.from_wav(wav_buffer)
                mp3_buffer = io.BytesIO()
                audio_segment.export(mp3_buffer, format='mp3', bitrate=mp3_bitrate)
                mp3_buffer.seek(0)
                audio_bytes = mp3_buffer.read()
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                return segment_idx, f"data:audio/mpeg;base64,{audio_base64}", None
            except Exception:
                pass
        
        # Fallback to WAV
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, segment_data_int16, sr, format='WAV', subtype='PCM_16')
        wav_buffer.seek(0)
        audio_bytes = wav_buffer.read()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        return segment_idx, f"data:audio/wav;base64,{audio_base64}", None
        
    except Exception as e:
        return segment_idx, None, f"Error: {str(e)}"


def extract_audio_segments(audio_path: str, segments: List[Dict]) -> List[Tuple[str, str, Optional[str]]]:
    """Extract multiple segments from audio file with parallel processing."""
    try:
        audio_data, sr = load_audio_ffmpeg(audio_path)
        
        tasks = []
        use_mp3 = HAS_PYDUB
        
        for i, seg in enumerate(segments):
            start_time = seg.get('Start', seg.get('start', seg.get('Start time', 0)))
            end_time = seg.get('End', seg.get('end', seg.get('End time', 0)))
            if start_time is not None and end_time is not None:
                tasks.append((audio_data, sr, float(start_time), float(end_time), i, use_mp3))
        
        results = []
        max_workers = os.cpu_count() or 4
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(clip_and_encode_audio, *task): task[4]
                for task in tasks
            }
            for future in as_completed(futures):
                results.append(future.result())
        
        results.sort(key=lambda x: x[0])
        
        audio_segments = []
        for i, (idx, audio_src, error_msg) in enumerate(results):
            if idx < len(segments):
                seg = segments[idx]
                label = f"Segment {idx + 1}"
                audio_segments.append((label, audio_src, error_msg))
        
        return audio_segments
        
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return []


# ============================================================================
# API Client
# ============================================================================

class VibeVoiceAPIClient:
    """Client for VibeVoice vLLM API."""
    
    def __init__(self, api_url: str = "http://localhost:8000", model_name: str = None):
        self.api_url = api_url.rstrip("/")
        self._model_name = model_name  # User-specified model name (can be None for auto-detect)
        self._available_models: List[str] = []  # Cached available models
        self.endpoint = f"{self.api_url}/v1/chat/completions"
    
    @property
    def model_name(self) -> str:
        """Get the model name (auto-detected if not specified)."""
        if self._model_name:
            return self._model_name
        if self._available_models:
            return self._available_models[0]
        return "vibevoice"  # Fallback default
    
    @model_name.setter
    def model_name(self, value: str):
        """Set the model name."""
        self._model_name = value
    
    def get_available_models_sync(self) -> List[str]:
        """Fetch available models from vLLM API (synchronous)."""
        try:
            response = httpx.get(f"{self.api_url}/v1/models", timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = [m.get('id') for m in data.get('data', []) if m.get('id')]
                self._available_models = models
                print(f"📋 Available models: {models}")
                return models
            return []
        except Exception as e:
            print(f"⚠️ Failed to fetch models: {e}")
            return []
    
    async def get_available_models(self) -> List[str]:
        """Fetch available models from vLLM API (async)."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.api_url}/v1/models", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    models = [m.get('id') for m in data.get('data', []) if m.get('id')]
                    self._available_models = models
                    return models
            return []
        except Exception as e:
            print(f"⚠️ Failed to fetch models: {e}")
            return []
    
    async def check_health(self) -> Tuple[bool, str]:
        """Check if the API server is healthy."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.api_url}/health", timeout=5)
                if response.status_code == 200:
                    return True, "API server is healthy"
                return False, f"API returned status {response.status_code}"
        except httpx.ConnectError:
            return False, "Cannot connect to API server"
        except Exception as e:
            return False, f"Health check failed: {e}"
    
    def check_health_sync(self) -> Tuple[bool, str]:
        """Synchronous health check for startup."""
        try:
            response = httpx.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                return True, "API server is healthy"
            return False, f"API returned status {response.status_code}"
        except httpx.ConnectError:
            return False, "Cannot connect to API server"
        except Exception as e:
            return False, f"Health check failed: {e}"
    
    async def transcribe_streaming(
        self,
        audio_path: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 1.0,
        context_info: str = None,
        timeout: int = 1200,
    ) -> AsyncGenerator[Tuple[str, Optional[Dict]], None]:
        """
        Transcribe audio using streaming API (async version).
        
        Yields:
            Tuple of (accumulated_text, final_result_or_none)
        """
        # Get audio duration
        duration = _get_duration_seconds_ffprobe(audio_path)
        
        # Read and encode audio
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        
        # Build prompt
        show_keys = ["Start", "End", "Speaker", "Content"]
        prompt_text = (
            f"This is a {duration:.2f} seconds audio, please transcribe it with these keys: "
            + ", ".join(show_keys)
        )
        
        # Add context info if provided
        if context_info and context_info.strip():
            prompt_text += f"\n\nContext information (hotwords, speaker names, etc.):\n{context_info.strip()}"
        
        # Build request payload
        mime = _guess_mime_type(audio_path)
        data_url = f"data:{mime};base64,{audio_b64}"
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that transcribes audio input into text output in JSON format."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "audio_url", "audio_url": {"url": data_url}},
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
            "stream_options": {"include_usage": True},  # Enable token statistics
        }
        
        # Send request with streaming using async httpx
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
                async with client.stream(
                    "POST",
                    self.endpoint,
                    json=payload,
                ) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        error_msg = f"API error: {response.status_code} - {error_text.decode()}"
                        yield error_msg, {"error": error_msg}
                        return
                    
                    accumulated_text = ""
                    usage_info = None
                    stopped = False
                    
                    # Declare global at function level for proper access
                    global stop_generation_flag
                    
                    # Process streaming lines with periodic stop flag checking
                    async for line in response.aiter_lines():
                        # Check stop flag after each line
                        if stop_generation_flag:
                            stopped = True
                            print("[INFO] Stop flag detected, breaking out of stream...")
                            break
                        
                        if line:
                            if line.startswith("data: "):
                                json_str = line[6:]
                                if json_str.strip() == "[DONE]":
                                    break
                                
                                try:
                                    data = json.loads(json_str)
                                    
                                    # Check for usage info (sent in final chunk)
                                    if 'usage' in data and data['usage']:
                                        usage_info = data['usage']
                                    
                                    if 'choices' in data and data['choices']:
                                        delta = data['choices'][0].get('delta', {})
                                        content = delta.get('content', '')
                                        
                                        if content:
                                            # Handle incremental or full text
                                            if content.startswith(accumulated_text):
                                                accumulated_text = content
                                            else:
                                                accumulated_text += content
                                            
                                            yield accumulated_text, None
                                        
                                except json.JSONDecodeError:
                                    pass
                    
                    # If stopped, try to close the response to stop receiving more data
                    if stopped:
                        try:
                            await response.aclose()
                            print("[INFO] Response closed after stop")
                        except Exception:
                            pass
            
            # Parse final result with partial parsing support
            segments, parse_warning = self._parse_segments(accumulated_text)
            if segments is None:
                segments = []
            
            final_result = {
                "raw_text": accumulated_text,
                "segments": segments,
                "duration": duration,
                "usage": usage_info,  # Include token statistics
                "stopped": stopped,  # Whether generation was stopped by user
                "parse_warning": parse_warning,  # Warning if partial parse
            }
            
            yield accumulated_text, final_result
            
        except httpx.TimeoutException:
            yield "Request timed out", {"error": "timeout"}
        except Exception as e:
            yield f"Error: {str(e)}", {"error": str(e)}

    def _parse_segments(self, raw_text: str) -> Tuple[Optional[List[Dict]], Optional[str]]:
        """
        Parse segments from raw API response.
        Handles truncated responses by extracting complete segments.
        
        Returns:
            Tuple of (segments_list, warning_message)
            - If fully successful: (segments, None)
            - If partially successful: (segments, warning_message)
            - If failed: (None, error_message)
        """
        if not raw_text:
            return None, "Empty response"
        
        # Try to find JSON array in the response
        text = raw_text.strip()
        
        # Try direct parse first
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result, None
            elif isinstance(result, dict) and "segments" in result:
                return result["segments"], None
            elif isinstance(result, dict):
                # Single segment
                return [result], None
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON array from text
        import re
        
        # Try to find array pattern
        array_match = re.search(r'\[[\s\S]*\]', text)
        if array_match:
            try:
                result = json.loads(array_match.group())
                if isinstance(result, list):
                    return result, None
            except json.JSONDecodeError:
                pass
        
        # Try to find object with segments
        obj_match = re.search(r'\{[\s\S]*"segments"[\s\S]*\}', text)
        if obj_match:
            try:
                result = json.loads(obj_match.group())
                if "segments" in result:
                    return result["segments"], None
            except json.JSONDecodeError:
                pass
        
        # ===== Handle truncated response =====
        # Try to parse individual complete segments from truncated array
        segments = self._parse_truncated_segments(text)
        if segments:
            return segments, f"⚠️ Partial parse: {len(segments)} segments recovered from truncated response"
        
        return None, "Cannot parse JSON from response"

    def _parse_truncated_segments(self, text: str) -> Optional[List[Dict]]:
        """
        Parse complete segments from a truncated JSON array response.
        This handles cases where the response is cut off mid-segment.
        
        Strategy:
        1. Find all complete JSON objects {...} that look like segments
        2. Validate each has expected keys (Start, End, Content or Speaker)
        3. Try to recover incomplete last segment (e.g., repetition truncation)
        """
        # Check if text starts with array
        text = text.strip()
        if not text.startswith('['):
            # Try to find array start
            array_start = text.find('[')
            if array_start == -1:
                return None
            text = text[array_start:]
        
        # Find all complete JSON objects
        # Pattern: {...} that are properly closed
        segments = []
        depth = 0
        obj_start = -1
        in_string = False
        escape_next = False
        
        for i, char in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            
            if in_string:
                continue
            
            if char == '{':
                if depth == 0:
                    obj_start = i
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0 and obj_start != -1:
                    # Found a complete object
                    obj_str = text[obj_start:i+1]
                    try:
                        obj = json.loads(obj_str)
                        # Validate it looks like a segment
                        if self._is_valid_segment(obj):
                            segments.append(obj)
                    except json.JSONDecodeError:
                        pass
                    obj_start = -1
        
        # Try to recover incomplete last segment (truncated due to repetition)
        if obj_start != -1:
            incomplete_text = text[obj_start:]
            recovered = self._recover_incomplete_segment(incomplete_text)
            if recovered and self._is_valid_segment(recovered):
                segments.append(recovered)
        
        return segments if segments else None

    def _recover_incomplete_segment(self, incomplete_text: str) -> Optional[Dict]:
        """
        Try to recover an incomplete segment that was truncated.
        Handles cases like repetition where Content is cut off mid-string.
        
        Example input:
        {"Start":198.36,"End":206.86,"Speaker":0,"Content":"I'm not gonna do it, I'm not gonna do it, I'm not...
        """
        import re
        
        # Try to extract available fields
        segment = {}
        
        # Extract Start
        start_match = re.search(r'"Start"\s*:\s*([0-9.]+)', incomplete_text)
        if start_match:
            segment['Start'] = float(start_match.group(1))
        
        # Extract End
        end_match = re.search(r'"End"\s*:\s*([0-9.]+)', incomplete_text)
        if end_match:
            segment['End'] = float(end_match.group(1))
        
        # Extract Speaker
        speaker_match = re.search(r'"Speaker"\s*:\s*([0-9]+)', incomplete_text)
        if speaker_match:
            segment['Speaker'] = int(speaker_match.group(1))
        
        # Extract Content - handle truncated string
        content_match = re.search(r'"Content"\s*:\s*"', incomplete_text)
        if content_match:
            content_start = content_match.end()
            # Find the content, may be truncated
            content_text = incomplete_text[content_start:]
            
            # Check for repetition pattern and clean it
            cleaned_content = self._clean_repetition(content_text)
            if cleaned_content:
                segment['Content'] = cleaned_content
                segment['_truncated'] = True  # Mark as recovered from truncation
        
        # Must have at least Start, End to be valid
        if 'Start' in segment and 'End' in segment:
            return segment
        
        return None

    def _clean_repetition(self, content: str) -> Optional[str]:
        """
        Clean content from truncated string.
        For repetition cases, keep first 500 characters.
        """
        # Remove trailing incomplete quote if any
        content = content.rstrip('"')
        
        if not content:
            return None
        
        # Keep first 500 characters for repetition cases
        if len(content) > 500:
            return content[:500] + "..."
        
        return content

    def _is_valid_segment(self, obj: Dict) -> bool:
        """
        Check if a dict looks like a valid ASR segment.
        Must have Start, End, and either Content or Speaker.
        """
        if not isinstance(obj, dict):
            return False
        
        # Check for time boundaries (various possible key names)
        has_start = any(k in obj for k in ['Start', 'start', 'Start time'])
        has_end = any(k in obj for k in ['End', 'end', 'End time'])
        
        if not (has_start and has_end):
            return False
        
        # Should have content or speaker
        has_content = any(k in obj for k in ['Content', 'content', 'text'])
        has_speaker = any(k in obj for k in ['Speaker', 'speaker', 'Speaker ID'])
        
        return has_content or has_speaker


# ============================================================================
# Global State
# ============================================================================

api_client: Optional[VibeVoiceAPIClient] = None

# Global stop flag for generation
stop_generation_flag = False

# Event to signal stop for async operations
stop_event: Optional[asyncio.Event] = None


# ============================================================================
# Gradio Interface Functions
# ============================================================================

async def transcribe_audio(
    media_input,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    context_info: str = "",
    max_video_size_mb: float = DEFAULT_MAX_VIDEO_SIZE_MB
) -> AsyncGenerator[Tuple[str, str, Optional[str], Optional[str], Optional[str]], None]:
    """
    Transcribe audio/video using API and return results (async streaming version).
    
    Args:
        media_input: Audio/Video file path or tuple (sample_rate, audio_data) for microphone
        max_new_tokens: Maximum tokens to generate
        temperature: Temperature for sampling
        top_p: Top-p for nucleus sampling
        do_sample: Whether to use sampling (affects temperature)
        context_info: Optional context information
        max_video_size_mb: Maximum video file size in MB
    
    Yields:
        Tuple of (raw_text, audio_segments_html, srt_content, video_path, vtt_content)
    """
    global api_client, stop_generation_flag
    
    # Reset stop flag at the start of each transcription
    stop_generation_flag = False
    print("[INFO] Stop flag reset at transcribe_audio start")
    
    if api_client is None:
        yield "❌ API client not initialized. Please check API URL.", "", None, None, None
        return
    
    # Check API health (async)
    healthy, msg = await api_client.check_health()
    if not healthy:
        yield f"❌ API server not available: {msg}", "", None, None, None
        return
    
    if media_input is None:
        yield "❌ Please provide an audio or video input.", "", None, None, None
        return
    
    try:
        print("[INFO] Transcription requested via API")
        
        # Determine audio path and track if input is video
        audio_path = None
        original_video_path = None  # Track original video for playback with subtitles
        temp_file_to_cleanup = None
        extracted_audio_to_cleanup = None  # Track extracted audio from video
        is_video_input = False
        
        # Handle media input
        if isinstance(media_input, str):
            # Check if uploaded file is a video
            if is_video_file(media_input):
                is_video_input = True
                original_video_path = media_input  # Keep video path for subtitle playback
                video_size = get_file_size_mb(media_input)
                print(f"[INFO] Uploaded video file size: {video_size:.2f} MB (limit: {max_video_size_mb} MB)")
                
                if video_size > max_video_size_mb:
                    yield f"❌ Video file too large: {video_size:.2f} MB. Maximum allowed: {max_video_size_mb} MB", "", None, None, None
                    return
                
                yield f"🎬 Extracting audio from video ({video_size:.2f} MB)...", "", None, None, None
                extracted_path, extract_error = extract_audio_from_video(media_input)
                
                if extract_error:
                    yield f"❌ Failed to extract audio from video: {extract_error}", "", None, None, None
                    return
                
                audio_path = extracted_path
                extracted_audio_to_cleanup = extracted_path
                print(f"[INFO] Extracted audio from video: {audio_path}")
            else:
                # Audio file
                audio_path = media_input
                print(f"[INFO] Using uploaded audio file: {audio_path}")
        elif isinstance(media_input, tuple):
            # Gradio microphone input: (sample_rate, audio_array)
            sample_rate, audio_array = media_input
            audio_array = np.array(audio_array, dtype=np.float32)
            
            # Debug: log input audio info
            print(f"[DEBUG] Microphone input: shape={audio_array.shape}, sr={sample_rate}")
            print(f"[DEBUG] Microphone audio range: min={audio_array.min():.6f}, max={audio_array.max():.6f}")
            
            # Normalize to [-1, 1] range properly
            max_val = max(abs(audio_array.max()), abs(audio_array.min()))
            if max_val > 10.0:
                # Data is likely int16 or similar, normalize it
                audio_array = audio_array / max_val
                print(f"[DEBUG] Normalized microphone audio (int format detected), original max_val={max_val}")
            elif max_val > 1.0:
                # Float format with slight overflow, just clip it
                audio_array = np.clip(audio_array, -1.0, 1.0)
                print(f"[DEBUG] Clipped microphone audio (slight overflow), original max_val={max_val}")
            
            # Check for silent audio
            if max_val == 0:
                print(f"[WARNING] Microphone audio appears to be completely silent!")
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=CUSTOM_TEMP_DIR)
            temp_file.close()
            audio_int16 = (audio_array * 32767.0).astype(np.int16)
            sf.write(temp_file.name, audio_int16, sample_rate, subtype='PCM_16')
            audio_path = temp_file.name
            temp_file_to_cleanup = temp_file.name
            print(f"[INFO] Saved microphone input to: {audio_path}")
        
        # Final check - if we still don't have audio_path, something went wrong
        if audio_path is None:
            yield "❌ Invalid audio input format.", "", None, None, None
            return
        
        # Set temperature based on sampling mode
        actual_temp = temperature if do_sample else 0.0
        
        # Start streaming transcription
        print("[INFO] Starting API transcription (streaming mode)")
        start_time = time.time()
        
        final_result = None
        token_count = 0
        accumulated_text = ""  # Track accumulated text for stop case
        
        async for text, result in api_client.transcribe_streaming(
            audio_path=audio_path,
            max_tokens=max_new_tokens,
            temperature=actual_temp,
            top_p=top_p,
            context_info=context_info,
        ):
            # Track accumulated text
            if text:
                accumulated_text = text
            
            # Check stop flag at higher level too (already declared global at function start)
            if stop_generation_flag:
                print("[INFO] Stop flag detected in transcribe_audio, breaking...")
                # Create a stopped result - parse whatever we have so far
                stopped_segments, stopped_warning = api_client._parse_segments(accumulated_text) if accumulated_text else ([], None)
                final_result = {
                    "raw_text": accumulated_text or "",
                    "segments": stopped_segments or [],
                    "duration": 0,
                    "usage": None,
                    "stopped": True,
                    "parse_warning": stopped_warning,
                }
                break
            
            if result is not None:
                final_result = result
            else:
                # Streaming update - format for readability
                token_count = len(text.split())  # Rough estimate
                formatted_text = text.replace('},', '},\n')
                yield f"🔄 Transcribing... ({token_count} tokens)\n---\n{formatted_text}", "", None, None, None
        
        generation_time = time.time() - start_time
        
        if final_result is None or "error" in final_result:
            error_msg = final_result.get("error", "Unknown error") if final_result else "No response"
            yield f"❌ Transcription failed: {error_msg}", "", None, None, None
            return
        
        # Check if stopped by user
        was_stopped = final_result.get('stopped', False)
        
        # Format final output with token statistics
        if was_stopped:
            raw_output = f"--- ⏹️ Transcription Stopped ---\n"
        else:
            raw_output = f"--- ✅ Transcription Complete ---\n"
        raw_output += f"⏱️ Time: {generation_time:.2f}s | 🎵 Audio: {final_result.get('duration', 0):.2f}s\n"
        
        # Add token statistics if available
        usage = final_result.get('usage')
        if usage:
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            total_tokens = usage.get('total_tokens', 0)
            tokens_per_sec = completion_tokens / generation_time if generation_time > 0 else 0
            raw_output += f"📊 Tokens: {prompt_tokens} (prompt) + {completion_tokens} (completion) = {total_tokens} (total)\n"
            raw_output += f"⚡ Speed: {tokens_per_sec:.1f} tokens/s\n"
        
        # Add parse warning if partial parsing was used
        parse_warning = final_result.get('parse_warning')
        if parse_warning:
            raw_output += f"{parse_warning}\n"
        
        raw_output += f"---\n"
        formatted_raw_text = final_result['raw_text'].replace('},', '},\n')
        raw_output += formatted_raw_text
        
        # Generate audio segments HTML
        segments = final_result.get('segments', [])
        audio_segments_html = ""
        num_segments = len(segments)
        
        if segments:
            # Extract audio clips for each segment
            audio_clips = extract_audio_segments(audio_path, segments)
            
            # Calculate approximate total size
            total_duration = sum(
                (float(seg.get('End', seg.get('end', 0))) - float(seg.get('Start', seg.get('start', 0))))
                for seg in segments
                if seg.get('End') is not None and seg.get('Start') is not None
            )
            approx_size_kb = total_duration * 4  # ~4KB per second at 32kbps
            
            # Add CSS for theme-aware styling (matching original demo)
            theme_css = """
            <style>
            :root {
                --segment-bg: #f8f9fa;
                --segment-border: #e1e5e9;
                --segment-text: #495057;
                --segment-meta: #6c757d;
                --content-bg: white;
                --content-border: #007bff;
                --warning-bg: #fff3cd;
                --warning-border: #ffc107;
                --warning-text: #856404;
            }
            
            @media (prefers-color-scheme: dark) {
                :root {
                    --segment-bg: #2d3748;
                    --segment-border: #4a5568;
                    --segment-text: #e2e8f0;
                    --segment-meta: #a0aec0;
                    --content-bg: #1a202c;
                    --content-border: #4299e1;
                    --warning-bg: #744210;
                    --warning-border: #d69e2e;
                    --warning-text: #faf089;
                }
            }
            
            .audio-segments-container {
                max-height: 600px;
                overflow-y: auto;
                padding: 10px;
            }
            
            .audio-segment {
                margin-bottom: 15px;
                padding: 15px;
                border: 2px solid var(--segment-border);
                border-radius: 8px;
                background-color: var(--segment-bg);
                transition: all 0.3s ease;
            }
            
            .audio-segment:hover {
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            }
            
            .segment-header {
                margin-bottom: 10px;
            }
            
            .segment-title {
                margin: 0;
                color: var(--segment-text);
                font-size: 16px;
                font-weight: 600;
            }
            
            .segment-meta {
                margin-top: 5px;
                font-size: 14px;
                color: var(--segment-meta);
            }
            
            .segment-content {
                margin-bottom: 10px;
                padding: 12px;
                background-color: var(--content-bg);
                border-radius: 6px;
                border-left: 4px solid var(--content-border);
                color: var(--segment-text);
                line-height: 1.5;
            }
            
            .segment-audio {
                width: 100%;
                margin-top: 10px;
                border-radius: 4px;
            }
            
            .segment-warning {
                margin-top: 10px;
                padding: 10px;
                background-color: var(--warning-bg);
                border-radius: 4px;
                border-left: 4px solid var(--warning-border);
                color: var(--warning-text);
                font-size: 13px;
            }
            
            .segments-title {
                color: var(--segment-text);
                margin-bottom: 10px;
            }
            
            .segments-description {
                color: var(--segment-meta);
                margin-bottom: 20px;
            }
            
            .size-badge {
                display: inline-block;
                background: linear-gradient(135deg, #6c757d, #495057);
                color: white;
                padding: 4px 10px;
                border-radius: 12px;
                font-size: 12px;
                margin-left: 10px;
            }
            </style>
            """
            
            # Build HTML
            format_info = "MP3 32kbps 16kHz mono" if HAS_PYDUB else "WAV 16kHz"
            audio_segments_html = theme_css
            audio_segments_html += "<div class='audio-segments-container'>"
            audio_segments_html += f"<h3 class='segments-title'>🔊 Audio Segments ({num_segments} segments)"
            audio_segments_html += f"<span class='size-badge'>📦 ~{approx_size_kb:.0f}KB ({format_info})</span></h3>"
            audio_segments_html += "<p class='segments-description'>🎵 Click the play button to listen to each segment directly!</p>"
            
            for i, seg in enumerate(segments):
                start = seg.get('Start', seg.get('start', seg.get('Start time', None)))
                end = seg.get('End', seg.get('end', seg.get('End time', None)))
                speaker = seg.get('Speaker', seg.get('speaker', seg.get('Speaker ID', None)))
                content = seg.get('Content', seg.get('content', seg.get('text', '')))
                
                # Format times nicely
                start_str = f"{float(start):.2f}" if start is not None else "N/A"
                end_str = f"{float(end):.2f}" if end is not None else "N/A"
                speaker_str = str(speaker) if speaker is not None else "N/A"
                
                # Get audio clip
                audio_html = ""
                error_html = ""
                if i < len(audio_clips):
                    _, audio_src, error_msg = audio_clips[i]
                    if audio_src:
                        audio_type = 'audio/mp3' if 'audio/mp3' in audio_src or 'audio/mpeg' in audio_src else 'audio/wav'
                        audio_html = f"""
                        <audio controls class='segment-audio' preload='none'>
                            <source src='{audio_src}' type='{audio_type}'>
                            Your browser does not support the audio element.
                        </audio>
                        """
                    elif error_msg:
                        error_html = f"""
                        <div class='segment-warning'>
                            <small>❌ {error_msg}</small>
                        </div>
                        """
                
                audio_segments_html += f"""
                <div class='audio-segment'>
                    <div class='segment-header'>
                        <h4 class='segment-title'>🎤 Speaker {speaker_str}</h4>
                        <div class='segment-meta'>
                            ⏱️ {start_str}s - {end_str}s
                        </div>
                    </div>
                    <div class='segment-content'>
                        {content}
                    </div>
                    {audio_html}
                    {error_html}
                </div>
                """
            
            audio_segments_html += "</div>"
        else:
            audio_segments_html = """
            <style>
            :root {
                --no-segments-text: #6c757d;
            }
            
            @media (prefers-color-scheme: dark) {
                :root {
                    --no-segments-text: #a0aec0;
                }
            }
            
            .no-segments-container {
                padding: 20px;
                text-align: center;
                color: var(--no-segments-text);
                line-height: 1.6;
            }
            </style>
            <div class='no-segments-container'>
                <p>❌ No audio segments available.</p>
                <p>This could happen if the model output doesn't contain valid time stamps.</p>
            </div>
            """
        
        # Cleanup temp files
        if temp_file_to_cleanup and os.path.exists(temp_file_to_cleanup):
            try:
                os.unlink(temp_file_to_cleanup)
            except Exception:
                pass
        
        if extracted_audio_to_cleanup and os.path.exists(extracted_audio_to_cleanup):
            try:
                os.unlink(extracted_audio_to_cleanup)
            except Exception:
                pass
        
        # Generate SRT and VTT content if we have segments
        srt_content = None
        vtt_content = None
        if segments:
            srt_content = segments_to_srt(segments)
            vtt_content = segments_to_vtt(segments)
        
        yield raw_output, audio_segments_html, srt_content, original_video_path, vtt_content
        
    except Exception as e:
        print(f"Error during transcription: {e}")
        print(traceback.format_exc())
        yield f"❌ Error: {str(e)}", "", None, None, None


def create_gradio_interface(api_url: str, model_name: str = None, default_max_tokens: int = 4096, max_video_size_mb: float = DEFAULT_MAX_VIDEO_SIZE_MB):
    """Create and launch Gradio interface."""
    global api_client
    
    # Initialize API client
    api_client = VibeVoiceAPIClient(api_url=api_url, model_name=model_name)
    
    # Check API health and fetch available models (sync for startup)
    healthy, health_msg = api_client.check_health_sync()
    available_models = []
    if healthy:
        available_models = api_client.get_available_models_sync()
        if available_models:
            # Auto-select first model if not specified
            if not model_name:
                print(f"🎯 Auto-selected model: {api_client.model_name}")
            api_status = f"✅ Connected to API: {api_url} | Model: {api_client.model_name}"
        else:
            api_status = f"⚠️ Connected but no models found at: {api_url}"
    else:
        api_status = f"⚠️ API not available: {health_msg}"
    print(api_status)
    
    # Custom CSS for button styling
    custom_css = """
    #transcribe-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        color: white !important;
    }
    #transcribe-btn:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
    }
    #stop-btn {
        background-color: #dc3545 !important;
        border-color: #dc3545 !important;
    }
    #stop-btn:hover {
        background-color: #c82333 !important;
    }
    /* Fix tab layout on small screens */
    .tabs > .tab-nav {
        flex-wrap: nowrap !important;
        overflow-x: auto !important;
    }
    .tabs > .tab-nav > button {
        white-space: nowrap !important;
        flex-shrink: 0 !important;
        font-size: 13px !important;
        padding: 8px 12px !important;
    }
    """
    
    with gr.Blocks(title="VibeVoice ASR Demo") as demo:
        gr.Markdown("# 🎙️ VibeVoice ASR Demo")
        gr.Markdown("Upload audio/video files or record from microphone to get speech-to-text transcription with speaker diarization.")
        
        # Store max video size for use in transcribe function
        max_video_size_state = gr.State(value=max_video_size_mb)
        
        # Hidden slider for max_tokens (use default value from args)
        max_tokens_slider = gr.Slider(
            minimum=512,
            maximum=32768,
            value=default_max_tokens,
            step=512,
            label="Max New Tokens",
            visible=False
        )
        
        # Define example files
        # Look for demo files relative to repo root (/app in container)
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        _repo_root = os.path.dirname(os.path.dirname(_script_dir))
        example_dir = os.path.join(_repo_root, "demo", "asr_demo")
        example_files = {
            "chat_audio": os.path.join(example_dir, "demo1-chat.mp3"),
            "chat_video": os.path.join(example_dir, "demo1-chat.mp4"),
            "song_audio": os.path.join(example_dir, "demo2-song.mp3"),
            "song_video": os.path.join(example_dir, "demo2-song.mp4"),
            "hotword": os.path.join(example_dir, "demo3-hotwords.wav"),
        }
        
        with gr.Row():
            # Left column: Media Input (1/3)
            with gr.Column(scale=1):
                # Examples section
                gr.Markdown("## 🎯 Examples")
                with gr.Row():
                    example1_btn = gr.Button("🗣️ Chat", size="sm", scale=1)
                    example2_btn = gr.Button("🎵 Song", size="sm", scale=1)
                    example3_btn = gr.Button("📝 Hotword", size="sm", scale=1)
                
                # Media input section (combined audio/video)
                gr.Markdown("## 🎵 Media Input")
                gr.Markdown(f"*Upload or record audio/video. For video (max {max_video_size_mb} MB), audio will be extracted.*")
                
                # Tabs for Upload File and Record
                with gr.Tabs():
                    with gr.TabItem("📁 Upload File"):
                        media_input = gr.File(
                            label="Upload Audio or Video File",
                            file_types=list(COMMON_AUDIO_EXTS) + list(COMMON_VIDEO_EXTS),
                            type="filepath"
                        )
                    with gr.TabItem("🎙️ Record Audio"):
                        audio_record = gr.Audio(
                            label="Record Audio",
                            sources=["microphone"],
                            type="filepath",
                            interactive=True
                        )
                    with gr.TabItem("🎥 Record Video"):
                        video_record = gr.Video(
                            label="Record Video (auto-converts to 480p@30fps)",
                            sources=["webcam"],
                            include_audio=True,
                            interactive=True
                        )
                
                # Preview section - expanded by default
                with gr.Accordion("👁️ Media Preview", open=True):
                    audio_preview = gr.Audio(
                        label="Audio Preview",
                        interactive=False,
                        visible=False
                    )
                    video_preview = gr.Video(
                        label="Video Preview", 
                        interactive=False,
                        visible=False
                    )
            
            # Right column: Context + Sampling + Results (2/3)
            with gr.Column(scale=2):
                # Context info and Sampling in one row
                with gr.Row(equal_height=True):
                    # Context information section
                    with gr.Column(scale=1):
                        gr.Markdown("## 📋 Customized Context")
                        context_info_input = gr.Textbox(
                            label="Add your customized terms in bellow for better recognition. ",
                            placeholder="VibeVoice \nMicrosoft \nAzure ... ",
                            lines=5,
                            max_lines=6,
                            interactive=True,
                        )
                    
                    # Sampling parameters - side by side with Hotwords
                    with gr.Column(scale=1):
                        gr.Markdown("## 🎲 Sampling")
                        do_sample_checkbox = gr.Checkbox(
                            value=False,
                            label="Enable Sampling"
                        )
                        temperature_slider = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            value=0.0,
                            step=0.1,
                            label="Temperature"
                        )
                        top_p_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=1.0,
                            step=0.05,
                            label="Top-p"
                        )
                
                # Transcribe buttons
                with gr.Row():
                    transcribe_button = gr.Button("🎯 Transcribe", variant="primary", size="lg", scale=3, elem_id="transcribe-btn")
                    stop_button = gr.Button("⏹️ Stop", variant="secondary", size="lg", scale=1, elem_id="stop-btn")
                
                # Results section
                gr.Markdown("## 📝 Results")
                
                with gr.Tabs():
                    with gr.TabItem("📝 Raw Output"):
                        raw_output = gr.Textbox(
                            label="Raw Transcription Output",
                            lines=15,
                            max_lines=30,
                            interactive=False
                        )
                    
                    with gr.TabItem("🔊 Audio Segments", visible=False) as audio_segments_tab:
                        audio_segments_output = gr.HTML(
                            label="Play individual segments to verify accuracy"
                        )
                    
                    with gr.TabItem("🎬 Video with Subtitles", visible=False) as video_subs_tab:
                        gr.Markdown("*Video playback with generated subtitles (only available for video input)*")
                        video_with_subs_output = gr.HTML(
                            label="Video Player with Subtitles"
                        )
                    
                    with gr.TabItem("📥 Download Subtitles", visible=False) as download_subs_tab:
                        gr.Markdown("*Download generated subtitles in SRT format*")
                        srt_download = gr.File(
                            label="Download SRT Subtitle File",
                            interactive=False
                        )
        
        # Event handlers
        def async_copy_uploaded_file(file_path: str, save_dir: str = DEFAULT_UPLOAD_SAVE_DIR):
            """Asynchronously copy uploaded file to save directory with unique filename."""
            def _copy_file():
                try:
                    # Create save directory if not exists
                    save_path = os.path.join(os.path.dirname(__file__), save_dir)
                    os.makedirs(save_path, exist_ok=True)
                    
                    # Generate unique filename: timestamp_uuid_originalname
                    original_name = os.path.basename(file_path)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    unique_id = str(uuid.uuid4())[:8]
                    new_filename = f"{timestamp}_{unique_id}_{original_name}"
                    dest_path = os.path.join(save_path, new_filename)
                    
                    # Copy file
                    shutil.copy2(file_path, dest_path)
                    print(f"[INFO] Uploaded file saved to: {dest_path}")
                except Exception as e:
                    print(f"[WARNING] Failed to save uploaded file: {e}")
            
            # Run copy in background thread
            thread = threading.Thread(target=_copy_file, daemon=True)
            thread.start()
        
        def update_media_preview(file_path):
            """Update media preview based on uploaded file type. 
            Compress video to 480p for preview.
            Convert unsupported audio formats (opus, ogg, flac) to MP3.
            """
            if file_path is None:
                # Don't clear preview when input is None (e.g., when cleared by example button)
                return gr.update(), gr.update()
            
            # Async copy uploaded file to save directory
            async_copy_uploaded_file(file_path)
            
            ext = os.path.splitext(file_path)[1].lower()
            if ext in COMMON_VIDEO_EXTS:
                # Video file - compress to 480p for preview, then show
                print(f"[INFO] Compressing uploaded video for preview: {file_path}")
                compressed_path, error = convert_video_to_mp4(file_path, height=480, crf=28, fps=30)
                if compressed_path:
                    print(f"[INFO] Video compressed for preview: {compressed_path}")
                    return gr.update(value=None, visible=False), gr.update(value=compressed_path, visible=True)
                else:
                    # Fallback to original if compression fails
                    print(f"[WARNING] Video compression failed: {error}, using original")
                    return gr.update(value=None, visible=False), gr.update(value=file_path, visible=True)
            elif ext in AUDIO_FORMATS_NEED_CONVERSION:
                # Audio format needs conversion (opus, ogg, flac, etc.) - convert to MP3
                print(f"[INFO] Converting {ext} audio to MP3 for better compatibility: {file_path}")
                converted_path, error = convert_audio_to_mp3(file_path)
                if converted_path:
                    print(f"[INFO] Audio converted to MP3: {converted_path}")
                    return gr.update(value=converted_path, visible=True), gr.update(value=None, visible=False)
                else:
                    # Fallback to original if conversion fails
                    print(f"[WARNING] Audio conversion failed: {error}, using original")
                    return gr.update(value=file_path, visible=True), gr.update(value=None, visible=False)
            else:
                # Audio file in supported format - show directly
                return gr.update(value=file_path, visible=True), gr.update(value=None, visible=False)
        
        def update_audio_preview(audio_path):
            """Update preview when audio is recorded."""
            if audio_path is None:
                # Don't clear preview when input is None (e.g., when cleared by example button)
                return gr.update(), gr.update()
            return gr.update(value=audio_path, visible=True), gr.update(value=None, visible=False)
        
        def update_video_preview(video_path):
            """Update preview when video is recorded."""
            if video_path is None:
                # Don't clear preview when input is None (e.g., when cleared by example button)
                return gr.update(), gr.update()
            return gr.update(value=None, visible=False), gr.update(value=video_path, visible=True)
        
        # Update preview when file is uploaded
        media_input.change(
            fn=update_media_preview,
            inputs=[media_input],
            outputs=[audio_preview, video_preview]
        )
        
        # Update preview when audio is recorded
        audio_record.change(
            fn=update_audio_preview,
            inputs=[audio_record],
            outputs=[audio_preview, video_preview]
        )
        
        # Update preview when video is recorded
        video_record.change(
            fn=update_video_preview,
            inputs=[video_record],
            outputs=[audio_preview, video_preview]
        )
        
        # Example button handlers - clear upload/record inputs when example is selected
        def load_example_chat():
            """Load chat example with video preview, clear other inputs."""
            video_path = example_files["chat_video"]
            if os.path.exists(video_path):
                return (
                    gr.update(value=None, visible=False),       # audio_preview
                    gr.update(value=video_path, visible=True),  # video_preview
                    "",                                          # context_info (no hotwords)
                    gr.update(value=None),                       # media_input (clear)
                    gr.update(value=None),                       # audio_record (clear)
                    gr.update(value=None),                       # video_record (clear)
                )
            return gr.update(), gr.update(), "", gr.update(), gr.update(), gr.update()
        
        def load_example_song():
            """Load song example with video preview, clear other inputs."""
            video_path = example_files["song_video"]
            if os.path.exists(video_path):
                return (
                    gr.update(value=None, visible=False),       # audio_preview
                    gr.update(value=video_path, visible=True),  # video_preview
                    "",                                          # context_info (no hotwords)
                    gr.update(value=None),                       # media_input (clear)
                    gr.update(value=None),                       # audio_record (clear)
                    gr.update(value=None),                       # video_record (clear)
                )
            return gr.update(), gr.update(), "", gr.update(), gr.update(), gr.update()
        
        def load_example_hotword():
            """Load hotword example with VibeVoice in context, clear other inputs."""
            audio_path = example_files["hotword"]
            if os.path.exists(audio_path):
                return (
                    gr.update(value=audio_path, visible=True),  # audio_preview
                    gr.update(value=None, visible=False),       # video_preview
                    "VibeVoice",                                 # context_info with hotword
                    gr.update(value=None),                       # media_input (clear)
                    gr.update(value=None),                       # audio_record (clear)
                    gr.update(value=None),                       # video_record (clear)
                )
            return gr.update(), gr.update(), "VibeVoice", gr.update(), gr.update(), gr.update()
        
        example1_btn.click(
            fn=load_example_chat,
            inputs=[],
            outputs=[audio_preview, video_preview, context_info_input, media_input, audio_record, video_record]
        )
        
        example2_btn.click(
            fn=load_example_song,
            inputs=[],
            outputs=[audio_preview, video_preview, context_info_input, media_input, audio_record, video_record]
        )
        
        example3_btn.click(
            fn=load_example_hotword,
            inputs=[],
            outputs=[audio_preview, video_preview, context_info_input, media_input, audio_record, video_record]
        )
        
        def reset_stop_flag():
            """Reset stop flag before starting transcription."""
            global stop_generation_flag
            stop_generation_flag = False
            print("[INFO] Stop flag reset")
        
        def set_stop_flag():
            """Set stop flag to interrupt generation."""
            global stop_generation_flag
            stop_generation_flag = True
            print("[INFO] Stop flag set - stopping generation...")
            return "⏹️ Stop requested, waiting for current chunk to complete..."
        
        def get_media_input(file_input, audio_rec, video_rec, audio_prev, video_prev):
            """Get the media input from preview (which shows what will be transcribed).
            
            Priority: preview content (video_prev > audio_prev) since that's what user sees.
            Recorded videos are automatically converted to 480p MP4 to reduce file size.
            """
            # Always use preview content - it shows what will be transcribed
            if video_prev is not None:
                # Check if it's a recorded video that needs conversion
                if video_rec is not None and video_prev == video_rec:
                    print(f"[INFO] Recorded video detected: {video_rec}")
                    converted_path, error = convert_video_to_mp4(video_rec, height=480, crf=28, fps=30)
                    if converted_path:
                        print(f"[INFO] Recorded video converted to 480p@30fps: {converted_path}")
                        return converted_path
                    else:
                        print(f"[WARNING] Failed to convert recorded video: {error}, using original")
                return video_prev
            if audio_prev is not None:
                return audio_prev
            return None
        
        async def transcribe_wrapper(
            file_input, audio_rec, video_rec, audio_prev, video_prev, max_tokens, temp, top_p, do_sample, context_info, max_video_size
        ):
            """Wrapper to handle file/recording input and process results."""
            media = get_media_input(file_input, audio_rec, video_rec, audio_prev, video_prev)
            
            video_html = ""
            srt_file_path = None
            
            async for raw_text, segments_html, srt_content, video_path, vtt_content in transcribe_audio(
                media, max_tokens, temp, top_p, do_sample, context_info, max_video_size
            ):
                # Generate video player HTML with subtitles if video was uploaded
                if video_path and vtt_content:
                    # Create a temp VTT file for the video player
                    vtt_b64 = base64.b64encode(vtt_content.encode('utf-8')).decode('utf-8')
                    vtt_data_url = f"data:text/vtt;base64,{vtt_b64}"
                    
                    # Read video file and create data URL
                    with open(video_path, 'rb') as f:
                        video_bytes = f.read()
                    video_b64 = base64.b64encode(video_bytes).decode('utf-8')
                    ext = os.path.splitext(video_path)[1].lower()
                    mime_type = 'video/mp4' if ext == '.mp4' else f'video/{ext[1:]}'
                    video_data_url = f"data:{mime_type};base64,{video_b64}"
                    
                    video_html = f'''
                    <style>
                    .video-container {{
                        width: 100%;
                        max-width: 800px;
                        margin: 0 auto;
                    }}
                    .video-container video {{
                        width: 100%;
                        border-radius: 8px;
                    }}
                    .video-container video::cue {{
                        background-color: rgba(0, 0, 0, 0.7);
                        color: white;
                        font-size: 16px;
                    }}
                    </style>
                    <div class="video-container">
                        <video controls>
                            <source src="{video_data_url}" type="{mime_type}">
                            <track kind="subtitles" src="{vtt_data_url}" srclang="en" label="Subtitles" default>
                            Your browser does not support the video element.
                        </video>
                    </div>
                    '''
                else:
                    video_html = '''
                    <div style="text-align: center; padding: 40px; color: #6c757d;">
                        <p>🎬 No video input detected.</p>
                        <p>Upload a video file to see playback with subtitles.</p>
                    </div>
                    '''
                
                # Create SRT file for download if available
                if srt_content:
                    srt_temp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.srt', encoding='utf-8', dir=CUSTOM_TEMP_DIR)
                    srt_temp.write(srt_content)
                    srt_temp.close()
                    srt_file_path = srt_temp.name
                
                # Determine tab visibility based on whether we have final results
                # Show tabs only when we have actual content (not during streaming)
                has_segments = segments_html and '<div class' in segments_html
                has_video = video_path is not None
                has_srt = srt_content is not None
                
                # Return all outputs including tab visibility
                yield (
                    raw_text, 
                    segments_html, 
                    video_html, 
                    srt_file_path,
                    gr.update(visible=has_segments),  # audio_segments_tab
                    gr.update(visible=has_video),     # video_subs_tab  
                    gr.update(visible=has_srt)        # download_subs_tab
                )
        
        transcribe_button.click(
            fn=reset_stop_flag,
            inputs=[],
            outputs=[],
            queue=False
        ).then(
            fn=transcribe_wrapper,
            inputs=[
                media_input,
                audio_record,
                video_record,
                audio_preview,
                video_preview,
                max_tokens_slider,
                temperature_slider,
                top_p_slider,
                do_sample_checkbox,
                context_info_input,
                max_video_size_state
            ],
            outputs=[
                raw_output, 
                audio_segments_output, 
                video_with_subs_output, 
                srt_download,
                audio_segments_tab,
                video_subs_tab,
                download_subs_tab
            ]
        )
        
        stop_button.click(
            fn=set_stop_flag,
            inputs=[],
            outputs=[raw_output],
            queue=False
        )
    
    return demo, custom_css


def main():
    parser = argparse.ArgumentParser(description="VibeVoice ASR Gradio Demo")
    parser.add_argument(
        "--api_url",
        type=str,
        default="http://localhost:8000",
        help="URL of the vLLM API server"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name as registered in vLLM server (auto-detected if not specified)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=4096,
        help="Default max new tokens for generation"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to bind the server to"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public link via Gradio"
    )
    parser.add_argument(
        "--cloudflared",
        action="store_true",
        help="Create a public link using cloudflared tunnel"
    )
    parser.add_argument(
        "--max_video_size",
        type=float,
        default=DEFAULT_MAX_VIDEO_SIZE_MB,
        help=f"Maximum video file size in MB (default: {DEFAULT_MAX_VIDEO_SIZE_MB})"
    )
    
    args = parser.parse_args()
    
    # Create interface
    demo, custom_css = create_gradio_interface(
        api_url=args.api_url,
        model_name=args.model_name,
        default_max_tokens=args.max_new_tokens,
        max_video_size_mb=args.max_video_size
    )
    
    print(f"🚀 Starting VibeVoice ASR Demo")
    print(f"📍 Server will be available at: http://{args.host}:{args.port}")
    print(f"🔗 API Endpoint: {args.api_url}")
    
    # Cloudflared tunnel support
    cloudflared_process = None
    if args.cloudflared:
        cloudflared_process = start_cloudflared_tunnel(args.port)
    
    # Gradio 6.0+ moved theme/css to launch()
    launch_kwargs = {
        "server_name": args.host,
        "server_port": args.port,
        "share": args.share,
        "show_error": True,
        "theme": gr.themes.Soft(),
        "css": custom_css,
    }
    
    try:
        # Enable queue for concurrent request handling
        demo.queue(default_concurrency_limit=10)
        demo.launch(**launch_kwargs)
    finally:
        if cloudflared_process:
            cloudflared_process.terminate()


if __name__ == "__main__":
    main()
