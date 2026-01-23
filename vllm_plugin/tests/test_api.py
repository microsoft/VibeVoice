#!/usr/bin/env python3
"""
Test VibeVoice vLLM API with Streaming (Real-time output).

Usage:
    python test_api.py [audio_path] [--url URL]
    
Examples:
    python test_api.py                                    # Use default audio
    python test_api.py /path/to/audio.wav                 # Specify audio file
    python test_api.py /path/to/audio.mp3 --url http://localhost:8000  # Custom URL
"""
import requests
import json
import base64
import time
import sys
import os
import subprocess
import argparse


def _guess_mime_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        return "audio/wav"
    if ext in (".mp3",):
        return "audio/mpeg"
    if ext in (".m4a",):
        return "audio/mp4"
    if ext in (".mp4", ".m4v", ".mov", ".webm"):
        return "video/mp4"
    if ext in (".flac",):
        return "audio/flac"
    if ext in (".ogg", ".opus"):
        return "audio/ogg"
    return "application/octet-stream"


def _get_duration_seconds_ffprobe(path: str) -> float:
    """Get audio duration using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8").strip()
    return float(out)


def _extract_audio_from_video(video_path: str) -> str:
    """
    Extract audio from video file (mp4/mov/webm) to a temporary mp3 file.
    Returns the path to the extracted audio file.
    """
    import tempfile
    # Create temp file with .mp3 extension
    fd, audio_path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn",  # No video
        "-acodec", "libmp3lame",
        "-q:a", "2",  # High quality
        audio_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return audio_path


def _is_video_file(path: str) -> bool:
    """Check if the file is a video file that needs audio extraction."""
    ext = os.path.splitext(path)[1].lower()
    return ext in (".mp4", ".m4v", ".mov", ".webm", ".avi", ".mkv")


def test_transcription(audio_path: str, base_url: str = "http://localhost:8000"):
    """Test ASR transcription with streaming output."""
    
    print(f"Loading audio from: {audio_path}")
    
    # Handle video files: extract audio first
    temp_audio_path = None
    actual_audio_path = audio_path
    if _is_video_file(audio_path):
        print(f"Detected video file, extracting audio...")
        temp_audio_path = _extract_audio_from_video(audio_path)
        actual_audio_path = temp_audio_path
        print(f"Audio extracted to: {temp_audio_path}")
    
    try:
        duration = _get_duration_seconds_ffprobe(actual_audio_path)
        print(f"Audio duration: {duration:.2f} seconds")
        
        with open(actual_audio_path, "rb") as f:
            audio_bytes = f.read()
        
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        print(f"Audio size: {len(audio_bytes)} bytes")
        
    except Exception as e:
        print(f"Error preparing audio: {e}")
        return

    # Build the request
    url = f"{base_url}/v1/chat/completions"
    
    show_keys = ["Start time", "End time", "Speaker ID", "Content"]
    prompt_text = (
        f"This is a {duration:.2f} seconds audio, please transcribe it with these keys: "
        + ", ".join(show_keys)
    )

    mime = _guess_mime_type(actual_audio_path)
    data_url = f"data:{mime};base64,{audio_b64}"

    payload = {
        "model": "vibevoice",
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
        "max_tokens": 32768,       
        "temperature": 0.0,      
        "stream": True,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
    }
    
    print(f"\nSending request to {url} (Streaming Mode)...")
    print(f"Prompt: {prompt_text}")
    print("-" * 60)
    
    t0 = time.time()
    try:
        
        response = requests.post(url, json=payload, stream=True, timeout=12000)
        
        if response.status_code == 200:
            print("Response received. Streaming content:\n")

            printed = ""
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    
                    if decoded_line.startswith("data: "):
                        json_str = decoded_line[6:] 
                        if json_str.strip() == "[DONE]":
                            print("\n\n[Finished]")
                            break
                        try:
                            data = json.loads(json_str)
                           
                            delta = data['choices'][0]['delta']
                            content = delta.get('content', '')
                            if content:

                                # vLLM/OpenAI-compatible streams may emit either
                                # incremental deltas OR the full accumulated text.
                                # Only print the newly-added part to avoid repeats.
                                if content.startswith(printed):
                                    to_print = content[len(printed):]
                                else:
                                    to_print = content

                                if to_print:
                                    print(to_print, end='', flush=True)
                                    printed += to_print
                        except json.JSONDecodeError:
                            pass
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.Timeout:
        print("\nRequest timed out!")
    except Exception as e:
        print(f"\nError: {e}")
        
    print(f"\n{'-'*60}")
    print(f"Total time elapsed: {time.time() - t0:.2f}s")
    
    # Cleanup temp audio file if created
    if temp_audio_path and os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)
        print(f"Cleaned up temp file: {temp_audio_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test VibeVoice vLLM API with streaming output"
    )
    parser.add_argument(
        "audio_path",
        nargs="?",
        default=None,
        help="Path to audio file (wav, mp3, flac, etc.) or video file"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="vLLM server base URL (default: http://localhost:8000)"
    )
    
    args = parser.parse_args()
    
    # Find default audio if not specified
    audio_path = args.audio_path
    if audio_path is None:
        # Try to find a sample audio in common locations
        possible_paths = [
            # In VibeVoice demo folder
            os.path.join(os.path.dirname(__file__), "..", "..", "demo", "voices", "en-Carter_man.wav"),
            os.path.join(os.path.dirname(__file__), "..", "..", "demo", "voices", "zh-Anchen_man_bgm.wav"),
            # Relative to current directory
            "demo/voices/en-Carter_man.wav",
            "demo/voices/zh-Anchen_man_bgm.wav",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                audio_path = path
                break
        
        if audio_path is None:
            print("Error: No audio file specified and no default audio found.")
            print("Usage: python test_api.py <audio_path>")
            sys.exit(1)
    
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)
    
    test_transcription(audio_path, args.url)


if __name__ == "__main__":
    main()
