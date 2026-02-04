#!/usr/bin/env python3
"""
Test VibeVoice vLLM API with Streaming and Optional Hotwords Support.

This script tests ASR transcription via the vLLM OpenAI-compatible API.
By default, it runs standard transcription without hotwords.

Optionally, you can provide hotwords (context_info) to improve recognition
of domain-specific content like proper nouns, technical terms, and speaker names.
Hotwords are embedded in the prompt as "with extra info: {hotwords}".

Usage:
    python test_api_with_hotwords.py [audio_path] [--url URL] [--hotwords "word1,word2"]
    
Examples:
    # Standard transcription (no hotwords)
    python3 test_api.py audio.wav
    
    # With hotwords for better recognition of specific terms
    python3 test_api.py audio.wav --hotwords "Microsoft,Azure,VibeVoice"
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
    """Guess MIME type from file extension."""
    ext = os.path.splitext(path)[1].lower()
    mime_map = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".mp4": "video/mp4",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".opus": "audio/ogg",
    }
    return mime_map.get(ext, "application/octet-stream")


def _get_duration_seconds_ffprobe(path: str) -> float:
    """Get audio duration using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8").strip()
    return float(out)


def _is_video_file(path: str) -> bool:
    """Check if the file is a video file that needs audio extraction."""
    ext = os.path.splitext(path)[1].lower()
    return ext in (".mp4", ".m4v", ".mov", ".webm", ".avi", ".mkv")


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


def test_transcription_with_hotwords(
    audio_path: str,
    context_info: str = None,
    base_url: str = "http://localhost:8000",
):
    """
    Test ASR transcription with customized hotwords.
    
    Hotwords are embedded in the prompt text as "with extra info: {hotwords}".
    This helps the model recognize domain-specific terms more accurately.
    
    Args:
        audio_path: Path to the audio file
        context_info: Hotwords string (e.g., "Microsoft,Azure,VibeVoice")
        base_url: vLLM server URL
    """
    
    print(f"=" * 70)
    print(f"Testing Customized Hotwords Support")
    print(f"=" * 70)
    print(f"Input file: {audio_path}")
    print(f"Hotwords: {context_info or '(none)'}")
    print()
    
    # Handle video files: extract audio first
    temp_audio_path = None
    actual_audio_path = audio_path
    if _is_video_file(audio_path):
        print(f"🎬 Detected video file, extracting audio...")
        temp_audio_path = _extract_audio_from_video(audio_path)
        actual_audio_path = temp_audio_path
        print(f"✅ Audio extracted to: {temp_audio_path}")
    
    # Load audio
    try:
        duration = _get_duration_seconds_ffprobe(actual_audio_path)
        print(f"Audio duration: {duration:.2f} seconds")
        
        with open(actual_audio_path, "rb") as f:
            audio_bytes = f.read()
        
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        print(f"Audio size: {len(audio_bytes)} bytes")
        
    except Exception as e:
        print(f"Error preparing audio: {e}")
        # Cleanup temp file if created
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        return

    # Build the request
    url = f"{base_url}/v1/chat/completions"
    
    show_keys = ["Start time", "End time", "Speaker ID", "Content"]
    
    # Build prompt with optional hotwords
    # Hotwords are embedded as "with extra info: {hotwords}" in the prompt
    if context_info and context_info.strip():
        prompt_text = (
            f"This is a {duration:.2f} seconds audio, with extra info: {context_info.strip()}\n\n"
            f"Please transcribe it with these keys: " + ", ".join(show_keys)
        )
        print(f"\n📝 Hotwords embedded in prompt: '{context_info}'")
    else:
        prompt_text = (
            f"This is a {duration:.2f} seconds audio, please transcribe it with these keys: "
            + ", ".join(show_keys)
        )
        print(f"\n📝 No hotwords provided")

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
    }
    
    print(f"\n{'=' * 70}")
    print(f"Sending request to {url}")
    print(f"{'=' * 70}")
    
    t0 = time.time()
    try:
        response = requests.post(url, json=payload, stream=True, timeout=12000)
        
        if response.status_code == 200:
            print("\n✅ Response received. Streaming content:\n")
            print("-" * 50)

            printed = ""
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    
                    if decoded_line.startswith("data: "):
                        json_str = decoded_line[6:] 
                        if json_str.strip() == "[DONE]":
                            print("\n" + "-" * 50)
                            print("✅ [Finished]")
                            break
                        try:
                            data = json.loads(json_str)
                            delta = data['choices'][0]['delta']
                            content = delta.get('content', '')
                            if content:
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
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out!")
    except Exception as e:
        print(f"❌ Error: {e}")
        
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"⏱️  Total time elapsed: {elapsed:.2f}s")
    print(f"📊 RTF (Real-Time Factor): {elapsed / duration:.2f}x")
    print(f"{'=' * 70}")
    
    # Cleanup temp audio file if created
    if temp_audio_path and os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)
        print(f"🗑️  Cleaned up temp file: {temp_audio_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test VibeVoice vLLM API with Customized Hotwords"
    )
    parser.add_argument(
        "audio_path",
        help="Path to audio file (wav, mp3, flac, etc.) or video file"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="vLLM server URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--hotwords",
        type=str,
        default=None,
        help="Hotwords to improve recognition (e.g., 'Microsoft,Azure,VibeVoice')"
    )
    
    args = parser.parse_args()
    
    # Run test
    test_transcription_with_hotwords(
        audio_path=args.audio_path,
        context_info=args.hotwords,
        base_url=args.url,
    )


if __name__ == "__main__":
    main()
