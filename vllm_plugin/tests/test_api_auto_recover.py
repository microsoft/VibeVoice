#!/usr/bin/env python3
"""
Test VibeVoice vLLM API with Streaming, Hotwords, and Auto-Recovery.

This script tests ASR transcription with automatic recovery from repetition loops.
Supports optional hotwords to improve recognition of domain-specific terms.

Features:
- Streaming output with real-time repetition detection
- Auto-recovery when model enters repetition loops
- Optional hotwords support (embedded in prompt as "with extra info: {hotwords}")
- Video file support (auto-extracts audio)

Recovery Strategy:
1. First attempt: greedy decoding (temperature=0, top_p=1.0)
2. If loop detected: retry with temperature=0.2/0.3/0.4, top_p=0.95
3. Max 3 retries, truncate to last complete segment boundary

Usage:
    python test_api_auto_recover.py <audio_path> [output_path] [--url URL] [--hotwords "word1,word2"] [--debug]

Examples:
    # Basic usage
    python3 test_api_auto_recover.py audio.wav
    
    # With hotwords
    python3 test_api_auto_recover.py audio.wav --hotwords "Microsoft,VibeVoice"
    
    # Save result to file
    python3 test_api_auto_recover.py audio.wav result.txt
    
    # Debug mode (show recovery info)
    python3 test_api_auto_recover.py audio.wav --debug
"""
import requests
import json
import base64
import time
import sys
import os
import subprocess
import re
import argparse
from collections import Counter


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
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
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


def _find_last_segment_boundary(text: str) -> int:
    """
    Find the position after the last complete segment boundary (},).
    Returns -1 if no complete segment found.
    """
    # Find last "}, " or "}," pattern (segment separator)
    pos = text.rfind("},")
    if pos != -1:
        return pos + 2  # Include the },
    return -1


def _find_safe_print_boundary(text: str, max_pos: int) -> int:
    """
    Find the last complete segment boundary before max_pos.
    Returns 0 if no complete segment found before max_pos.
    """
    search_text = text[:max_pos]
    pos = search_text.rfind("},")
    if pos != -1:
        return pos + 2  # Include the },
    return 0


class RepetitionDetector:
    """Detect repetition patterns in streaming text."""
    
    def __init__(self, 
                 min_pattern_len: int = 10,      # Minimum chars for a pattern
                 min_repeats: int = 3,           # Minimum repetitions to trigger
                 window_size: int = 500):        # Window to check for patterns
        self.min_pattern_len = min_pattern_len
        self.min_repeats = min_repeats
        self.window_size = window_size
        self.text = ""
        
    def add_text(self, new_text: str):
        """Add new text and return (is_looping, good_text_end_pos)."""
        self.text += new_text
        return self._check_repetition()
    
    def _check_repetition(self):
        """Check if the recent text contains repetition loops."""
        if len(self.text) < self.min_pattern_len * self.min_repeats:
            return False, len(self.text)
        
        # Check the recent window
        window = self.text[-self.window_size:] if len(self.text) > self.window_size else self.text
        
        # Method 1: Check for repeated substrings
        for pattern_len in range(self.min_pattern_len, len(window) // self.min_repeats + 1):
            # Get the last pattern_len characters as potential pattern
            pattern = window[-pattern_len:]
            
            # Count how many times this pattern appears at the end
            count = 0
            pos = len(window)
            while pos >= pattern_len:
                if window[pos - pattern_len:pos] == pattern:
                    count += 1
                    pos -= pattern_len
                else:
                    break
            
            if count >= self.min_repeats:
                # Found repetition! Calculate where the good text ends
                repetition_start = len(self.text) - (count * pattern_len)
                # Keep one instance of the pattern (or none if it's garbage)
                good_end = repetition_start + pattern_len if self._is_meaningful(pattern) else repetition_start
                return True, good_end
        
        # Method 2: Check for repeated short phrases (like "you're not, you're not")
        # Look for patterns like "X, X, X" or "X X X"
        words = window.split()
        if len(words) >= self.min_repeats * 2:
            # Check last N words for repetition
            for phrase_len in range(2, 6):  # 2-5 word phrases
                if len(words) < phrase_len * self.min_repeats:
                    continue
                
                phrase = " ".join(words[-phrase_len:])
                count = 0
                idx = len(words)
                while idx >= phrase_len:
                    candidate = " ".join(words[idx - phrase_len:idx])
                    if candidate == phrase:
                        count += 1
                        idx -= phrase_len
                    else:
                        break
                
                if count >= self.min_repeats:
                    # Calculate position in original text
                    repeated_text = (phrase + " ") * count
                    good_end = len(self.text) - len(repeated_text.rstrip()) + len(phrase)
                    return True, max(0, good_end)
        
        return False, len(self.text)
    
    def _is_meaningful(self, pattern: str) -> bool:
        """Check if pattern is meaningful content (not just garbage)."""
        # Filter out patterns that are just punctuation, spaces, or very repetitive
        clean = pattern.strip()
        if not clean:
            return False
        if len(set(clean)) < 3:  # Too few unique characters
            return False
        return True
    
    def get_good_text(self, end_pos: int) -> str:
        """Get text up to the specified position."""
        return self.text[:end_pos]
    
    def reset(self, keep_text: str = ""):
        """Reset detector, optionally keeping some text."""
        self.text = keep_text


def stream_with_recovery(
    url: str,
    base_messages: list,
    audio_data_url: str,
    prompt_text: str,
    max_tokens: int = 32768,
    max_retries: int = 3,
    timeout: int = 12000,
    debug: bool = False,
):
    """
    Stream transcription with automatic recovery from repetition loops.
    
    Args:
        url: API endpoint
        base_messages: Base messages (system + user with audio)
        audio_data_url: The audio data URL for the request
        prompt_text: The text prompt
        max_tokens: Maximum tokens to generate
        max_retries: Maximum recovery attempts (default 3)
        timeout: Request timeout
        debug: If True, show recovery debug info to stderr
    
    Recovery strategy:
        - First attempt: temperature=0.0, top_p=1.0 (greedy)
        - Recovery: temperature=0.2/0.3/0.4 for retry 1/2/3, top_p=0.95
        - If has complete segments: use assistant prefix
        - If no complete segments: restart from scratch
        - Max 3 retries, if all fail output error message
    
    Returns:
        Final transcription text
    """
    import sys as _sys
    
    def _log(msg):
        """Log to stderr only if debug."""
        if debug:
            print(msg, file=_sys.stderr)
    
    detector = RepetitionDetector(
        min_pattern_len=10,   # At least 10 chars for a pattern
        min_repeats=10,       # Must repeat 10+ times
        window_size=400,      # Check last 400 chars (can detect 10-40 char patterns repeated 10 times)
    )
    
    accumulated_text = ""
    retry_count = 0
    user_safe_printed_len = 0  # Track how much we've safely shown to user (at segment boundaries)
    is_recovery = False  # Whether we're in recovery mode
    
    while retry_count <= max_retries:
        # Build request payload
        messages = list(base_messages)  # Copy base messages
        
        # If we have accumulated text from previous attempt, add it as partial assistant response
        if accumulated_text:
            # Add the good content as a partial assistant message
            # vLLM will continue from here
            messages.append({
                "role": "assistant",
                "content": accumulated_text
            })
        
        # Set sampling parameters based on recovery state
        if is_recovery:
            # Recovery: increase temperature each retry to break loops
            recovery_temp = 0.1 + 0.1 * retry_count  # 0.2, 0.3, 0.4 for retry 1, 2, 3
            payload = {
                "model": "vibevoice",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": recovery_temp,
                "top_p": 0.95,
                "stream": True,
            }
            if accumulated_text:
                _log(f"[RECOVERY #{retry_count}] Continuing from {len(accumulated_text)} chars with temp={recovery_temp}, top_p=0.95")
            else:
                _log(f"[RECOVERY #{retry_count}] Restarting from scratch with temp={recovery_temp}, top_p=0.95")
        else:
            # First attempt: greedy decoding
            payload = {
                "model": "vibevoice",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
                "stream": True,
            }
        
        try:
            response = requests.post(url, json=payload, stream=True, timeout=timeout)
            
            if response.status_code != 200:
                _log(f"[ERROR] {response.status_code} - {response.text[:500]}")
                return accumulated_text
            
            new_text = ""
            printed = ""  # Track what we've already received to handle vLLM duplicates
            
            for line in response.iter_lines():
                if not line:
                    continue
                
                decoded_line = line.decode('utf-8')
                if not decoded_line.startswith("data: "):
                    continue
                
                json_str = decoded_line[6:]
                if json_str.strip() == "[DONE]":
                    # Successfully finished without loops
                    full_result = accumulated_text + new_text
                    # Print any remaining content that wasn't printed yet
                    if len(full_result) > user_safe_printed_len:
                        remaining = full_result[user_safe_printed_len:]
                        print(remaining, end='', flush=True)
                    print()  # Final newline
                    return full_result
                
                try:
                    data = json.loads(json_str)
                    delta = data['choices'][0].get('delta', {})
                    content = delta.get('content', '')
                    
                    if content:
                        # vLLM/OpenAI-compatible streams may emit either
                        # incremental deltas OR the full accumulated text.
                        # Only track the newly-added part.
                        if content.startswith(printed):
                            to_add = content[len(printed):]
                        else:
                            to_add = content
                        
                        if to_add:
                            printed += to_add
                            new_text += to_add
                            
                            # When continuing from prefix, model may add "[" or "[{" at start
                            # or repeat the ending "}, " from prefix
                            # We need to handle these to maintain valid JSON array format
                            if accumulated_text and new_text:
                                stripped = new_text.lstrip()
                                # Case 1: Model added "[{" - remove the "["
                                if stripped.startswith("[{"):
                                    new_text = stripped[1:]
                                    _log("[STRIPPED leading '[' from continuation]")
                                # Case 2: Model added just "[" - remove it
                                elif stripped.startswith("["):
                                    new_text = stripped[1:]
                                    _log("[STRIPPED leading '[' from continuation]")
                                # Case 3: Model repeated "}," from prefix ending
                                elif stripped.startswith("},"):
                                    new_text = stripped[2:]
                                    _log("[STRIPPED leading '},' from continuation]")
                                # Case 4: Model repeated "}" from prefix ending
                                elif stripped.startswith("}") and not stripped.startswith("}]"):
                                    new_text = stripped[1:]
                                    _log("[STRIPPED leading '}' from continuation]")
                                
                                # Fix malformed JSON: {"2.99,... -> {"Start":2.99,...
                                # This happens when model skips "Start": key
                                import re
                                malformed = re.match(r'^\{"(\d+\.?\d*),', new_text)
                                if malformed:
                                    time_val = malformed.group(1)
                                    new_text = '{"Start":' + time_val + ',' + new_text[malformed.end():]
                                    _log(f"[FIXED malformed JSON: added Start key]")
                            
                            # Check for repetition in the combined text
                            full_text = accumulated_text + new_text
                            detector.text = full_text
                            is_looping, good_end = detector._check_repetition()
                            
                            if is_looping:
                                _log(f"[LOOP DETECTED at char {good_end}]")
                                
                                # Use what user has already seen as prefix for retry
                                # user_safe_printed_len is always at a segment boundary
                                if user_safe_printed_len > 0:
                                    accumulated_text = full_text[:user_safe_printed_len]
                                    _log(f"[RETRY from user-visible content at {user_safe_printed_len}]")
                                else:
                                    # No complete segment shown to user yet - restart from scratch
                                    accumulated_text = ""
                                    _log(f"[NO CONTENT SHOWN TO USER - restart from scratch]")
                                
                                detector.reset(accumulated_text)
                                is_recovery = True
                                
                                if debug:
                                    print("\n[...recovering...]", end='', flush=True, file=sys.stderr)
                                
                                retry_count += 1
                                
                                if retry_count > max_retries:
                                    _log(f"[MAX RETRIES REACHED]")
                                    print("\n[Error] Transcription failed due to model output anomaly. Please try another audio or contact support.", flush=True)
                                    return None
                                
                                # Break inner loop to retry
                                break
                            else:
                                # No loop detected - stream content to user
                                # Only print up to (full_text_len - window_size) at segment boundaries
                                # This ensures user never sees content that might be rolled back
                                safe_end = max(0, len(full_text) - detector.window_size)
                                safe_boundary = _find_safe_print_boundary(full_text, safe_end)
                                
                                if safe_boundary > user_safe_printed_len:
                                    # Print new safe content
                                    to_print = full_text[user_safe_printed_len:safe_boundary]
                                    print(to_print, end='', flush=True)
                                    user_safe_printed_len = safe_boundary
                                
                except json.JSONDecodeError:
                    continue
            else:
                # Loop completed without break (no repetition detected)
                full_result = accumulated_text + new_text
                
                # Print any remaining content that wasn't printed yet
                if len(full_result) > user_safe_printed_len:
                    remaining = full_result[user_safe_printed_len:]
                    print(remaining, end='', flush=True)
                
                print()  # Final newline
                return full_result
                
        except requests.exceptions.Timeout:
            _log("[TIMEOUT]")
            print()
            return accumulated_text
        except Exception as e:
            _log(f"[ERROR: {e}]")
            print()
            return accumulated_text
    
    # All retries exhausted
    print("\n[Error] Transcription failed due to model output anomaly. Please try another audio or contact support.", flush=True)
    return None


def test_transcription_with_recovery(
    audio_path: str,
    output_path: str = None,
    base_url: str = "http://localhost:8000",
    hotwords: str = None,
    debug: bool = False,
):
    """
    Test ASR transcription with auto-recovery from repetition loops.
    
    Args:
        audio_path: Path to the audio file
        output_path: Optional path to save transcription result
        base_url: vLLM server URL
        hotwords: Hotwords string (e.g., "Microsoft,Azure,VibeVoice")
        debug: Show recovery debug info
    """
    
    print(f"=" * 70)
    print(f"Testing with Auto-Recovery")
    print(f"=" * 70)
    print(f"Input file: {audio_path}")
    print(f"Hotwords: {hotwords or '(none)'}")
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
        print(f"❌ Error preparing audio: {e}")
        # Cleanup temp file if created
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        return

    url = f"{base_url}/v1/chat/completions"
    
    show_keys = ["Start time", "End time", "Speaker ID", "Content"]
    
    # Build prompt with optional hotwords
    if hotwords and hotwords.strip():
        prompt_text = (
            f"This is a {duration:.2f} seconds audio, with extra info: {hotwords.strip()}\n\n"
            f"Please transcribe it with these keys: " + ", ".join(show_keys)
        )
        print(f"\n📝 Hotwords embedded in prompt: '{hotwords}'")
    else:
        prompt_text = (
            f"This is a {duration:.2f} seconds audio, please transcribe it with these keys: "
            + ", ".join(show_keys)
        )
        print(f"\n📝 No hotwords provided")

    mime = _guess_mime_type(actual_audio_path)
    data_url = f"data:{mime};base64,{audio_b64}"

    # Base messages (without assistant continuation)
    base_messages = [
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
    ]
    
    print(f"\n{'=' * 70}")
    print(f"Sending request to {url}")
    print(f"{'=' * 70}")
    
    t0 = time.time()
    print("\n✅ Response received. Streaming content:\n")
    print("-" * 50)
    
    result = stream_with_recovery(
        url=url,
        base_messages=base_messages,
        audio_data_url=data_url,
        prompt_text=prompt_text,
        max_tokens=32768,
        max_retries=3,
        debug=debug,
    )
    
    elapsed = time.time() - t0
    print("-" * 50)
    print("✅ [Finished]")
    print(f"\n{'=' * 70}")
    print(f"⏱️  Total time elapsed: {elapsed:.2f}s")
    print(f"{'=' * 70}")
    
    if result is None:
        print("❌ Transcription failed")
        return
    
    print(f"📄 Final output length: {len(result)} chars")
    
    # Optionally save result
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"💾 Result saved to: {output_path}")
    
    # Cleanup temp audio file if created
    if temp_audio_path and os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)
        print(f"🗑️  Cleaned up temp file: {temp_audio_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test VibeVoice vLLM API with auto-recovery from repetition loops"
    )
    parser.add_argument(
        "audio_path",
        help="Path to audio file (wav, mp3, flac, etc.) or video file"
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        default=None,
        help="Optional path to save transcription result"
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show recovery debug info"
    )
    
    args = parser.parse_args()
    
    # Run test
    test_transcription_with_recovery(
        audio_path=args.audio_path,
        output_path=args.output_path,
        base_url=args.url,
        hotwords=args.hotwords,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
