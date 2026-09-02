#!/usr/bin/env python3
"""
Test VibeVoice vLLM Streaming ASR API over WebSocket, with Optional Hotwords.

This script tests the streaming ASR server: the audio is pushed as PCM frames
and a transcript comes back once per chunk, while the rest of the file is still
being sent. Text appears as it is recognized rather than at the end.

Optionally, you can provide hotwords (context_info) to improve recognition
of domain-specific content like proper nouns, technical terms, and speaker names.
Hotwords are embedded in the prompt as "with extra info: {hotwords}".

Usage:
    python test_api_streaming.py [audio_path] [--url URL] [--hotwords "word1,word2"]

Examples:
    # Standard streaming transcription (no hotwords)
    python3 test_api_streaming.py audio.wav

    # With hotwords for better recognition of specific terms
    python3 test_api_streaming.py audio.wav --hotwords "Microsoft,Azure,VibeVoice"
"""
import asyncio
import json
import os
import sys
import time
import argparse
from urllib.parse import urlparse

import requests
import websockets

# Bare import off vllm_plugin/: the package __init__ imports vLLM, which this
# client does not need. Decoding here rather than in ffmpeg alone keeps the
# resampler identical to the server's, so the transcript matches.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asr_streaming import load_audio_bytes  # noqa: E402


def _ws_url(base_url: str) -> str:
    """http://host:port -> ws://host:port/v1/stream."""
    u = urlparse(base_url)
    scheme = "wss" if u.scheme == "https" else "ws"
    return f"{scheme}://{u.netloc}/v1/stream"


async def test_streaming_with_hotwords(
    audio_path: str,
    context_info: str = None,
    base_url: str = "http://localhost:8000",
):
    """
    Test streaming ASR transcription with customized hotwords.

    Args:
        audio_path: Path to the audio file
        context_info: Hotwords string (e.g., "Microsoft,Azure,VibeVoice")
        base_url: Streaming ASR server URL
    """

    print(f"=" * 70)
    print(f"Testing Streaming Transcription")
    print(f"=" * 70)
    print(f"Input file: {audio_path}")
    print(f"Hotwords: {context_info or '(none)'}")
    print()

    # The chunk length is a property of the checkpoint, not of this client.
    try:
        config = requests.get(f"{base_url}/v1/config", timeout=10).json()
    except Exception as e:
        print(f"❌ Cannot reach {base_url}: {e}")
        return
    sample_rate = config["sample_rate"]
    print(f"Model: {config['model']}")
    print(f"Chunk: {config['chunk_seconds']:.4f}s "
          f"({config['chunk_frames']} frames + {config['lookahead_frames']} lookahead)")

    # Load audio (video files work too: ffmpeg takes the audio track)
    try:
        with open(audio_path, "rb") as f:
            audio = load_audio_bytes(f.read(), sample_rate)
    except Exception as e:
        print(f"Error preparing audio: {e}")
        return
    duration = len(audio) / sample_rate
    print(f"Audio duration: {duration:.2f} seconds")

    if context_info and context_info.strip():
        print(f"\n📝 Hotwords embedded in prompt: '{context_info}'")
    else:
        print(f"\n📝 No hotwords provided")

    url = _ws_url(base_url)
    print(f"\n{'=' * 70}")
    print(f"Streaming to {url}")
    print(f"{'=' * 70}")

    t0 = time.time()
    result = None
    try:
        async with websockets.connect(url, max_size=None) as ws:
            await ws.send(json.dumps({
                "context_info": context_info,
                "max_tokens": 256,
                "temperature": 0.0,
                "top_p": 1.0,
            }))

            async def send_audio():
                # Half a second per frame, as a microphone would deliver it.
                step = sample_rate // 2
                for i in range(0, len(audio), step):
                    await ws.send(audio[i:i + step].astype("<f4").tobytes())
                await ws.send("end")

            sender = asyncio.create_task(send_audio())

            print("\n✅ Connected. Streaming content:\n")
            print("-" * 50)
            async for raw in ws:
                msg = json.loads(raw)
                if "error" in msg:
                    print(f"\n❌ Error: {msg['error']}")
                    break
                if msg.get("done"):
                    result = msg
                    print("\n" + "-" * 50)
                    print(f"✅ [Finished] {msg['total_chunks']} chunks")
                    break
                # Each chunk carries only its own text; the timestamp shows how
                # far behind the audio the transcript is running.
                print(f"[{time.time() - t0:6.2f}s] {msg['text']}", flush=True)
            await sender
    except Exception as e:
        print(f"❌ Error: {e}")

    elapsed = time.time() - t0
    if result:
        print(f"\n{'=' * 70}")
        print("Segments")
        print(f"{'=' * 70}")
        for seg in result["segments"]:
            print(f"[{seg['Start']:7.2f} - {seg['End']:7.2f}] "
                  f"Speaker {seg['Speaker']}: {seg['Content']}")

    print(f"\n{'=' * 70}")
    print(f"⏱️  Total time elapsed: {elapsed:.2f}s")
    print(f"📊 RTF (Real-Time Factor): {elapsed / duration:.2f}x")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Test VibeVoice vLLM Streaming ASR API"
    )
    parser.add_argument(
        "audio_path",
        help="Path to audio file (wav, mp3, flac, etc.) or video file"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Streaming ASR server URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--hotwords",
        type=str,
        default=None,
        help="Hotwords to improve recognition (e.g., 'Microsoft,Azure,VibeVoice')"
    )

    args = parser.parse_args()

    # Run test
    asyncio.run(test_streaming_with_hotwords(
        audio_path=args.audio_path,
        context_info=args.hotwords,
        base_url=args.url,
    ))


if __name__ == "__main__":
    main()
