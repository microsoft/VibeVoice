"""
Simple VibeVoice Inference Example

This script demonstrates basic usage of the VibeVoice Python API.

Run from VibeVoice root:
    python examples/simple_inference.py
"""

from vibevoice import VibeVoiceStreamingTTS, AudioPlayer


def main():
    print("="*60)
    print("VibeVoice Simple Inference Example")
    print("="*60)
    print()

    # Configuration
    MODEL_PATH = "microsoft/VibeVoice-Realtime-0.5B"
    VOICE_PROMPT_PATH = "demo/voices/streaming_model/en-Emma_woman.pt"  # Optional
    DEVICE = "cuda"  # or "cpu" or "mps"

    # Initialize TTS
    print("Initializing VibeVoice...")
    tts = VibeVoiceStreamingTTS(
        model_path=MODEL_PATH,
        voice_prompt_path=VOICE_PROMPT_PATH,
        device=DEVICE,
        inference_steps=5  # Fast inference
    )
    print()

    # Initialize audio player
    print("Initializing audio player...")
    player = AudioPlayer()
    print()

    # List available devices
    print("Available audio devices:")
    AudioPlayer.list_devices()
    print()

    # Generate text
    def text_generator():
        """Simple text generator"""
        text = "Hello! This is VibeVoice speaking. I can generate speech in real time."
        for word in text.split():
            yield word

    # Generate and play
    print("Generating and playing speech...")
    print("Text: 'Hello! This is VibeVoice speaking. I can generate speech in real time.'")
    print()

    audio_stream = tts.text_to_speech_streaming(text_generator())
    player.play_stream(audio_stream, realtime=True)

    print()
    print("="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
