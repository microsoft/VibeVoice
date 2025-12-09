"""
VibeVoice Python Inference Module

High-level API for easy text-to-speech inference with streaming support.
"""

import copy
from pathlib import Path
from typing import Iterator, Generator, Optional
from threading import Thread, Lock

import torch
import numpy as np

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("Warning: sounddevice not installed. Audio playback features will be disabled.")
    print("Install with: pip install sounddevice")

from .modular.streamer import AudioStreamer
from .modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference
)
from .processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor


class VibeVoiceStreamingTTS:
    """
    High-level wrapper for VibeVoice streaming text-to-speech.

    This class provides an easy-to-use interface for real-time TTS generation
    with support for voice cloning and streaming output.

    Example:
        >>> tts = VibeVoiceStreamingTTS(
        ...     model_path="microsoft/VibeVoice-Realtime-0.5B",
        ...     voice_prompt_path="path/to/voice.pt",
        ...     device="cuda"
        ... )
        >>>
        >>> def text_gen():
        ...     for word in ["Hello", "world"]:
        ...         yield word
        >>>
        >>> for audio_chunk in tts.text_to_speech_streaming(text_gen()):
        ...     # Process audio chunk
        ...     pass
    """

    def __init__(
        self,
        model_path: str,
        voice_prompt_path: Optional[str] = None,
        device: str = "cuda",
        inference_steps: int = 5,
    ):
        """
        Initialize VibeVoice streaming TTS.

        Args:
            model_path: Path to VibeVoice model or HuggingFace model ID
            voice_prompt_path: Optional path to voice prompt (.pt file) for voice cloning.
                              If None, will automatically use a default voice from demo/voices/streaming_model/
            device: Device to run on ('cuda', 'mps', or 'cpu')
            inference_steps: Number of diffusion inference steps (lower = faster, higher = better quality)
        """
        print(f"Loading VibeVoice model from {model_path}...")

        # Load processor
        self.processor = VibeVoiceStreamingProcessor.from_pretrained(model_path)

        # Determine dtype and attention implementation
        if device == "cuda":
            dtype = torch.bfloat16
            attn_impl = "flash_attention_2"
            device_map = "cuda"
        elif device == "mps":
            dtype = torch.float32
            attn_impl = "sdpa"
            device_map = None
        else:
            dtype = torch.float32
            attn_impl = "sdpa"
            device_map = "cpu"

        # Load model
        self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
            attn_implementation=attn_impl
        )

        if device == "mps":
            self.model.to("mps")

        self.model.eval()
        self.model.set_ddpm_inference_steps(num_steps=inference_steps)

        # Load voice prompt
        self.voice_prompt = None
        if voice_prompt_path and Path(voice_prompt_path).exists():
            print(f"Loading voice prompt from {voice_prompt_path}")
            self.voice_prompt = torch.load(
                voice_prompt_path,
                map_location=device,
                weights_only=False
            )
        else:
            # Try to find default voice prompts
            default_voice_dir = Path(__file__).parent.parent / "demo" / "voices" / "streaming_model"
            if default_voice_dir.exists():
                # Look for a default voice (prefer en-Mike_man.pt or first available)
                default_voices = list(default_voice_dir.glob("*.pt"))
                if default_voices:
                    # Prefer en-Mike_man.pt if available
                    preferred = default_voice_dir / "en-Mike_man.pt"
                    voice_path = preferred if preferred.exists() else default_voices[0]
                    print(f"Loading default voice prompt from {voice_path.name}")
                    self.voice_prompt = torch.load(
                        voice_path,
                        map_location=device,
                        weights_only=False
                    )

        if self.voice_prompt is None:
            raise RuntimeError(
                "No voice prompt provided and no default voices found. "
                "Please provide a voice_prompt_path or ensure demo/voices/streaming_model/*.pt exists."
            )

        self.device = device
        self.sample_rate = 24000
        print("Model loaded successfully!")

    def text_to_speech_streaming(
        self,
        text_iterator: Iterator[str],
        cfg_scale: float = 1.5,
    ) -> Iterator[np.ndarray]:
        """
        Convert text from an iterator to speech chunks in real-time.

        Args:
            text_iterator: Iterator/generator that yields text tokens/chunks
            cfg_scale: Classifier-free guidance scale (1.0-2.0, higher = better quality)

        Yields:
            numpy arrays containing audio chunks (float32, 1D, normalized to [-1.0, 1.0])

        Example:
            >>> def text_gen():
            ...     for word in ["Hello", "world"]:
            ...         yield word
            >>>
            >>> for audio_chunk in tts.text_to_speech_streaming(text_gen()):
            ...     print(f"Received chunk with {len(audio_chunk)} samples")
        """
        # Collect text from iterator
        text_chunks = list(text_iterator)
        full_text = " ".join(text_chunks)

        if not full_text.strip():
            return

        print(f"Generating speech for: '{full_text}'")

        # Create audio streamer
        audio_streamer = AudioStreamer(batch_size=1)

        # Process input
        inputs = self.processor.process_input_with_cached_prompt(
            text=full_text,
            cached_prompt=self.voice_prompt,
            padding=True,
            return_tensors="pt",
        )

        # Move inputs to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        # Start generation in background thread for real-time streaming
        def run_generation():
            with torch.no_grad():
                self.model.generate(
                    **inputs,
                    audio_streamer=audio_streamer,
                    cfg_scale=cfg_scale,
                    tokenizer=self.processor.tokenizer,
                    generation_config={'do_sample': False},
                    all_prefilled_outputs=copy.deepcopy(self.voice_prompt),
                )

        generation_thread = Thread(target=run_generation, daemon=True)
        generation_thread.start()

        # Yield audio chunks as they arrive from the model
        stream = audio_streamer.get_stream(0)
        for audio_chunk in stream:
            # Convert to numpy array (float32 for compatibility)
            if torch.is_tensor(audio_chunk):
                audio_chunk = audio_chunk.detach().cpu().to(torch.float32).numpy()
            else:
                audio_chunk = np.asarray(audio_chunk, dtype=np.float32)

            # Reshape to 1D if needed
            if audio_chunk.ndim > 1:
                audio_chunk = audio_chunk.reshape(-1)

            # Normalize if peak is above 1.0 to prevent distortion
            peak = np.max(np.abs(audio_chunk)) if audio_chunk.size else 0.0
            if peak > 1.0:
                audio_chunk = audio_chunk / peak

            # Clip to valid range [-1.0, 1.0]
            audio_chunk = np.clip(audio_chunk, -1.0, 1.0)

            yield audio_chunk.astype(np.float32, copy=False)

        # Wait for generation to complete
        generation_thread.join()

    def save_audio(self, audio: np.ndarray, output_path: str):
        """
        Save generated audio to a WAV file.

        Args:
            audio: Audio data as numpy array
            output_path: Path to save the WAV file

        Example:
            >>> chunks = list(tts.text_to_speech_streaming(text_gen()))
            >>> full_audio = np.concatenate(chunks)
            >>> tts.save_audio(full_audio, "output.wav")
        """
        self.processor.save_audio(
            audio,
            output_path=output_path,
            sampling_rate=self.sample_rate
        )


class AudioPlayer:
    """
    Audio player with speaker selection support.

    Provides easy playback of audio streams with automatic device management
    and real-time streaming support.

    Example:
        >>> player = AudioPlayer()
        >>> audio_stream = tts.text_to_speech_streaming(text_gen())
        >>> player.play_stream(audio_stream, realtime=True)
    """

    def __init__(self, device_id: Optional[int] = None, sample_rate: int = 24000):
        """
        Initialize audio player.

        Args:
            device_id: Speaker device ID (None for default)
            sample_rate: Audio sample rate in Hz (default 24000)
        """
        if not SOUNDDEVICE_AVAILABLE:
            raise ImportError(
                "sounddevice is required for audio playback. "
                "Install with: pip install sounddevice"
            )

        self.device_id = device_id
        self.sample_rate = sample_rate

    @staticmethod
    def list_devices(show_all: bool = False):
        """
        List available audio output devices.

        Args:
            show_all: If True, show all devices including duplicates.
                     If False, show only unique output devices.

        Example:
            >>> AudioPlayer.list_devices()
            Available Audio Output Devices:
            [3] Microsoft Sound Mapper - Output ⭐ DEFAULT
            [4] Speakers (USB Audio Device)
        """
        if not SOUNDDEVICE_AVAILABLE:
            print("sounddevice not installed. Cannot list devices.")
            return []

        print("\nAvailable Audio Output Devices:")
        print("-" * 60)
        devices = sd.query_devices()
        default_output = sd.default.device[1]

        if show_all:
            for i, device in enumerate(devices):
                if device['max_output_channels'] > 0:
                    default_marker = " (DEFAULT)" if i == default_output else ""
                    print(f"[{i}] {device['name']}{default_marker}")
                    print(f"    Channels: {device['max_output_channels']}")
                    print(f"    Sample Rate: {device['default_samplerate']} Hz")
                    print()
        else:
            seen_names = set()
            output_devices = []

            for i, device in enumerate(devices):
                if device['max_output_channels'] > 0:
                    name = device['name']
                    if name not in seen_names:
                        seen_names.add(name)
                        is_default = (i == default_output)
                        output_devices.append((i, name, is_default))

            for i, name, is_default in output_devices:
                default_marker = " ⭐ DEFAULT" if is_default else ""
                print(f"[{i}] {name}{default_marker}")

            print()
            print(f"Default device ID: {default_output}")
            print("Tip: Use device_id=None to use the default device")
            print()

        return devices

    @staticmethod
    def get_default_output_device():
        """
        Get the default output device ID.

        Returns:
            int: Default output device ID

        Example:
            >>> device_id = AudioPlayer.get_default_output_device()
            >>> player = AudioPlayer(device_id=device_id)
        """
        if not SOUNDDEVICE_AVAILABLE:
            return None
        return sd.default.device[1]

    def play_stream(self, audio_iterator: Iterator[np.ndarray], realtime: bool = True):
        """
        Play audio from an iterator of chunks.

        Args:
            audio_iterator: Iterator yielding audio chunks (numpy arrays)
            realtime: If True, use streaming mode with minimal buffering (~100ms latency).
                     If False, collect all chunks first for smooth playback.

        Example:
            >>> # Real-time streaming (low latency)
            >>> player.play_stream(audio_stream, realtime=True)
            >>>
            >>> # Buffered playback (smooth, no gaps)
            >>> player.play_stream(audio_stream, realtime=False)
        """
        if realtime:
            # Real-time streaming with callback-based continuous playback
            PREBUFFER_SECONDS = 0.1  # 100ms prebuffer
            BLOCKSIZE = 2048          # ~85ms chunks at 24kHz

            prebuffer_samples = int(self.sample_rate * PREBUFFER_SECONDS)

            buffer = np.array([], dtype=np.float32)
            buffer_lock = Lock()
            iterator_finished = False
            has_started = False

            def fill_buffer():
                nonlocal buffer, iterator_finished
                for audio_chunk in audio_iterator:
                    with buffer_lock:
                        buffer = np.concatenate([buffer, audio_chunk])
                iterator_finished = True

            fill_thread = Thread(target=fill_buffer, daemon=True)
            fill_thread.start()

            def audio_callback(outdata, frames, time_info, status):
                nonlocal buffer, has_started

                if status:
                    print(f"Audio callback status: {status}", flush=True)

                with buffer_lock:
                    if not has_started:
                        if len(buffer) >= prebuffer_samples or iterator_finished:
                            has_started = True
                            print("Starting playback (prebuffer ready)...", flush=True)
                        else:
                            outdata.fill(0)
                            return

                    if len(buffer) >= frames:
                        outdata[:] = buffer[:frames].reshape(-1, 1)
                        buffer = buffer[frames:]
                    elif len(buffer) > 0:
                        outdata[:len(buffer)] = buffer.reshape(-1, 1)
                        outdata[len(buffer):] = 0
                        buffer = np.array([], dtype=np.float32)
                    else:
                        outdata.fill(0)

            try:
                with sd.OutputStream(
                    samplerate=self.sample_rate,
                    blocksize=BLOCKSIZE,
                    device=self.device_id,
                    channels=1,
                    dtype='float32',
                    callback=audio_callback
                ):
                    print("Audio stream started...", flush=True)
                    fill_thread.join()

                    while True:
                        with buffer_lock:
                            if len(buffer) == 0 and iterator_finished:
                                break
                        sd.sleep(100)

                    sd.sleep(200)  # Final audio drain

            except Exception as e:
                print(f"Error during audio streaming: {e}", flush=True)
                raise

            print("Playback completed!", flush=True)

        else:
            # Buffered playback
            chunks = []
            print("Collecting audio chunks...", end="", flush=True)
            for audio_chunk in audio_iterator:
                chunks.append(audio_chunk)
                print(".", end="", flush=True)
            print(" Done!")

            if chunks:
                print("Playing audio...")
                full_audio = np.concatenate(chunks)
                sd.play(full_audio, samplerate=self.sample_rate, device=self.device_id)
                sd.wait()

    def stop(self):
        """Stop current playback."""
        if SOUNDDEVICE_AVAILABLE:
            sd.stop()


def list_default_voices() -> list[str]:
    """
    List available default voice prompts.

    Returns:
        List of available voice names (without .pt extension)

    Example:
        >>> from vibevoice import list_default_voices
        >>> voices = list_default_voices()
        >>> print(f"Available voices: {', '.join(voices)}")
    """
    default_voice_dir = Path(__file__).parent.parent / "demo" / "voices" / "streaming_model"
    if not default_voice_dir.exists():
        return []

    voice_files = sorted(default_voice_dir.glob("*.pt"))
    return [v.stem for v in voice_files]


def synthesize_speech(
    text: Iterator[str] | str,
    model_path: str = "microsoft/VibeVoice-Realtime-0.5B",
    voice_prompt_path: Optional[str] = None,
    device: str = "cuda",
    output_file: Optional[str] = None,
    play_audio: bool = True,
    speaker_device_id: Optional[int] = None,
    inference_steps: int = 5,
    cfg_scale: float = 1.5,
    realtime: bool = True,
) -> Optional[np.ndarray]:
    """
    High-level function to synthesize speech from text.

    This is a convenience function that handles model loading, generation,
    and playback in a single call.

    Args:
        text: Text to synthesize (string or iterator of strings)
        model_path: Path to VibeVoice model or HuggingFace model ID
        voice_prompt_path: Optional path to voice prompt for voice cloning.
                          If None, will automatically use a default voice.
        device: Device to run on ('cuda', 'mps', 'cpu')
        output_file: Optional path to save audio to file
        play_audio: If True, play audio through speakers
        speaker_device_id: Speaker device ID (None for default)
        inference_steps: Number of diffusion steps (5=fast, 50=quality)
        cfg_scale: Classifier-free guidance scale (1.0-2.0)
        realtime: If True, use streaming playback mode

    Returns:
        np.ndarray: Generated audio if output_file is specified, else None

    Example:
        >>> # Simple usage
        >>> synthesize_speech("Hello world!")
        >>>
        >>> # Save to file
        >>> synthesize_speech("Hello world!", output_file="output.wav")
        >>>
        >>> # Voice cloning
        >>> synthesize_speech(
        ...     "Hello from my cloned voice",
        ...     voice_prompt_path="voices/speaker.pt"
        ... )
        >>>
        >>> # High quality
        >>> synthesize_speech(
        ...     "High quality speech",
        ...     inference_steps=50,
        ...     cfg_scale=2.0
        ... )
    """
    # Initialize TTS
    print(f"Loading model from {model_path}...")
    tts = VibeVoiceStreamingTTS(
        model_path=model_path,
        voice_prompt_path=voice_prompt_path,
        device=device,
        inference_steps=inference_steps,
    )

    # Simple text generator
    def text_gen():
        if isinstance(text, str):
            yield text
        else:
            for chunk in text:
                yield chunk

    # Generate audio
    print(f"Generating speech for: '{text}'")
    audio_stream = tts.text_to_speech_streaming(text_gen(), cfg_scale=cfg_scale)

    # Collect chunks if we need to save
    if output_file or not play_audio:
        chunks = []
        for chunk in audio_stream:
            chunks.append(chunk)
        full_audio = np.concatenate(chunks)

        if output_file:
            print(f"Saving audio to {output_file}...")
            tts.save_audio(full_audio, output_file)

        if play_audio and SOUNDDEVICE_AVAILABLE:
            print("Playing audio...")
            player = AudioPlayer(device_id=speaker_device_id)
            sd.play(full_audio, samplerate=tts.sample_rate, device=speaker_device_id)
            sd.wait()

        return full_audio if output_file else None

    # Stream and play directly
    if play_audio:
        if not SOUNDDEVICE_AVAILABLE:
            print("Warning: sounddevice not available, cannot play audio")
            return None

        print("Playing audio in real-time...")
        player = AudioPlayer(device_id=speaker_device_id)
        player.play_stream(audio_stream, realtime=realtime)
    full_audio = [chunk for chunk in audio_stream]

    return full_audio if output_file else None
