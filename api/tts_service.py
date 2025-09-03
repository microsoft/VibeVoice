import torch
import soundfile as sf
import librosa
import numpy as np
import os
from typing import List, Dict, Any, AsyncGenerator

# Import VibeVoice classes
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.streamer import AsyncAudioStreamer # Using the async streamer
from transformers import set_seed

class TTSService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(TTSService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path: str = "microsoft/VibeVoice-1.5B", inference_steps: int = 20, device: str = "cuda"):
        if self._initialized:
            return

        print("Initializing TTS Service...")
        self.model_path = model_path
        self.inference_steps = inference_steps
        self.device = device
        self.model = None
        self.processor = None
        self.voice_presets = {}

        self.load_model()
        self.setup_voice_presets()

        self._initialized = True
        print("TTS Service initialized.")

    def load_model(self):
        # Adapted from Gradio demo
        print(f"Loading processor & model from {self.model_path}")
        self.processor = VibeVoiceProcessor.from_pretrained(self.model_path)

        # Use flash_attention_2 if available, otherwise sdpa
        try:
            print("Attempting to load model with flash_attention_2...")
            self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                attn_implementation='flash_attention_2'
            )
            print("Model loaded with flash_attention_2.")
        except Exception as e:
            print(f"Failed to load with flash_attention_2: {e}")
            print("Falling back to sdpa. Performance may be slower.")
            self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                attn_implementation='sdpa'
            )
            print("Model loaded with sdpa.")

        self.model.eval()
        # Set noise scheduler and inference steps
        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config,
            algorithm_type='sde-dpmsolver++',
            beta_schedule='squaredcos_cap_v2'
        )
        self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)
        print(f"Model loaded successfully. Inference steps set to {self.inference_steps}.")

    def setup_voice_presets(self):
        # Adapted from Gradio demo
        voices_dir = "demo/voices"
        if not os.path.exists(voices_dir):
            print(f"Warning: Voices directory not found at {voices_dir}")
            return

        for f in os.listdir(voices_dir):
            if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac')):
                name = os.path.splitext(f)[0]
                self.voice_presets[name] = os.path.join(voices_dir, f)

        print(f"Loaded {len(self.voice_presets)} voice presets.")

    def _read_audio(self, audio_path: str, target_sr: int = 24000) -> np.ndarray:
        # Adapted from Gradio demo
        wav, sr = sf.read(audio_path)
        if len(wav.shape) > 1:
            wav = np.mean(wav, axis=1)
        if sr != target_sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        return wav

    def _prepare_inputs(self, script: str, speaker_voices: List[str]):
        # Load voice samples
        voice_samples = []
        for voice_name in speaker_voices:
            if voice_name not in self.voice_presets:
                # Attempt to find a partial match
                found = False
                for key in self.voice_presets:
                    if voice_name.lower() in key.lower():
                        voice_samples.append(self._read_audio(self.voice_presets[key]))
                        found = True
                        break
                if not found:
                    raise ValueError(f"Voice preset '{voice_name}' not found.")
            else:
                audio_data = self._read_audio(self.voice_presets[voice_name])
                voice_samples.append(audio_data)

        # Prepare model inputs
        inputs = self.processor(
            text=[script],
            voice_samples=[voice_samples],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        return inputs.to(self.device)

    async def generate_stream_async(self, script: str, speaker_voices: List[str], cfg_scale: float = 1.3) -> AsyncGenerator[bytes, None]:
        inputs = self._prepare_inputs(script, speaker_voices)

        # Use the AsyncAudioStreamer
        streamer = AsyncAudioStreamer(batch_size=1)

        # Run generation in a separate thread
        import threading
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=None,
            cfg_scale=cfg_scale,
            tokenizer=self.processor.tokenizer,
            generation_config={'do_sample': False},
            audio_streamer=streamer,
            verbose=False,
            refresh_negative=True,
        )
        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # Yield audio chunks from the async streamer
        async for audio_chunk in streamer.get_stream(0):
            # Convert tensor to 16-bit PCM bytes
            audio_np = audio_chunk.float().cpu().numpy().astype(np.float32)
            if np.max(np.abs(audio_np)) > 1.0:
                audio_np = audio_np / np.max(np.abs(audio_np))
            audio_16bit = (audio_np * 32767).astype(np.int16)
            yield audio_16bit.tobytes()

        thread.join() # Ensure thread is cleaned up

    def generate_batch(self, script: str, speaker_voices: List[str], cfg_scale: float = 1.3) -> (np.ndarray, int):
        inputs = self._prepare_inputs(script, speaker_voices)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=cfg_scale,
            tokenizer=self.processor.tokenizer,
            generation_config={'do_sample': False},
            verbose=True,
        )

        sample_rate = 24000
        audio_output = outputs.speech_outputs[0]
        # Convert tensor to numpy array
        audio_np = audio_output.float().cpu().numpy()
        return audio_np, sample_rate

# Create a function to get the service instance, to be used with FastAPI's dependency injection
def get_tts_service():
    # This ensures the service is created only once
    return TTSService(
        model_path="microsoft/VibeVoice-1.5B",
        inference_steps=20
    )
