#!/usr/bin/env python3
"""
CPU-based inference for VibeVoice
For systems without CUDA/GPU support
"""

import argparse
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor


class VibeVoiceCPUInference:
    """CPU-optimized inference for VibeVoice"""
    
    def __init__(self, model_path: str = "microsoft/VibeVoice-1.5B"):
        """
        Initialize VibeVoice for CPU inference
        
        Args:
            model_path: Path to model or HuggingFace model ID
        """
        self.model_path = model_path
        self.device = "cpu"
        self.sample_rate = 24000
        
        print(f"Initializing VibeVoice for CPU inference...")
        self._load_model()
        
    def _load_model(self):
        """Load model and processor"""
        print(f"Loading model from {self.model_path}...")
        
        # Load processor
        self.processor = VibeVoiceProcessor.from_pretrained(self.model_path)
        
        # Load model with CPU optimization
        self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,  # Use float32 for CPU
            attn_implementation="eager",  # Most compatible attention
            low_cpu_mem_usage=True,
        )
        
        self.model.eval()
        
        # Configure for faster inference (trade quality for speed)
        self.model.set_ddpm_inference_steps(num_steps=10)
        
        print("✓ Model loaded successfully")
    
    def load_voice_sample(self, voice_path: str) -> Optional[np.ndarray]:
        """
        Load and preprocess voice sample
        
        Args:
            voice_path: Path to voice audio file
            
        Returns:
            Preprocessed audio array or None if loading fails
        """
        try:
            # Load audio with librosa
            audio_data, sr = librosa.load(voice_path, sr=self.sample_rate, mono=True)
            
            # Normalize
            if audio_data.max() > 0:
                audio_data = audio_data / np.abs(audio_data).max()
            
            duration = len(audio_data) / sr
            print(f"✓ Loaded voice sample: {duration:.2f}s at {sr}Hz")
            return audio_data
            
        except Exception as e:
            print(f"Warning: Failed to load voice sample: {e}")
            return None
    
    def find_default_voice(self) -> Optional[str]:
        """Find a default voice file from the demo voices directory"""
        voices_dir = Path(__file__).parent / "voices"
        
        if not voices_dir.exists():
            return None
        
        # Prefer English voices for better compatibility
        preferred_patterns = ["en-*.wav", "*.wav"]
        
        for pattern in preferred_patterns:
            voices = list(voices_dir.glob(pattern))
            if voices:
                return str(voices[0])
        
        return None
    
    @torch.no_grad()
    def generate(
        self,
        text: str,
        voice_path: Optional[str] = None,
        max_new_tokens: int = 1000,
        cfg_scale: float = 1.5,
        show_progress: bool = True
    ) -> Tuple[Optional[np.ndarray], int]:
        """
        Generate speech from text
        
        Args:
            text: Input text to synthesize
            voice_path: Optional path to voice sample for voice cloning
            max_new_tokens: Maximum number of tokens to generate
            cfg_scale: Classifier-free guidance scale
            show_progress: Show progress bar during generation
            
        Returns:
            Tuple of (audio_array, sample_rate) or (None, 0) if generation fails
        """
        print("\n" + "="*50)
        print("Generating speech...")
        print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        
        # Handle voice sample
        voice_samples = None
        if voice_path:
            if os.path.exists(voice_path):
                voice_samples = self.load_voice_sample(voice_path)
                if voice_samples is not None:
                    print(f"Using voice: {Path(voice_path).name}")
            else:
                print(f"Warning: Voice file not found: {voice_path}")
        
        if voice_samples is None and voice_path is None:
            # Try to find a default voice
            default_voice = self.find_default_voice()
            if default_voice:
                voice_samples = self.load_voice_sample(default_voice)
                if voice_samples is not None:
                    print(f"Using default voice: {Path(default_voice).name}")
        
        # Prepare inputs
        print("Processing input...")
        
        # Format text with speaker tag
        formatted_text = f"Speaker 1: {text}"
        
        # Process with or without voice
        if voice_samples is not None:
            inputs = self.processor(
                text=[formatted_text],
                voice_samples=[voice_samples],  # Correct parameter name
                return_tensors="pt"
            )
        else:
            print("Note: Generating without voice sample (may have lower quality)")
            inputs = self.processor(
                text=[formatted_text],
                voice_samples=None,
                return_tensors="pt"
            )
            
            # Add placeholder tensors if needed
            if inputs.get('speech_tensors') is None:
                batch_size = inputs['input_ids'].shape[0]
                inputs['speech_tensors'] = torch.zeros(batch_size, 1, 1)
                inputs['speech_masks'] = torch.zeros(batch_size, 1, dtype=torch.bool)
                if 'speech_input_mask' not in inputs:
                    inputs['speech_input_mask'] = torch.zeros_like(
                        inputs['input_ids'], dtype=torch.bool
                    )
        
        print(f"Input tokens: {inputs['input_ids'].shape[-1]}")
        
        # Generate
        print("Running inference (this may take a few minutes on CPU)...")
        start_time = time.time()
        
        try:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                cfg_scale=cfg_scale,
                tokenizer=self.processor.tokenizer,
                generation_config={'do_sample': False},
                show_progress_bar=show_progress,
                verbose=False,
            )
            
            generation_time = time.time() - start_time
            
            # Extract audio
            if outputs.speech_outputs and len(outputs.speech_outputs) > 0:
                audio = outputs.speech_outputs[0]
                
                if isinstance(audio, torch.Tensor):
                    audio = audio.cpu().numpy()
                
                # Calculate statistics
                duration = len(audio) / self.sample_rate
                rtf = generation_time / duration if duration > 0 else 0
                
                print(f"\n✓ Generation completed in {generation_time:.1f}s")
                print(f"  Audio duration: {duration:.2f}s")
                print(f"  Real-time factor: {rtf:.1f}x")
                print("="*50)
                
                return audio, self.sample_rate
            else:
                print("Error: No audio output generated")
                return None, 0
                
        except Exception as e:
            print(f"Error during generation: {e}")
            import traceback
            traceback.print_exc()
            return None, 0
    
    def save_audio(
        self,
        audio: np.ndarray,
        output_path: str,
        sample_rate: int = 24000
    ) -> bool:
        """
        Save audio to file
        
        Args:
            audio: Audio array to save
            output_path: Output file path
            sample_rate: Audio sample rate
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure numpy array
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
            
            # Ensure 1D
            audio = audio.squeeze()
            
            # Normalize
            if audio.max() > 0:
                audio = audio / np.abs(audio).max() * 0.95
            audio = np.clip(audio, -1.0, 1.0)
            
            # Save
            sf.write(output_path, audio, sample_rate)
            print(f"✓ Audio saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error saving audio: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="CPU-based VibeVoice inference for non-CUDA systems"
    )
    parser.add_argument(
        "--text", 
        type=str,
        required=True,
        help="Text to synthesize"
    )
    parser.add_argument(
        "--voice", 
        type=str,
        default=None,
        help="Path to voice sample WAV file (optional)"
    )
    parser.add_argument(
        "--output", 
        type=str,
        default="output.wav",
        help="Output audio file path (default: output.wav)"
    )
    parser.add_argument(
        "--model_path", 
        type=str,
        default="microsoft/VibeVoice-1.5B",
        help="Model path or HuggingFace ID (default: microsoft/VibeVoice-1.5B)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1000,
        help="Maximum tokens to generate (default: 1000)"
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=1.5,
        help="Classifier-free guidance scale (default: 1.5)"
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable progress bar"
    )
    
    args = parser.parse_args()
    
    # Initialize model
    model = VibeVoiceCPUInference(model_path=args.model_path)
    
    # Generate speech
    audio, sample_rate = model.generate(
        text=args.text,
        voice_path=args.voice,
        max_new_tokens=args.max_tokens,
        cfg_scale=args.cfg_scale,
        show_progress=not args.no_progress
    )
    
    # Save output
    if audio is not None:
        success = model.save_audio(audio, args.output, sample_rate)
        if success:
            print(f"\n✨ Success! Play the audio with:")
            print(f"  {'afplay' if os.uname().sysname == 'Darwin' else 'aplay'} {args.output}")
            return 0
    
    return 1


if __name__ == "__main__":
    exit(main())