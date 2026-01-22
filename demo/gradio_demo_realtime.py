"""
VibeVoice Realtime Gradio Demo - Streaming TTS Interface for VibeVoice-Realtime-0.5B
"""

import argparse
import os
import sys
import time
import copy
from pathlib import Path
from typing import List, Dict, Any, Iterator
import threading
import numpy as np
import gradio as gr
import torch
import traceback
import re

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
from transformers.utils import logging
from transformers import set_seed

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


class VibeVoiceRealtimeDemo:
    def __init__(self, model_path: str, device: str = "mps", inference_steps: int = 5):
        """Initialize the VibeVoice Realtime demo with model loading."""
        self.model_path = model_path
        self.device = device
        self.inference_steps = inference_steps
        self.is_generating = False
        self.stop_generation = False
        self.load_model()
        self.setup_voice_presets()
        self.load_example_scripts()

    def load_model(self):
        """Load the VibeVoice Streaming model and processor."""
        print(f"Loading processor & model from {self.model_path}")
        
        # Normalize device
        if self.device.lower() == "mpx":
            print("Note: device 'mpx' detected, treating it as 'mps'.")
            self.device = "mps"
        if self.device == "mps" and not torch.backends.mps.is_available():
            print("Warning: MPS not available. Falling back to CPU.")
            self.device = "cpu"
        print(f"Using device: {self.device}")
        
        # Load processor
        self.processor = VibeVoiceStreamingProcessor.from_pretrained(self.model_path)
        
        # Decide dtype & attention
        if self.device == "mps":
            load_dtype = torch.float32
            attn_impl = "sdpa"
        elif self.device == "cuda":
            load_dtype = torch.bfloat16
            attn_impl = "flash_attention_2"
        else:
            load_dtype = torch.float32
            attn_impl = "sdpa"
        
        print(f"Using device: {self.device}, torch_dtype: {load_dtype}, attn_implementation: {attn_impl}")
        
        # Load model
        try:
            if self.device == "mps":
                self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    attn_implementation=attn_impl,
                    device_map=None,
                )
                self.model.to("mps")
            elif self.device == "cuda":
                self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map="cuda",
                    attn_implementation=attn_impl,
                )
            else:
                self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map="cpu",
                    attn_implementation=attn_impl,
                )
        except Exception as e:
            if attn_impl == 'flash_attention_2':
                print(f"[ERROR]: {type(e).__name__}: {e}")
                print("Falling back to SDPA attention...")
                self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map=(self.device if self.device in ("cuda", "cpu") else None),
                    attn_implementation='sdpa'
                )
                if self.device == "mps":
                    self.model.to("mps")
            else:
                raise e
        
        self.model.eval()
        self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)
        
        if hasattr(self.model.model, 'language_model'):
            print(f"Language model attention: {self.model.model.language_model.config._attn_implementation}")

    def setup_voice_presets(self):
        """Setup voice presets by scanning the voices/streaming_model directory."""
        voices_dir = os.path.join(os.path.dirname(__file__), "voices/streaming_model")
        
        if not os.path.exists(voices_dir):
            print(f"Warning: Voices directory not found at {voices_dir}")
            self.voice_presets = {}
            self.available_voices = {}
            return
        
        self.voice_presets = {}
        
        # Get all .pt files (voice embeddings for streaming model)
        pt_files = [f for f in os.listdir(voices_dir) 
                    if f.lower().endswith('.pt') and os.path.isfile(os.path.join(voices_dir, f))]
        
        for pt_file in pt_files:
            name = os.path.splitext(pt_file)[0]
            full_path = os.path.join(voices_dir, pt_file)
            self.voice_presets[name] = full_path
        
        self.voice_presets = dict(sorted(self.voice_presets.items()))
        self.available_voices = {
            name: path for name, path in self.voice_presets.items()
            if os.path.exists(path)
        }
        
        if not self.available_voices:
            raise gr.Error("No voice presets found. Please add .pt files to demo/voices/streaming_model/")
        
        print(f"Found {len(self.available_voices)} voice files in {voices_dir}")
        print(f"Available voices: {', '.join(self.available_voices.keys())}")

    def load_example_scripts(self):
        """Load example scripts from the text_examples directory."""
        examples_dir = os.path.join(os.path.dirname(__file__), "text_examples")
        self.example_scripts = []
        
        if not os.path.exists(examples_dir):
            print(f"Warning: text_examples directory not found at {examples_dir}")
            return
        
        txt_files = sorted([f for f in os.listdir(examples_dir) 
                          if f.lower().endswith('.txt') and f.startswith('1p_')])
        
        for txt_file in txt_files:
            file_path = os.path.join(examples_dir, txt_file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    script_content = f.read().strip()
                
                script_content = '\n'.join(line for line in script_content.split('\n') if line.strip())
                
                if script_content:
                    self.example_scripts.append(script_content)
                    print(f"Loaded example: {txt_file}")
            except Exception as e:
                print(f"Error loading {txt_file}: {e}")
        
        print(f"Successfully loaded {len(self.example_scripts)} example scripts")

    def generate_speech(self, text: str, voice: str, cfg_scale: float = 1.5) -> Iterator[tuple]:
        """Generate speech from text using the selected voice."""
        try:
            self.stop_generation = False
            self.is_generating = True
            
            if not text.strip():
                self.is_generating = False
                raise gr.Error("Please provide text to synthesize.")
            
            if not voice or voice not in self.available_voices:
                self.is_generating = False
                raise gr.Error("Please select a valid voice.")
            
            # Clean text
            text = text.replace("'", "'").replace('"', '"').replace('"', '"')
            
            log = f"🎙️ Generating speech...\n"
            log += f"📊 Parameters: CFG Scale={cfg_scale}, Inference Steps={self.inference_steps}\n"
            log += f"🎭 Voice: {voice}\n"
            
            yield None, log
            
            if self.stop_generation:
                self.is_generating = False
                yield None, "🛑 Generation stopped by user"
                return
            
            # Load voice embedding
            target_device = self.device if self.device in ("cuda", "mps") else "cpu"
            voice_path = self.available_voices[voice]
            all_prefilled_outputs = torch.load(voice_path, map_location=target_device, weights_only=False)
            
            log += "✅ Loaded voice embedding\n"
            yield None, log
            
            # Prepare inputs
            inputs = self.processor.process_input_with_cached_prompt(
                text=text,
                cached_prompt=all_prefilled_outputs,
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            
            # Move tensors to device
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(target_device)
            
            log += "🔄 Processing with VibeVoice Realtime...\n"
            yield None, log
            
            if self.stop_generation:
                self.is_generating = False
                yield None, "🛑 Generation stopped by user"
                return
            
            start_time = time.time()
            
            # Generate audio
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=self.processor.tokenizer,
                generation_config={'do_sample': False},
                verbose=False,
                all_prefilled_outputs=copy.deepcopy(all_prefilled_outputs),
            )
            
            generation_time = time.time() - start_time
            
            self.is_generating = False
            
            if self.stop_generation:
                yield None, "🛑 Generation stopped by user"
                return
            
            # Process output
            if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
                audio_data = outputs.speech_outputs[0]
                
                # Convert to numpy
                if torch.is_tensor(audio_data):
                    if audio_data.dtype == torch.bfloat16:
                        audio_data = audio_data.float()
                    audio_np = audio_data.cpu().numpy().astype(np.float32)
                else:
                    audio_np = np.array(audio_data, dtype=np.float32)
                
                # Ensure 1D
                if len(audio_np.shape) > 1:
                    audio_np = audio_np.squeeze()
                
                # Normalize
                if np.max(np.abs(audio_np)) > 1.0:
                    audio_np = audio_np / np.max(np.abs(audio_np))
                
                sample_rate = 24000
                audio_duration = len(audio_np) / sample_rate
                rtf = generation_time / audio_duration if audio_duration > 0 else float('inf')
                
                final_log = log + f"⏱️ Generation completed in {generation_time:.2f} seconds\n"
                final_log += f"🎵 Audio duration: {audio_duration:.2f} seconds\n"
                final_log += f"📊 RTF (Real Time Factor): {rtf:.2f}x\n"
                final_log += "✨ Generation successful!"
                
                yield (sample_rate, audio_np), final_log
            else:
                yield None, log + "❌ No audio was generated."
        
        except gr.Error as e:
            self.is_generating = False
            yield None, f"❌ Error: {str(e)}"
        except Exception as e:
            self.is_generating = False
            error_msg = f"❌ Unexpected error: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            yield None, error_msg

    def stop_audio_generation(self):
        """Stop the current audio generation."""
        self.stop_generation = True
        print("🛑 Audio generation stop requested")


def create_demo_interface(demo_instance: VibeVoiceRealtimeDemo):
    """Create the Gradio interface."""
    
    custom_css = """
    .gradio-container {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-header {
        background: linear-gradient(90deg, #10b981 0%, #059669 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(16, 185, 129, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
    }
    
    .generate-btn {
        background: linear-gradient(135deg, #059669 0%, #0d9488 100%);
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 20px rgba(5, 150, 105, 0.4);
    }
    
    .stop-btn {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        color: white;
        font-weight: 600;
    }
    """
    
    with gr.Blocks(
        title="VibeVoice Realtime - TTS Demo",
        css=custom_css,
        theme=gr.themes.Soft(
            primary_hue="green",
            secondary_hue="teal",
            neutral_hue="slate",
        )
    ) as interface:
        
        gr.HTML("""
        <div class="main-header">
            <h1>🎙️ VibeVoice Realtime</h1>
            <p>Real-time Text-to-Speech with Streaming Support (~300ms latency)</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎛️ **Settings**")
                
                voice_dropdown = gr.Dropdown(
                    choices=list(demo_instance.available_voices.keys()),
                    value=list(demo_instance.available_voices.keys())[0] if demo_instance.available_voices else None,
                    label="Voice",
                )
                
                cfg_scale = gr.Slider(
                    minimum=1.0,
                    maximum=2.0,
                    value=1.5,
                    step=0.1,
                    label="CFG Scale",
                )
            
            with gr.Column(scale=2):
                gr.Markdown("### 📝 **Text Input**")
                
                text_input = gr.Textbox(
                    label="Text to Synthesize",
                    placeholder="Enter text here...",
                    lines=6,
                    max_lines=12,
                )
                
                with gr.Row():
                    random_btn = gr.Button("🎲 Random Example", variant="secondary")
                    generate_btn = gr.Button("🚀 Generate Speech", variant="primary", elem_classes="generate-btn")
                
                stop_btn = gr.Button("🛑 Stop", variant="stop", elem_classes="stop-btn", visible=False)
                
                gr.Markdown("### 🎵 **Generated Audio**")
                
                audio_output = gr.Audio(
                    label="Output Audio",
                    type="numpy",
                    autoplay=True,
                )
                
                log_output = gr.Textbox(
                    label="Generation Log",
                    lines=6,
                    interactive=False,
                )
        
        def generate_wrapper(text, voice, cfg):
            yield gr.update(visible=False), gr.update(visible=True), None, "Starting..."
            
            for audio, log in demo_instance.generate_speech(text, voice, cfg):
                if audio is not None:
                    yield gr.update(visible=True), gr.update(visible=False), audio, log
                else:
                    yield gr.update(visible=True), gr.update(visible=False), None, log
        
        def stop_handler():
            demo_instance.stop_audio_generation()
            return "🛑 Stopped", gr.update(visible=True), gr.update(visible=False)
        
        def load_random_example():
            import random
            if demo_instance.example_scripts:
                return random.choice(demo_instance.example_scripts)
            return "Hello, this is a test of the VibeVoice realtime text-to-speech system."
        
        generate_btn.click(
            fn=generate_wrapper,
            inputs=[text_input, voice_dropdown, cfg_scale],
            outputs=[generate_btn, stop_btn, audio_output, log_output],
        )
        
        stop_btn.click(
            fn=stop_handler,
            inputs=[],
            outputs=[log_output, generate_btn, stop_btn],
        )
        
        random_btn.click(
            fn=load_random_example,
            inputs=[],
            outputs=[text_input],
        )
        
        # Examples
        if demo_instance.example_scripts:
            gr.Examples(
                examples=[[ex] for ex in demo_instance.example_scripts[:5]],
                inputs=[text_input],
                label="Example Scripts"
            )
        
        gr.Markdown("""
        ### 💡 **Tips**
        - This model is optimized for **single-speaker** real-time TTS
        - First audio chunk generated in ~300ms (hardware dependent)
        - Supports English text; other languages may produce unexpected results
        - CFG Scale controls guidance strength (1.0-2.0)
        """)
    
    return interface


def parse_args():
    parser = argparse.ArgumentParser(description="VibeVoice Realtime Gradio Demo")
    model_path_default = os.environ.get("VIBEVOICE_MODEL_PATH", "models/VibeVoice-Realtime-0.5B")
    device_default = os.environ.get("VIBEVOICE_DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
    inference_steps_default = int(os.environ.get("VIBEVOICE_INFERENCE_STEPS", "5"))
    share_default = os.environ.get("VIBEVOICE_SHARE", "").lower() in ("1", "true", "yes")
    port_default = int(os.environ.get("VIBEVOICE_PORT", "7860"))
    
    parser.add_argument("--model_path", type=str, default=model_path_default)
    parser.add_argument("--device", type=str, default=device_default)
    parser.add_argument("--inference_steps", type=int, default=inference_steps_default)
    parser.add_argument("--share", action="store_true", default=share_default)
    parser.add_argument("--port", type=int, default=port_default)
    
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)
    
    print("🎙️ Initializing VibeVoice Realtime Demo...")
    
    demo_instance = VibeVoiceRealtimeDemo(
        model_path=args.model_path,
        device=args.device,
        inference_steps=args.inference_steps
    )
    
    interface = create_demo_interface(demo_instance)
    
    print(f"🚀 Launching demo on port {args.port}")
    print(f"📁 Model path: {args.model_path}")
    print(f"🎭 Available voices: {len(demo_instance.available_voices)}")
    
    try:
        interface.queue(max_size=10, default_concurrency_limit=1).launch(
            share=args.share,
            server_port=args.port,
            server_name="0.0.0.0" if args.share else "127.0.0.1",
            show_error=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    except Exception as e:
        print(f"❌ Server error: {e}")
        raise


if __name__ == "__main__":
    main()
