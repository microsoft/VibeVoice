import argparse
import os
import re
import traceback
from typing import List, Tuple, Union, Dict, Any
import time
import torch

from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


class VoiceMapper:
    """Maps speaker names to voice file paths"""
    
    def __init__(self):
        self.setup_voice_presets()

        # change name according to our preset wav file
        new_dict = {}
        for name, path in self.voice_presets.items():
            
            if '_' in name:
                name = name.split('_')[0]
            
            if '-' in name:
                name = name.split('-')[-1]

            new_dict[name] = path
        self.voice_presets.update(new_dict)
        # print(list(self.voice_presets.keys()))

    def setup_voice_presets(self):
        """Setup voice presets by scanning the voices directory."""
        voices_dir = os.path.join(os.path.dirname(__file__), "voices")
        
        # Check if voices directory exists
        if not os.path.exists(voices_dir):
            print(f"Warning: Voices directory not found at {voices_dir}")
            self.voice_presets = {}
            self.available_voices = {}
            return
        
        # Scan for all WAV files in the voices directory
        self.voice_presets = {}
        
        # Get all .wav files in the voices directory
        wav_files = [f for f in os.listdir(voices_dir) 
                    if f.lower().endswith('.wav') and os.path.isfile(os.path.join(voices_dir, f))]
        
        # Create dictionary with filename (without extension) as key
        for wav_file in wav_files:
            # Remove .wav extension to get the name
            name = os.path.splitext(wav_file)[0]
            # Create full path
            full_path = os.path.join(voices_dir, wav_file)
            self.voice_presets[name] = full_path
        
        # Sort the voice presets alphabetically by name for better UI
        self.voice_presets = dict(sorted(self.voice_presets.items()))
        
        # Filter out voices that don't exist (this is now redundant but kept for safety)
        self.available_voices = {
            name: path for name, path in self.voice_presets.items()
            if os.path.exists(path)
        }
        
        print(f"Found {len(self.available_voices)} voice files in {voices_dir}")
        print(f"Available voices: {', '.join(self.available_voices.keys())}")

    def get_voice_path(self, speaker_name: str) -> str:
        """Get voice file path for a given speaker name"""
        # First try exact match
        if speaker_name in self.voice_presets:
            return self.voice_presets[speaker_name]
        
        # Try partial matching (case insensitive)
        speaker_lower = speaker_name.lower()
        for preset_name, path in self.voice_presets.items():
            if preset_name.lower() in speaker_lower or speaker_lower in preset_name.lower():
                return path
        
        # Default to first voice if no match found
        default_voice = list(self.voice_presets.values())[0]
        print(f"Warning: No voice preset found for '{speaker_name}', using default voice: {default_voice}")
        return default_voice


def parse_txt_script(txt_content: str) -> Tuple[List[str], List[str]]:
    """
    Parse txt script content and extract speakers and their text
    Fixed pattern: Speaker 1, Speaker 2, Speaker 3, Speaker 4
    Returns: (scripts, speaker_numbers)
    """
    lines = txt_content.strip().split('\n')
    scripts = []
    speaker_numbers = []
    
    # Pattern to match "Speaker X:" format where X is a number
    speaker_pattern = r'^Speaker\s+(\d+):\s*(.*)$'
    
    current_speaker = None
    current_text = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        match = re.match(speaker_pattern, line, re.IGNORECASE)
        if match:
            # If we have accumulated text from previous speaker, save it
            if current_speaker and current_text:
                scripts.append(f"Speaker {current_speaker}: {current_text.strip()}")
                speaker_numbers.append(current_speaker)
            
            # Start new speaker
            current_speaker = match.group(1).strip()
            current_text = match.group(2).strip()
        else:
            # Continue text for current speaker
            if current_text:
                current_text += " " + line
            else:
                current_text = line
    
    # Don't forget the last speaker
    if current_speaker and current_text:
        scripts.append(f"Speaker {current_speaker}: {current_text.strip()}")
        speaker_numbers.append(current_speaker)
    
    return scripts, speaker_numbers


def parse_args():
    parser = argparse.ArgumentParser(description="VibeVoice Processor TXT Input Test")
    parser.add_argument(
        "--model_path",
        type=str,
        default="microsoft/VibeVoice-1.5b",
        help="Path to the HuggingFace model directory",
    )
    
    parser.add_argument(
        "--txt_path",
        type=str,
        default="demo/text_examples/1p_abs.txt",
        help="Path to the txt file containing the script",
    )
    parser.add_argument(
        "--speaker_names",
        type=str,
        nargs='+',
        default='Andrew',
        help="Speaker names in order (e.g., --speaker_names Andrew Ava 'Bill Gates')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save output audio files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")),
        help="Device for inference: cuda | mps | cpu",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=1.3,
        help="CFG (Classifier-Free Guidance) scale for generation (default: 1.3)",
    )
    
    return parser.parse_args()

def main():
    args = parse_args()

    # Normalize potential 'mpx' typo to 'mps'
    if args.device.lower() == "mpx":
        print("Note: device 'mpx' detected, treating it as 'mps'.")
        args.device = "mps"

    # Validate mps availability if requested
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS not available. Falling back to CPU.")
        args.device = "cpu"

    print(f"Using device: {args.device}")

    # Initialize voice mapper
    voice_mapper = VoiceMapper()
    
    # Check if txt file exists
    if not os.path.exists(args.txt_path):
        print(f"Error: txt file not found: {args.txt_path}")
        return
    
    # Read and parse txt file
    print(f"Reading script from: {args.txt_path}")
    with open(args.txt_path, 'r', encoding='utf-8') as f:
        txt_content = f.read()
    
    # Parse the txt content to get speaker numbers
    scripts, speaker_numbers = parse_txt_script(txt_content)
    
    if not scripts:
        print("Error: No valid speaker scripts found in the txt file")
        return
    
    print(f"Found {len(scripts)} speaker segments:")
    for i, (script, speaker_num) in enumerate(zip(scripts, speaker_numbers)):
        print(f"  {i+1}. Speaker {speaker_num}")
        print(f"     Text preview: {script[:100]}...")
    
    # Map speaker numbers to provided speaker names
    speaker_name_mapping = {}
    speaker_names_list = args.speaker_names if isinstance(args.speaker_names, list) else [args.speaker_names]
    for i, name in enumerate(speaker_names_list, 1):
        speaker_name_mapping[str(i)] = name
    
    print(f"\nSpeaker mapping:")
    for speaker_num in set(speaker_numbers):
        mapped_name = speaker_name_mapping.get(speaker_num, f"Speaker {speaker_num}")
        print(f"  Speaker {speaker_num} -> {mapped_name}")
    
    # Map speakers to voice files using the provided speaker names
    voice_samples = []
    actual_speakers = []
    
    # Get unique speaker numbers in order of first appearance
    unique_speaker_numbers = []
    seen = set()
    for speaker_num in speaker_numbers:
        if speaker_num not in seen:
            unique_speaker_numbers.append(speaker_num)
            seen.add(speaker_num)
    
    for speaker_num in unique_speaker_numbers:
        speaker_name = speaker_name_mapping.get(speaker_num, f"Speaker {speaker_num}")
        voice_path = voice_mapper.get_voice_path(speaker_name)
        voice_samples.append(voice_path)
        actual_speakers.append(speaker_name)
        print(f"Speaker {speaker_num} ('{speaker_name}') -> Voice: {os.path.basename(voice_path)}")
    
    # Prepare data for model
    full_script = '\n'.join(scripts)
    full_script = full_script.replace("’", "'")        
    
    print(f"Loading processor & model from {args.model_path}")
    processor = VibeVoiceProcessor.from_pretrained(args.model_path)


    # Decide dtype & attention implementation
    if args.device == "mps":
        load_dtype = torch.float32  # MPS requires float32
        attn_impl_primary = "sdpa"  # flash_attention_2 not supported on MPS
    elif args.device == "cuda":
        load_dtype = torch.bfloat16
        attn_impl_primary = "flash_attention_2"
    else:  # cpu
        load_dtype = torch.float32
        attn_impl_primary = "sdpa"
    print(f"Using device: {args.device}, torch_dtype: {load_dtype}, attn_implementation: {attn_impl_primary}")
    # Load model with device-specific logic
    try:
        if args.device == "mps":
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                args.model_path,
                torch_dtype=load_dtype,
                attn_implementation=attn_impl_primary,
                device_map=None,  # load then move
            )
            model.to("mps")
        elif args.device == "cuda":
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                args.model_path,
                torch_dtype=load_dtype,
                device_map="cuda",
                attn_implementation=attn_impl_primary,
            )
        else:  # cpu
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                args.model_path,
                torch_dtype=load_dtype,
                device_map="cpu",
                attn_implementation=attn_impl_primary,
            )
    except Exception as e:
        if attn_impl_primary == 'flash_attention_2':
            print(f"[ERROR] : {type(e).__name__}: {e}")
            print(traceback.format_exc())
            print("Error loading the model. Trying to use SDPA. However, note that only flash_attention_2 has been fully tested, and using SDPA may result in lower audio quality.")
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                args.model_path,
                torch_dtype=load_dtype,
                device_map=(args.device if args.device in ("cuda", "cpu") else None),
                attn_implementation='sdpa'
            )
            if args.device == "mps":
                model.to("mps")
        else:
            raise e


    model.eval()
    model.set_ddpm_inference_steps(num_steps=10)

    if hasattr(model.model, 'language_model'):
       print(f"Language model attention: {model.model.language_model.config._attn_implementation}")
       
    # Prepare inputs for the model
    inputs = processor(
        text=[full_script], # Wrap in list for batch processing
        voice_samples=[voice_samples], # Wrap in list for batch processing
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    # Move tensors to target device
    target_device = args.device if args.device != "cpu" else "cpu"
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(target_device)

    print(f"Starting generation with cfg_scale: {args.cfg_scale}")

    # Generate audio
    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=None,
        cfg_scale=args.cfg_scale,
        tokenizer=processor.tokenizer,
        generation_config={'do_sample': False},
        verbose=True,
    )
    generation_time = time.time() - start_time
    print(f"Generation time: {generation_time:.2f} seconds")
    
    # Calculate audio duration and additional metrics
    if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
        # Assuming 24kHz sample rate (common for speech synthesis)
        sample_rate = 24000
        audio_samples = outputs.speech_outputs[0].shape[-1] if len(outputs.speech_outputs[0].shape) > 0 else len(outputs.speech_outputs[0])
        audio_duration = audio_samples / sample_rate
        rtf = generation_time / audio_duration if audio_duration > 0 else float('inf')
        
        print(f"Generated audio duration: {audio_duration:.2f} seconds")
        print(f"RTF (Real Time Factor): {rtf:.2f}x")
    else:
        print("No audio output generated")
    
    # Calculate token metrics
    input_tokens = inputs['input_ids'].shape[1]  # Number of input tokens
    output_tokens = outputs.sequences.shape[1]  # Total tokens (input + generated)
    generated_tokens = output_tokens - input_tokens
    
    print(f"Prefilling tokens: {input_tokens}")
    print(f"Generated tokens: {generated_tokens}")
    print(f"Total tokens: {output_tokens}")

    # Save output (processor handles device internally)
    txt_filename = os.path.splitext(os.path.basename(args.txt_path))[0]
    output_path = os.path.join(args.output_dir, f"{txt_filename}_generated.wav")
    os.makedirs(args.output_dir, exist_ok=True)
    
    processor.save_audio(
        outputs.speech_outputs[0], # First (and only) batch item
        output_path=output_path,
    )
    print(f"Saved output to {output_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("GENERATION SUMMARY")
    print("="*50)
    print(f"Input file: {args.txt_path}")
    print(f"Output file: {output_path}")
    print(f"Speaker names: {args.speaker_names}")
    print(f"Number of unique speakers: {len(set(speaker_numbers))}")
    print(f"Number of segments: {len(scripts)}")
    print(f"Prefilling tokens: {input_tokens}")
    print(f"Generated tokens: {generated_tokens}")
    print(f"Total tokens: {output_tokens}")
    print(f"Generation time: {generation_time:.2f} seconds")
    print(f"Audio duration: {audio_duration:.2f} seconds")
    print(f"RTF (Real Time Factor): {rtf:.2f}x")
    
    print("="*50)

if __name__ == "__main__":
    main()
