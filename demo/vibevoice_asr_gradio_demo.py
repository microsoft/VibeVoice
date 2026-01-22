#!/usr/bin/env python
"""
VibeVoice ASR Gradio Demo
"""

import os
import sys
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
import argparse
import time
import json
import gradio as gr
from typing import List, Dict, Tuple, Optional, Generator
import tempfile
import base64
import io
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import TextIteratorStreamer for streaming generation
from transformers import TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList

try:
    from liger_kernel.transformers import apply_liger_kernel_to_qwen2
    # Only apply RoPE, RMSNorm, SwiGLU patches (these affect the underlying Qwen2 layers)
    apply_liger_kernel_to_qwen2(
        rope=True,
        rms_norm=True,
        swiglu=True,
        cross_entropy=False,
    )
    print("✅ Liger Kernel applied to Qwen2 components (RoPE, RMSNorm, SwiGLU)")
except Exception as e:
    print(f"⚠️ Failed to apply Liger Kernel: {e}, you can install it with: pip install liger-kernel")
    
# Try to import pydub for MP3 conversion
try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False
    print("⚠️ Warning: pydub not available, falling back to WAV format")

from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
from vibevoice.processor.audio_utils import load_audio_use_ffmpeg, COMMON_AUDIO_EXTS


class VibeVoiceASRInference:
    """Simple inference wrapper for VibeVoice ASR model."""
    
    def __init__(self, model_path: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16, attn_implementation: str = "flash_attention_2"):
        """
        Initialize the ASR inference pipeline.
        
        Args:
            model_path: Path to the pretrained model (HuggingFace format directory or model name)
            device: Device to run inference on
            dtype: Data type for model weights
            attn_implementation: Attention implementation to use ('flash_attention_2', 'sdpa', 'eager')
        """
        print(f"Loading VibeVoice ASR model from {model_path}")
        
        # Load processor
        self.processor = VibeVoiceASRProcessor.from_pretrained(model_path)
        
        # Load model
        print(f"Using attention implementation: {attn_implementation}")
        self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=device if device == "auto" else None,
            attn_implementation=attn_implementation,
            trust_remote_code=True
        )
        
        if device != "auto":
            self.model = self.model.to(device)
        
        self.device = device if device != "auto" else next(self.model.parameters()).device
        self.model.eval()
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"✅ Model loaded successfully on {self.device}")
        print(f"📊 Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    
    def transcribe(
        self, 
        audio_path: str = None,
        audio_array: np.ndarray = None,
        sample_rate: int = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
        num_beams: int = 1,
        repetition_penalty: float = 1.0,
        context_info: str = None,
        streamer: Optional[TextIteratorStreamer] = None,
    ) -> dict:
        """
        Transcribe audio to text.
        
        Args:
            audio_path: Path to audio file
            audio_array: Audio array (if not loading from file)
            sample_rate: Sample rate of audio array
            max_new_tokens: Maximum tokens to generate
            temperature: Temperature for sampling (0 for greedy)
            top_p: Top-p for nucleus sampling (1.0 for no filtering)
            do_sample: Whether to use sampling
            num_beams: Number of beams for beam search (1 for greedy)
            repetition_penalty: Repetition penalty (1.0 for no penalty)
            context_info: Optional context information (e.g., hotwords, speaker names, topics) to help transcription
            streamer: Optional TextIteratorStreamer for streaming output
            
        Returns:
            Dictionary with transcription results
        """
        # Process audio
        inputs = self.processor(
            audio=audio_path,
            sampling_rate=sample_rate,
            return_tensors="pt",
            add_generation_prompt=True,
            context_info=context_info
        )
        
        # Move to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
        
        # Generate
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature if temperature > 0 else None,
            "top_p": top_p if do_sample else None,
            "do_sample": do_sample,
            "num_beams": num_beams,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self.processor.pad_id,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
        }
        
        # Add streamer if provided
        if streamer is not None:
            generation_config["streamer"] = streamer
        
        # Add stopping criteria for stop button support
        generation_config["stopping_criteria"] = StoppingCriteriaList([StopOnFlag()])
        
        # Remove None values
        generation_config = {k: v for k, v in generation_config.items() if v is not None}
        
        start_time = time.time()
        
        # Calculate input token statistics before generation
        input_ids = inputs['input_ids'][0]  # Shape: [seq_len]
        total_input_tokens = input_ids.shape[0]
        
        # Count padding tokens (tokens equal to pad_id)
        pad_id = self.processor.pad_id
        padding_mask = (input_ids == pad_id)
        num_padding_tokens = padding_mask.sum().item()
        
        # Count speech tokens (tokens between speech_start_id and speech_end_id)
        speech_start_id = self.processor.speech_start_id
        speech_end_id = self.processor.speech_end_id
        
        # Find speech regions
        input_ids_list = input_ids.tolist()
        num_speech_tokens = 0
        in_speech = False
        for token_id in input_ids_list:
            if token_id == speech_start_id:
                in_speech = True
                num_speech_tokens += 1  # Count speech_start token
            elif token_id == speech_end_id:
                in_speech = False
                num_speech_tokens += 1  # Count speech_end token
            elif in_speech:
                num_speech_tokens += 1
        
        # Text tokens = total - speech - padding
        num_text_tokens = total_input_tokens - num_speech_tokens - num_padding_tokens
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                **generation_config
            )
        
        generation_time = time.time() - start_time
        
        # Decode output
        generated_ids = output_ids[0, inputs['input_ids'].shape[1]:]
        generated_text = self.processor.decode(generated_ids, skip_special_tokens=True)
        
        # Parse structured output
        try:
            transcription_segments = self.processor.post_process_transcription(generated_text)
        except Exception as e:
            print(f"Warning: Failed to parse structured output: {e}")
            transcription_segments = []
        
        return {
            "raw_text": generated_text,
            "segments": transcription_segments,
            "generation_time": generation_time,
            "input_tokens": {
                "total": total_input_tokens,
                "speech": num_speech_tokens,
                "text": num_text_tokens,
                "padding": num_padding_tokens,
            },
        }


def clip_and_encode_audio(
    audio_data: np.ndarray,
    sr: int,
    start_time: float,
    end_time: float,
    segment_idx: int,
    use_mp3: bool = True,
    target_sr: int = 16000,  # Downsample to 16kHz for smaller size
    mp3_bitrate: str = "32k"  # Use low bitrate for minimal transfer
) -> Tuple[int, Optional[str], Optional[str]]:
    """
    Clip audio segment and encode to base64.
    
    Args:
        audio_data: Full audio array
        sr: Sample rate
        start_time: Start time in seconds
        end_time: End time in seconds
        segment_idx: Segment index for identification
        use_mp3: Whether to use MP3 format (smaller size)
        target_sr: Target sample rate for downsampling (lower = smaller)
        mp3_bitrate: MP3 bitrate (lower = smaller, e.g., "24k", "32k", "48k")
        
    Returns:
        Tuple of (segment_idx, base64_string, error_message)
    """
    try:
        # Convert time to sample indices
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        # Ensure indices are within bounds
        start_sample = max(0, start_sample)
        end_sample = min(len(audio_data), end_sample)
        
        if start_sample >= end_sample:
            return segment_idx, None, f"Invalid time range: [{start_time:.2f}s - {end_time:.2f}s]"
        
        # Extract segment
        segment_data = audio_data[start_sample:end_sample]
        
        # Downsample if needed (reduces data size significantly)
        if sr != target_sr and target_sr < sr:
            # Simple downsampling using linear interpolation
            duration = len(segment_data) / sr
            new_length = int(duration * target_sr)
            indices = np.linspace(0, len(segment_data) - 1, new_length)
            segment_data = np.interp(indices, np.arange(len(segment_data)), segment_data)
            sr = target_sr
        
        # Convert float32 audio to int16 for encoding
        segment_data_int16 = (segment_data * 32768.0).astype(np.int16)
        
        # Convert to MP3 if pydub is available and use_mp3 is True
        if use_mp3 and HAS_PYDUB:
            try:
                # Write to WAV in memory
                wav_buffer = io.BytesIO()
                sf.write(wav_buffer, segment_data_int16, sr, format='WAV', subtype='PCM_16')
                wav_buffer.seek(0)
                
                # Convert to MP3 with low bitrate
                audio_segment = AudioSegment.from_wav(wav_buffer)
                # Convert to mono if stereo (halves the size)
                if audio_segment.channels > 1:
                    audio_segment = audio_segment.set_channels(1)
                mp3_buffer = io.BytesIO()
                audio_segment.export(mp3_buffer, format='mp3', bitrate=mp3_bitrate)
                mp3_buffer.seek(0)
                
                # Encode to base64
                audio_bytes = mp3_buffer.read()
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                audio_src = f"data:audio/mp3;base64,{audio_base64}"
                
                return segment_idx, audio_src, None
            except Exception as e:
                # Fall back to WAV on error
                print(f"MP3 conversion failed for segment {segment_idx}, using WAV: {e}")
        
        # Fall back to WAV format (no temp file, use in-memory buffer)
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, segment_data_int16, sr, format='WAV', subtype='PCM_16')
        wav_buffer.seek(0)
        
        audio_bytes = wav_buffer.read()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        audio_src = f"data:audio/wav;base64,{audio_base64}"
        
        return segment_idx, audio_src, None
        
    except Exception as e:
        error_msg = f"Error clipping segment {segment_idx}: {str(e)}"
        print(error_msg)
        return segment_idx, None, error_msg


def extract_audio_segments(audio_path: str, segments: List[Dict]) -> List[Tuple[str, str, Optional[str]]]:
    """
    Extract multiple segments from audio file efficiently with parallel processing.
    
    Args:
        audio_path: Path to original audio file
        segments: List of segment dictionaries with start_time, end_time, etc.
    
    Returns:
        List of tuples (segment_label, audio_base64_src, error_msg)
    """
    try:
        # Read audio file once using ffmpeg for better format support
        print(f"📂 Loading audio file: {audio_path}")
        audio_data, sr = load_audio_use_ffmpeg(audio_path, resample=False)
        print(f"✅ Audio loaded: {len(audio_data)} samples, {sr} Hz")
        
        # Prepare tasks
        tasks = []
        use_mp3 = HAS_PYDUB  # Use MP3 if available
        
        for i, seg in enumerate(segments):
            start_time = seg.get('start_time')
            end_time = seg.get('end_time')
            
            # Skip if times are not available or invalid
            if (not isinstance(start_time, (int, float)) or 
                not isinstance(end_time, (int, float)) or 
                start_time >= end_time):
                tasks.append((i, None, None, None, None, None))  # Will be filtered later
                continue
            
            tasks.append((audio_data, sr, start_time, end_time, i, use_mp3))
        
        # Process in parallel using ThreadPoolExecutor
        results = []
        total_segments = len(tasks)
        completed_count = 0
        
        # Use CPU count for max workers
        max_workers = os.cpu_count() or 4
        print(f"🚀 Starting parallel processing with {max_workers} threads...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for task in tasks:
                if task[0] is None:  # Skip invalid tasks
                    continue
                future = executor.submit(clip_and_encode_audio, *task)
                futures[future] = task[4]  # segment_idx
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    completed_count += 1
                    # Log progress every 100 segments or at completion
                    if completed_count % 100 == 0 or completed_count == len(futures):
                        print(f"Progress: {completed_count}/{len(futures)} segments processed ({completed_count*100//len(futures)}%)")
                except Exception as e:
                    idx = futures[future]
                    results.append((idx, None, f"Processing error: {str(e)}"))
                    completed_count += 1
                    print(f"Error on segment {idx}: {e}")
        
        print(f"✅ Completed processing all {len(futures)} valid segments")
        
        # Sort by segment index to maintain order
        results.sort(key=lambda x: x[0])
        
        # Build output list with labels
        audio_segments = []
        for i, (idx, audio_src, error_msg) in enumerate(results):
            seg = segments[idx] if idx < len(segments) else {}
            start_time = seg.get('start_time', 'N/A')
            end_time = seg.get('end_time', 'N/A')
            speaker_id = seg.get('speaker_id', 'N/A')
            
            segment_label = f"Segment {idx+1}: [{start_time:.2f}s - {end_time:.2f}s] Speaker {speaker_id}"
            audio_segments.append((segment_label, audio_src, error_msg))
        
        return audio_segments
        
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return []


# Global variable to store the ASR model
asr_model = None

# Global stop flag for generation
stop_generation_flag = False


class StopOnFlag(StoppingCriteria):
    """Custom stopping criteria that checks a global flag."""
    def __call__(self, input_ids, scores, **kwargs):
        global stop_generation_flag
        return stop_generation_flag


def parse_time_to_seconds(val: Optional[str]) -> Optional[float]:
    """Parse seconds or hh:mm:ss to float seconds."""
    if val is None:
        return None
    val = val.strip()
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        pass
    if ":" in val:
        parts = val.split(":")
        if not all(p.strip().replace(".", "", 1).isdigit() for p in parts):
            return None
        parts = [float(p) for p in parts]
        if len(parts) == 3:
            h, m, s = parts
        elif len(parts) == 2:
            h = 0
            m, s = parts
        else:
            return None
        return h * 3600 + m * 60 + s
    return None


def slice_audio_to_temp(
    audio_data: np.ndarray,
    sample_rate: int,
    start_sec: Optional[float],
    end_sec: Optional[float]
) -> Tuple[Optional[str], Optional[str]]:
    """Slice audio_data to [start_sec, end_sec) and write to a temp WAV file."""
    n_samples = len(audio_data)
    full_duration = n_samples / float(sample_rate)
    start = 0.0 if start_sec is None else max(0.0, start_sec)
    end = full_duration if end_sec is None else min(full_duration, end_sec)
    if end <= start:
        return None, f"Invalid time range: start={start:.2f}s, end={end:.2f}s"
    start_idx = int(start * sample_rate)
    end_idx = int(end * sample_rate)
    segment = audio_data[start_idx:end_idx]
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_file.close()
    segment_int16 = (segment * 32768.0).astype(np.int16)
    sf.write(temp_file.name, segment_int16, sample_rate, subtype='PCM_16')
    return temp_file.name, None


def initialize_model(model_path: str, device: str = "cuda", attn_implementation: str = "flash_attention_2"):
    """Initialize the ASR model."""
    global asr_model
    try:
        dtype = torch.bfloat16 if device != "cpu" else torch.float32
        asr_model = VibeVoiceASRInference(
            model_path=model_path,
            device=device,
            dtype=dtype,
            attn_implementation=attn_implementation
        )
        return f"✅ Model loaded successfully from {model_path}"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error loading model: {str(e)}"


def transcribe_audio(
    audio_input,
    audio_path_input: str,
    start_time_input: str,
    end_time_input: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    repetition_penalty: float = 1.0,
    context_info: str = ""
) -> Generator[Tuple[str, str], None, None]:
    """
    Transcribe audio and return results with audio segments (streaming version).
    
    Args:
        audio_input: Audio file path or tuple (sample_rate, audio_data)
        max_new_tokens: Maximum tokens to generate
        temperature: Temperature for sampling (0 for greedy)
        top_p: Top-p for nucleus sampling
        do_sample: Whether to use sampling
        context_info: Optional context information (e.g., hotwords, speaker names, topics)
    
    Yields:
        Tuple of (raw_text, audio_segments_html)
    """
    if asr_model is None:
        yield "❌ Please load a model first!", ""
        return
    
    if not audio_path_input and audio_input is None:
        yield "❌ Please provide audio input!", ""
        return
    
    try:
        print("[INFO] Transcription requested")
        start_sec = parse_time_to_seconds(start_time_input)
        end_sec = parse_time_to_seconds(end_time_input)
        print(f"[INFO] Parsed time range: start={start_sec}, end={end_sec}")
        if (start_time_input and start_sec is None) or (end_time_input and end_sec is None):
            yield "❌ Invalid time format. Use seconds or hh:mm:ss.", ""
            return

        audio_path = None
        audio_array = None
        sample_rate = None

        if audio_path_input:
            candidate = Path(audio_path_input.strip())
            if not candidate.exists():
                yield f"❌ Provided path does not exist: {candidate}", ""
                return
            audio_path = str(candidate)
            print(f"[INFO] Using provided audio path: {audio_path}")
        # Get audio file path (Gradio Audio component returns tuple (sample_rate, audio_data) or file path)
        elif isinstance(audio_input, str):
            audio_path = audio_input
            print(f"[INFO] Using uploaded audio path: {audio_path}")
        elif isinstance(audio_input, tuple):
            # Audio from microphone: (sample_rate, audio_data)
            sample_rate, audio_array = audio_input
            print(f"[INFO] Received microphone audio with sample_rate={sample_rate}")
        elif audio_path is None:
            yield "❌ Invalid audio input format!", ""
            return

        # If slicing is requested, load and slice audio
        if start_sec is not None or end_sec is not None:
            print("[INFO] Slicing audio per requested time range")
            if audio_array is None or sample_rate is None:
                try:
                    audio_array, sample_rate = load_audio_use_ffmpeg(audio_path, resample=False)
                    print("[INFO] Loaded audio for slicing via ffmpeg")
                except Exception as exc:
                    yield f"❌ Failed to load audio for slicing: {exc}", ""
                    return
            sliced_path, err = slice_audio_to_temp(audio_array, sample_rate, start_sec, end_sec)
            if err:
                yield f"❌ {err}", ""
                return
            audio_path = sliced_path
            print(f"[INFO] Sliced audio written to temp file: {audio_path}")
        elif audio_array is not None and sample_rate is not None:
            # no slicing but microphone input: write to temp file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            audio_path = temp_file.name
            temp_file.close()
            audio_data_int16 = (audio_array * 32768.0).astype(np.int16)
            sf.write(audio_path, audio_data_int16, sample_rate, subtype='PCM_16')
            print(f"[INFO] Microphone audio saved to temp file: {audio_path}")
        
        # Create streamer for real-time output
        streamer = TextIteratorStreamer(
            asr_model.processor.tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True
        )
        
        # Store result in a mutable container for the thread
        result_container = {"result": None, "error": None}
        
        def run_transcription():
            try:
                result_container["result"] = asr_model.transcribe(
                    audio_path=audio_path,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty,
                    context_info=context_info if context_info and context_info.strip() else None,
                    streamer=streamer
                )
            except Exception as e:
                result_container["error"] = str(e)
                traceback.print_exc()
        
        # Start transcription in background thread
        print("[INFO] Starting model transcription (streaming mode)")
        start_time = time.time()
        transcription_thread = threading.Thread(target=run_transcription)
        transcription_thread.start()
        
        # Yield streaming output
        generated_text = ""
        token_count = 0
        for new_text in streamer:
            generated_text += new_text
            token_count += 1
            elapsed = time.time() - start_time
            # Show streaming output with live stats, format for readability
            formatted_text = generated_text.replace('},', '},\n')
            streaming_output = f"--- 🔴 LIVE Streaming Output (tokens: {token_count}, time: {elapsed:.1f}s) ---\n{formatted_text}"
            yield streaming_output, "<div style='padding: 20px; text-align: center; color: #6c757d;'>⏳ Generating transcription... Audio segments will appear after completion.</div>"
        
        # Wait for thread to complete
        transcription_thread.join()
        
        if result_container["error"]:
            yield f"❌ Error during transcription: {result_container['error']}", ""
            return
        
        result = result_container["result"]
        generation_time = time.time() - start_time
        
        # Get input token statistics
        input_tokens = result.get('input_tokens', {})
        speech_tokens = input_tokens.get('speech', 0)
        text_tokens = input_tokens.get('text', 0)
        padding_tokens = input_tokens.get('padding', 0)
        total_input = input_tokens.get('total', 0)
        
        # Format final raw output with input/output token stats
        raw_output = f"--- ✅ Raw Output ---\n"
        raw_output += f"📥 Input: {total_input} tokens (🎤 speech: {speech_tokens}, 📝 text: {text_tokens}, ⬜ pad: {padding_tokens})\n"
        raw_output += f"📤 Output: {token_count} tokens | ⏱️ Time: {generation_time:.2f}s\n"
        raw_output += f"---\n"
        # Format raw text for better readability: add newline after each dict (},)
        formatted_raw_text = result['raw_text'].replace('},', '},\n')
        raw_output += formatted_raw_text
        
        # Debug: print raw output to console
        print(f"[DEBUG] Raw model output:")
        print(f"[DEBUG] {result['raw_text']}")
        print(f"[DEBUG] Found {len(result['segments'])} segments")
        
        # Create audio segments with server-side encoding (low quality for minimal transfer)
        # Using: 16kHz mono MP3 @ 32kbps = ~4KB per second of audio
        audio_segments_html = ""
        segments = result['segments']
        
        if segments:
            num_segments = len(segments)
            print(f"[INFO] Creating per-segment audio clips ({num_segments} segments, 16kHz mono MP3 @ 32kbps)")
            
            # Extract all audio segments efficiently (load audio only once)
            audio_segments = extract_audio_segments(audio_path, segments)
            print("[INFO] Completed creating audio clips")
            
            # Calculate approximate total size
            total_duration = sum(
                (seg.get('end_time', 0) - seg.get('start_time', 0)) 
                for seg in segments 
                if isinstance(seg.get('start_time'), (int, float)) and isinstance(seg.get('end_time'), (int, float))
            )
            approx_size_kb = total_duration * 4  # ~4KB per second at 32kbps
            
            # Add CSS for theme-aware styling
            theme_css = """
            <style>
            :root {
                --segment-bg: #f8f9fa;
                --segment-border: #e1e5e9;
                --segment-text: #495057;
                --segment-meta: #6c757d;
                --content-bg: white;
                --content-border: #007bff;
                --warning-bg: #fff3cd;
                --warning-border: #ffc107;
                --warning-text: #856404;
            }
            
            @media (prefers-color-scheme: dark) {
                :root {
                    --segment-bg: #2d3748;
                    --segment-border: #4a5568;
                    --segment-text: #e2e8f0;
                    --segment-meta: #a0aec0;
                    --content-bg: #1a202c;
                    --content-border: #4299e1;
                    --warning-bg: #744210;
                    --warning-border: #d69e2e;
                    --warning-text: #faf089;
                }
            }
            
            .audio-segments-container {
                max-height: 600px;
                overflow-y: auto;
                padding: 10px;
            }
            
            .audio-segment {
                margin-bottom: 15px;
                padding: 15px;
                border: 2px solid var(--segment-border);
                border-radius: 8px;
                background-color: var(--segment-bg);
                transition: all 0.3s ease;
            }
            
            .audio-segment:hover {
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            }
            
            .segment-header {
                margin-bottom: 10px;
            }
            
            .segment-title {
                margin: 0;
                color: var(--segment-text);
                font-size: 16px;
                font-weight: 600;
            }
            
            .segment-meta {
                margin-top: 5px;
                font-size: 14px;
                color: var(--segment-meta);
            }
            
            .segment-content {
                margin-bottom: 10px;
                padding: 12px;
                background-color: var(--content-bg);
                border-radius: 6px;
                border-left: 4px solid var(--content-border);
                color: var(--segment-text);
                line-height: 1.5;
            }
            
            .segment-audio {
                width: 100%;
                margin-top: 10px;
                border-radius: 4px;
            }
            
            .segment-warning {
                margin-top: 10px;
                padding: 10px;
                background-color: var(--warning-bg);
                border-radius: 4px;
                border-left: 4px solid var(--warning-border);
                color: var(--warning-text);
                font-size: 13px;
            }
            
            .segments-title {
                color: var(--segment-text);
                margin-bottom: 10px;
            }
            
            .segments-description {
                color: var(--segment-meta);
                margin-bottom: 20px;
            }
            
            .size-badge {
                display: inline-block;
                background: linear-gradient(135deg, #6c757d, #495057);
                color: white;
                padding: 4px 10px;
                border-radius: 12px;
                font-size: 12px;
                margin-left: 10px;
            }
            </style>
            """
            
            audio_segments_html = theme_css
            audio_segments_html += f"<div class='audio-segments-container'>"
            
            # Add format info
            format_info = "MP3 32kbps 16kHz mono" if HAS_PYDUB else "WAV 16kHz"
            audio_segments_html += f"<h3 class='segments-title'>🔊 Audio Segments ({num_segments} segments)"
            audio_segments_html += f"<span class='size-badge'>📦 ~{approx_size_kb:.0f}KB ({format_info})</span></h3>"
            audio_segments_html += "<p class='segments-description'>🎵 Click the play button to listen to each segment directly!</p>"
            
            for i, (label, audio_src, error_msg) in enumerate(audio_segments):
                seg = segments[i] if i < len(segments) else {}
                start_time = seg.get('start_time', 'N/A')
                end_time = seg.get('end_time', 'N/A')
                speaker_id = seg.get('speaker_id', 'N/A')
                content = seg.get('text', '')
                
                # Format times nicely
                start_str = f"{start_time:.2f}" if isinstance(start_time, (int, float)) else str(start_time)
                end_str = f"{end_time:.2f}" if isinstance(end_time, (int, float)) else str(end_time)
                
                audio_segments_html += f"""
                <div class='audio-segment'>
                    <div class='segment-header'>
                        <h4 class='segment-title'>Segment {i+1}</h4>
                        <div class='segment-meta'>
                            <strong>Time:</strong> [{start_str}s - {end_str}s] | 
                            <strong>Speaker:</strong> {speaker_id}
                        </div>
                    </div>
                    
                    <div class='segment-content'>
                        {content}
                    </div>
                """
                
                if audio_src:
                    # Detect format from data URI
                    audio_type = 'audio/mp3' if 'audio/mp3' in audio_src else 'audio/wav'
                    audio_segments_html += f"""
                    <audio controls class='segment-audio' preload='none'>
                        <source src='{audio_src}' type='{audio_type}'>
                        Your browser does not support the audio element.
                    </audio>
                    """
                elif error_msg:
                    audio_segments_html += f"""
                    <div class='segment-warning'>
                        <small>❌ {error_msg}</small>
                    </div>
                    """
                else:
                    audio_segments_html += """
                    <div class='segment-warning'>
                        <small>Audio playback unavailable for this segment</small>
                    </div>
                    """
                
                audio_segments_html += "</div>"
            
            audio_segments_html += "</div>"
        else:
            audio_segments_html = """
            <style>
            :root {
                --no-segments-text: #6c757d;
            }
            
            @media (prefers-color-scheme: dark) {
                :root {
                    --no-segments-text: #a0aec0;
                }
            }
            
            .no-segments-container {
                padding: 20px;
                text-align: center;
                color: var(--no-segments-text);
                line-height: 1.6;
            }
            </style>
            <div class='no-segments-container'>
                <p>❌ No audio segments available.</p>
                <p>This could happen if the model output doesn't contain valid time stamps.</p>
            </div>
            """
        
        # Final yield with complete results
        yield raw_output, audio_segments_html
        
    except Exception as e:
        print(f"Error during transcription: {e}")
        print(traceback.format_exc())
        yield f"❌ Error during transcription: {str(e)}", ""


def create_gradio_interface(model_path: str, default_max_tokens: int = 8192, attn_implementation: str = "flash_attention_2"):
    """Create and launch Gradio interface.
    
    Args:
        model_path: Path to the model (HuggingFace format directory or model name)
        default_max_tokens: Default value for max_new_tokens slider
        attn_implementation: Attention implementation to use ('flash_attention_2', 'sdpa', 'eager')
    """
    
    # Initialize model at startup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_status = initialize_model(model_path, device, attn_implementation)
    print(model_status)
    
    # Exit if model loading failed
    if model_status.startswith("❌"):
        print("\n" + "="*80)
        print("💥 FATAL ERROR: Model loading failed!")
        print("="*80)
        print("Cannot start demo without a valid model. Please check:")
        print("  1. Model path is correct")
        print("  2. Model files are not corrupted")
        print("  3. You have enough GPU memory")
        print("  4. CUDA is properly installed (if using GPU)")
        print("="*80)
        sys.exit(1)
    
    # Custom CSS for Stop button styling
    custom_css = """
    #stop-btn {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
        border: none !important;
        color: white !important;
    }
    #stop-btn:hover {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%) !important;
    }
    """
    
    # Gradio 6.0+ moved theme/css to launch()
    with gr.Blocks(title="VibeVoice ASR Demo") as demo:
        gr.Markdown("# 🎙️ VibeVoice ASR Demo")
        gr.Markdown("Upload audio files or record from microphone to get speech-to-text transcription with speaker diarization.")
        gr.Markdown(f"**Model loaded from:** `{model_path}`")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Generation parameters
                gr.Markdown("## ⚙️ Generation Parameters")
                max_tokens_slider = gr.Slider(
                    minimum=4096,
                    maximum=65536,
                    value=default_max_tokens,
                    step=4096,
                    label="Max New Tokens"
                )
                
                # Sampling parameters
                gr.Markdown("### 🎲 Sampling")
                do_sample_checkbox = gr.Checkbox(
                    value=False,
                    label="Enable Sampling",
                    info="Enable random sampling instead of deterministic decoding"
                )
                
                with gr.Column(visible=False) as sampling_params:
                    temperature_slider = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.0,
                        step=0.1,
                        label="Temperature",
                        info="0 = greedy, higher = more random"
                    )
                    top_p_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=1.0,
                        step=0.05,
                        label="Top-p (Nucleus Sampling)",
                        info="1.0 = no filtering"
                    )
                
                # Repetition penalty (works with both greedy and sampling)
                repetition_penalty_slider = gr.Slider(
                    minimum=1.0,
                    maximum=1.2,
                    value=1.0,
                    step=0.01,
                    label="Repetition Penalty",
                    info="1.0 = no penalty, higher = less repetition (works with greedy & sampling)"
                )
                
                # Context information section
                gr.Markdown("## 📋 Context Info (Optional)")
                context_info_input = gr.Textbox(
                    label="Context Information",
                    placeholder="Enter hotwords, speaker names, topics, or other context to help transcription...\nExample:\nJohn Smith\nMachine Learning\nOpenAI",
                    lines=4,
                    max_lines=8,
                    interactive=True,
                    info="Provide context like proper nouns, technical terms, or speaker names to improve accuracy"
                )
            
            with gr.Column(scale=2):
                # Audio input section
                gr.Markdown("## 🎵 Audio Input")
                audio_input = gr.Audio(
                    label="Upload Audio File or Record from Microphone",
                    sources=["upload", "microphone"],
                    type="filepath",
                    interactive=True,
                    buttons=["download"]
                )
                
                with gr.Accordion("📂 Advanced: Remote Path & Time Slicing", open=False):
                    audio_path_input = gr.Textbox(
                        label="Audio path (optional)",
                        placeholder="Enter remote audio file path",
                        lines=1
                    )
                    with gr.Row():
                        start_time_input = gr.Textbox(
                            label="Start time",
                            placeholder="e.g., 0 or 00:00:00",
                            lines=1,
                            info="Leave empty to start from the beginning"
                        )
                        end_time_input = gr.Textbox(
                            label="End time",
                            placeholder="e.g., 30.5 or 00:00:30.5",
                            lines=1,
                            info="Leave empty to use full length"
                        )
                
                with gr.Row():
                    transcribe_button = gr.Button("🎯 Transcribe", variant="primary", size="lg", scale=3)
                    stop_button = gr.Button("⏹️ Stop", variant="secondary", size="lg", scale=1, elem_id="stop-btn")
                
                # Results section
                gr.Markdown("## 📝 Results")
                
                with gr.Tabs():
                    with gr.TabItem("Raw Output"):
                        raw_output = gr.Textbox(
                            label="Raw Transcription Output",
                            lines=8,
                            max_lines=20,
                            interactive=False
                        )
                    
                    with gr.TabItem("Audio Segments"):
                        audio_segments_output = gr.HTML(
                            label="Play individual segments to verify accuracy"
                        )
        
        # Event handlers
        do_sample_checkbox.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[do_sample_checkbox],
            outputs=[sampling_params]
        )
        
        def reset_stop_flag():
            """Reset stop flag before starting transcription."""
            global stop_generation_flag
            stop_generation_flag = False
        
        def set_stop_flag():
            """Set stop flag to interrupt generation."""
            global stop_generation_flag
            stop_generation_flag = True
            return "⏹️ Stop requested..."
        
        transcribe_button.click(
            fn=reset_stop_flag,
            inputs=[],
            outputs=[],
            queue=False
        ).then(
            fn=transcribe_audio,
            inputs=[
                audio_input,
                audio_path_input,
                start_time_input,
                end_time_input,
                max_tokens_slider,
                temperature_slider,
                top_p_slider,
                do_sample_checkbox,
                repetition_penalty_slider,
                context_info_input
            ],
            outputs=[raw_output, audio_segments_output]
        )
        
        stop_button.click(
            fn=set_stop_flag,
            inputs=[],
            outputs=[raw_output],
            queue=False
        )
        
        # Add examples
        gr.Markdown("## 📋 Instructions")
        gr.Markdown(f"""
        1. **Upload Audio**: Use the audio component to upload a file or record from microphone
           - **Supported formats**: {', '.join(sorted(set([ext.lower() for ext in COMMON_AUDIO_EXTS])))}
           - Optionally set **Start/End time** (seconds or hh:mm:ss) to clip before transcription
        2. **Context Info (Optional)**: Provide context to improve transcription accuracy
           - Add hotwords, proper nouns, speaker names, or technical terms
           - One item per line or comma-separated
           - Examples: "John Smith", "OpenAI", "machine learning"
        3. **Adjust Parameters**: Configure generation parameters as needed
        4. **Transcribe**: Click "Transcribe" to get results
        5. **Review Results**: 
           - **Raw Output**: View the model's original output
           - **Audio Segments**: Play individual segments directly to verify accuracy
        
        **Audio Segments**: Each segment shows the time range, speaker ID, transcribed content, and an embedded audio player for immediate verification.
        """)
    
    return demo, custom_css


def main():
    parser = argparse.ArgumentParser(description="VibeVoice ASR Gradio Demo")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="microsoft/VibeVoice-ASR",
        help="Path to the model (HuggingFace format directory or model name)"
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        help="Attention implementation to use (default: flash_attention_2)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32768,
        help="Default max new tokens for generation (default: 32768)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to bind the server to"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public link"
    )
    
    args = parser.parse_args()
    
    # Create and launch interface
    demo, custom_css = create_gradio_interface(
        model_path=args.model_path,
        default_max_tokens=args.max_new_tokens,
        attn_implementation=args.attn_implementation
    )
    
    print(f"🚀 Starting VibeVoice ASR Demo...")
    print(f"📍 Server will be available at: http://{args.host}:{args.port}")
    
    # Gradio 6.0+ moved theme/css to launch()
    launch_kwargs = {
        "server_name": args.host,
        "server_port": args.port,
        "share": args.share,
        "show_error": True,
        "theme": gr.themes.Soft(),
        "css": custom_css,
    }
    
    # Enable queue for concurrent request handling
    demo.queue(default_concurrency_limit=3)
    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
