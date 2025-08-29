import os
import re
from typing import List, Dict, Any, Optional, Tuple
import logging
import torch
from pathlib import Path

from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

logger = logging.getLogger(__name__)

class PodcastAudioGenerator:
    """Generates multi-speaker podcast audio using VibeVoice"""
    
    def __init__(self, model_path: str = "microsoft/VibeVoice-1.5B"):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.voice_presets = {}
        self._setup_voice_presets()
        self._load_model()
    
    def _setup_voice_presets(self):
        """Setup voice presets by scanning the voices directory"""
        voices_dir = Path(__file__).parent.parent / "demo" / "voices"
        
        if not voices_dir.exists():
            logger.warning(f"Voices directory not found at {voices_dir}")
            return
        
        # Scan for English voice files
        for wav_file in voices_dir.glob("*.wav"):
            if wav_file.name.startswith("en-"):
                # Extract speaker name from filename
                # Format: en-{Name}_{gender}.wav
                name_part = wav_file.stem.replace("en-", "").split("_")[0]
                self.voice_presets[name_part] = str(wav_file)
        
        logger.info(f"Available voices: {list(self.voice_presets.keys())}")
    
    def _load_model(self):
        """Load VibeVoice model and processor"""
        try:
            logger.info(f"Loading VibeVoice model: {self.model_path}")
            
            # Load processor
            self.processor = VibeVoiceProcessor.from_pretrained(self.model_path)
            
            # Load model with correct parameters (same as inference_from_file.py)
            attn_implementation = "flash_attention_2"  # Recommended for better audio quality
            self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map='cuda',
                attn_implementation=attn_implementation
            )
            
            self.model.eval()
            self.model.set_ddpm_inference_steps(num_steps=10)
            
            if hasattr(self.model.model, 'language_model'):
                logger.info(f"Language model attention: {self.model.model.language_model.config._attn_implementation}")
            
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def parse_dialogue(self, dialogue_text: str) -> List[Tuple[str, str]]:
        """Parse dialogue text into (speaker, text) pairs"""
        # Debug logging to see what dialogue_text is being received
        logger.info(f"Parsing dialogue text (length: {len(dialogue_text)})")
        logger.info(f"First 500 characters: {dialogue_text[:500]}")
        
        lines = dialogue_text.strip().split('\n')
        parsed_dialogue = []
        
        logger.info(f"Split into {len(lines)} lines")
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            logger.debug(f"Processing line {i+1}: '{line}'")
            
            # Look for various dialogue formats
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    speaker = parts[0].strip()
                    text = parts[1].strip()
                    
                    # Clean up speaker name - remove asterisk prefixes
                    # Handle formats like "**Speaker 1:**" or "**Speaker 2:**"
                    if speaker.startswith('**'):
                        speaker = speaker[2:].strip()  # Remove ** prefix
                        logger.debug(f"Cleaned speaker name from asterisk prefix: '{speaker}'")
                    elif speaker.startswith('*'):
                        speaker = speaker[1:].strip()  # Remove * prefix
                        logger.debug(f"Cleaned speaker name from asterisk prefix: '{speaker}'")
                    
                    # Clean up text - remove asterisk prefixes from text as well
                    if text.startswith('**'):
                        text = text[2:].strip()  # Remove ** prefix from text
                        logger.debug(f"Cleaned text from asterisk prefix")
                    elif text.startswith('*'):
                        text = text[1:].strip()  # Remove * prefix from text
                        logger.debug(f"Cleaned text from asterisk prefix")
                    
                    # Accept various speaker formats:
                    # - "Speaker 1", "Speaker 2", etc. (VibeVoice format)
                    # - "Host", "Analyst", "Reporter", etc. (named speakers)
                    # - Any text followed by colon
                    if text and speaker:  # Only add non-empty text and speaker
                        # Normalize speaker names for consistency
                        if speaker.startswith('Speaker '):
                            # Keep VibeVoice format as-is, but ensure proper format
                            normalized_speaker = speaker
                        else:
                            # Convert other formats to Speaker format for voice assignment
                            # Map common speaker names to Speaker format
                            speaker_mapping = {
                                'Host': 'Speaker 1',
                                'Analyst': 'Speaker 2', 
                                'Reporter': 'Speaker 1',
                                'Guest': 'Speaker 2'
                            }
                            normalized_speaker = speaker_mapping.get(speaker, speaker)
                        
                        parsed_dialogue.append((normalized_speaker, text))
                        logger.debug(f"Added dialogue pair: {normalized_speaker} -> {text[:50]}...")
                    else:
                        logger.debug(f"Skipped line with empty speaker or text: '{line}'")
            else:
                logger.debug(f"Line has no colon, skipping: '{line}'")
        
        logger.info(f"Parsed {len(parsed_dialogue)} dialogue pairs")
        return parsed_dialogue
    
    def assign_voices_to_speakers(self, speakers: List[str], voice_preferences: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Assign voice files to speakers"""
        voice_assignment = {}
        available_voices = list(self.voice_presets.keys())
        
        if not available_voices:
            logger.warning("No English voices available")
            return voice_assignment
        
        # Use preferences if provided
        if voice_preferences:
            for speaker, preferred_voice in voice_preferences.items():
                if preferred_voice in self.voice_presets:
                    voice_assignment[speaker] = self.voice_presets[preferred_voice]
        
        # Assign remaining speakers to available voices
        used_voices = set(voice_assignment.values())
        remaining_voices = [v for k, v in self.voice_presets.items() if v not in used_voices]
        
        for speaker in speakers:
            if speaker not in voice_assignment:
                if remaining_voices:
                    voice_assignment[speaker] = remaining_voices.pop(0)
                else:
                    # Reuse voices if we run out
                    voice_assignment[speaker] = list(self.voice_presets.values())[
                        len(voice_assignment) % len(self.voice_presets)
                    ]
        
        logger.info(f"Voice assignments: {voice_assignment}")
        return voice_assignment
    
    def generate_audio_segment(self, text: str, voice_path: str) -> torch.Tensor:
        """Generate audio for a single text segment with specified voice"""
        try:
            # Process the input - use correct format with lists
            inputs = self.processor(
                text=[text],  # Wrap in list for batch processing
                voice_samples=[voice_path],  # Wrap in list for batch processing
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            
            # Generate audio with correct parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=1.3,
                    tokenizer=self.processor.tokenizer,
                    generation_config={'do_sample': False},
                    verbose=True,
                )
            
            # Extract audio from outputs.speech_outputs[0]
            if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
                return outputs.speech_outputs[0]
            else:
                logger.error("No audio output generated")
                return None
            
        except Exception as e:
            logger.error(f"Error generating audio for text '{text[:50]}...': {e}")
            return None
    
    def concatenate_audio_segments(self, audio_segments: List[torch.Tensor]) -> torch.Tensor:
        """Concatenate multiple audio segments with brief pauses"""
        if not audio_segments:
            return torch.tensor([])
        
        # Filter out None segments
        valid_segments = [seg for seg in audio_segments if seg is not None]
        
        if not valid_segments:
            return torch.tensor([])
        
        # Add brief silence between segments (0.5 seconds at 24kHz)
        sample_rate = 24000
        silence_duration = int(0.5 * sample_rate)
        silence = torch.zeros(silence_duration, dtype=valid_segments[0].dtype, device=valid_segments[0].device)
        
        concatenated_segments = []
        for i, segment in enumerate(valid_segments):
            # Ensure segment is properly shaped
            if segment.dim() > 1:
                segment = segment.squeeze()
            
            concatenated_segments.append(segment)
            
            # Add silence between segments (except after the last one)
            if i < len(valid_segments) - 1:
                concatenated_segments.append(silence)
        
        return torch.cat(concatenated_segments, dim=0)
    
    def generate_podcast_audio(
        self, 
        dialogue_text: str, 
        output_path: str,
        voice_preferences: Optional[Dict[str, str]] = None
    ) -> bool:
        """Generate complete podcast audio from dialogue text"""
        
        if not self.model or not self.processor:
            logger.error("Model not loaded")
            return False
        
        try:
            # Parse the dialogue
            dialogue_pairs = self.parse_dialogue(dialogue_text)
            if not dialogue_pairs:
                logger.error("No valid dialogue found")
                return False
            
            # Get unique speakers
            speakers = list(set(pair[0] for pair in dialogue_pairs))
            logger.info(f"Found speakers: {speakers}")
            
            # Assign voices to speakers
            voice_assignment = self.assign_voices_to_speakers(speakers, voice_preferences)
            
            if not voice_assignment:
                logger.error("No voice assignments made")
                return False
            
            # Generate audio for each dialogue segment
            audio_segments = []
            logger.info(f"Generating audio for {len(dialogue_pairs)} segments")
            
            for i, (speaker, text) in enumerate(dialogue_pairs):
                logger.info(f"Generating segment {i+1}/{len(dialogue_pairs)}: {speaker}")
                
                voice_path = voice_assignment.get(speaker)
                if not voice_path:
                    logger.warning(f"No voice assigned to speaker: {speaker}")
                    continue
                
                audio_segment = self.generate_audio_segment(text, voice_path)
                if audio_segment is not None:
                    audio_segments.append(audio_segment)
                else:
                    logger.warning(f"Failed to generate audio for segment {i+1}")
            
            if not audio_segments:
                logger.error("No audio segments generated")
                return False
            
            # Concatenate all segments
            logger.info("Concatenating audio segments")
            final_audio = self.concatenate_audio_segments(audio_segments)
            
            # Save the audio
            self.save_audio(final_audio, output_path)
            logger.info(f"Podcast audio saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating podcast audio: {e}")
            return False
    
    def save_audio(self, audio_tensor: torch.Tensor, output_path: str, sample_rate: int = 24000):
        """Save audio tensor to file using processor.save_audio"""
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Use processor.save_audio method (same as inference_from_file.py)
            self.processor.save_audio(
                audio_tensor,
                output_path=output_path,
            )
            
        except Exception as e:
            logger.error(f"Error saving audio: {e}")
    
    def get_available_voices(self) -> Dict[str, str]:
        """Get available voice presets"""
        return self.voice_presets.copy()
    
    def get_default_voice_preferences(self, speakers: List[str]) -> Dict[str, str]:
        """Get default voice preferences for speakers"""
        available_voices = list(self.voice_presets.keys())
        preferences = {}
        
        # Default speaker-to-voice mappings for VibeVoice format
        voice_mapping = {
            'Speaker 1': 'Alice',
            'Speaker 2': 'Carter', 
            'Speaker 3': 'Frank',
            'Speaker 4': 'Maya'
        }
        
        for speaker in speakers:
            if speaker in voice_mapping and voice_mapping[speaker] in available_voices:
                preferences[speaker] = voice_mapping[speaker]
            elif available_voices:
                # Assign round-robin style
                idx = len(preferences) % len(available_voices)
                preferences[speaker] = available_voices[idx]
        
        return preferences

if __name__ == "__main__":
    # Test the audio generator
    generator = PodcastAudioGenerator()
    
    # Test with sample dialogue
    sample_dialogue = """Speaker 1: Welcome to today's hot news podcast! We have some fascinating developments to discuss.
Speaker 2: Absolutely! There's been quite a lot happening in the tech world today.
Speaker 1: Let's dive into the main stories. What caught your attention?
Speaker 2: The AI breakthroughs and new programming frameworks are particularly interesting.
Speaker 1: Great insights! Thanks for joining us today."""
    
    print("Available voices:", generator.get_available_voices())
    
    # Generate podcast audio
    output_file = "/tmp/test_podcast.wav"
    success = generator.generate_podcast_audio(sample_dialogue, output_file)
    
    if success:
        print(f"Test podcast generated successfully: {output_file}")
    else:
        print("Failed to generate test podcast")