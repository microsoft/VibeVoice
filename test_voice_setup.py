#!/usr/bin/env python3
"""
Test voice assignments and audio generation setup (without actually generating audio)
"""
import sys
import os

# Add the parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_voice_setup():
    """Test voice setup without loading the full model"""
    
    print("🎙️ Testing Voice Setup")
    print("=" * 30)
    
    try:
        # Test voice presets discovery
        from pathlib import Path
        voices_dir = Path("demo/voices")
        
        if voices_dir.exists():
            voice_files = list(voices_dir.glob("en-*.wav"))
            print(f"✅ Found {len(voice_files)} English voice files:")
            for voice_file in voice_files:
                name = voice_file.stem.replace("en-", "").split("_")[0]
                print(f"   - {name}: {voice_file.name}")
        else:
            print("❌ Voice directory not found!")
            return False
        
        # Test dialogue parsing
        sample_dialogue = """Host: Welcome to today's hot news podcast!
Expert: Thank you for having me on the show.
Host: Let's dive into the main stories.
Expert: Absolutely, there's quite a lot to discuss today."""
        
        from news_podcast.audio_generator import PodcastAudioGenerator
        
        # This will work without loading the model
        generator = PodcastAudioGenerator.__new__(PodcastAudioGenerator)
        generator.voice_presets = {}
        generator._setup_voice_presets()
        
        print(f"\n🎭 Available voices: {list(generator.voice_presets.keys())}")
        
        # Test dialogue parsing
        dialogue_pairs = generator.parse_dialogue(sample_dialogue)
        print(f"\n📝 Parsed {len(dialogue_pairs)} dialogue pairs:")
        for speaker, text in dialogue_pairs:
            print(f"   {speaker}: {text[:40]}...")
        
        # Test voice assignment
        speakers = list(set(pair[0] for pair in dialogue_pairs))
        voice_assignment = generator.assign_voices_to_speakers(speakers)
        print(f"\n🗣️ Voice assignments:")
        for speaker, voice_path in voice_assignment.items():
            voice_name = Path(voice_path).stem if voice_path else "None"
            print(f"   {speaker}: {voice_name}")
        
        print("\n✅ Voice setup test completed successfully!")
        print("\n📋 Summary:")
        print(f"   - Available voices: {len(generator.voice_presets)}")
        print(f"   - Dialogue segments: {len(dialogue_pairs)}")
        print(f"   - Speakers: {len(speakers)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_voice_setup()
    sys.exit(0 if success else 1)