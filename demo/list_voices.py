#!/usr/bin/env python3
"""
VibeVoice Voice Listing Utility

This script lists all available voice presets for VibeVoice models,
including metadata like language and gender parsed from filenames.

Usage:
    python demo/list_voices.py [--format {table|json|simple}] [--lang LANG]

Examples:
    # List all voices in table format (default)
    python demo/list_voices.py

    # List only English voices
    python demo/list_voices.py --lang en

    # Output as JSON for programmatic use
    python demo/list_voices.py --format json

    # Simple list of voice names only
    python demo/list_voices.py --format simple
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional


class VoiceInfo:
    """Represents metadata about a voice preset"""

    def __init__(self, filename: str, path: Path):
        self.filename = filename
        self.path = path
        self.name = filename  # Full name without extension

        # Parse language code and speaker info from filename
        # Expected format: {lang}-{SpeakerName}_{gender}.pt
        # Examples: en-Carter_man.pt, de-Spk0_woman.pt
        parts = filename.split('-', 1)

        if len(parts) == 2:
            self.language = parts[0]
            speaker_part = parts[1]

            # Extract speaker name and gender
            if '_' in speaker_part:
                self.speaker_name, self.gender = speaker_part.rsplit('_', 1)
            else:
                self.speaker_name = speaker_part
                self.gender = 'unknown'
        else:
            # Fallback for non-standard naming
            self.language = 'unknown'
            self.speaker_name = filename
            self.gender = 'unknown'

        # Get file size
        try:
            self.size_mb = self.path.stat().st_size / (1024 * 1024)
        except Exception:
            self.size_mb = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'name': self.name,
            'language': self.language,
            'speaker': self.speaker_name,
            'gender': self.gender,
            'size_mb': round(self.size_mb, 2),
            'path': str(self.path),
        }

    @staticmethod
    def get_language_name(code: str) -> str:
        """Convert language code to full name"""
        lang_map = {
            'en': 'English',
            'de': 'German',
            'fr': 'French',
            'it': 'Italian',
            'jp': 'Japanese',
            'kr': 'Korean',
            'nl': 'Dutch',
            'pl': 'Polish',
            'pt': 'Portuguese',
            'sp': 'Spanish',
            'in': 'International',
        }
        return lang_map.get(code, code.upper())


class VoiceManager:
    """Manages voice presets and provides listing functionality"""

    def __init__(self, voices_dir: Optional[Path] = None):
        if voices_dir is None:
            # Default to demo/voices/streaming_model
            script_dir = Path(__file__).parent
            voices_dir = script_dir / "voices" / "streaming_model"

        self.voices_dir = Path(voices_dir)
        self.voices: List[VoiceInfo] = []
        self._load_voices()

    def _load_voices(self):
        """Load all voice presets from the voices directory"""
        if not self.voices_dir.exists():
            print(f"Warning: Voices directory not found at {self.voices_dir}")
            return

        # Find all .pt files
        for pt_file in sorted(self.voices_dir.glob("*.pt")):
            voice_info = VoiceInfo(pt_file.stem, pt_file)
            self.voices.append(voice_info)

    def filter_by_language(self, lang_code: str) -> List[VoiceInfo]:
        """Filter voices by language code"""
        return [v for v in self.voices if v.language.lower() == lang_code.lower()]

    def get_by_name(self, name: str) -> Optional[VoiceInfo]:
        """Get a specific voice by name"""
        for voice in self.voices:
            if voice.name == name or voice.speaker_name == name:
                return voice
        return None

    def print_table(self, voices: Optional[List[VoiceInfo]] = None):
        """Print voices in a formatted table"""
        if voices is None:
            voices = self.voices

        if not voices:
            print("No voices found.")
            return

        # Calculate column widths
        name_width = max(len(v.name) for v in voices) + 2
        speaker_width = max(len(v.speaker_name) for v in voices) + 2
        lang_width = max(len(VoiceInfo.get_language_name(v.language)) for v in voices) + 2

        # Ensure minimum widths
        name_width = max(name_width, 20)
        speaker_width = max(speaker_width, 15)
        lang_width = max(lang_width, 12)

        # Print header
        print(f"\n{'Name':<{name_width}} {'Speaker':<{speaker_width}} {'Language':<{lang_width}} {'Gender':<8} {'Size (MB)':<10}")
        print("=" * (name_width + speaker_width + lang_width + 28))

        # Print voices
        for voice in voices:
            lang_name = VoiceInfo.get_language_name(voice.language)
            print(f"{voice.name:<{name_width}} {voice.speaker_name:<{speaker_width}} {lang_name:<{lang_width}} {voice.gender:<8} {voice.size_mb:>8.2f}")

        print(f"\nTotal: {len(voices)} voice(s)")

    def print_simple(self, voices: Optional[List[VoiceInfo]] = None):
        """Print simple list of voice names"""
        if voices is None:
            voices = self.voices

        for voice in voices:
            print(voice.name)

    def to_json(self, voices: Optional[List[VoiceInfo]] = None) -> str:
        """Convert voices to JSON format"""
        if voices is None:
            voices = self.voices

        data = {
            'total': len(voices),
            'voices_directory': str(self.voices_dir),
            'voices': [v.to_dict() for v in voices]
        }
        return json.dumps(data, indent=2)

    def get_statistics(self) -> Dict:
        """Get statistics about available voices"""
        if not self.voices:
            return {
                'total': 0,
                'by_language': {},
                'by_gender': {},
            }

        # Count by language
        by_lang = {}
        for voice in self.voices:
            lang = voice.language
            by_lang[lang] = by_lang.get(lang, 0) + 1

        # Count by gender
        by_gender = {}
        for voice in self.voices:
            gender = voice.gender
            by_gender[gender] = by_gender.get(gender, 0) + 1

        return {
            'total': len(self.voices),
            'by_language': by_lang,
            'by_gender': by_gender,
        }


def main():
    parser = argparse.ArgumentParser(
        description="List available VibeVoice voice presets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--format',
        choices=['table', 'json', 'simple'],
        default='table',
        help='Output format (default: table)'
    )
    parser.add_argument(
        '--lang',
        type=str,
        help='Filter by language code (e.g., en, de, fr)'
    )
    parser.add_argument(
        '--voices-dir',
        type=Path,
        help='Path to voices directory (default: demo/voices/streaming_model)'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show statistics about available voices'
    )

    args = parser.parse_args()

    # Initialize voice manager
    manager = VoiceManager(voices_dir=args.voices_dir)

    # Filter by language if specified
    voices = manager.voices
    if args.lang:
        voices = manager.filter_by_language(args.lang)
        if not voices:
            print(f"No voices found for language: {args.lang}")
            return

    # Show statistics if requested
    if args.stats:
        stats = manager.get_statistics()
        print("\n=== Voice Statistics ===")
        print(f"Total voices: {stats['total']}")
        print("\nBy Language:")
        for lang, count in sorted(stats['by_language'].items()):
            lang_name = VoiceInfo.get_language_name(lang)
            print(f"  {lang_name} ({lang}): {count}")
        print("\nBy Gender:")
        for gender, count in sorted(stats['by_gender'].items()):
            print(f"  {gender.capitalize()}: {count}")
        print()
        return

    # Display in requested format
    if args.format == 'table':
        manager.print_table(voices)
    elif args.format == 'json':
        print(manager.to_json(voices))
    elif args.format == 'simple':
        manager.print_simple(voices)


if __name__ == '__main__':
    main()
