# VibeVoice Voice Files

This directory contains voice preset files for VibeVoice-Realtime model.

## Downloading Voice Files

Voice files (`.pt` format) need to be downloaded from the official VibeVoice repository:

https://github.com/microsoft/VibeVoice/tree/main/demo/voices/streaming_model

## Available Voices

The following voice presets are available:
- `en-Carter_man.pt` - Male voice (English)
- `en-Davis_man.pt` - Male voice (English)
- `en-Emma_woman.pt` - Female voice (English)
- `en-Frank_man.pt` - Male voice (English)
- `en-Grace_woman.pt` - Female voice (English)
- `en-Mike_man.pt` - Male voice (English)
- `de-Spk0_man.pt` - Male voice (German)
- `de-Spk1_woman.pt` - Female voice (German)
- `fr-Spk0_man.pt` - Male voice (French)
- `fr-Spk1_woman.pt` - Female voice (French)
- `in-Samuel_man.pt` - Male voice (Hindi)
- `it-Spk0_woman.pt` - Female voice (Italian)
- `it-Spk1_man.pt` - Male voice (Italian)
- `jp-Spk0_man.pt` - Male voice (Japanese)
- `jp-Spk1_woman.pt` - Female voice (Japanese)
- `kr-Spk0_woman.pt` - Female voice (Korean)
- `kr-Spk1_man.pt` - Male voice (Korean)
- `nl-Spk0_man.pt` - Male voice (Dutch)
- `nl-Spk1_woman.pt` - Female voice (Dutch)
- `pl-Spk0_man.pt` - Male voice (Polish)
- `pl-Spk1_woman.pt` - Female voice (Polish)
- `pt-Spk0_woman.pt` - Female voice (Portuguese)
- `pt-Spk1_man.pt` - Male voice (Portuguese)
- `sp-Spk0_woman.pt` - Female voice (Spanish)
- `sp-Spk1_man.pt` - Male voice (Spanish)

## Usage

Download voice files to this directory, then use the demo script:

```bash
python demo\chunked_markdown_tts_realtime.py --markdown demo\example_scripts\simple.md --speaker Carter --output test.wav --depth 3 --device cuda
```

## Creating Custom Voices

To create your own voice file, see the documentation at:
https://github.com/microsoft/VibeVoice#creating-voice-presets
