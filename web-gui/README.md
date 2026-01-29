# VibeVoice-Narrator Web GUI

A modern web-based interface for VibeVoice-Narrator, a Markdown-to-Speech conversion tool. This is a fully functional implementation that provides a user-friendly interface for converting Markdown documents to natural-sounding speech audio.

## Features

### Core Functionality
- 📝 **Markdown Editor** - Built-in markdown editor with live preview
- 🎭 **Voice Selection** - Choose from multiple voice presets with preview functionality
- ⚙️ **Configurable Settings** - Adjust chunk depth, pause duration, device selection, and more
- 🎵 **TTS Conversion** - Convert markdown documents to speech with progress feedback
- 💾 **Audio Management** - Playback, download, and history tracking of generated audio
- 💾 **Session Persistence** - Autosave session data with version history (up to 20 versions)

### User Interface
- 🎨 **Light/Dark Theme Support** - Switch between light and dark themes, with system preference detection
- 📱 **Responsive Design** - Optimized for desktop and tablet devices
- 🔄 **Tab-based Navigation** - Organized interface with Document, Voice, Settings, and Player tabs
- 🔔 **Toast Notifications** - User feedback for all actions
- 📊 **Audio History** - Track and replay all generated audio files

### Backend Features
- 🚀 **RESTful API** - Clean API endpoints for TTS conversion and voice management
- 💾 **Model Caching** - Models and processors cached for improved performance
- 🔊 **Voice Preview** - Generate short preview clips for voice testing
- 📁 **Static File Serving** - Efficient serving of generated audio files

## Prerequisites

- Python 3.9 or higher
- Node.js 18 or higher
- pip (Python package manager)
- npm (Node package manager)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Dazlarus/VibeVoice-Narrator.git
cd VibeVoice-Narrator/web-gui
```

2. Install backend dependencies:
```bash
cd backend
pip install -r requirements.txt
```

For development and running tests, also install the development/test requirements:
```bash
# Preferred: install compiled dev requirements (if available)
pip install -r dev-requirements.txt  # recommended
# Or install directly from the development requirements input file
pip install -r dev-requirements.in
```

3. Install frontend dependencies:
```bash
cd ../frontend
npm install
```

4. Download voice files

The TTS backend requires voice model files (PyTorch `.pt` checkpoints). You can either copy the demo voices included in this repository for development, or download official voice files/releases for production testing.

Required (minimum):
- At least one voice checkpoint such as `en-Carter_man.pt` or `en-Emma_woman.pt` (examples included under `../demo/voices/streaming_model/`).

Optional:
- `extra-voices.zip` (additional voices or bundles)

Approximate file sizes (varies by model):
- Single voice checkpoint (`*.pt`): ~150-800 MB
- Voice bundles / `extra-voices.zip`: ~1–4 GB

Examples

a) Use demo voices included in this repository (recommended for local dev):

```bash
# Create target directory and copy demo models
mkdir -p data/voices
cp -v ../demo/voices/streaming_model/*.pt data/voices/
```

On Windows (PowerShell / cmd):

```powershell
mkdir data\voices
xcopy "..\demo\voices\streaming_model\*.pt" data\voices\ /Y
```

b) Download from a release (replace `v1.0` and filename with the actual tag/name):

```bash
# Example using curl
curl -L -o data/voices/en-Carter_man.pt \
  https://github.com/microsoft/VibeVoice/releases/download/v1.0/en-Carter_man.pt

# Example using wget
wget -O data/voices/en-Carter_man.pt \
  https://github.com/microsoft/VibeVoice/releases/download/v1.0/en-Carter_man.pt
```

If you obtain a ZIP bundle (e.g., `extra-voices.zip`), extract it into `data/voices/` so that the final layout looks like:

```
data/voices/
├── en-Carter_man.pt
├── en-Emma_woman.pt
└── extra-voices/
    ├── voice-A/
    │   └── model-files...
    └── voice-B/
        └── model-files...
```

Using the demo voices

- You can either copy files from `../demo/voices/streaming_model/` into `data/voices/` (recommended for quick start), or create a symlink:

```bash
ln -s ../../demo/voices/streaming_model data/voices
```

On Windows (Developer PowerShell / cmd with admin rights):

```powershell
mklink /J data\voices ..\demo\voices\streaming_model
```

Verification

Confirm voice files are visible to the backend by listing the directory:

```bash
ls -lh data/voices
# or on Windows
dir data\voices
```

Notes

- Keep storage and download sizes in mind when fetching multiple voices; some models are large and may take time to download.
- For production or CI, prefer hosting voices on an internal artifact server or use official GitHub Releases where available.

## Running the Application

### Windows

Double-click `start.bat` or run from command line:
```cmd
start.bat
```

### Linux/Mac

Make script executable and run:
```bash
chmod +x start.sh
./start.sh
```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000

## Development

### Running Backend Only

```bash
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Running Frontend Only

```bash
cd frontend
npm run dev
```

## Project Structure

```
web-gui/
├── frontend/                 # Next.js application
│   ├── app/
│   │   ├── page.tsx        # Main dashboard page
│   │   ├── layout.tsx      # Root layout
│   │   └── globals.css     # Global styles
│   ├── components/          # React components
│   │   ├── ui/             # shadcn/ui components
│   │   ├── audio-player.tsx
│   │   ├── configuration-panel.tsx
│   │   ├── document-upload.tsx
│   │   ├── voice-selection.tsx
│   │   └── theme-switcher.tsx
│   ├── lib/               # Utilities
│   │   ├── api.ts          # API client functions
│   │   └── utils.ts        # Helper functions
│   ├── providers/          # React context providers
│   │   └── theme-provider.tsx
│   └── package.json
├── backend/               # FastAPI application
│   ├── main.py           # FastAPI app entry point
│   ├── config.py          # Configuration settings
│   ├── routes/           # API endpoints
│   │   ├── tts.py         # TTS conversion endpoints
│   │   └── export.py      # Export functionality
│   └── requirements.txt
├── data/                 # Local data storage
│   ├── voices/          # Voice files (.pt)
│   ├── audio/           # Generated audio (.wav)
│   │   └── preview/      # Preview audio clips
│   └── documents/        # Uploaded documents
├── start.bat            # Windows launcher
├── start.sh             # Unix launcher
└── README.md
```

## Technology Stack

### Frontend
- **Next.js 14** - React framework with App Router
- **React 18** - UI library with hooks
- **TypeScript** - Type safety and better IDE support
- **Tailwind CSS** - Utility-first CSS for rapid styling
- **shadcn/ui** - Accessible component library built on Radix UI
- **Lucide React** - Icon library
- **Sonner** - Toast notification library

### Backend
- **FastAPI** - Modern Python web framework with automatic API documentation
- **Pydantic** - Data validation using Python type annotations
- **Uvicorn** - ASGI server for production deployment
- **VibeVoice Streaming Model** - TTS generation using VibeVoice-Realtime model

## API Documentation

For detailed API documentation, see [`../docs/web-gui-implementation.md`](../docs/web-gui-implementation.md).

### Main Endpoints

- `POST /api/tts/convert` - Convert markdown to speech
- `POST /api/tts/preview` - Generate voice preview
- `GET /voices` - List available voices
- `GET /config` - Get backend configuration
- `POST /api/export/download` - Export audio file

## Usage Guide

### 1. Upload Document
- Click the "Document" tab
- Upload a Markdown file or paste content directly
- The document is displayed with live preview
- Content is automatically saved to session

### 2. Select Voice
- Click the "Voice" tab
- Browse available voices grouped by language
- Click "Preview" to hear a sample of each voice
- Select your preferred voice

### 3. Configure Settings
- Click the "Settings" tab
- Adjust chunk depth (1-6 headings)
- Set pause duration between chunks (0-5000ms)
- Toggle heading inclusion in speech
- Select device (auto, cuda, cpu, mps)
- Set number of iterations (1 - server max; frontend adapts to server-provided max). Per-document and total quotas are derived from this value.

### 4. Convert to Speech
- Click "Convert to Speech" button
- Progress indicator shows conversion status
- Audio is generated and saved to `data/audio/`
- Audio appears in history list

### 5. Playback and Export
- Click the "Player" tab to play generated audio
- Use "Download" button to save audio as WAV file
- History shows all generated audio with timestamps

### 6. Session Management
- Session data is autosaved every 400ms
- Click "Save Version" to manually save current state
- Up to 20 versions are stored in browser localStorage
- Click "Reset Session" to clear all data

## Troubleshooting

### Common Issues

**Backend not starting**
- Check Python version (3.9+ required)
- Verify dependencies: `pip install -r backend/requirements.txt`
- Check port 8000 is not in use

**Frontend not starting**
- Check Node.js version (18+ required)
- Verify dependencies: `npm install`
- Check port 3000 is not in use

**Voices not loading**
- Verify voice files exist in `data/voices/` or `../demo/voices/streaming_model/`
- Check file permissions

**Conversion fails**
- Verify VibeVoice dependencies are installed in parent project
- Check device compatibility (cuda requires NVIDIA GPU)
- Review error messages in backend console

## License

This project is licensed under the **MIT License**.

See the top-level `LICENSE` file for the full text.
