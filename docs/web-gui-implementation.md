# VibeVoice-Narrator: Web GUI Implementation Status

## Overview

The web-based GUI for VibeVoice-Narrator has been successfully implemented and is fully functional. This document describes the current implementation status, architecture, and features of the web GUI.

**Deployment Note:** This application is designed for local deployment only. Users launch the application via a `.bat` file (Windows) or `.sh` script (Unix) and access it via `http://localhost:port` in their browser.

---

## 1. Current Implementation Status

### 1.1 Completed Features

✅ **Core Functionality**
- Document upload and markdown editor with live preview
- Voice selection with preview functionality
- Configurable TTS settings (chunk depth, pause duration, etc.)
- TTS conversion with progress feedback
- Audio playback and download
- Session persistence (autosave)
- Version history management

✅ **User Interface**
- Light/Dark theme support with system preference detection
- Responsive design for desktop and tablet devices
- Tab-based navigation (Document, Voice, Settings, Player)
- Audio history with playback and download
- Toast notifications for user feedback

✅ **Backend API**
- RESTful API endpoints for TTS conversion
- Voice preview generation
- Model and processor caching for performance
- Static file serving for audio
- CORS configuration for local development

### 1.2 Technology Stack (Implemented)

**Frontend**
- **Next.js 14** - React framework with App Router
- **React 18** - UI library
- **TypeScript** - Type safety
- **Tailwind CSS** - Utility-first CSS
- **shadcn/ui** - Accessible component library
- **Lucide React** - Icon library
- **Sonner** - Toast notifications

**Backend**
- **FastAPI** - Modern Python web framework
- **Pydantic** - Data validation
- **Uvicorn** - ASGI server
- **SQLite** - Local data storage (via filesystem)
- **VibeVoice Streaming Model** - TTS generation

---

## 2. Architecture

### 2.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    User Browser (localhost)                  │
│                           │                               │
│                           ▼                               │
│                    Next.js Frontend                     │
│                           │                               │
│                           ▼                               │
│                    FastAPI Backend                       │
│                           │                               │
│         ┌─────────────────┼─────────────────┐            │
│         │                 │                 │            │
│         ▼                 ▼                 ▼            │
│   VibeVoice Model    Voice Cache      Audio Storage       │
│   (Streaming)        (.pt files)      (.wav files)        │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Project Structure

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
│   │   ├── api.ts          # API client
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
└── README.md            # Installation and usage guide
```

---

## 3. API Endpoints

### 3.1 TTS Endpoints (`/api/tts`)

#### POST `/api/tts/convert`
Convert markdown document to speech audio.

**Request Body:**
```json
{
  "content": "string (markdown content)",
  "voice_id": "string (voice filename without extension)",
  "filename": "string (optional, output filename)",
  "chunk_depth": "int (default: 1)",
  "pause_ms": "int (default: 500)",
  "include_heading": "bool (default: false)",
  "strip_markdown": "bool (default: true)",
  "device": "string (default: 'auto')",
  "iterations": "int (default: 1). The server exposes a dynamic per-request maximum (e.g., 10). The frontend derives quotas from this value: per-document cap = 5 × per-request max; total cap = per-document cap × 3. Example: per-request max=10 → per-document=50 → total=150."
}
```

**Response:**
```json
{
  "success": "bool",
  "message": "string",
  "audio_url": "string (relative path)",
  "duration": "float (seconds)",
  "outputs": [
    {
      "audio_url": "string",
      "duration": "float",
      "filename": "string",
      "iteration": "int"
    }
  ]
}
```

#### POST `/api/tts/preview`
Generate a short voice preview clip.

**Request Body:**
```json
{
  "voice_id": "string",
  "text": "string (optional, default: 'Hello, this is a preview...')",
  "device": "string (default: 'auto')",
  "model": "string (optional, default from config)"
}
```

**Response:**
```json
{
  "success": "bool",
  "message": "string",
  "audio_url": "string",
  "duration": "float"
}
```

#### GET `/api/tts/status/{job_id}`
Get status of a TTS conversion job (placeholder for future WebSocket implementation).

### 3.2 Export Endpoints (`/api/export`)

#### POST `/api/export/download`
Export generated audio file for download.

**Request Body:**
```json
{
  "audio_url": "string (relative path)",
  "filename": "string",
  "format": "string (default: 'wav')"
}
```

### 3.3 Utility Endpoints

#### GET `/voices`
List available voices with metadata.

**Response:**
```json
{
  "voices": [
    {
      "id": "string (voice filename)",
      "name": "string (speaker name)",
      "language": "string",
      "gender": "string ('male' | 'female')",
      "path": "string (full file path)"
    }
  ]
}
```

#### GET `/config`
Get current backend configuration.

**Response:**
```json
{
  "default_model": "string",
  "default_device": "string",
  "sample_rate": "int",
  "default_pause_ms": "int",
  "default_chunk_depth": "int"
}
```

---

## 4. Configuration

### 4.1 Backend Configuration (`backend/config.py`)

```python
# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
VOICES_DIR = DATA_DIR / "voices"
DOCUMENTS_DIR = DATA_DIR / "documents"
AUDIO_DIR = DATA_DIR / "audio"

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
FRONTEND_URL = "http://localhost:3000"

# Model Configuration
DEFAULT_MODEL = "microsoft/VibeVoice-Realtime-0.5B"
DEFAULT_DEVICE = "auto"

# File Upload Limits
MAX_UPLOAD_SIZE = 1024 * 1024 * 1024  # 1GB

# Audio Settings
SAMPLE_RATE = 24000
DEFAULT_PAUSE_MS = 500
DEFAULT_CHUNK_DEPTH = 1

# Warmup Settings
WARMUP_PREVIEW = True
WARMUP_VOICE_ID = None
```

### 4.2 Frontend Configuration

The frontend uses environment variables and runtime configuration:

- **API Base URL**: Configured in `lib/api.ts`
- **Theme**: Stored in localStorage with system preference detection
- **Session Data**: Autosaved to localStorage with version history

---

## 5. Installation and Usage

### 5.1 Prerequisites

- Python 3.9 or higher
- Node.js 18 or higher
- pip (Python package manager)
- npm (Node package manager)

### 5.2 Installation Steps

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

If you are developing or running tests, install development/test dependencies as well:
```bash
pip install -r dev-requirements.txt  # preferred
# or
pip install -r dev-requirements.in
```

3. Install frontend dependencies:
```bash
cd ../frontend
npm install
```

4. Download voice files:
```bash
# Copy voice files to data/voices/
# Or use existing voices from ../demo/voices/streaming_model/
```

### 5.3 Running the Application

#### Windows
Double-click `start.bat` or run from command line:
```cmd
start.bat
```

#### Linux/Mac
Make script executable and run:
```bash
chmod +x start.sh
./start.sh
```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000

### 5.4 Development Mode

#### Running Backend Only
```bash
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### Running Frontend Only
```bash
cd frontend
npm run dev
```

---

## 6. Features and User Flow

### 6.1 Primary User Journey

1. **Document Upload**
   - User uploads a Markdown file or pastes content directly
   - Document is displayed with live preview
   - Content is autosaved to session storage

2. **Voice Selection**
   - User browses available voices
   - Voices are grouped by language
   - User can preview voices before selection
   - Selected voice is saved to session

3. **Configuration**
   - User adjusts chunk depth (1-6)
   - User sets pause duration between chunks (0-5000ms)
   - User toggles heading inclusion
   - User selects device (auto, cuda, cpu, mps)
   - User sets number of iterations (1 - server-side max). The server exposes `max_iterations_per_request` which is used by the frontend to derive larger quotas (per-document and total).

4. **Conversion**
   - User clicks "Convert to Speech"
   - Progress indicator shows conversion status
   - Audio is generated and saved to `data/audio/`
   - Audio is added to history

5. **Playback and Export**
   - User plays generated audio in the player tab
   - User can download audio as WAV file
   - History shows all generated audio files

### 6.2 Session Management

- **Autosave**: Session data is autosaved every 400ms
- **Version History**: Up to 20 versions are saved in localStorage
- **Reset**: User can clear session and history
- **Persistence**: Session survives page refresh

---

## 7. Performance Optimizations

### 7.1 Backend Caching

- **Model Cache**: Models are cached per device to avoid reloading
- **Processor Cache**: Processors are cached per model name
- **Voice Cache**: Voice files are cached in memory after first load
- **Thread Safety**: Cache access is protected with locks

### 7.2 Frontend Optimizations

- **Debouncing**: Session save is debounced to reduce writes
- **Lazy Loading**: Components load on demand
- **Abort Controllers**: Requests can be cancelled to prevent memory leaks

---

## 8. Known Limitations

### 8.1 Current Limitations

- **Single User**: No multi-user support (local deployment only)
- **No Authentication**: No user accounts or authentication
- **No Cloud Storage**: All data is stored locally
- **No Real-time Streaming**: WebSocket support is placeholder only
- **Batch Processing**: Limited to 10 iterations per request
- **File Size**: Maximum 1 GB per document

### 8.2 Future Enhancements

- WebSocket support for real-time progress updates
- Multi-document batch processing
- Custom voice upload and management
- Advanced audio editing and trimming
- Export to multiple formats (MP3, OGG, etc.)
- User accounts and preferences persistence

---

## 9. Troubleshooting

### 9.1 Common Issues

**Backend not starting**
- Check Python version (3.9+ required)
- Verify dependencies are installed: `pip install -r backend/requirements.txt`
- Check port 8000 is not in use

**Frontend not starting**
- Check Node.js version (18+ required)
- Verify dependencies are installed: `npm install`
- Check port 3000 is not in use

**Voices not loading**
- Verify voice files exist in `data/voices/` or `demo/voices/streaming_model/`
- Check file permissions

**Conversion fails**
- Verify VibeVoice dependencies are installed in parent project
- Check device compatibility (cuda requires NVIDIA GPU)
- Review error messages in backend console

### 9.2 Debug Mode

Enable verbose logging in backend by setting environment variables:
```bash
export VIBEVOICE_DEBUG=1
```

You can also enable automatic reload for the backend server (useful during development). Set `RELOAD=1` or `RELOAD=true` in your environment or `.env` file to enable uvicorn reload. If `RELOAD` is not set, the server will default to enabling reload when `VIBEVOICE_DEBUG` is true.

---

## 10. License

This project follows the same license as the original VibeVoice repository.
