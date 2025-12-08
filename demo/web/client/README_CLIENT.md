# VibeVoice-Realtime Client

React + Vite frontend for VibeVoice-Realtime TTS Demo.

## Features

- ✅ Real-time TTS streaming via WebSocket
- ✅ Interactive audio playback with Web Audio API
- ✅ Voice selection and parameter controls (CFG, Inference Steps)
- ✅ Streaming text preview with typing animation
- ✅ Audio recording and WAV export
- ✅ Real-time metrics and logging
- ✅ Tailwind CSS styling

## Development

### Prerequisites

- Node.js 16+ 
- Python backend running on `http://localhost:3000`

### Install Dependencies

```bash
npm install
```

### Run Development Server

```bash
npm run dev
```

The app will be available at `http://localhost:5173`

### Build for Production

```bash
npm run build
```

## Project Structure

```
src/
├── components/          # React components
│   ├── TextInput.jsx
│   ├── StreamingPreview.jsx
│   ├── Controls.jsx
│   ├── Metrics.jsx
│   └── Logs.jsx
├── audio/              # Web Audio API logic
│   └── AudioPlayer.js
├── websocket/          # WebSocket client
│   └── StreamingClient.js
├── utils/              # Utility functions
│   ├── audioUtils.js
│   └── logger.js
├── App.jsx             # Main app component
├── main.jsx            # Entry point
└── index.css           # Global styles
```

## Configuration

The Vite dev server proxies API requests to the Python backend:

- `/config` → `http://localhost:3000/config`
- `/stream` → `ws://localhost:3000/stream`

See `vite.config.js` for proxy configuration.

## Usage

1. Start the Python backend server
2. Start the Vite dev server
3. Open `http://localhost:5173`
4. Enter text and click "Start" to generate speech
5. Use controls to adjust CFG scale and inference steps
6. Save generated audio as WAV file
