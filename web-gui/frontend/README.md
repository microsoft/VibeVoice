# VibeVoice-Narrator — Frontend

The frontend is a Next.js application that provides the web UI for VibeVoice-Narrator. It allows users to upload or edit Markdown documents, preview the chunked content, select voices, and download generated audio produced by the backend TTS service.

## Quick start (development)

Install dependencies and start the dev server (choose your package manager):

```bash
# Install
npm install
# or
# yarn
# pnpm install
# bun install

# Start development server
npm run dev
# or
# yarn dev
# pnpm dev
# bun dev
```

The app will be available at http://localhost:3000 by default.

## Backend API connection

Configure the backend base URL using `NEXT_PUBLIC_API_URL`:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

If your backend requires authentication, set an `AUTH_TOKEN` environment variable and the frontend will send it as a Bearer token (`Authorization: Bearer <AUTH_TOKEN>`).

API client utilities are found in `src/lib` (see `src/lib/api.ts`).

The frontend expects backend routes like:
- `/api/tts/convert` — convert markdown to audio
- `/api/tts/preview` — preview a voice clip
- `/api/export/download` — generate a download link (export)

## Environment examples

Create `.env.local` inside the `frontend` directory with:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
AUTH_TOKEN=your-token-if-needed
```

## Key files to edit

- `app/page.tsx` — main entry for top-level UI
- `src/components/` — UI components (upload, preview, controls)
- `src/lib/` — client API helpers and integrations
- `styles/` or `globals.css` — global styles

## Tests

Run Playwright tests with:

```bash
npx playwright test
```

## Deployment & Architecture

For full architecture and deployment instructions, see the parent repository README:

- ../README.md — main repo docs (deployment, infra, service responsibilities)

## Contributing

When adding features, include tests and update docs. Follow existing UI and API patterns and add a short note to this README describing any new environment variables or runtime requirements.
