#!/usr/bin/env python
"""
VibeVoice ASR Streaming FastAPI Demo

This demo uses the streaming ASR server instead of loading the model directly.
The browser holds one WebSocket and pushes raw PCM down it; between the
microphone and the model there is a socket and a relay, nothing else. Upload
uses the same socket -- a file is just PCM that already exists -- so the two
paths cannot drift apart.

  GET  /              the UI, one self-contained page
  GET  /config        the chunk geometry, read from the upstream server
  GET  /healthz       readiness, includes upstream status
  WS   /ws/asr        audio up, transcript down

Usage:
    python fastapi_asr_streaming_demo_api.py --api_url http://localhost:8000
"""
import argparse
import asyncio
import json
import logging
import os
import re
import subprocess
import threading
import time
import urllib.request
from urllib.parse import urlparse

import httpx
import uvicorn
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger("vibevoice.demo")


def _ws_url(api_url: str) -> str:
    """http://host:port -> ws://host:port/v1/stream."""
    u = urlparse(api_url)
    scheme = "wss" if u.scheme == "https" else "ws"
    return f"{scheme}://{u.netloc}/v1/stream"


async def _pump_up(browser: WebSocket, upstream) -> None:
    """Browser -> server: PCM frames, then the "end" marker."""
    while True:
        msg = await browser.receive()
        if msg.get("type") == "websocket.disconnect":
            await upstream.close()
            return
        if msg.get("bytes") is not None:
            await upstream.send(msg["bytes"])
        elif msg.get("text") is not None:
            await upstream.send(msg["text"])
            if msg["text"] == "end":
                return


async def _pump_down(browser: WebSocket, upstream) -> None:
    """Server -> browser: one message per chunk, verbatim.

    The server hangs up as soon as it has sent the final message, often without
    a closing handshake. That is the end of the relay, not a failure -- a
    session that really was cut short is missing its "done" message, which the
    page already treats as unfinished.
    """
    try:
        async for raw in upstream:
            await browser.send_text(raw if isinstance(raw, str) else raw.decode())
    except ConnectionClosed:
        return


def create_app(api_url: str, max_new_tokens: int) -> FastAPI:
    """Build the demo around one upstream server."""
    app = FastAPI(title="VibeVoice streaming ASR demo")
    upstream_ws = _ws_url(api_url)

    async def upstream_config() -> dict:
        """The geometry the server reads off its checkpoint.

        Fetched rather than hard-coded: the page captures at ``sample_rate``,
        and the two released checkpoints differ in chunk length. Cached after
        the first success so the page does not re-probe on every reload.
        """
        if not hasattr(app.state, "config"):
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{api_url}/v1/config")
                resp.raise_for_status()
                app.state.config = resp.json()
        return app.state.config

    @app.get("/healthz")
    async def healthz():
        try:
            cfg = await upstream_config()
        except Exception as e:
            return JSONResponse({"status": "upstream unreachable",
                                 "api_url": api_url, "error": str(e)},
                                status_code=503)
        return {"status": "ok", "api_url": api_url, "model": cfg.get("model")}

    @app.get("/config")
    async def config():
        cfg = dict(await upstream_config())
        cfg["max_new_tokens"] = max_new_tokens
        return JSONResponse(cfg)

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return PAGE

    @app.websocket("/ws/asr")
    async def ws_asr(browser: WebSocket):
        """Relay one browser session to one upstream session.

        Both directions run concurrently: the browser keeps pushing audio while
        transcripts for earlier chunks are still coming back, which is the whole
        point of streaming.
        """
        await browser.accept()
        try:
            opts = json.loads(await browser.receive_text())
        except Exception:
            await browser.close()
            return

        opts.setdefault("max_tokens", max_new_tokens)
        try:
            async with websockets.connect(upstream_ws, max_size=None) as upstream:
                await upstream.send(json.dumps(opts))
                up = asyncio.create_task(_pump_up(browser, upstream))
                down = asyncio.create_task(_pump_down(browser, upstream))
                done, _ = await asyncio.wait(
                    {up, down}, return_when=asyncio.FIRST_COMPLETED)
                if down in done:
                    # Upstream is finished or gone; nothing left to relay.
                    up.cancel()
                else:
                    # The browser sent "end", but the tail chunks and the final
                    # "done" message are still upstream. Cancelling here would
                    # drop exactly the part the user waits for.
                    await down
                for task in (up, down):
                    if task.done() and not task.cancelled():
                        task.result()
        except WebSocketDisconnect:
            return
        except Exception as e:
            logger.exception("relay failed")
            try:
                await browser.send_text(json.dumps({"error": str(e)}))
            except Exception:
                pass

    return app


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VibeVoice Streaming ASR</title>
<style>
  :root {
    color-scheme: dark;
    --bg:#0f1117; --panel:#171a23; --line:#262b38; --fg:#e6e8ef;
    --muted:#8b93a7; --faint:#5c6478; --accent:#5b8cff; --rec:#ff4d5e;
    --ok:#31c48d; --ghost:#232838; --field:#10131c;
    --hover:#1d2230; --sel:#1d2436;
  }
  :root[data-theme="light"] {
    color-scheme: light;
    --bg:#f4f6fa; --panel:#ffffff; --line:#dde2ec; --fg:#171a22;
    --muted:#5b6377; --faint:#8b93a7; --accent:#2f5fd0; --rec:#d92d3f;
    --ok:#0e9250; --ghost:#eaeef6; --field:#f7f9fc;
    --hover:#eef2f9; --sel:#e5edfb;
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font-size:15px;
         font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,
                     "Helvetica Neue","PingFang SC","Microsoft YaHei",sans-serif; }
  header { padding:18px 24px; border-bottom:1px solid var(--line);
           display:flex; align-items:center; gap:12px; flex-wrap:wrap; }
  h1 { font-size:17px; margin:0; font-weight:600; }
  .dot { width:8px; height:8px; border-radius:50%; background:var(--muted); }
  .dot.ok { background:var(--ok); }
  .dot.rec { background:var(--rec); animation:pulse 1.2s infinite; }
  @keyframes pulse { 50% { opacity:.25; } }
  main { display:grid; grid-template-columns:320px 1fr; gap:20px; padding:20px 24px;
         min-height:calc(100vh - 118px); align-content:stretch; }
  main > * { min-width:0; }
  .panel { background:var(--panel); border:1px solid var(--line);
           border-radius:10px; padding:16px; }
  .panel h2 { font-size:13px; text-transform:uppercase; letter-spacing:.06em;
              color:var(--muted); margin:0 0 12px; font-weight:600; }
  button { font:inherit; border:0; border-radius:8px; padding:10px 16px;
           background:var(--accent); color:#fff; cursor:pointer; min-height:40px; }
  button.ghost { background:var(--ghost); color:var(--fg); }
  button:disabled { opacity:.45; cursor:not-allowed; }
  #theme { margin-left:auto; background:transparent; color:var(--muted);
           border:1px solid var(--line); padding:0 11px; min-height:32px;
           font-size:15px; line-height:1; }
  label { display:block; font-size:13px; color:var(--muted); margin:14px 0 6px; }
  input[type=text], input[type=number] { width:100%; padding:9px 11px;
    border-radius:8px; border:1px solid var(--line); background:var(--field);
    color:var(--fg); font:inherit; }
  input[type=file] { width:100%; font-size:13px; color:var(--muted); }
  .row { display:flex; gap:10px; flex-wrap:wrap; }
  .seg { border-left:2px solid var(--accent); padding:2px 0 2px 12px;
         margin:0 0 16px; border-radius:0 6px 6px 0; }
  .seg.play { cursor:pointer; }
  @media (hover:hover) { .seg.play:hover { background:var(--hover); } }
  .seg.on { background:var(--sel); }
  .seg .meta { font-size:12px; color:var(--muted); margin-bottom:4px; }
  .seg.on .meta { color:var(--accent); }
  .seg .body { line-height:1.75; white-space:pre-wrap; word-break:break-word; }
  #out:empty::before { content:"Transcript appears here as you speak.";
                       color:var(--muted); }
  #out { min-height:320px; }
  .status { font-size:13px; color:var(--muted); margin-top:12px; }
  .bar { display:flex; align-items:center; justify-content:space-between;
         gap:10px; flex-wrap:wrap; margin-bottom:12px; }
  .bar button { min-height:32px; padding:6px 12px; font-size:13px; }
  .mut { color:var(--faint); font-weight:400; }

  @media (max-width: 720px) {
    header {
      padding: 14px 16px;
    }

    main {
      grid-template-columns: 1fr;
      gap: 12px;
      padding: 12px;
      min-height: auto;
    }

    .panel {
      padding: 14px;
    }

    button {
      min-height: 44px;
    }

    .panel > .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      width: 100%;
    }

    .panel > .row button {
      width: 100%;
    }

    .bar {
      align-items: flex-start;
    }

    .bar > .row {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      width: 100%;
    }

    .bar button {
      width: 100%;
    }

    input[type=text],
    input[type=number],
    input[type=file] {
      font-size: 16px;
    }

    #out {
      min-height: 240px;
    }
  }
</style>
<script>
// Runs before first paint so the page does not flash the wrong theme. Follows
// the OS until the header button is pressed, after which the choice sticks.
(function () {
  var t = null;
  try { t = localStorage.getItem('vv-theme'); } catch (e) {}
  if (t !== 'light' && t !== 'dark')
    t = matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
  document.documentElement.dataset.theme = t;
})();
</script>
</head>
<body>
<header>
  <h1>VibeVoice Streaming ASR</h1>
  <span class="dot" id="dot"></span>
  <span class="status" id="conn" style="margin:0">idle</span>
  <button id="theme" type="button"></button>
</header>
<main>
  <div class="panel">
    <h2>Input</h2>
    <div class="row">
      <button id="rec">Record</button>
      <button id="stop" class="ghost" disabled>Stop</button>
    </div>
    <label>Or transcribe a file</label>
    <input type="file" id="file" accept="audio/*,video/*">
    <div class="row" style="margin-top:10px">
      <button id="send" disabled>Transcribe</button>
      <button id="cancel" class="ghost" disabled>Cancel</button>
    </div>
    <label>Hotwords (context_info)</label>
    <input type="text" id="ctx" placeholder="Microsoft,VibeVoice">
    <label>Temperature <span class="mut">0 = greedy</span></label>
    <input type="number" id="temp" value="0" min="0" max="2" step="0.1">
    <label>Top-p <span class="mut" id="toppNote">needs temperature &gt; 0</span></label>
    <input type="number" id="topp" value="1" min="0.05" max="1" step="0.05" disabled>
    <div class="status" id="status">ready</div>
  </div>
  <div class="panel">
    <div class="bar">
      <h2 style="margin:0">Transcript</h2>
      <div class="row">
        <button class="ghost" id="copy" disabled>Copy</button>
        <button class="ghost" id="srt" disabled>SRT</button>
        <button class="ghost" id="clear" disabled>Clear</button>
      </div>
    </div>
    <div id="out"></div>
  </div>
</main>
<script>
const $ = id => document.getElementById(id);
let SR = 24000, MAXTOK = 256, sock = null, ctxAudio = null, node = null,
    media = null, lastSrt = '', segments = [], pcm = null;

// Kept so a card can be played back. The upload is already one Float32Array;
// the mic is stored as Int16 in arrival order, which halves the memory for
// something whose only destination is a speaker. Past the cap audio is dropped
// and those cards stop being clickable, rather than the tab dying.
const PLAYBACK_MAX_S = 1800;
let playPcm = null, micChunks = [], micSamples = 0;
let playCtx = null, player = null, playing = null;

function playable() { return !!(playPcm || micSamples); }

function keepMic(frame) {
  if (micSamples >= PLAYBACK_MAX_S * SR) return;
  const q = new Int16Array(frame.length);
  for (let i = 0; i < frame.length; i++)
    q[i] = Math.max(-1, Math.min(1, frame[i])) * 32767;
  micChunks.push(q);
  micSamples += q.length;
}

// Samples [a, b) of this run, whichever source it came from.
function sliceAudio(a, b) {
  if (playPcm)
    return a >= playPcm.length ? null
                               : playPcm.subarray(a, Math.min(b, playPcm.length));
  b = Math.min(b, micSamples);
  if (a >= b) return null;
  const out = new Float32Array(b - a);
  let pos = 0;
  for (const c of micChunks) {
    const s = Math.max(a, pos), e = Math.min(b, pos + c.length);
    for (let i = s; i < e; i++) out[i - a] = c[i - pos] / 32768;
    pos += c.length;
    if (pos >= b) break;
  }
  return out;
}

function stopPlay() {
  if (player) {
    player.onended = null;
    try { player.stop(); } catch (e) {}
    player.disconnect();
    player = null;
  }
  playing = null;
}

function playSegment(i) {
  const s = segments[i];
  if (!s || !playable()) return;
  if (playing === i) { stopPlay(); render(); return; }
  const clip = sliceAudio(Math.floor(s.Start * SR), Math.ceil(s.End * SR));
  if (!clip || !clip.length) { $('status').textContent = 'no audio kept for this card'; return; }
  stopPlay();
  playCtx = playCtx || new AudioContext();
  const buf = playCtx.createBuffer(1, clip.length, SR);
  buf.getChannelData(0).set(clip);
  player = playCtx.createBufferSource();
  player.buffer = buf;
  player.connect(playCtx.destination);
  player.onended = () => { if (playing === i) { playing = null; render(); } };
  player.start();
  playing = i;
  render();
}

$('out').onclick = ev => {
  const el = ev.target.closest('.seg');
  if (el) playSegment(+el.dataset.i);
};

fetch('/config').then(r => r.json()).then(c => {
  SR = c.sample_rate; MAXTOK = c.max_new_tokens;
  $('status').textContent = 'ready -- ' + c.chunk_seconds.toFixed(2) + 's chunks';
});

function setConn(text, cls) {
  $('conn').textContent = text;
  $('dot').className = 'dot' + (cls ? ' ' + cls : '');
}
function options() {
  return { context_info: $('ctx').value.trim() || null,
           temperature: parseFloat($('temp').value) || 0,
           top_p: parseFloat($('topp').value) || 1,
           max_tokens: MAXTOK };
}
// Top-p only bites once sampling is on, so it follows the temperature box.
$('temp').oninput = () => {
  const on = (parseFloat($('temp').value) || 0) > 0;
  $('topp').disabled = !on;
  $('toppNote').textContent = on ? 'nucleus sampling' : 'needs temperature > 0';
};
// textContent throughout: a transcript is model output, and building this with
// innerHTML would let a spoken "<script>" run in the page.
function render() {
  const out = $('out');
  out.textContent = '';
  const canPlay = playable();
  segments.forEach((seg, i) => {
    const card = document.createElement('div');
    card.className = 'seg' + (canPlay ? ' play' : '') + (playing === i ? ' on' : '');
    card.dataset.i = i;
    if (canPlay) card.title = 'Click to play this segment';
    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.textContent = (playing === i ? '▶  ' : '')
                     + 'Speaker ' + seg.Speaker + '  ·  '
                     + seg.Start.toFixed(2) + 's - ' + seg.End.toFixed(2) + 's';
    const body = document.createElement('div');
    body.className = 'body';
    body.textContent = seg.Content;
    card.appendChild(meta); card.appendChild(body); out.appendChild(card);
  });
  const empty = segments.length === 0;
  $('copy').disabled = empty;
  $('clear').disabled = empty;
  $('srt').disabled = !lastSrt;
}
function connect() {
  return new Promise((resolve, reject) => {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(proto + '://' + location.host + '/ws/asr');
    ws.onopen = () => { ws.send(JSON.stringify(options())); setConn('connected', 'ok'); resolve(ws); };
    ws.onerror = () => reject(new Error('connection failed'));
    ws.onclose = () => setConn('idle', '');
    ws.onmessage = ev => {
      const msg = JSON.parse(ev.data);
      if (msg.error) { $('status').textContent = 'error: ' + msg.error; finish(); return; }
      if (msg.segments) { segments = msg.segments; render(); }
      $('status').textContent = msg.done
        ? 'done -- ' + msg.total_chunks + ' chunks'
        : 'receiving -- ' + (msg.chunk + 1) + ' chunks';
      if (msg.done) { lastSrt = msg.srt || ''; render(); finish(); }
    };
  });
}
function finish() {
  $('rec').disabled = false; $('stop').disabled = true; $('file').disabled = false;
  $('send').disabled = !$('file').files.length; $('cancel').disabled = true;
  setConn('idle', '');
  if (node) { node.disconnect(); node = null; }
  if (media) { media.getTracks().forEach(t => t.stop()); media = null; }
  if (ctxAudio) { ctxAudio.close(); ctxAudio = null; }
}

function reset(msg) {
  segments = []; lastSrt = '';
  stopPlay();
  playPcm = null; micChunks = []; micSamples = 0;
  render();
  $('status').textContent = msg;
}

$('rec').onclick = async () => {
  reset('starting...');
  try {
    media = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) { $('status').textContent = 'microphone denied'; return; }
  sock = await connect();
  $('rec').disabled = true; $('stop').disabled = false; $('file').disabled = true;
  $('send').disabled = true; $('cancel').disabled = false;
  setConn('recording', 'rec');
  ctxAudio = new AudioContext({ sampleRate: SR });
  const src = ctxAudio.createMediaStreamSource(media);
  node = ctxAudio.createScriptProcessor(4096, 1, 1);
  node.onaudioprocess = e => {
    if (!sock || sock.readyState !== WebSocket.OPEN) return;
    const frame = new Float32Array(e.inputBuffer.getChannelData(0));
    // Kept before the send, so a card still plays what the microphone heard
    // even if the network dropped that frame on the way out.
    keepMic(frame);
    sock.send(frame.buffer);
  };
  src.connect(node); node.connect(ctxAudio.destination);
};

$('stop').onclick = () => {
  $('status').textContent = 'flushing...';
  if (sock && sock.readyState === WebSocket.OPEN) sock.send('end');
  $('stop').disabled = true;
};

// Decoding happens on pick, sending on the button: options set after choosing a
// file still apply, and a mis-picked file costs nothing.
$('file').onchange = async ev => {
  pcm = null; $('send').disabled = true;
  const f = ev.target.files[0];
  if (!f) { $('status').textContent = 'ready'; return; }
  $('status').textContent = 'decoding...';
  try {
    const raw = await f.arrayBuffer();
    const tmp = new AudioContext();
    const decoded = await tmp.decodeAudioData(raw);
    tmp.close();
    const off = new OfflineAudioContext(1, Math.ceil(decoded.duration * SR), SR);
    const s = off.createBufferSource();
    s.buffer = decoded; s.connect(off.destination); s.start();
    pcm = (await off.startRendering()).getChannelData(0);
  } catch (e) { $('status').textContent = 'cannot decode: ' + e.message; return; }
  $('send').disabled = false;
  $('status').textContent = 'ready -- ' + (pcm.length / SR).toFixed(1) + 's decoded';
};

$('send').onclick = async () => {
  if (!pcm) return;
  reset('streaming...');
  playPcm = pcm;
  sock = await connect();
  $('rec').disabled = true; $('file').disabled = true;
  $('send').disabled = true; $('cancel').disabled = false;
  // Sent in the cadence a microphone would produce, so text arrives gradually.
  for (let i = 0; i < pcm.length; i += SR / 2) {
    if (!sock || sock.readyState !== WebSocket.OPEN) return;
    sock.send(new Float32Array(pcm.slice(i, i + SR / 2)).buffer);
    await new Promise(r => setTimeout(r, 10));
  }
  if (sock && sock.readyState === WebSocket.OPEN) sock.send('end');
  $('status').textContent = 'sent, waiting for the tail...';
};

$('cancel').onclick = () => {
  if (sock) sock.close();
  sock = null;
  finish();
  $('status').textContent = 'cancelled';
};

$('clear').onclick = () => { reset('ready'); };

$('copy').onclick = () => {
  navigator.clipboard.writeText(
    segments.map(s => 'Speaker ' + s.Speaker + ': ' + s.Content).join('\\n'));
  $('status').textContent = 'copied ' + segments.length + ' segments';
};

$('srt').onclick = () => {
  const url = URL.createObjectURL(new Blob([lastSrt], { type: 'text/plain' }));
  const a = document.createElement('a');
  a.href = url; a.download = 'transcript.srt'; a.click();
  URL.revokeObjectURL(url);
};

// The glyph shows what pressing it gets you, not what you are looking at.
const root = document.documentElement;
function paintTheme() {
  const light = root.dataset.theme === 'light';
  $('theme').textContent = light ? '☾' : '☀';
  $('theme').title = light ? 'Switch to dark' : 'Switch to light';
  $('theme').setAttribute('aria-label', $('theme').title);
}
$('theme').onclick = () => {
  root.dataset.theme = root.dataset.theme === 'light' ? 'dark' : 'light';
  try { localStorage.setItem('vv-theme', root.dataset.theme); } catch (e) {}
  paintTheme();
};
paintTheme();
render();
</script>
</body>
</html>
"""


CLOUDFLARED_PATH = os.path.expanduser("~/.local/bin/cloudflared")
CLOUDFLARED_URL = ("https://github.com/cloudflare/cloudflared/releases/latest"
                   "/download/cloudflared-linux-amd64")
TUNNEL_RE = re.compile(r"https://[-a-z0-9]+\.trycloudflare\.com")


def download_cloudflared() -> bool:
    """Fetch the cloudflared binary unless it is already there."""
    if os.path.exists(CLOUDFLARED_PATH):
        return True
    print("Downloading cloudflared...")
    os.makedirs(os.path.dirname(CLOUDFLARED_PATH), exist_ok=True)
    try:
        urllib.request.urlretrieve(CLOUDFLARED_URL, CLOUDFLARED_PATH)
        os.chmod(CLOUDFLARED_PATH, 0o755)
        return True
    except Exception as e:
        print(f"Failed to download cloudflared: {e}")
        return False


def start_cloudflared_tunnel(port: int):
    """Publish the demo on a public https URL, printed once it is assigned.

    https matters beyond convenience: getUserMedia only runs in a secure
    context, so a plain http address on anything but localhost cannot reach
    the microphone at all.
    """
    if not download_cloudflared():
        return None
    process = subprocess.Popen(
        [CLOUDFLARED_PATH, "tunnel", "--url", f"http://localhost:{port}",
         "--no-autoupdate"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def watch():
        for line in process.stdout:
            found = TUNNEL_RE.search(line)
            if found:
                print(f"\n  public URL: {found.group(0)}\n", flush=True)

    threading.Thread(target=watch, daemon=True).start()
    time.sleep(3)
    return process


def main():
    parser = argparse.ArgumentParser(description="VibeVoice ASR Streaming FastAPI Demo")
    parser.add_argument(
        "--api_url",
        type=str,
        default="http://localhost:8000",
        help="URL of the streaming ASR server"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Default max new tokens per chunk"
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
        "--cloudflared",
        action="store_true",
        help="Create a public link using cloudflared tunnel"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    tunnel = start_cloudflared_tunnel(args.port) if args.cloudflared else None
    app = create_app(args.api_url, args.max_new_tokens)
    print(f"relaying to {args.api_url}, open http://localhost:{args.port}")
    try:
        uvicorn.run(app, host=args.host, port=args.port)
    finally:
        if tunnel:
            tunnel.terminate()


if __name__ == "__main__":
    main()
