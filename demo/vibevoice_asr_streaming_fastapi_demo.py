"""FastAPI streaming ASR demo for VibeVoice-ASR-Streaming.

Loads the checkpoint in-process and serves a page that transcribes while the
audio is still arriving. Run it, then open the printed URL.
"""

import argparse
import asyncio
import json
import os

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

state = {}
gpu_lock = asyncio.Lock()


def load_model(model_path: str, device: str, attn_implementation: str):
    # A local checkpoint directory or a HF repo id, the same as the
    # from_pretrained calls below.
    name = "preprocessor_config.json"
    if os.path.isdir(model_path):
        config_path = os.path.join(model_path, name)
    else:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(model_path, name)

    with open(config_path) as f:
        cfg = json.load(f)

    missing = [k for k in ("chunk_frames", "lookahead_frames") if k not in cfg]
    if missing:
        raise SystemExit(
            f"{config_path} has no {', '.join(missing)}, so {model_path} is not "
            "a streaming checkpoint. Serve it with "
            "demo/vibevoice_asr_gradio_demo.py instead.")

    sample_rate = cfg["target_sample_rate"]
    # Streaming model: the frame size is the checkpoint's, never a constant --
    # a wrong one mis-cuts every window without raising.
    frame_samples = cfg["speech_tok_compress_ratio"]
    chunk_samples = cfg["chunk_frames"] * frame_samples
    lookahead_samples = cfg["lookahead_frames"] * frame_samples

    processor = VibeVoiceASRProcessor.from_pretrained(model_path)
    if processor.tokenizer.text_chunk_end_id is None:
        raise SystemExit(
            f"{model_path} is not a streaming checkpoint: its tokenizer has no "
            "<|text_chunk_end|>, so no chunk would ever end.")

    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=device,
        attn_implementation=attn_implementation,
    ).eval()

    state.update(
        model=model,
        tokenizer=processor.tokenizer,
        sample_rate=sample_rate,
        chunk_samples=chunk_samples,
        window_samples=chunk_samples + lookahead_samples,
        chunk_seconds=chunk_samples / sample_rate,
    )
    print(f"chunk {cfg['chunk_frames']} frames ({chunk_samples / sample_rate:.4f}s), "
          f"lookahead {cfg['lookahead_frames']} frames")


def transcribe_window(window: np.ndarray, session: dict) -> str:
    """Encode one window and advance the model by one chunk."""
    model = state["model"]
    audio = torch.from_numpy(window).to(next(model.parameters()).device)
    features = model.encode_speech(audio.unsqueeze(0))
    text, _ = model.streaming_generate_step(
        audio_features=features,
        streaming_state=session["stream"],
        tokenizer=state["tokenizer"],
        max_new_tokens=session["max_tokens"],
        temperature=session["temperature"],
    )
    return text


app = FastAPI(title="VibeVoice streaming ASR demo")


@app.get("/healthz")
async def healthz():
    return JSONResponse({"status": "ok" if state else "loading"})


@app.get("/config")
async def config():
    return JSONResponse({
        "sample_rate": state["sample_rate"],
        "chunk_seconds": state["chunk_seconds"],
    })


@app.get("/", response_class=HTMLResponse)
async def index():
    return PAGE


@app.websocket("/ws/asr")
async def ws_asr(ws: WebSocket):
    await ws.accept()
    try:
        opts = json.loads(await ws.receive_text())
    except Exception:
        await ws.close()
        return

    model = state["model"]
    window_samples = state["window_samples"]
    chunk_samples = state["chunk_samples"]

    async with gpu_lock:
        session = {
            "stream": model.init_streaming_state(
                state["tokenizer"], context_info=opts.get("context_info") or None),
            "max_tokens": int(opts.get("max_tokens") or 256),
            "temperature": float(opts.get("temperature") or 0.0),
        }

    buffer = np.zeros(0, dtype=np.float32)
    texts = []

    async def drain(flush: bool):
        nonlocal buffer
        while True:
            available = len(buffer)
            if available >= window_samples:
                window = buffer[:window_samples]
            elif flush and available > 0:
                window = np.zeros(window_samples, dtype=np.float32)
                window[:available] = buffer
            else:
                return
            async with gpu_lock:
                text = await asyncio.to_thread(transcribe_window, window, session)
            texts.append(text)
            # Drop the advance, keep the lookahead: it is the next chunk's
            # leading audio. Consuming rather than indexing is what bounds a
            # long recording -- an index-only cursor keeps every sample of the
            # session alive and re-copies all of it on each incoming frame.
            buffer = buffer[chunk_samples:]
            await ws.send_text(json.dumps({"chunks": len(texts),
                                           "text": "".join(texts)}))
            if flush and len(buffer) <= 0:
                return

    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                return
            if message.get("bytes") is not None:
                pcm = np.frombuffer(message["bytes"], dtype="<f4")
                buffer = np.concatenate([buffer, pcm])
                await drain(flush=False)
            elif message.get("text") == "end":
                await drain(flush=True)
                await ws.send_text(json.dumps({"chunks": len(texts),
                                               "text": "".join(texts),
                                               "done": True,
                                               "total_chunks": len(texts)}))
                return
    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"error": str(e)}))
        except Exception:
            pass


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
    --muted:#8b93a7; --accent:#5b8cff; --rec:#ff4d5e; --ok:#31c48d;
    --ghost:#232838; --field:#10131c;
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
  label { display:block; font-size:13px; color:var(--muted); margin:14px 0 6px; }
  input[type=text], input[type=number] { width:100%; padding:9px 11px;
    border-radius:8px; border:1px solid var(--line); background:var(--field);
    color:var(--fg); font:inherit; }
  input[type=file] { width:100%; font-size:13px; color:var(--muted); }
  .row { display:flex; gap:10px; flex-wrap:wrap; }
  #out { white-space:pre-wrap; word-break:break-word; line-height:1.75;
         min-height:320px; }
  #out:empty::before { content:"Transcript appears here as you speak.";
                       color:var(--muted); }
  .status { font-size:13px; color:var(--muted); margin-top:12px; }
</style>
</head>
<body>
<header>
  <h1>VibeVoice Streaming ASR</h1>
  <span class="dot" id="dot"></span>
  <span class="status" id="conn" style="margin:0">idle</span>
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
    <label>Hotwords (context_info)</label>
    <input type="text" id="ctx" placeholder="Microsoft,VibeVoice">
    <label>Temperature</label>
    <input type="number" id="temp" value="0" min="0" max="2" step="0.1">
    <div class="status" id="status">ready</div>
  </div>
  <div class="panel">
    <h2>Transcript</h2>
    <div id="out"></div>
  </div>
</main>
<script>
const $ = id => document.getElementById(id);
let SR = 24000, sock = null, ctxAudio = null, node = null, media = null;

fetch('/config').then(r => r.json()).then(c => { SR = c.sample_rate; });

function setConn(text, cls) {
  $('conn').textContent = text;
  $('dot').className = 'dot' + (cls ? ' ' + cls : '');
}
function options() {
  return { context_info: $('ctx').value.trim() || null,
           temperature: parseFloat($('temp').value) || 0,
           max_tokens: 256 };
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
      if (msg.error) { $('status').textContent = 'error: ' + msg.error; return; }
      $('out').textContent = msg.text || '';
      $('status').textContent = msg.done
        ? 'done -- ' + msg.total_chunks + ' chunks'
        : 'receiving -- ' + msg.chunks + ' chunks';
      if (msg.done) finish();
    };
  });
}
function finish() {
  $('rec').disabled = false; $('stop').disabled = true; $('file').disabled = false;
  setConn('idle', '');
  if (node) { node.disconnect(); node = null; }
  if (media) { media.getTracks().forEach(t => t.stop()); media = null; }
  if (ctxAudio) { ctxAudio.close(); ctxAudio = null; }
}

$('rec').onclick = async () => {
  $('out').textContent = ''; $('status').textContent = 'starting...';
  try {
    media = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) { $('status').textContent = 'microphone denied'; return; }
  sock = await connect();
  $('rec').disabled = true; $('stop').disabled = false; $('file').disabled = true;
  setConn('recording', 'rec');
  ctxAudio = new AudioContext({ sampleRate: SR });
  const src = ctxAudio.createMediaStreamSource(media);
  node = ctxAudio.createScriptProcessor(4096, 1, 1);
  node.onaudioprocess = e => {
    if (!sock || sock.readyState !== WebSocket.OPEN) return;
    sock.send(new Float32Array(e.inputBuffer.getChannelData(0)).buffer);
  };
  src.connect(node); node.connect(ctxAudio.destination);
};

$('stop').onclick = () => {
  $('status').textContent = 'flushing...';
  if (sock && sock.readyState === WebSocket.OPEN) sock.send('end');
  $('stop').disabled = true;
};

$('file').onchange = async ev => {
  const f = ev.target.files[0];
  if (!f) return;
  $('out').textContent = ''; $('status').textContent = 'decoding...';
  const raw = await f.arrayBuffer();
  const tmp = new AudioContext();
  const decoded = await tmp.decodeAudioData(raw);
  tmp.close();
  const off = new OfflineAudioContext(1, Math.ceil(decoded.duration * SR), SR);
  const s = off.createBufferSource();
  s.buffer = decoded; s.connect(off.destination); s.start();
  const pcm = (await off.startRendering()).getChannelData(0);

  sock = await connect();
  $('rec').disabled = true; $('file').disabled = true;
  $('status').textContent = 'streaming...';
  // Send in the cadence a microphone would produce, so text arrives gradually.
  for (let i = 0; i < pcm.length; i += SR / 2) {
    sock.send(new Float32Array(pcm.slice(i, i + SR / 2)).buffer);
    await new Promise(r => setTimeout(r, 10));
  }
  sock.send('end');
};
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True,
                        help="Path to a streaming checkpoint")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7870)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn_implementation", default="sdpa",
                        choices=["sdpa", "flash_attention_2", "eager"])
    args = parser.parse_args()

    load_model(args.model_path, args.device, args.attn_implementation)
    print(f"open http://localhost:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
