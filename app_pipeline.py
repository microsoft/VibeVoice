"""
VibeVoice + Llama Pipeline - Phase 2 (FIXED v0.2.2)
Complete unified pipeline with LLM + TTS + WebSocket proxy support

Key Fixes:
- Detects proxy headers and generates correct WebSocket URLs
- Works on both local (ws://localhost) and proxy (wss://proxy-url)
- No more localhost hardcoding
"""

import logging
import os
import torch
import json
import numpy as np
from datetime import datetime
from typing import Optional
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

LLM_PORT = 8005
TTS_PORT = 8000
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 256

app = FastAPI(title="VibeVoice + Llama Pipeline", version="0.2.2")

# Add CORS middleware for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
pipeline_state = {
    "llm_loaded": False,
    "tts_loaded": False,
    "llm_model": None,
    "llm_tokenizer": None,
    "pipeline": None,
    "tts_voices": [],
    "default_voice": "en-Carter_man",
    "stats": {
        "total_requests": 0,
        "total_llm_time": 0,
        "total_tts_time": 0,
    }
}

# ============================================================================
# INITIALIZATION
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup"""
    logger.info("=" * 80)
    logger.info("INITIALIZING PHASE 2 PIPELINE (FIXED VERSION - v0.2.2)")
    logger.info("=" * 80)
    
    try:
        # System info
        logger.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        if torch.cuda.is_available():
            logger.info(f"CUDA: {torch.version.cuda}")
        logger.info(f"PyTorch: {torch.__version__}")
        
        # Load LLM
        logger.info("Loading Llama-3.2-3B...")
        try:
            from vibevoice.llm_integration import LlamaLLM
            llm = LlamaLLM(model_id=MODEL_ID)
            pipeline_state["llm_model"] = llm
            pipeline_state["llm_loaded"] = True
            logger.info("✅ Llama-3.2-3B loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load LLM: {e}")
            pipeline_state["llm_loaded"] = False
        
        # Check TTS
        logger.info("Checking VibeVoice TTS...")
        try:
            response = requests.get(f"http://localhost:{TTS_PORT}/config", timeout=5)
            if response.status_code == 200:
                config = response.json()
                pipeline_state["tts_loaded"] = True
                pipeline_state["tts_voices"] = config.get("available_voices", ["en-Carter_man"])
                pipeline_state["default_voice"] = config.get("default_voice", "en-Carter_man")
                
                logger.info("✅ VibeVoice TTS detected on port 8000")
                logger.info(f"   Available voices: {pipeline_state['tts_voices']}")
                logger.info(f"   Default voice: {pipeline_state['default_voice']}")
            else:
                logger.warning(f"⚠️  TTS returned non-200 status: {response.status_code}")
                pipeline_state["tts_loaded"] = False
        except Exception as e:
            logger.error(f"❌ TTS health check failed: {e}")
            pipeline_state["tts_loaded"] = False
        
        logger.info("=" * 80)
        logger.info(f"Pipeline Status: LLM={pipeline_state['llm_loaded']}, TTS={pipeline_state['tts_loaded']}")
        logger.info(f"Port {LLM_PORT}: Ready to accept requests")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down pipeline...")
    if pipeline_state["llm_model"]:
        try:
            del pipeline_state["llm_model"]
            torch.cuda.empty_cache()
        except:
            pass

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_websocket_url(request: Request, text: str, voice: str = None, cfg: float = 1.5, steps: int = 5) -> str:
    """
    Generate correct WebSocket URL based on connection type.
    
    KEY FIX: Detects if running behind proxy and generates appropriate URL.
    """
    voice = voice or pipeline_state["default_voice"]
    
    # Check if request is coming through proxy
    x_forwarded_proto = request.headers.get('x-forwarded-proto')
    x_forwarded_host = request.headers.get('x-forwarded-host')
    
    logger.info(f"[WEBSOCKET] x-forwarded-proto: {x_forwarded_proto}")
    logger.info(f"[WEBSOCKET] x-forwarded-host: {x_forwarded_host}")
    
    if x_forwarded_proto == 'https' and x_forwarded_host:
        # Running on HTTPS proxy - use secure WebSocket with proxy URL
        # Extract just the port prefix (e.g., "17wlelvk973qxz-8000" -> "17wlelvk973qxz")
        host_parts = x_forwarded_host.split(':')
        base_host = host_parts[0].rsplit('-', 1)[0] if '-' in host_parts[0] else host_parts[0]
        
        # Build secure WebSocket URL for proxy
        ws_url = f"wss://{base_host}-8000.proxy.runpod.net/stream?text={text}&cfg={cfg}&steps={steps}&voice={voice}"
        logger.info(f"[WEBSOCKET] Using HTTPS proxy URL: {ws_url[:80]}...")
    else:
        # Local development - use WebSocket with localhost
        ws_url = f"ws://localhost:{TTS_PORT}/stream?text={text}&cfg={cfg}&steps={steps}&voice={voice}"
        logger.info(f"[WEBSOCKET] Using local URL: {ws_url[:80]}...")
    
    return ws_url

# ============================================================================
# HEALTH & STATUS ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "llm_loaded": pipeline_state["llm_loaded"],
        "tts_loaded": pipeline_state["tts_loaded"],
        "tts_port": TTS_PORT,
        "llm_port": LLM_PORT,
    }

@app.get("/stats")
async def get_stats():
    """Get pipeline statistics"""
    avg_llm_time = (
        pipeline_state["stats"]["total_llm_time"] / pipeline_state["stats"]["total_requests"]
        if pipeline_state["stats"]["total_requests"] > 0 else 0
    )
    avg_tts_time = (
        pipeline_state["stats"]["total_tts_time"] / pipeline_state["stats"]["total_requests"]
        if pipeline_state["stats"]["total_requests"] > 0 else 0
    )
    
    return {
        "total_requests": pipeline_state["stats"]["total_requests"],
        "avg_llm_latency_ms": round(avg_llm_time, 2),
        "avg_tts_latency_ms": round(avg_tts_time, 2),
        "gpu_memory": f"{torch.cuda.memory_allocated() / 1e9:.2f}GB" if torch.cuda.is_available() else "N/A",
    }

@app.get("/interface")
async def get_interface():
    """Serve HTML interface"""
    html_file = "/workspace/app/repo/vibevoice_llm_interface_v031.html"
    if os.path.exists(html_file):
        with open(html_file, 'r') as f:
            return f.read()
    return {"error": "Interface file not found"}

# ============================================================================
# MAIN PIPELINE ENDPOINTS
# ============================================================================

@app.post("/generate_response")
async def generate_response(
    prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
):
    """
    Generate LLM response only (no TTS)
    """
    if not pipeline_state["llm_loaded"]:
        raise HTTPException(status_code=503, detail="LLM not loaded")
    
    try:
        import time
        start_time = time.time()
        
        logger.info("=" * 80)
        logger.info("[LLM] Generating response...")
        logger.info(f"[LLM] Prompt: {prompt}")
        logger.info(f"[LLM] Temperature: {temperature}, Max tokens: {max_tokens}")
        
        # Generate response
        response = pipeline_state["llm_model"].generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        latency = (time.time() - start_time) * 1000
        
        logger.info(f"[LLM] ✅ Response generated in {latency:.2f}ms")
        logger.info("=" * 80)
        
        pipeline_state["stats"]["total_requests"] += 1
        pipeline_state["stats"]["total_llm_time"] += latency
        
        return {
            "status": "success",
            "prompt": prompt,
            "llm_response": response,
            "generation_ms": round(latency, 2),
        }
    
    except Exception as e:
        logger.error(f"[LLM] ❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pipeline/text_to_speech")
async def pipeline_text_to_speech(
    request: Request,
    prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
):
    """
    Complete pipeline: LLM response → TTS → WebSocket URL
    
    KEY FIX: Now generates correct WebSocket URL based on proxy detection
    """
    if not pipeline_state["llm_loaded"]:
        raise HTTPException(status_code=503, detail="LLM not loaded")
    
    try:
        import time
        start_time = time.time()
        
        logger.info("=" * 80)
        logger.info("[PIPELINE] Starting complete pipeline")
        logger.info(f"[PIPELINE] Input prompt: {prompt}...")
        logger.info("=" * 80)
        
        # STEP 1: Generate LLM response
        step1_start = time.time()
        logger.info("[PIPELINE] STEP 1: Generating LLM response...")
        
        llm_response = pipeline_state["llm_model"].generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        llm_latency = (time.time() - step1_start) * 1000
        logger.info(f"[PIPELINE] STEP 1 ✅ Complete in {llm_latency:.2f}ms")
        logger.info(f"[PIPELINE] LLM Response preview: {llm_response[:50]}...")
        
        # STEP 2: Generate WebSocket URL for TTS
        step2_start = time.time()
        logger.info("[PIPELINE] STEP 2: Generating WebSocket URL for TTS...")
        
        if pipeline_state["tts_loaded"]:
            # Get correct WebSocket URL (KEY FIX!)
            audio_url = get_websocket_url(
                request=request,
                text=llm_response,
                voice=pipeline_state["default_voice"],
                cfg=1.5,
                steps=5
            )
            logger.info(f"[PIPELINE] WebSocket streaming URL: {audio_url[:100]}...")
        else:
            logger.warning("[PIPELINE] ⚠️  TTS not available, but continuing...")
            audio_url = None
        
        tts_latency = (time.time() - step2_start) * 1000
        logger.info(f"[PIPELINE] STEP 2 Complete in {tts_latency:.2f}ms")
        
        total_latency = (time.time() - start_time) * 1000
        
        logger.info("=" * 80)
        logger.info("[PIPELINE] ✅ COMPLETE")
        logger.info(f"[PIPELINE] Total latency: {total_latency:.2f}ms")
        logger.info("=" * 80)
        
        pipeline_state["stats"]["total_requests"] += 1
        pipeline_state["stats"]["total_llm_time"] += llm_latency
        pipeline_state["stats"]["total_tts_time"] += tts_latency
        
        return {
            "status": "success",
            "prompt": prompt,
            "llm_response": llm_response,
            "audio_url": audio_url,
            "llm_latency_ms": round(llm_latency, 2),
            "tts_latency_ms": round(tts_latency, 2),
            "total_latency_ms": round(total_latency, 2),
        }
    
    except Exception as e:
        logger.error(f"[PIPELINE] ❌ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    logger.info(f"🚀 Starting pipeline server on 0.0.0.0:{LLM_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=LLM_PORT, log_level="info")
