#!/usr/bin/env python3
"""
VibeVoice + Llama-3.2 Complete Pipeline
Text → LLM Response → TTS Audio (End-to-end)

PHASE 2: UNIFIED PIPELINE
- Port 8005: Accepts user input
- Step 1: Sends to Llama (generates response)
- Step 2: Sends response to VibeVoice TTS (port 8000)
- Step 3: Returns audio URL to user on port 8005
"""

import os
import sys
import logging
import asyncio
import time
import requests
from typing import Optional
from pathlib import Path

# Setup environment
os.environ['HF_HOME'] = '/workspace/models/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/models/huggingface/transformers'
os.environ['TORCH_HOME'] = '/workspace/models/torch'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
from fastapi import FastAPI, HTTPException, WebSocket, BackgroundTasks
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import custom modules
sys.path.insert(0, '/workspace/app/repo/src')
from vibevoice.llm_integration import LlamaStreamer, LLMConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# REQUEST/RESPONSE SCHEMAS
# =============================================================================

class GenerateRequest(BaseModel):
    """LLM Generation request"""
    prompt: str = Field(..., min_length=1, max_length=1000)
    temperature: float = Field(default=0.7, ge=0.1, le=1.0)
    max_tokens: int = Field(default=256, ge=10, le=512)

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    tts_loaded: bool
    llm_loaded: bool
    gpu_memory_gb: Optional[float] = None
    tts_port: int = 8000
    llm_port: int = 8005

class LLMResponse(BaseModel):
    """LLM generation response"""
    status: str
    prompt: str
    llm_response: str
    generation_ms: float

class PipelineResponse(BaseModel):
    """Complete pipeline response (LLM + TTS)"""
    status: str
    prompt: str
    llm_response: str
    audio_url: Optional[str] = None
    llm_latency_ms: float
    tts_latency_ms: float
    total_latency_ms: float

# =============================================================================
# INITIALIZATION
# =============================================================================

app = FastAPI(
    title="VibeVoice + Llama Pipeline",
    description="TTS + LLM for speech-to-speech on port 8005",
    version="0.2.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
llm: Optional[LlamaStreamer] = None
TTS_LOADED = False
LLM_LOADED = False

# TTS Configuration
TTS_HOST = "http://localhost:8000"
TTS_SYNTHESIZE_ENDPOINT = f"{TTS_HOST}/synthesize"
TTS_HEALTH_ENDPOINT = f"{TTS_HOST}/health"

# =============================================================================
# STARTUP
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global llm, LLM_LOADED, TTS_LOADED
    
    logger.info("=" * 80)
    logger.info("INITIALIZING PHASE 2 PIPELINE")
    logger.info("=" * 80)
    
    # GPU info
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA: {torch.version.cuda}")
        logger.info(f"PyTorch: {torch.__version__}")
    
    # Initialize LLM
    try:
        logger.info("Loading Llama-3.2-3B...")
        llm = LlamaStreamer(LLMConfig())
        llm.load_model()
        LLM_LOADED = True
        logger.info("✅ Llama-3.2-3B loaded successfully")
    except Exception as e:
        logger.error(f"❌ Llama load failed: {e}")
        LLM_LOADED = False
    
    # Check TTS availability
    try:
        response = requests.get(TTS_HEALTH_ENDPOINT, timeout=5)
        if response.status_code == 200:
            TTS_LOADED = True
            logger.info("✅ VibeVoice TTS detected on port 8000")
        else:
            TTS_LOADED = False
            logger.warning("⚠️  TTS returned non-200 status")
    except Exception as e:
        TTS_LOADED = False
        logger.warning(f"⚠️  TTS not available on port 8000: {e}")
    
    logger.info("=" * 80)
    logger.info(f"Pipeline Status: LLM={LLM_LOADED}, TTS={TTS_LOADED}")
    logger.info("Port 8005: Ready to accept requests")
    logger.info("=" * 80)

# =============================================================================
# SHUTDOWN
# =============================================================================

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down server...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Cleanup complete")

# =============================================================================
# HEALTH & STATUS ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "VibeVoice + Llama Pipeline",
        "version": "0.2.0",
        "endpoints": {
            "health": "/health",
            "interface": "/interface",
            "generate_llm": "/generate_response (POST)",
            "pipeline_tts": "/pipeline/text_to_speech (POST)",
            "stats": "/stats"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    gpu_memory = None
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    return HealthResponse(
        status="ok",
        tts_loaded=TTS_LOADED,
        llm_loaded=LLM_LOADED,
        gpu_memory_gb=gpu_memory,
        tts_port=8000,
        llm_port=8005
    )

@app.get("/stats")
async def get_stats():
    """System statistics"""
    
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        percent = (used / total) * 100
    else:
        used = total = percent = 0
    
    return {
        "status": "ok",
        "gpu_used_gb": round(used, 2),
        "gpu_total_gb": round(total, 2),
        "gpu_percent": round(percent, 2),
        "models": {
            "tts_port_8000": TTS_LOADED,
            "llm_port_8005": LLM_LOADED
        },
        "tts_endpoint": TTS_SYNTHESIZE_ENDPOINT,
        "message": "Connect both ports for full pipeline"
    }

# =============================================================================
# WEB INTERFACE
# =============================================================================

@app.get("/interface", response_class=HTMLResponse)
async def interface():
    """Serve web interface"""
    html_path = Path("/workspace/app/repo/demo/vibevoice_llm_demo.html")
    if html_path.exists():
        return FileResponse(html_path)
    return """
    <html>
        <body style="font-family: Arial; margin: 20px;">
            <h1>VibeVoice + Llama Pipeline</h1>
            <p>Interface file not found</p>
            <p>Expected location: /workspace/app/repo/demo/vibevoice_llm_demo.html</p>
            <hr>
            <h2>API Endpoints:</h2>
            <ul>
                <li><strong>GET /health</strong> - Check system status</li>
                <li><strong>GET /stats</strong> - System statistics</li>
                <li><strong>POST /generate_response</strong> - LLM only</li>
                <li><strong>POST /pipeline/text_to_speech</strong> - LLM + TTS</li>
            </ul>
        </body>
    </html>
    """

# =============================================================================
# LLM ONLY ENDPOINT (Phase 2 - Step 1)
# =============================================================================

@app.post("/generate_response", response_model=LLMResponse)
async def generate_response(request: GenerateRequest):
    """
    LLM Only: Generate response from prompt
    
    Input:  { "prompt": "What is AI?" }
    Output: { "llm_response": "AI is...", "generation_ms": 1200 }
    
    This is PHASE 2 STEP 1: Generate LLM response only
    """
    
    if not LLM_LOADED:
        raise HTTPException(status_code=503, detail="LLM not loaded")
    
    try:
        start = time.time()
        
        logger.info(f"[LLM Only] Processing: {request.prompt[:50]}...")
        
        # Generate LLM response
        response = llm.generate_simple(
            text=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        elapsed = (time.time() - start) * 1000
        
        logger.info(f"[LLM Only] ✅ Response generated in {elapsed:.0f}ms")
        
        return LLMResponse(
            status="success",
            prompt=request.prompt,
            llm_response=response,
            generation_ms=elapsed
        )
        
    except Exception as e:
        logger.error(f"[LLM Only] ❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# UNIFIED PIPELINE ENDPOINT (Phase 2 - COMPLETE)
# THIS IS THE MAIN ENDPOINT YOU NEED
# =============================================================================

@app.post("/pipeline/text_to_speech")
async def text_to_speech_pipeline(request: GenerateRequest):
    """
    COMPLETE PIPELINE: Text → LLM → TTS → Audio
    
    *** THIS IS THE MAIN ENDPOINT FOR PHASE 2 ***
    
    Flow:
    1. User sends text to port 8005
    2. Llama-3.2-3B generates response (port 8005)
    3. Response sent to VibeVoice TTS (port 8000)
    4. TTS generates audio
    5. Audio URL returned to port 8005
    6. User can play/download audio
    
    Input:
    {
        "prompt": "What is artificial intelligence?",
        "max_tokens": 256,
        "temperature": 0.7
    }
    
    Output:
    {
        "status": "success",
        "prompt": "What is...",
        "llm_response": "AI is...",
        "audio_url": "http://localhost:8000/audio/output_xxx.wav",
        "llm_latency_ms": 1200,
        "tts_latency_ms": 450,
        "total_latency_ms": 1650
    }
    """
    
    if not LLM_LOADED:
        raise HTTPException(status_code=503, detail="LLM not loaded")
    
    if not TTS_LOADED:
        logger.warning("[Pipeline] ⚠️  TTS not available, but continuing...")
    
    try:
        pipeline_start = time.time()
        
        logger.info("=" * 80)
        logger.info(f"[PIPELINE] Starting complete pipeline")
        logger.info(f"[PIPELINE] Input prompt: {request.prompt[:60]}...")
        logger.info("=" * 80)
        
        # =====================================================================
        # STEP 1: Generate LLM Response
        # =====================================================================
        logger.info("[PIPELINE] STEP 1: Generating LLM response...")
        
        llm_start = time.time()
        
        llm_response = llm.generate_simple(
            text=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        llm_elapsed = (time.time() - llm_start) * 1000
        
        logger.info(f"[PIPELINE] STEP 1 ✅ Complete in {llm_elapsed:.0f}ms")
        logger.info(f"[PIPELINE] LLM Response preview: {llm_response[:80]}...")
        
        # =====================================================================
        # STEP 2: Send LLM Response to TTS (port 8000)
        # =====================================================================
        logger.info("[PIPELINE] STEP 2: Sending response to TTS on port 8000...")
        
        tts_start = time.time()
        audio_url = None
        
        try:
            # Call TTS endpoint on port 8000
            logger.info(f"[PIPELINE] TTS Request to: {TTS_SYNTHESIZE_ENDPOINT}")
            
            tts_response = requests.post(
                TTS_SYNTHESIZE_ENDPOINT,
                json={
                    "text": llm_response,
                    "speaker": "default",
                    "speed": 1.0
                },
                timeout=60
            )
            
            logger.info(f"[PIPELINE] TTS Response Status: {tts_response.status_code}")
            
            if tts_response.status_code == 200:
                tts_data = tts_response.json()
                audio_url = tts_data.get("audio_url")
                logger.info(f"[PIPELINE] STEP 2 ✅ Audio generated: {audio_url}")
            else:
                logger.error(f"[PIPELINE] TTS returned status {tts_response.status_code}")
                logger.error(f"[PIPELINE] TTS response: {tts_response.text}")
                # Continue anyway, return response without audio
                
        except requests.exceptions.ConnectionError as e:
            logger.error(f"[PIPELINE] ❌ Cannot connect to TTS on port 8000: {e}")
            logger.error("[PIPELINE] Make sure TTS is running: python demo/vibevoice_realtime_demo.py --port 8000")
            # Continue anyway, return response without audio
            
        except requests.exceptions.Timeout:
            logger.error("[PIPELINE] ❌ TTS timeout (>60 seconds)")
            # Continue anyway
            
        except Exception as e:
            logger.error(f"[PIPELINE] ❌ TTS error: {e}")
            # Continue anyway
        
        tts_elapsed = (time.time() - tts_start) * 1000
        
        if audio_url:
            logger.info(f"[PIPELINE] STEP 2 ✅ Complete in {tts_elapsed:.0f}ms")
        else:
            logger.warning(f"[PIPELINE] STEP 2 ⚠️  Partial: No audio URL (TTS may be down)")
        
        # =====================================================================
        # STEP 3: Return Complete Response
        # =====================================================================
        total_elapsed = (time.time() - pipeline_start) * 1000
        
        logger.info("=" * 80)
        logger.info("[PIPELINE] ✅ COMPLETE")
        logger.info(f"[PIPELINE] Total latency: {total_elapsed:.0f}ms")
        logger.info("=" * 80)
        
        return PipelineResponse(
            status="success",
            prompt=request.prompt,
            llm_response=llm_response,
            audio_url=audio_url,
            llm_latency_ms=llm_elapsed,
            tts_latency_ms=tts_elapsed,
            total_latency_ms=total_elapsed
        )
        
    except Exception as e:
        logger.error(f"[PIPELINE] ❌ Fatal error: {e}")
        logger.error(f"[PIPELINE] Traceback: ", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {str(e)}"
        )

# =============================================================================
# ALTERNATIVE: STREAMING ENDPOINT (Optional - for Phase 3)
# =============================================================================

@app.post("/pipeline/stream")
async def stream_pipeline(request: GenerateRequest):
    """
    Streaming version: For future real-time audio streaming
    (Not fully implemented in Phase 2)
    """
    
    if not LLM_LOADED:
        raise HTTPException(status_code=503, detail="LLM not loaded")
    
    try:
        logger.info("[STREAM] Starting streaming pipeline...")
        
        llm_response = llm.generate_simple(
            text=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        return {
            "status": "success",
            "prompt": request.prompt,
            "llm_response": llm_response,
            "mode": "streaming",
            "note": "Audio streaming coming in Phase 3"
        }
        
    except Exception as e:
        logger.error(f"[STREAM] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    port = int(os.getenv("DEMO_PORT", 8005))
    host = os.getenv("DEMO_HOST", "0.0.0.0")
    
    logger.info(f"🚀 Starting pipeline server on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )
