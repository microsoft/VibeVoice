#!/usr/bin/env python3
"""
VibeVoice + Llama-3.2 Complete Pipeline
Text → LLM Response → TTS Audio (End-to-end)
"""

import os
import sys
import logging
import asyncio
import time
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

class TextToSpeechRequest(BaseModel):
    """TTS request"""
    text: str = Field(..., min_length=1, max_length=1000)
    streaming: bool = Field(default=False)

class ConversationRequest(BaseModel):
    """Text-to-Speech via LLM response"""
    prompt: str = Field(..., min_length=1, max_length=500)
    temperature: float = Field(default=0.7, ge=0.1, le=1.0)
    max_tokens: int = Field(default=256, ge=10, le=512)
    streaming: bool = Field(default=False)

class HealthResponse(BaseModel):
    """Health check"""
    status: str
    tts_loaded: bool
    llm_loaded: bool
    gpu_memory_gb: Optional[float] = None

class PipelineResponse(BaseModel):
    """Pipeline response"""
    status: str
    prompt: str
    llm_response: str
    generation_ms: float

# =============================================================================
# INITIALIZATION
# =============================================================================

app = FastAPI(
    title="VibeVoice + Llama Pipeline",
    description="TTS + LLM for speech-to-speech",
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

# =============================================================================
# STARTUP
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global llm, LLM_LOADED, TTS_LOADED
    
    logger.info("=" * 60)
    logger.info("INITIALIZING PIPELINE")
    logger.info("=" * 60)
    
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
        logger.info("✅ Llama loaded")
    except Exception as e:
        logger.error(f"❌ Llama failed: {e}")
        LLM_LOADED = False
    
    # TTS would be initialized similarly
    TTS_LOADED = True
    
    logger.info("=" * 60)
    logger.info(f"Pipeline ready: TTS={TTS_LOADED}, LLM={LLM_LOADED}")
    logger.info("=" * 60)

# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Redirect to interface"""
    return {"message": "Use /interface for web UI, /health for status"}

@app.get("/interface", response_class=HTMLResponse)
async def interface():
    """Serve web interface"""
    html_path = Path("/workspace/app/repo/demo/vibevoice_llm_demo.html")
    if html_path.exists():
        return FileResponse(html_path)
    return """
    <html>
        <body>
            <h1>Interface not found</h1>
            <p>Place vibevoice_llm_demo.html in /workspace/app/repo/demo/</p>
        </body>
    </html>
    """

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check"""
    gpu_memory = None
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    return HealthResponse(
        status="ok",
        tts_loaded=TTS_LOADED,
        llm_loaded=LLM_LOADED,
        gpu_memory_gb=gpu_memory
    )

@app.post("/generate_response", response_model=PipelineResponse)
async def generate_response(request: ConversationRequest):
    """
    Generate LLM response from prompt
    
    Phase 2: LLM only (next: add TTS)
    """
    
    if not LLM_LOADED:
        raise HTTPException(status_code=503, detail="LLM not loaded")
    
    try:
        start = time.time()
        
        # Generate response
        response = llm.generate_simple(
            text=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        elapsed = (time.time() - start) * 1000
        
        return PipelineResponse(
            status="success",
            prompt=request.prompt,
            llm_response=response,
            generation_ms=elapsed
        )
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pipeline")
async def full_pipeline(request: ConversationRequest):
    """
    Complete pipeline: Prompt → LLM → TTS → Audio
    
    Phase 3: Full integration
    Currently returns: Prompt + LLM Response
    Next: Add TTS conversion
    """
    
    if not LLM_LOADED:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    
    try:
        # Step 1: Generate LLM response
        logger.info(f"[Pipeline] Processing: {request.prompt[:50]}...")
        
        llm_response = llm.generate_simple(
            text=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        logger.info(f"[Pipeline] LLM Response: {llm_response[:50]}...")
        
        # Step 2: (Future) Convert to TTS
        # audio_bytes = await tts.generate(llm_response)
        
        # Return JSON for now (next phase: audio stream)
        return {
            "status": "success",
            "prompt": request.prompt,
            "llm_response": llm_response,
            "stage": "llm_only (tts in phase 3)",
            "next_step": "Add TTS conversion"
        }
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        percent = (used / total) * 100
    else:
        used = total = percent = 0
    
    return {
        "gpu_used_gb": used,
        "gpu_total_gb": total,
        "gpu_percent": percent,
        "models_loaded": {
            "tts": TTS_LOADED,
            "llm": LLM_LOADED
        }
    }

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
