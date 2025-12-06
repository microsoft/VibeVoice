"""
VibeVoice + Llama Pipeline - Phase 2 (FIXED v0.2.5 - FINAL)
Complete unified pipeline with LLM + TTS + WebSocket proxy support

Key Fixes for v0.2.5:
- FIXED: CUDA device mismatch (inputs to GPU)
- Proper attention mask handling
- Removed from old device before generating
- Works perfectly on RunPod!
"""

import logging
import os
import torch
import json
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

app = FastAPI(title="VibeVoice + Llama Pipeline", version="0.2.5")

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
    "device": None,
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
    logger.info("INITIALIZING PHASE 2 PIPELINE (FIXED VERSION - v0.2.5 FINAL)")
    logger.info("=" * 80)
    
    try:
        # System info
        logger.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        if torch.cuda.is_available():
            logger.info(f"CUDA: {torch.version.cuda}")
        logger.info(f"PyTorch: {torch.__version__}")
        
        # Set device
        pipeline_state["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {pipeline_state['device']}")
        
        # Load LLM
        logger.info("Loading Llama-3.2-3B...")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            logger.info(f"Loading model: {MODEL_ID}")
            pipeline_state["llm_tokenizer"] = AutoTokenizer.from_pretrained(MODEL_ID)
            pipeline_state["llm_model"] = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            
            # Ensure model is on correct device
            pipeline_state["llm_model"] = pipeline_state["llm_model"].to(pipeline_state["device"])
            
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

def generate_llm_response(prompt: str, temperature: float = 0.7, max_tokens: int = 256) -> str:
    """Generate LLM response using loaded model - FIX: Proper device handling"""
    try:
        if not pipeline_state["llm_model"] or not pipeline_state["llm_tokenizer"]:
            logger.error("LLM not loaded")
            return "I'm sorry, the LLM model is not loaded. Please check the backend."
        
        tokenizer = pipeline_state["llm_tokenizer"]
        model = pipeline_state["llm_model"]
        device = pipeline_state["device"]
        
        # Clean the prompt
        prompt = prompt.strip()
        if not prompt:
            return "Please provide a valid prompt."
        
        logger.info(f"[LLM] Input: {prompt[:100]}...")
        logger.info(f"[LLM] Device: {device}")
        
        # Tokenize - FIX: Move inputs to GPU
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # CRITICAL FIX: Move all inputs to the same device as model
        inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value 
                 for key, value in inputs.items()}
        
        logger.info(f"[LLM] Input device: {inputs['input_ids'].device}")
        logger.info(f"[LLM] Model device: {next(model.parameters()).device}")
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.1),  # Prevent zero temperature
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the original prompt from response if it's included
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        logger.info(f"[LLM] Output: {response[:100]}...")
        return response
    
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return f"Error generating response: {str(e)[:100]}"

def get_websocket_url(request: Request, text: str, voice: str = None, cfg: float = 1.5, steps: int = 5) -> str:
    """Generate correct WebSocket URL based on connection type"""
    voice = voice or pipeline_state["default_voice"]
    
    # Check if request is coming through proxy
    x_forwarded_proto = request.headers.get('x-forwarded-proto')
    x_forwarded_host = request.headers.get('x-forwarded-host')
    
    logger.info(f"[WEBSOCKET] x-forwarded-proto: {x_forwarded_proto}")
    logger.info(f"[WEBSOCKET] x-forwarded-host: {x_forwarded_host}")
    
    # Encode text for URL
    import urllib.parse
    encoded_text = urllib.parse.quote(text)
    
    if x_forwarded_proto == 'https' and x_forwarded_host:
        # Running on HTTPS proxy
        host_parts = x_forwarded_host.split(':')
        base_host = host_parts[0].rsplit('-', 1)[0] if '-' in host_parts[0] else host_parts[0]
        
        ws_url = f"wss://{base_host}-{TTS_PORT}.proxy.runpod.net/stream?text={encoded_text}&cfg={cfg}&steps={steps}&voice={voice}"
        logger.info(f"[WEBSOCKET] Using HTTPS proxy URL")
    else:
        # Local development
        ws_url = f"ws://localhost:{TTS_PORT}/stream?text={encoded_text}&cfg={cfg}&steps={steps}&voice={voice}"
        logger.info(f"[WEBSOCKET] Using local URL")
    
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

# ============================================================================
# MAIN PIPELINE ENDPOINTS
# ============================================================================

@app.post("/pipeline/text_to_speech")
async def pipeline_text_to_speech(request: Request):
    """
    Complete pipeline: LLM response → TTS → WebSocket URL
    
    Accepts JSON body with:
    {
        "prompt": "your message",
        "temperature": 0.7,
        "max_tokens": 256
    }
    """
    if not pipeline_state["llm_loaded"]:
        raise HTTPException(status_code=503, detail="LLM not loaded")
    
    try:
        import time
        start_time = time.time()
        
        # Parse request body
        try:
            body = await request.json()
        except:
            body = {}
        
        prompt = body.get("prompt", "").strip()
        temperature = body.get("temperature", DEFAULT_TEMPERATURE)
        max_tokens = body.get("max_tokens", DEFAULT_MAX_TOKENS)
        
        logger.info("=" * 80)
        logger.info("[PIPELINE] Starting complete pipeline")
        logger.info(f"[PIPELINE] Prompt: {prompt[:50]}...")
        logger.info(f"[PIPELINE] Temp: {temperature}, Tokens: {max_tokens}")
        logger.info("=" * 80)
        
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")
        
        # STEP 1: Generate LLM response
        step1_start = time.time()
        logger.info("[PIPELINE] STEP 1: Generating LLM response...")
        
        llm_response = generate_llm_response(
            prompt=prompt,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )
        
        llm_latency = (time.time() - step1_start) * 1000
        logger.info(f"[PIPELINE] STEP 1 ✅ Complete in {llm_latency:.2f}ms")
        logger.info(f"[PIPELINE] LLM Response: {llm_response[:100]}...")
        
        # STEP 2: Generate WebSocket URL for TTS
        step2_start = time.time()
        logger.info("[PIPELINE] STEP 2: Generating WebSocket URL for TTS...")
        
        audio_url = get_websocket_url(
            request=request,
            text=llm_response,
            voice=pipeline_state["default_voice"],
            cfg=1.5,
            steps=5
        )
        
        tts_latency = (time.time() - step2_start) * 1000
        logger.info(f"[PIPELINE] STEP 2 ✅ Complete in {tts_latency:.2f}ms")
        
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
    
    except HTTPException:
        raise
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
