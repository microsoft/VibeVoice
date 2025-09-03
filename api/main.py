import asyncio
import io
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field
from typing import List
import uvicorn

# Import the service and the dependency injector function
from .tts_service import TTSService, get_tts_service

# --- FastAPI App Initialization ---
app = FastAPI(
    title="VibeVoice API",
    description="A powerful API for generating long-form, multi-speaker conversational audio with VibeVoice.",
    version="1.0.0",
)

# --- Pydantic Models for API Contracts ---
class TTSRequest(BaseModel):
    script: str = Field(
        ...,
        description="The full script for the podcast. Use 'Speaker X:' to denote turns, where X is a 0-indexed integer.",
        example="Speaker 0: Welcome to our podcast today!\nSpeaker 1: Thanks for having me. I'm excited to discuss..."
    )
    speaker_voices: List[str] = Field(
        ...,
        description="A list of voice presets to use for the speakers. The order corresponds to the speaker index (e.g., speaker_voices[0] for Speaker 0).",
        example=["en-Alice_woman", "en-Carter_man"]
    )
    cfg_scale: float = Field(
        default=1.3,
        description="Classifier-Free Guidance scale. Higher values increase adherence to the text.",
        ge=1.0,
        le=2.0
    )

class HealthCheckResponse(BaseModel):
    status: str
    service_initialized: bool

# --- Request Queue and Batching (Concurrency Limiter) ---
# For now, we use a simple asyncio.Semaphore to limit concurrent requests to 1.
# This acts as a simple queue, ensuring the GPU is not overloaded.
# A more advanced implementation could batch requests together.
concurrency_limiter = asyncio.Semaphore(1)

# --- API Endpoints ---
@app.post("/api/generate/streaming", tags=["Generation"])
async def generate_streaming(
    request: TTSRequest,
    tts_service: TTSService = Depends(get_tts_service)
):
    """
    Generate audio in real-time and stream it back as raw 16-bit PCM audio chunks.
    This endpoint is ideal for applications that need to play audio as it's being generated.
    """
    async with concurrency_limiter:
        try:
            audio_generator = tts_service.generate_stream_async(
                script=request.script,
                speaker_voices=request.speaker_voices,
                cfg_scale=request.cfg_scale
            )
            # The media type 'application/octet-stream' is used for raw binary data.
            # The client should know to interpret this as 16-bit PCM at 24000Hz.
            return StreamingResponse(audio_generator, media_type="application/octet-stream")
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.post("/api/generate/batch", tags=["Generation"])
async def generate_batch(
    request: TTSRequest,
    tts_service: TTSService = Depends(get_tts_service)
):
    """
    Generate the full audio and return it as a complete WAV file.
    This endpoint is suitable for background processing or when the full audio file is needed before playback.
    """
    async with concurrency_limiter:
        try:
            # Run the synchronous, CPU/GPU-bound `generate_batch` in a separate thread
            # to avoid blocking the asyncio event loop.
            audio_np, sample_rate = await asyncio.to_thread(
                tts_service.generate_batch,
                script=request.script,
                speaker_voices=request.speaker_voices,
                cfg_scale=request.cfg_scale
            )

            # Convert numpy array to WAV format in memory
            buffer = io.BytesIO()
            sf.write(buffer, audio_np, sample_rate, format='WAV')
            buffer.seek(0)

            # Return the WAV file as a response
            return Response(content=buffer.getvalue(), media_type="audio/wav")
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.get("/", tags=["Status"], response_model=HealthCheckResponse)
def read_root(tts_service: TTSService = Depends(get_tts_service)):
    """
    Root endpoint for health checks.
    Indicates if the API is running and if the TTS service is initialized.
    """
    return {
        "status": "VibeVoice API is running",
        "service_initialized": tts_service._initialized if tts_service else False
    }

# --- Singleton Model Loading on Application Startup ---
@app.on_event("startup")
async def startup_event():
    """
    Loads the TTS model into memory when the application starts.
    This is triggered by the first call to `get_tts_service`.
    """
    print("Application startup: Initializing TTS Service...")
    # This call will create and initialize the singleton instance
    get_tts_service()
    print("TTS Service initialization complete.")

if __name__ == "__main__":
    # This allows running the app directly for development/testing
    # In production, you would use a command like: uvicorn api.main:app --host 0.0.0.0 --port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
