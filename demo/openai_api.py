import io
import re
import time
import torch
import soundfile as sf
import traceback

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from inference_from_file import VoiceMapper

app = FastAPI()

class SpeechRequest(BaseModel):
    model: str
    input: str
    voice: str

MODEL_CONFIG = {
    "microsoft/VibeVoice-1.5B": {
        "id": "microsoft/VibeVoice-1.5B",
        "object": "model",
        "owned_by": "microsoft"
    } # Add more models as needed
}

print("Loading models...")
loaded_models = {}
for model_id in MODEL_CONFIG:
    print(f" - {model_id}")
    processor = VibeVoiceProcessor.from_pretrained(model_id)
    
    try:
        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map='cuda',
            attn_implementation='flash_attention_2'
        )
    except Exception:
        print(f"[WARNING] Failed to load with flash_attention_2. Falling back to sdpa.")
        traceback.print_exc()
        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map='cuda',
            attn_implementation='sdpa'
        )
    
    model.eval()
    model.set_ddpm_inference_steps(num_steps=10)
    loaded_models[model_id] = {
        "model": model,
        "processor": processor
    }

print("Models ready:", list(loaded_models.keys()))

voice_mapper = VoiceMapper()

def format_text(text: str) -> str:
    if not re.search(r'^Speaker\s+\d+:', text.strip(), re.IGNORECASE | re.MULTILINE):
        return f"Speaker 1: {text.strip()}"
    return text.strip()

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": list(MODEL_CONFIG.values())
    }

@app.post("/v1/audio/speech")
async def generate_speech(request: SpeechRequest):
    if request.model not in loaded_models:
        raise HTTPException(status_code=400, detail=f"Model '{request.model}' not available.")

    try:
        voice_path = voice_mapper.get_voice_path(request.voice)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    model_entry = loaded_models[request.model]
    processor, model = model_entry["processor"], model_entry["model"]

    text = format_text(request.input)
    voice_samples = [voice_path]

    inputs = processor(
        text=[text],
        voice_samples=[voice_samples],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    start_time = time.time()
    outputs = model.generate(
        **inputs,
        cfg_scale=1.3,
        tokenizer=processor.tokenizer,
        generation_config={"do_sample": False},
        verbose=False,
    )
    print(f"[{request.model}] Generated in {time.time() - start_time:.2f}s: '{text[:30]}...'")

    audio_np = outputs.speech_outputs[0].cpu().float().numpy()
    buffer = io.BytesIO()
    sf.write(buffer, audio_np.T, samplerate=24000, format='WAV')
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="audio/wav")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
