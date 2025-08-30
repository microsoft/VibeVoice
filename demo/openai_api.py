import io
import re
import time
import torch
import soundfile as sf
import traceback
from transformers import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from inference_from_file import VoiceMapper

logging.set_verbosity_info()
logger = logging.get_logger(__name__)

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
    }
    # ,"WestZhang/VibeVoice-Large-pt": {
    #     "id": "WestZhang/VibeVoice-Large-pt",
    #     "object": "model",
    #     "owned_by": "WestZhang"
    # }
}


print("Loading all models and processors...")

loaded_models = {}
for model_id in MODEL_CONFIG.keys():
    print(f" - Loading: {model_id}")
    processor = VibeVoiceProcessor.from_pretrained(model_id)
    try:
        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map='cuda',
            attn_implementation='flash_attention_2' # flash_attention_2 is recommended
        )
    except Exception as e:
        print(f"[ERROR] : {type(e).__name__}: {e}")
        print(traceback.format_exc())
        print("Error loading the model. Trying to use SDPA. However, note that only flash_attention_2 has been fully tested, and using SDPA may result in lower audio quality.")
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

def ensure_speaker_lines(text: str) -> str:
    """Wrap plain text with 'Speaker 1:' if no speaker lines found"""
    speaker_line_pattern = re.compile(r'^Speaker\s+\d+:', re.IGNORECASE)
    lines = text.strip().splitlines()
    has_speaker_lines = any(speaker_line_pattern.match(line.strip()) for line in lines)

    if not has_speaker_lines:
        return f"Speaker 1: {text.strip()}"
    return text

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
    processor = model_entry["processor"]
    model = model_entry["model"]

    text = ensure_speaker_lines(request.input.strip())
    voice_samples = [voice_path]

    inputs = processor(
        text=[text],
        voice_samples=[voice_samples],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    start = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=None,
        cfg_scale=1.3,
        tokenizer=processor.tokenizer,
        generation_config={"do_sample": False},
        verbose=False,
    )
    duration = time.time() - start

    audio = outputs.speech_outputs[0]
    audio_np = audio.cpu().float().numpy()
    buffer = io.BytesIO()
    sf.write(buffer, audio_np.T, samplerate=24000, format='WAV')
    buffer.seek(0)

    print(f"[{request.model}] Generated speech in {duration:.2f}s: '{text[:30]}...'")

    return StreamingResponse(buffer, media_type="audio/wav")

if __name__ == "__main__": 
    import uvicorn 
    uvicorn.run(app, host="0.0.0.0", port=8000)