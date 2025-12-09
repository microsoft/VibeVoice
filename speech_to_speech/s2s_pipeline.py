"""
Speech-to-Speech Pipeline with WebSocket Streaming
===================================================

Unified pipeline integrating:
- Silero VAD (MIT)
- faster-whisper ASR (MIT)
- Qwen2.5-1.5B LLM (Apache 2.0)
- VibeVoice TTS (MIT)

Target: <800ms end-to-end latency for real-time conversation.
"""

import logging
import asyncio
import time
import json
import os
import sys
from typing import Optional, Dict, Any, AsyncGenerator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import threading
from queue import Queue, Empty
from contextlib import asynccontextmanager

import numpy as np
import torch

# Add parent directory to path for VibeVoice imports
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketState
import uvicorn

logger = logging.getLogger(__name__)


class PipelineState(Enum):
    """Pipeline processing states"""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING_VAD = "processing_vad"
    TRANSCRIBING = "transcribing"
    GENERATING = "generating"
    SYNTHESIZING = "synthesizing"
    SPEAKING = "speaking"


# Domain-specific system prompts - designed for natural voice conversation
SYSTEM_PROMPTS = {
    "medical": """You are VEMI AI Medical Assistant, a caring and knowledgeable healthcare assistant created by Alvion Global Solutions.

CONVERSATION CONTEXT:
You are having an ongoing voice conversation. The user's previous messages are provided for context. Always remember what was discussed earlier and refer back to it naturally.

CRITICAL BEHAVIOR:
- NEVER say "Hello", "Hi", or any greeting UNLESS the user greets you first in this message
- If the user describes a symptom (like back pain), ASK CLARIFYING QUESTIONS:
  * "Which part of your back - upper, middle, or lower?"
  * "How long have you had this pain?"
  * "Is it sharp or dull? Does it come and go?"
  * "Did anything trigger it - lifting, sitting, injury?"
- After gathering information, suggest possible causes and recommend seeing a doctor if needed
- Remember the conversation context - if discussing back pain, stay on that topic

YOUR APPROACH:
1. Listen to the symptom
2. Ask 1-2 specific clarifying questions
3. Based on answers, provide targeted guidance
4. Recommend professional help when appropriate

IDENTITY:
- You are VEMI AI Medical Assistant - NEVER say "Chat Doctor" or any other name
- Be warm, empathetic, and professional
- Keep responses conversational (2-4 sentences)""",

    "automobile": """You are VEMI AI Automobile Assistant, a friendly and expert automotive technician created by Alvion Global Solutions.

CONVERSATION CONTEXT:
You are having an ongoing voice conversation. The user's previous messages are provided for context. Always remember what was discussed earlier and continue that topic naturally.

CRITICAL BEHAVIOR:
- NEVER say "Hello", "Hi", or any greeting UNLESS the user greets you first in this message
- If the user mentions a car problem (like flat tire, AC not working), ASK CLARIFYING QUESTIONS:
  * For AC: "Is it blowing warm air, or no air at all? Any strange noises?"
  * For tire: "Do you have a spare tire? Is the car in a safe location?"
  * For engine: "Any warning lights on? What sound is it making?"
- Remember conversation context - if you were discussing AC, questions like "how to test it" refer to the AC
- Provide step-by-step troubleshooting guidance

YOUR APPROACH:
1. Understand the specific problem
2. Ask 1-2 diagnostic questions
3. Give practical troubleshooting steps
4. Recommend a mechanic for complex/dangerous repairs

IDENTITY:
- You are VEMI AI Automobile Assistant - always identify as VEMI AI
- Be helpful like a knowledgeable mechanic friend
- Keep responses conversational (2-4 sentences)""",

    "general": """You are VEMI AI, a friendly and helpful voice assistant created by Alvion Global Solutions.

CONVERSATION CONTEXT:
You are having an ongoing voice conversation. Remember what was discussed earlier and maintain context.

CRITICAL BEHAVIOR:
- NEVER say "Hello", "Hi", or any greeting UNLESS the user greets you first in this message
- Remember conversation context and refer back to previous topics naturally
- Ask clarifying questions when the user's request is unclear

IDENTITY:
- You are VEMI AI - always identify yourself as VEMI AI
- Be conversational and natural
- Keep responses concise (2-3 sentences)"""
}


@dataclass
class PipelineConfig:
    """Configuration for S2S Pipeline"""
    # Component settings
    asr_model: str = "small.en"  # faster-whisper built-in model, fast and accurate
    llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    tts_model: str = "microsoft/VibeVoice-Realtime-0.5B"
    
    # Device settings
    device: str = "cuda"
    asr_compute_type: str = "int8"
    llm_dtype: str = "bfloat16"
    
    # Audio settings
    input_sample_rate: int = 16000
    output_sample_rate: int = 24000
    
    # VAD settings - tuned for responsive barge-in
    vad_threshold: float = 0.4  # Lower = more sensitive to speech (better barge-in)
    min_speech_ms: int = 150  # Shorter = faster detection of user speaking
    min_silence_ms: int = 200  # Shorter = quicker end-of-speech detection
    
    # LLM settings
    max_llm_tokens: int = 150  # Allow longer responses for detailed answers
    llm_temperature: float = 0.7
    use_finetuned_model: bool = False  # Fine-tuned model learned ChatDoctor patterns from training data
    
    # TTS settings
    tts_voice: str = "en-Carter_man"
    tts_cfg_scale: float = 1.5
    tts_inference_steps: int = 5
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8005
    
    # Agent settings
    default_agent: str = "general"


def clean_llm_response(text: str, user_greeted: bool = False) -> str:
    """
    Clean LLM response to remove unwanted patterns.
    Removes ChatDoctor references, unnecessary greetings, and other artifacts.
    
    Args:
        text: The LLM response text
        user_greeted: Whether the user greeted first (if False, remove greetings from response)
    """
    import re
    
    # Patterns to remove (case-insensitive)
    patterns_to_remove = [
        r"(?i)hi\s*,?\s*welcome\s+to\s+chat\s*doctor[^\n]*",
        r"(?i)thanks?\s+for\s+(calling|contacting|asking)\s+chat\s*doctor[^\n]*",
        r"(?i)chat\s*doctor\s*(forum|\.com|service)?\.?",
        r"(?i)best\s+regards[^\n]*chat\s*doctor[^\n]*",
        r"(?i)wishing\s+for\s+a\s+quick[^\n]*",
        r"(?i)hope\s+this\s+(may\s+)?answer[^\n]*",
        r"(?i)let\s+me\s+know\s+if\s+anything[^\n]*",
        r"(?i)take\s*care\.?\s*chat\s*doctor[^\n]*",
        r"(?i)regards\s*,?\s*chat\s*doctor[^\n]*",
    ]
    
    for pattern in patterns_to_remove:
        text = re.sub(pattern, "", text)
    
    # Remove greetings at the start if user didn't greet first
    if not user_greeted:
        # Remove common greetings at the start of responses
        greeting_patterns = [
            r"^(?i)(hello|hi|hey|greetings)[\s,!\.]*",
            r"^(?i)(good\s+(morning|afternoon|evening))[\s,!\.]*",
        ]
        for pattern in greeting_patterns:
            text = re.sub(pattern, "", text)
    
    # Clean up multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove trailing incomplete sentences
    if text and text[-1] not in '.!?':
        # Find last complete sentence
        for i in range(len(text) - 1, -1, -1):
            if text[i] in '.!?':
                text = text[:i + 1]
                break
    
    return text.strip()


def user_is_greeting(text: str) -> bool:
    """Check if user message is a greeting."""
    import re
    text_lower = text.lower().strip()
    greeting_patterns = [
        r"^(hello|hi|hey|greetings|good\s+(morning|afternoon|evening))[\s,!\.]*$",
        r"^(hello|hi|hey)[\s,!\.]+",
    ]
    for pattern in greeting_patterns:
        if re.match(pattern, text_lower):
            return True
    return False


@dataclass
class PipelineMetrics:
    """Metrics for pipeline performance"""
    vad_latency_ms: float = 0.0
    asr_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    tts_first_chunk_ms: float = 0.0
    total_latency_ms: float = 0.0
    tokens_generated: int = 0
    audio_duration_ms: float = 0.0


class S2SPipeline:
    """
    Speech-to-Speech Pipeline with streaming support.
    
    Processes audio through:
    1. VAD - Voice Activity Detection
    2. ASR - Speech Recognition
    3. LLM - Response Generation (sentence-level streaming)
    4. TTS - Speech Synthesis (streaming)
    
    Features:
    - Sentence-level LLM→TTS streaming for lower latency
    - Barge-in cancellation support
    - Async cancellation tokens
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.state = PipelineState.IDLE
        self._initialized = False
        
        # Components (lazy loaded)
        self._vad = None
        self._asr = None
        self._llm = None
        self._tts_service = None
        
        # Metrics
        self.last_metrics = PipelineMetrics()
        self.total_requests = 0
        
        # Cancellation support for barge-in
        self._cancel_event = asyncio.Event()
        self._is_generating = False
        
        # Agent type for domain-specific responses
        self._agent_type = self.config.default_agent
        
        logger.info("S2S Pipeline created")
    
    def set_agent(self, agent_type: str) -> None:
        """Set the agent type for domain-specific responses"""
        if agent_type in SYSTEM_PROMPTS:
            self._agent_type = agent_type
            # Update LLM system prompt if LLM is initialized
            if self._llm:
                self._llm.set_system_prompt(SYSTEM_PROMPTS[agent_type])
            logger.info(f"Agent type set to: {agent_type}")
        else:
            logger.warning(f"Unknown agent type: {agent_type}, using general")
            self._agent_type = "general"
    
    def get_agent(self) -> str:
        """Get current agent type"""
        return self._agent_type
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for current agent"""
        return SYSTEM_PROMPTS.get(self._agent_type, SYSTEM_PROMPTS["general"])
    
    async def initialize(self) -> None:
        """Initialize all pipeline components"""
        if self._initialized:
            return
        
        logger.info("Initializing S2S Pipeline components...")
        start_time = time.time()
        
        # Initialize VAD
        logger.info("Loading VAD...")
        from .vad_module import StreamingVAD, VADConfig
        vad_config = VADConfig(
            threshold=self.config.vad_threshold,
            min_speech_duration_ms=self.config.min_speech_ms,
            min_silence_duration_ms=self.config.min_silence_ms
        )
        self._vad = StreamingVAD(vad_config)
        self._vad.initialize()
        
        # Initialize ASR
        logger.info("Loading ASR...")
        from .asr_module import StreamingASR, ASRConfig
        asr_config = ASRConfig(
            model_size=self.config.asr_model,
            device=self.config.device,
            compute_type=self.config.asr_compute_type
        )
        self._asr = StreamingASR(asr_config)
        self._asr.initialize()
        
        # Initialize LLM
        logger.info("Loading LLM...")
        from .llm_module import StreamingLLM, LLMConfig
        
        # Determine which model to load
        llm_model_path = self.config.llm_model  # Default: HuggingFace base model
        
        # Only use fine-tuned models if explicitly enabled
        # Note: Fine-tuned models may have learned patterns from training data (e.g., ChatDoctor)
        # that override system prompts. Use base model with strong prompts for better control.
        if self.config.use_finetuned_model:
            import os
            script_dir = os.path.dirname(os.path.abspath(__file__))
            local_models = {
                "medical": os.path.join(script_dir, "models", "medical_llm"),
                "automobile": os.path.join(script_dir, "models", "automobile_llm"),
            }
            
            # Check if we have a fine-tuned model for the current agent
            if self._agent_type in local_models:
                local_path = local_models[self._agent_type]
                if os.path.exists(local_path) and os.path.isdir(local_path):
                    if any(f.endswith(('.safetensors', '.bin')) for f in os.listdir(local_path)):
                        llm_model_path = local_path
                        logger.info(f"Using fine-tuned {self._agent_type} model: {local_path}")
        else:
            logger.info(f"Using base model (fine-tuned models disabled): {llm_model_path}")
        
        llm_config = LLMConfig(
            model_name=llm_model_path,
            device=self.config.device,
            dtype=self.config.llm_dtype,
            max_new_tokens=self.config.max_llm_tokens,
            temperature=self.config.llm_temperature
        )
        self._llm = StreamingLLM(llm_config)
        self._llm.initialize()
        
        # Set initial system prompt based on agent type
        self._llm.set_system_prompt(SYSTEM_PROMPTS.get(self._agent_type, SYSTEM_PROMPTS["general"]))
        
        # Initialize TTS (connect to VibeVoice server)
        logger.info("Initializing TTS connection...")
        self._tts_service = TTSClient(
            model_path=self.config.tts_model,
            device=self.config.device,
            voice=self.config.tts_voice,
            cfg_scale=self.config.tts_cfg_scale,
            inference_steps=self.config.tts_inference_steps
        )
        await self._tts_service.initialize()
        
        # Warm up models to reduce first inference latency
        await self._warmup_models()
        
        init_time = (time.time() - start_time) * 1000
        self._initialized = True
        logger.info(f"S2S Pipeline initialized in {init_time:.0f}ms")
    
    async def _warmup_models(self) -> None:
        """Warm up LLM and TTS models with a test inference"""
        logger.info("Warming up models...")
        warmup_start = time.time()
        
        try:
            # Warm up LLM with a short prompt
            _ = self._llm.respond("Hi")
            
            # Warm up TTS with a short text (consume but don't use the audio)
            async for _ in self._tts_service.synthesize_streaming("Hello"):
                pass
            
            warmup_time = (time.time() - warmup_start) * 1000
            logger.info(f"Models warmed up in {warmup_time:.0f}ms")
        except Exception as e:
            logger.warning(f"Warmup failed (non-critical): {e}")
    
    async def process_audio(
        self,
        audio_bytes: bytes,
        sample_rate: int = 16000
    ) -> AsyncGenerator[bytes, None]:
        """
        Process audio input and stream audio response.
        
        Args:
            audio_bytes: Raw PCM16 audio bytes from client
            sample_rate: Audio sample rate
            
        Yields:
            PCM16 audio chunks for playback
        """
        if not self._initialized:
            await self.initialize()
        
        pipeline_start = time.time()
        self.state = PipelineState.PROCESSING_VAD
        
        # Step 1: VAD - Extract speech segments
        vad_start = time.time()
        speech_segments = self._vad.process_audio(audio_bytes, sample_rate)
        vad_time = (time.time() - vad_start) * 1000
        
        if not speech_segments:
            self.state = PipelineState.IDLE
            return
        
        # Process each speech segment
        for speech_audio in speech_segments:
            self.state = PipelineState.TRANSCRIBING
            
            # Step 2: ASR - Transcribe speech
            asr_start = time.time()
            transcription = self._asr.transcribe_segment(speech_audio, sample_rate)
            asr_time = (time.time() - asr_start) * 1000
            
            if not transcription.text.strip():
                continue
            
            logger.info(f"User: {transcription.text}")
            self.state = PipelineState.GENERATING
            
            # Check if user greeted (to allow greeting in response)
            user_greeted = user_is_greeting(transcription.text)
            
            # Step 3: LLM - Generate response
            llm_start = time.time()
            llm_response = self._llm.respond(transcription.text)
            llm_time = (time.time() - llm_start) * 1000
            
            # Clean response to remove unwanted patterns and greetings (if user didn't greet)
            cleaned_response = clean_llm_response(llm_response.text, user_greeted)
            llm_response.text = cleaned_response
            
            logger.info(f"Assistant: {llm_response.text}")
            self.state = PipelineState.SYNTHESIZING
            
            # Step 4: TTS - Synthesize speech
            tts_start = time.time()
            first_chunk = True
            
            async for audio_chunk in self._tts_service.synthesize_streaming(llm_response.text):
                if first_chunk:
                    tts_first_chunk = (time.time() - tts_start) * 1000
                    total_time = (time.time() - pipeline_start) * 1000
                    
                    # Update metrics
                    self.last_metrics = PipelineMetrics(
                        vad_latency_ms=vad_time,
                        asr_latency_ms=asr_time,
                        llm_latency_ms=llm_time,
                        tts_first_chunk_ms=tts_first_chunk,
                        total_latency_ms=total_time,
                        tokens_generated=llm_response.tokens_generated
                    )
                    
                    logger.info(f"E2E Latency: {total_time:.0f}ms "
                               f"(VAD:{vad_time:.0f} ASR:{asr_time:.0f} "
                               f"LLM:{llm_time:.0f} TTS:{tts_first_chunk:.0f})")
                    
                    first_chunk = False
                
                self.state = PipelineState.SPEAKING
                yield audio_chunk
        
        self.total_requests += 1
        self.state = PipelineState.IDLE
    
    async def process_speech_segment(
        self,
        speech_audio: np.ndarray,
        sample_rate: int = 16000
    ) -> AsyncGenerator[bytes, None]:
        """
        Process a complete speech segment (from VAD).
        
        Supports barge-in cancellation during TTS streaming.
        
        Args:
            speech_audio: Speech audio numpy array
            sample_rate: Audio sample rate
            
        Yields:
            PCM16 audio chunks
        """
        if not self._initialized:
            await self.initialize()
        
        # Reset cancellation state
        self._cancel_event.clear()
        self._is_generating = True
        
        pipeline_start = time.time()
        
        try:
            # ASR
            self.state = PipelineState.TRANSCRIBING
            asr_start = time.time()
            transcription = self._asr.transcribe_segment(speech_audio, sample_rate)
            asr_time = (time.time() - asr_start) * 1000
            
            if not transcription.text.strip():
                self.state = PipelineState.IDLE
                return
            
            logger.info(f"[ASR] '{transcription.text}' ({asr_time:.0f}ms)")
            
            # Check for cancellation after ASR
            if self._cancel_event.is_set():
                logger.info("Cancelled after ASR")
                return
            
            # Check if user greeted
            user_greeted = user_is_greeting(transcription.text)
            
            # LLM
            self.state = PipelineState.GENERATING
            llm_start = time.time()
            llm_response = self._llm.respond(transcription.text)
            llm_time = (time.time() - llm_start) * 1000
            
            # Clean response to remove unwanted patterns and greetings
            llm_response.text = clean_llm_response(llm_response.text, user_greeted)
            
            logger.info(f"[LLM] '{llm_response.text}' ({llm_time:.0f}ms)")
            
            # Check for cancellation after LLM
            if self._cancel_event.is_set():
                logger.info("Cancelled after LLM")
                return
            
            # TTS
            self.state = PipelineState.SYNTHESIZING
            tts_start = time.time()
            first_chunk = True
            
            async for audio_chunk in self._tts_service.synthesize_streaming(llm_response.text):
                # Check for cancellation during TTS streaming
                if self._cancel_event.is_set():
                    logger.info("Cancelled during TTS")
                    return
                
                if first_chunk:
                    tts_first = (time.time() - tts_start) * 1000
                    total = (time.time() - pipeline_start) * 1000
                    
                    self.last_metrics = PipelineMetrics(
                        asr_latency_ms=asr_time,
                        llm_latency_ms=llm_time,
                        tts_first_chunk_ms=tts_first,
                        total_latency_ms=total,
                        tokens_generated=llm_response.tokens_generated
                    )
                    
                    logger.info(f"[E2E] {total:.0f}ms (ASR:{asr_time:.0f} LLM:{llm_time:.0f} TTS:{tts_first:.0f})")
                    first_chunk = False
                
                self.state = PipelineState.SPEAKING
                yield audio_chunk
            
            self.total_requests += 1
        
        finally:
            self._is_generating = False
            self.state = PipelineState.IDLE
    
    async def process_speech_segment_streaming(
        self,
        speech_audio: np.ndarray,
        sample_rate: int = 16000
    ) -> AsyncGenerator[bytes, None]:
        """
        Process speech with sentence-level LLM→TTS streaming for lower latency.
        
        Instead of waiting for full LLM response, this streams sentences to TTS
        as soon as they are detected (at punctuation boundaries).
        
        Args:
            speech_audio: Speech audio numpy array
            sample_rate: Audio sample rate
            
        Yields:
            PCM16 audio chunks
        """
        if not self._initialized:
            await self.initialize()
        
        # Reset cancellation state
        self._cancel_event.clear()
        self._is_generating = True
        
        pipeline_start = time.time()
        
        try:
            # ASR
            self.state = PipelineState.TRANSCRIBING
            asr_start = time.time()
            transcription = self._asr.transcribe_segment(speech_audio, sample_rate)
            asr_time = (time.time() - asr_start) * 1000
            
            if not transcription.text.strip():
                self.state = PipelineState.IDLE
                return
            
            logger.info(f"[ASR] '{transcription.text}' ({asr_time:.0f}ms)")
            
            # Check for cancellation
            if self._cancel_event.is_set():
                logger.info("Cancelled after ASR")
                return
            
            # Check if user greeted
            user_greeted = user_is_greeting(transcription.text)
            
            # LLM with sentence-level streaming
            self.state = PipelineState.GENERATING
            llm_start = time.time()
            
            sentence_buffer = ""
            full_response = ""
            sentence_delimiters = {'.', '!', '?', '。', '！', '？'}
            first_audio_chunk = True
            tts_first_time = 0
            tokens_count = 0
            
            async for token in self._llm.respond_streaming_async(transcription.text):
                # Check for cancellation
                if self._cancel_event.is_set():
                    logger.info("Cancelled during LLM generation")
                    return
                
                sentence_buffer += token
                full_response += token
                tokens_count += 1
                
                # Check for sentence boundary
                if any(sentence_buffer.rstrip().endswith(d) for d in sentence_delimiters):
                    sentence = sentence_buffer.strip()
                    # Clean the sentence to remove unwanted patterns
                    sentence = clean_llm_response(sentence, user_greeted)
                    if sentence:
                        # Log first sentence timing
                        if first_audio_chunk:
                            llm_time = (time.time() - llm_start) * 1000
                            logger.info(f"[LLM] First sentence: '{sentence[:50]}...' ({llm_time:.0f}ms)")
                        
                        # Stream this sentence to TTS
                        self.state = PipelineState.SYNTHESIZING
                        tts_start = time.time()
                        
                        async for audio_chunk in self._tts_service.synthesize_streaming(sentence):
                            if self._cancel_event.is_set():
                                logger.info("Cancelled during TTS")
                                return
                            
                            if first_audio_chunk:
                                tts_first_time = (time.time() - tts_start) * 1000
                                total = (time.time() - pipeline_start) * 1000
                                
                                self.last_metrics = PipelineMetrics(
                                    asr_latency_ms=asr_time,
                                    llm_latency_ms=llm_time,
                                    tts_first_chunk_ms=tts_first_time,
                                    total_latency_ms=total,
                                    tokens_generated=tokens_count
                                )
                                
                                logger.info(f"[E2E] {total:.0f}ms (ASR:{asr_time:.0f} LLM:{llm_time:.0f} TTS:{tts_first_time:.0f})")
                                first_audio_chunk = False
                            
                            self.state = PipelineState.SPEAKING
                            yield audio_chunk
                    
                    sentence_buffer = ""
            
            # Process any remaining text in buffer
            remaining = sentence_buffer.strip()
            remaining = clean_llm_response(remaining, user_greeted)  # Clean remaining text
            if remaining and not self._cancel_event.is_set():
                self.state = PipelineState.SYNTHESIZING
                async for audio_chunk in self._tts_service.synthesize_streaming(remaining):
                    if self._cancel_event.is_set():
                        return
                    self.state = PipelineState.SPEAKING
                    yield audio_chunk
            
            # Log full response
            logger.info(f"[LLM] Full: '{full_response}'")
            
            self.total_requests += 1
            
        finally:
            self._is_generating = False
            self.state = PipelineState.IDLE
    
    def cancel(self) -> None:
        """Cancel ongoing generation (for barge-in)"""
        if self._is_generating:
            self._cancel_event.set()
            logger.info("Generation cancelled (barge-in)")
    
    def reset(self) -> None:
        """Reset pipeline state"""
        # Cancel any ongoing generation
        self.cancel()
        self.state = PipelineState.IDLE
        if self._vad:
            self._vad.reset()
        if self._llm:
            self._llm.reset()
    
    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status"""
        return {
            "state": self.state.value,
            "initialized": self._initialized,
            "total_requests": self.total_requests,
            "last_metrics": {
                "vad_ms": self.last_metrics.vad_latency_ms,
                "asr_ms": self.last_metrics.asr_latency_ms,
                "llm_ms": self.last_metrics.llm_latency_ms,
                "tts_first_chunk_ms": self.last_metrics.tts_first_chunk_ms,
                "total_ms": self.last_metrics.total_latency_ms,
            }
        }


class TTSClient:
    """
    TTS Client that connects to VibeVoice TTS service.
    Can run as embedded service or connect to external.
    """
    
    def __init__(
        self,
        model_path: str = "microsoft/VibeVoice-Realtime-0.5B",
        device: str = "cuda",
        voice: str = "en-Carter_man",
        cfg_scale: float = 1.5,
        inference_steps: int = 5
    ):
        self.model_path = model_path
        self.device = device
        self.voice = voice
        self.cfg_scale = cfg_scale
        self.inference_steps = inference_steps
        self._service = None
        self._initialized = False
        self.sample_rate = 24000
    
    async def initialize(self) -> None:
        """Initialize TTS service"""
        if self._initialized:
            return
        
        try:
            # Try to import VibeVoice components
            from vibevoice.modular.modeling_vibevoice_streaming_inference import (
                VibeVoiceStreamingForConditionalGenerationInference,
            )
            from vibevoice.processor.vibevoice_streaming_processor import (
                VibeVoiceStreamingProcessor,
            )
            
            logger.info(f"Loading VibeVoice TTS: {self.model_path}")
            
            # Determine dtype and attention
            if self.device == "cuda":
                load_dtype = torch.bfloat16
                attn_impl = "flash_attention_2"
            else:
                load_dtype = torch.float32
                attn_impl = "sdpa"
            
            # Load processor
            self._processor = VibeVoiceStreamingProcessor.from_pretrained(self.model_path)
            
            # Load model
            try:
                self._model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map=self.device if self.device == "cuda" else None,
                    attn_implementation=attn_impl,
                )
            except Exception:
                logger.warning("Flash attention not available, using SDPA")
                self._model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map=self.device if self.device == "cuda" else None,
                    attn_implementation="sdpa",
                )
            
            self._model.eval()
            
            # Configure scheduler
            self._model.model.noise_scheduler = self._model.model.noise_scheduler.from_config(
                self._model.model.noise_scheduler.config,
                algorithm_type="sde-dpmsolver++",
                beta_schedule="squaredcos_cap_v2",
            )
            self._model.set_ddpm_inference_steps(num_steps=self.inference_steps)
            
            # Load voice preset
            self._load_voice_preset()
            
            self._initialized = True
            logger.info("VibeVoice TTS initialized")
            
        except ImportError as e:
            logger.error(f"VibeVoice not available: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize TTS: {e}")
            raise
    
    def _load_voice_preset(self) -> None:
        """Load voice preset"""
        voices_dir = REPO_ROOT / "demo" / "voices" / "streaming_model"
        
        # Find voice file
        voice_file = voices_dir / f"{self.voice}.pt"
        if not voice_file.exists():
            # Try to find any voice
            voice_files = list(voices_dir.glob("*.pt"))
            if voice_files:
                voice_file = voice_files[0]
                self.voice = voice_file.stem
                logger.warning(f"Voice not found, using: {self.voice}")
            else:
                raise FileNotFoundError(f"No voice presets found in {voices_dir}")
        
        device = torch.device(self.device if self.device != "cpu" else "cpu")
        self._voice_preset = torch.load(voice_file, map_location=device, weights_only=False)
        logger.info(f"Loaded voice preset: {self.voice}")
    
    def set_voice(self, voice: str) -> None:
        """
        Change the TTS voice preset.
        
        Available voices:
        - en-Carter_man (default)
        - en-Davis_man
        - en-Frank_man
        - en-Mike_man
        - en-Emma_woman
        - en-Grace_woman
        - in-Samuel_man
        """
        if voice != self.voice:
            self.voice = voice
            self._load_voice_preset()
    
    def get_available_voices(self) -> list:
        """Get list of available voice presets"""
        voices_dir = REPO_ROOT / "demo" / "voices" / "streaming_model"
        if not voices_dir.exists():
            return []
        return [f.stem for f in voices_dir.glob("*.pt")]
    
    async def synthesize_streaming(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Synthesize speech from text with streaming.
        
        Args:
            text: Text to synthesize
            
        Yields:
            PCM16 audio bytes
        """
        if not self._initialized:
            await self.initialize()
        
        if not text.strip():
            return
        
        import copy
        from vibevoice.modular.streamer import AudioStreamer
        
        # Prepare inputs
        text = text.replace("'", "'")
        processor_kwargs = {
            "text": text.strip(),
            "cached_prompt": self._voice_preset,
            "padding": True,
            "return_tensors": "pt",
            "return_attention_mask": True,
        }
        
        processed = self._processor.process_input_with_cached_prompt(**processor_kwargs)
        device = torch.device(self.device if self.device != "cpu" else "cpu")
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in processed.items()
        }
        
        # Create streamer
        audio_streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
        stop_event = threading.Event()
        errors = []
        
        # Run generation in thread
        def run_generation():
            try:
                self._model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=self.cfg_scale,
                    tokenizer=self._processor.tokenizer,
                    generation_config={
                        "do_sample": False,
                        "temperature": 1.0,
                        "top_p": 1.0,
                    },
                    audio_streamer=audio_streamer,
                    stop_check_fn=stop_event.is_set,
                    verbose=False,
                    refresh_negative=True,
                    all_prefilled_outputs=copy.deepcopy(self._voice_preset),
                )
            except Exception as e:
                errors.append(e)
                audio_streamer.end()
        
        thread = threading.Thread(target=run_generation, daemon=True)
        thread.start()
        
        # Stream audio chunks
        try:
            stream = audio_streamer.get_stream(0)
            for audio_chunk in stream:
                if torch.is_tensor(audio_chunk):
                    audio_chunk = audio_chunk.detach().cpu().to(torch.float32).numpy()
                else:
                    audio_chunk = np.asarray(audio_chunk, dtype=np.float32)
                
                if audio_chunk.ndim > 1:
                    audio_chunk = audio_chunk.reshape(-1)
                
                # Normalize
                peak = np.max(np.abs(audio_chunk)) if audio_chunk.size else 0.0
                if peak > 1.0:
                    audio_chunk = audio_chunk / peak
                
                # Convert to PCM16 bytes
                audio_chunk = np.clip(audio_chunk, -1.0, 1.0)
                pcm16 = (audio_chunk * 32767.0).astype(np.int16)
                
                yield pcm16.tobytes()
                await asyncio.sleep(0)  # Yield control
                
        finally:
            stop_event.set()
            audio_streamer.end()
            thread.join(timeout=1.0)
            
            if errors:
                logger.error(f"TTS error: {errors[0]}")


# =============================================================================
# FastAPI Application
# =============================================================================

def create_app(config: Optional[PipelineConfig] = None) -> FastAPI:
    """Create FastAPI application"""
    
    # Pipeline instance (created before lifespan)
    pipeline_config = config or PipelineConfig()
    pipeline = S2SPipeline(pipeline_config)
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan context manager for startup/shutdown"""
        logger.info("Starting S2S Pipeline server...")
        await pipeline.initialize()
        logger.info(f"Server ready on port {pipeline_config.port}")
        yield
        # Cleanup on shutdown
        logger.info("Shutting down S2S Pipeline...")
    
    app = FastAPI(
        title="VibeVoice Speech-to-Speech",
        description="Real-time Speech-to-Speech with <800ms latency",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Store in app state
    app.state.pipeline = pipeline
    app.state.config = pipeline_config
    
    @app.get("/")
    async def index():
        """Serve agent selection page"""
        frontend_path = Path(__file__).parent / "frontend" / "agent_select.html"
        if frontend_path.exists():
            return FileResponse(frontend_path)
        # Fallback to chat interface
        chat_path = Path(__file__).parent / "frontend" / "index.html"
        if chat_path.exists():
            return FileResponse(chat_path)
        return JSONResponse({"message": "S2S Pipeline API", "status": "running"})
    
    @app.get("/chat")
    async def chat(agent: str = "medical"):
        """Serve chat interface with agent parameter"""
        frontend_path = Path(__file__).parent / "frontend" / "index.html"
        if frontend_path.exists():
            return FileResponse(frontend_path)
        return JSONResponse({"message": "Chat interface not found", "status": "error"})
    
    @app.get("/health")
    async def health():
        """Health check"""
        return {
            "status": "ok",
            "pipeline_state": pipeline.state.value,
            "initialized": pipeline._initialized
        }
    
    @app.get("/status")
    async def status():
        """Get pipeline status"""
        return pipeline.get_status()
    
    @app.get("/config")
    async def get_config():
        """Get pipeline configuration"""
        return {
            "asr_model": pipeline_config.asr_model,
            "llm_model": pipeline_config.llm_model,
            "tts_model": pipeline_config.tts_model,
            "input_sample_rate": pipeline_config.input_sample_rate,
            "output_sample_rate": pipeline_config.output_sample_rate,
        }
    
    @app.get("/voices")
    async def get_voices():
        """Get available TTS voices"""
        voices = pipeline._tts_service.get_available_voices() if pipeline._tts_service else []
        return {
            "voices": voices,
            "current_voice": pipeline._tts_service.voice if pipeline._tts_service else "en-Carter_man",
            "voice_info": {
                "en-Carter_man": {"name": "Carter", "gender": "Male", "accent": "American"},
                "en-Davis_man": {"name": "Davis", "gender": "Male", "accent": "American"},
                "en-Frank_man": {"name": "Frank", "gender": "Male", "accent": "American"},
                "en-Mike_man": {"name": "Mike", "gender": "Male", "accent": "American"},
                "en-Emma_woman": {"name": "Emma", "gender": "Female", "accent": "American"},
                "en-Grace_woman": {"name": "Grace", "gender": "Female", "accent": "American"},
                "in-Samuel_man": {"name": "Samuel", "gender": "Male", "accent": "Indian"},
            }
        }
    
    @app.websocket("/stream")
    async def websocket_stream(ws: WebSocket):
        """
        WebSocket endpoint for real-time S2S.
        
        Client sends: Binary PCM16 audio at 16kHz
        Server sends: Binary PCM16 audio at 24kHz + JSON status messages
        """
        await ws.accept()
        logger.info("Client connected to S2S stream")
        
        try:
            # Send ready message
            await ws.send_json({
                "type": "status",
                "state": "ready",
                "input_sample_rate": pipeline_config.input_sample_rate,
                "output_sample_rate": pipeline_config.output_sample_rate
            })
            
            # Audio buffer for VAD
            audio_buffer = []
            
            while True:
                try:
                    # Receive audio data
                    data = await ws.receive()
                    
                    if "bytes" in data:
                        # Binary audio data
                        audio_bytes = data["bytes"]
                        
                        # Convert to numpy
                        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                        
                        # Process through VAD
                        speech_segments = pipeline._vad.process_audio(audio_bytes, pipeline_config.input_sample_rate)
                        
                        # Process each speech segment (with barge-in cancellation support)
                        for speech_audio in speech_segments:
                            # Send transcribing status
                            await ws.send_json({
                                "type": "status",
                                "state": "processing"
                            })
                            
                            # Process through pipeline (with cancellation support)
                            async for audio_chunk in pipeline.process_speech_segment(
                                speech_audio,
                                pipeline_config.input_sample_rate
                            ):
                                # Send audio chunk
                                await ws.send_bytes(audio_chunk)
                            
                            # Send metrics
                            await ws.send_json({
                                "type": "metrics",
                                **pipeline.last_metrics.__dict__
                            })
                            
                            # Send done status
                            await ws.send_json({
                                "type": "status",
                                "state": "ready"
                            })
                    
                    elif "text" in data:
                        # JSON message
                        msg = json.loads(data["text"])
                        
                        if msg.get("type") == "reset":
                            pipeline.reset()
                            await ws.send_json({"type": "status", "state": "reset"})
                        
                        elif msg.get("type") == "cancel":
                            # Barge-in: User interrupted, cancel current response
                            # Use cancel() instead of reset() to properly signal cancellation
                            pipeline.cancel()
                            logger.info("Barge-in: Response cancelled by user")
                            await ws.send_json({"type": "status", "state": "cancelled"})
                        
                        elif msg.get("type") == "ping":
                            # Keepalive ping - respond with pong
                            await ws.send_json({"type": "pong"})
                        
                        elif msg.get("type") == "set_agent":
                            # Set agent type for domain-specific responses
                            agent = msg.get("agent", "general")
                            pipeline.set_agent(agent)
                            await ws.send_json({
                                "type": "status", 
                                "state": "agent_set", 
                                "agent": pipeline.get_agent()
                            })
                        
                        elif msg.get("type") == "set_voice":
                            # Change TTS voice
                            voice = msg.get("voice", "en-Carter_man")
                            try:
                                pipeline._tts_service.set_voice(voice)
                                logger.info(f"Voice changed to: {voice}")
                                await ws.send_json({"type": "status", "state": "voice_changed", "voice": voice})
                            except Exception as e:
                                logger.error(f"Failed to change voice: {e}")
                                await ws.send_json({"type": "error", "message": f"Failed to change voice: {e}"})
                        
                        elif msg.get("type") == "welcome":
                            # Play domain-specific welcome message
                            agent = pipeline.get_agent()
                            welcome_messages = {
                                "medical": "Hello, this is VEMI AI Medical Assistant. How can I help you today?",
                                "automobile": "Hello, this is VEMI AI Automobile Assistant. How can I help you today?",
                                "general": "Hello, this is VEMI AI. How can I help you today?"
                            }
                            welcome_text = welcome_messages.get(agent, welcome_messages["general"])
                            logger.info(f"Playing welcome message for agent: {agent}")
                            await ws.send_json({"type": "status", "state": "processing"})
                            
                            # Synthesize welcome message
                            async for audio_chunk in pipeline._tts_service.synthesize_streaming(welcome_text):
                                await ws.send_bytes(audio_chunk)
                            
                            # Add to conversation
                            await ws.send_json({
                                "type": "transcript",
                                "user": "",
                                "assistant": welcome_text
                            })
                            await ws.send_json({"type": "status", "state": "ready"})
                        
                        elif msg.get("type") == "text":
                            # Direct text input (skip ASR)
                            text = msg.get("text", "")
                            if text:
                                await ws.send_json({"type": "status", "state": "processing"})
                                
                                # Check if user greeted
                                user_greeted = user_is_greeting(text)
                                
                                # Generate response
                                llm_start = time.time()
                                response = pipeline._llm.respond(text)
                                llm_time = (time.time() - llm_start) * 1000
                                
                                # Clean response to remove unwanted patterns
                                response.text = clean_llm_response(response.text, user_greeted)
                                
                                # Synthesize
                                tts_start = time.time()
                                first_chunk = True
                                async for audio_chunk in pipeline._tts_service.synthesize_streaming(response.text):
                                    if first_chunk:
                                        tts_first = (time.time() - tts_start) * 1000
                                        await ws.send_json({
                                            "type": "transcript",
                                            "user": text,
                                            "assistant": response.text,
                                            "llm_ms": llm_time,
                                            "tts_first_ms": tts_first
                                        })
                                        first_chunk = False
                                    
                                    await ws.send_bytes(audio_chunk)
                                
                                await ws.send_json({"type": "status", "state": "ready"})
                
                except WebSocketDisconnect:
                    logger.info("Client disconnected")
                    break
                except RuntimeError as e:
                    # Handle "Cannot call receive once disconnect received" gracefully
                    if "disconnect" in str(e).lower():
                        break
                    logger.error(f"WebSocket runtime error: {e}")
                    break
                except Exception as e:
                    logger.error(f"WebSocket error: {e}")
                    # Only try to send error if connection is still open
                    try:
                        if ws.client_state.name == "CONNECTED":
                            await ws.send_json({"type": "error", "message": str(e)})
                    except Exception:
                        pass  # Connection already closed, ignore
                    break
        
        finally:
            pipeline.reset()
            logger.info("Client disconnected from S2S stream")
    
    return app


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run the S2S pipeline server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="VibeVoice S2S Pipeline Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8005)
    parser.add_argument("--asr-model", type=str, default="distil-whisper/distil-small.en")
    parser.add_argument("--llm-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--tts-model", type=str, default="microsoft/VibeVoice-Realtime-0.5B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-level", type=str, default="INFO")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create config
    config = PipelineConfig(
        host=args.host,
        port=args.port,
        asr_model=args.asr_model,
        llm_model=args.llm_model,
        tts_model=args.tts_model,
        device=args.device
    )
    
    # Create and run app
    app = create_app(config)
    
    logger.info(f"Starting S2S server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
