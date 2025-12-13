"""
VEMI AI - LLM Module using Qwen2.5-1.5B-Instruct
=================================================

Conversational AI backend for VEMI AI voice assistant.
Created by Alvion Global Solutions.

Apache 2.0 License - Commercially free, no attribution required.
Target Latency: <300ms for short responses.

Features:
- Streaming token generation for real-time voice
- Optimized for natural conversational AI
- BFloat16/INT8 quantization support
- VEMI AI personality and response guidelines
"""

import logging
import time
from typing import Optional, List, Iterator, Generator, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
import threading
import asyncio

import torch

logger = logging.getLogger(__name__)


class LLMModel(Enum):
    """Available LLM models (all Apache 2.0 or MIT)"""
    QWEN_0_5B = "Qwen/Qwen2.5-0.5B-Instruct"
    QWEN_1_5B = "Qwen/Qwen2.5-1.5B-Instruct"
    QWEN_3B = "Qwen/Qwen2.5-3B-Instruct"
    SMOLLM2_360M = "HuggingFaceTB/SmolLM2-360M-Instruct"
    SMOLLM2_1_7B = "HuggingFaceTB/SmolLM2-1.7B-Instruct"


@dataclass
class LLMConfig:
    """Configuration for LLM"""
    # Model settings
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    device: str = "cuda"
    dtype: str = "bfloat16"  # bfloat16, float16, float32, int8
    
    # Generation settings
    max_new_tokens: int = 400  # Increased for complete scenario questions
    min_new_tokens: int = 1
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    
    # Streaming settings
    do_sample: bool = True
    
    # System prompt for VEMI AI conversational voice agent
    system_prompt: str = """You are VEMI AI, a friendly voice assistant created by Alvion Global Solutions.

CRITICAL RULES:
- Your name is VEMI AI - always identify yourself as VEMI AI
- Keep responses SHORT (1-2 sentences) - this is a voice conversation
- Be conversational and natural, like talking to a helpful friend
- Do NOT greet unless the user greets you first
- NEVER say "Chat Doctor", "ChatDoctor", or any similar name - you are VEMI AI only

RESPONSE STYLE:
- Speak naturally as in a real conversation
- Use contractions (I'm, you're, that's)
- Be concise and direct
- If asked your name, say "I'm VEMI AI, your voice assistant"

NEVER DO:
- Never give long paragraphs
- Never use bullet points or lists
- Never ignore what the user actually asked"""

    # Early stop settings
    stop_strings: List[str] = None
    
    def __post_init__(self):
        if self.stop_strings is None:
            self.stop_strings = ["\n\n", "User:", "Human:", "Assistant:", "VEMI AI:", "VEMI:", "Chat Doctor", "ChatDoctor"]


@dataclass
class LLMResponse:
    """Response from LLM generation"""
    text: str
    tokens_generated: int
    generation_time_ms: float
    tokens_per_second: float
    is_complete: bool = True


class Qwen2LLM:
    """
    Qwen2.5-1.5B-Instruct LLM with streaming support.
    
    Usage:
        llm = Qwen2LLM(LLMConfig())
        llm.load_model()
        
        # Generate response
        response = llm.generate("Hello, how are you?")
        print(response.text)
        
        # Stream response
        for token in llm.generate_streaming("Tell me a joke"):
            print(token, end="", flush=True)
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.model = None
        self.tokenizer = None
        self._model_loaded = False
        self._lock = threading.Lock()
        self._device = None
        
        logger.info(f"LLM initialized: model={self.config.model_name}, "
                   f"device={self.config.device}, dtype={self.config.dtype}")
    
    def load_model(self) -> None:
        """Load the LLM model and tokenizer"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            logger.info(f"Loading LLM: {self.config.model_name}")
            start_time = time.time()
            
            # Determine device
            if self.config.device == "auto":
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self._device = torch.device(self.config.device)
            
            # Determine dtype
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            torch_dtype = dtype_map.get(self.config.dtype, torch.bfloat16)
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
            )
            
            # Ensure pad token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optimizations
            load_kwargs = {
                "torch_dtype": torch_dtype,
                "trust_remote_code": True,
                "device_map": "auto" if self.config.device == "cuda" else None,
            }
            
            # Add quantization if INT8
            if self.config.dtype == "int8":
                try:
                    from transformers import BitsAndBytesConfig
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )
                except ImportError:
                    logger.warning("bitsandbytes not available, using float16")
                    load_kwargs["torch_dtype"] = torch.float16
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                **load_kwargs
            )
            
            # Move to device if not using device_map
            if self.config.device != "cuda" or "device_map" not in load_kwargs:
                self.model = self.model.to(self._device)
            
            self.model.eval()
            
            load_time = (time.time() - start_time) * 1000
            self._model_loaded = True
            
            # Log model info
            param_count = sum(p.numel() for p in self.model.parameters()) / 1e9
            logger.info(f"LLM loaded in {load_time:.0f}ms ({param_count:.2f}B params)")
            
        except ImportError as e:
            raise ImportError(f"transformers not installed: {e}")
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            raise RuntimeError(f"LLM model loading failed: {e}")
    
    def generate(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        conversation_history: Optional[List[dict]] = None,
    ) -> LLMResponse:
        """
        Generate a complete response.
        
        Args:
            user_message: User's input message
            system_prompt: Override system prompt
            max_tokens: Override max tokens
            temperature: Override temperature
            conversation_history: List of previous turns [{"user": ..., "assistant": ...}]
            
        Returns:
            LLMResponse with generated text
        """
        if not self._model_loaded:
            raise RuntimeError("LLM not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        # Build messages with conversation history
        messages = self._build_messages(user_message, system_prompt, conversation_history)
        
        # Tokenize
        with self._lock:
            inputs = self._prepare_inputs(messages)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens or self.config.max_new_tokens,
                    min_new_tokens=self.config.min_new_tokens,
                    temperature=temperature or self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    repetition_penalty=self.config.repetition_penalty,
                    do_sample=self.config.do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode response
            input_length = inputs["input_ids"].shape[1]
            response_tokens = outputs[0][input_length:]
            response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        # Clean up response
        response_text = self._clean_response(response_text)
        
        generation_time = (time.time() - start_time) * 1000
        tokens_generated = len(response_tokens)
        tokens_per_second = tokens_generated / (generation_time / 1000) if generation_time > 0 else 0
        
        logger.debug(f"LLM generated {tokens_generated} tokens in {generation_time:.0f}ms "
                    f"({tokens_per_second:.1f} tok/s)")
        
        return LLMResponse(
            text=response_text,
            tokens_generated=tokens_generated,
            generation_time_ms=generation_time,
            tokens_per_second=tokens_per_second,
            is_complete=True
        )
    
    def generate_streaming(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        conversation_history: Optional[List[dict]] = None,
    ) -> Generator[str, None, None]:
        """
        Stream tokens as they are generated.
        
        Args:
            user_message: User's input message
            system_prompt: Override system prompt
            max_tokens: Override max tokens
            conversation_history: List of previous turns [{"user": ..., "assistant": ...}]
            
        Yields:
            Token strings as they are generated
        """
        if not self._model_loaded:
            raise RuntimeError("LLM not loaded. Call load_model() first.")
        
        from transformers import TextIteratorStreamer
        
        # Build messages with conversation history
        messages = self._build_messages(user_message, system_prompt, conversation_history)
        
        with self._lock:
            inputs = self._prepare_inputs(messages)
            
            # Create streamer
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )
            
            # Generate in background thread
            generation_kwargs = {
                **inputs,
                "max_new_tokens": max_tokens or self.config.max_new_tokens,
                "min_new_tokens": self.config.min_new_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "repetition_penalty": self.config.repetition_penalty,
                "do_sample": self.config.do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "streamer": streamer,
            }
            
            thread = threading.Thread(
                target=self._generate_thread,
                kwargs=generation_kwargs
            )
            thread.start()
            
            # Yield tokens as they arrive
            accumulated = ""
            for token in streamer:
                accumulated += token
                
                # Check for stop strings
                should_stop = False
                for stop_str in self.config.stop_strings:
                    if stop_str in accumulated:
                        # Yield up to stop string
                        idx = accumulated.find(stop_str)
                        if idx > 0:
                            yield accumulated[:idx]
                        should_stop = True
                        break
                
                if should_stop:
                    break
                    
                yield token
            
            thread.join()
    
    async def generate_streaming_async(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        conversation_history: Optional[List[dict]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Async streaming generation for WebSocket integration.
        """
        gen = self.generate_streaming(user_message, system_prompt, max_tokens, conversation_history)
        
        for token in gen:
            yield token
            await asyncio.sleep(0)  # Yield control
    
    def _build_messages(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[dict]] = None
    ) -> List[dict]:
        """Build chat messages format with optional conversation history"""
        messages = [
            {
                "role": "system",
                "content": system_prompt or self.config.system_prompt
            }
        ]
        
        # Add conversation history if provided
        if conversation_history:
            for turn in conversation_history:
                messages.append({"role": "user", "content": turn["user"]})
                messages.append({"role": "assistant", "content": turn["assistant"]})
        
        # Add current user message
        messages.append({
            "role": "user", 
            "content": user_message
        })
        
        return messages
    
    def _prepare_inputs(self, messages: List[dict]) -> dict:
        """Prepare inputs for model"""
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        # Move to device
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        return inputs
    
    def _generate_thread(self, **kwargs):
        """Thread target for streaming generation"""
        with torch.no_grad():
            self.model.generate(**kwargs)
    
    def _clean_response(self, text: str) -> str:
        """Clean up generated response"""
        # Remove stop strings
        for stop_str in self.config.stop_strings:
            if stop_str in text:
                text = text.split(stop_str)[0]
        
        # Clean whitespace
        text = text.strip()
        
        # Remove incomplete sentences at the end
        if text and text[-1] not in ".!?":
            # Find last complete sentence
            for punct in ".!?":
                idx = text.rfind(punct)
                if idx > 0:
                    text = text[:idx + 1]
                    break
        
        return text


class StreamingLLM:
    """
    High-level streaming LLM for S2S pipeline integration.
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.llm = Qwen2LLM(config)
        self.conversation_history = []
        self.max_history_turns = 5
        self._system_prompt = None  # Custom system prompt (None = use config default)
        
    def initialize(self) -> None:
        """Initialize the LLM model"""
        self.llm.load_model()
    
    def set_system_prompt(self, prompt: str) -> None:
        """Set a custom system prompt for domain-specific responses"""
        self._system_prompt = prompt
        logger.info(f"System prompt updated ({len(prompt)} chars)")
    
    def get_system_prompt(self) -> str:
        """Get current system prompt"""
        return self._system_prompt or self.llm.config.system_prompt
        
    def respond(
        self,
        user_message: str,
        include_history: bool = True  # Default to True for conversation context
    ) -> LLMResponse:
        """
        Generate a response to user message.
        
        Args:
            user_message: Transcribed user speech
            include_history: Whether to include conversation history (default True)
            
        Returns:
            LLM response
        """
        # Pass conversation history to LLM for proper multi-turn context
        history = self.conversation_history[-self.max_history_turns:] if include_history else None
        
        response = self.llm.generate(
            user_message, 
            system_prompt=self._system_prompt,
            conversation_history=history
        )
        
        # Update history with this turn
        self.conversation_history.append({
            "user": user_message,
            "assistant": response.text
        })
        
        # Trim history
        if len(self.conversation_history) > self.max_history_turns:
            self.conversation_history = self.conversation_history[-self.max_history_turns:]
        
        return response
    
    def respond_streaming(
        self,
        user_message: str,
        include_history: bool = True
    ) -> Generator[str, None, None]:
        """
        Stream response tokens with conversation history.
        """
        history = self.conversation_history[-self.max_history_turns:] if include_history else None
        
        full_response = ""
        for token in self.llm.generate_streaming(
            user_message, 
            system_prompt=self._system_prompt,
            conversation_history=history
        ):
            full_response += token
            yield token
        
        # Update history after streaming completes
        self.conversation_history.append({
            "user": user_message,
            "assistant": full_response
        })
        if len(self.conversation_history) > self.max_history_turns:
            self.conversation_history = self.conversation_history[-self.max_history_turns:]
    
    async def respond_streaming_async(
        self,
        user_message: str,
        include_history: bool = True
    ) -> AsyncGenerator[str, None]:
        """
        Async stream response tokens with conversation history.
        """
        history = self.conversation_history[-self.max_history_turns:] if include_history else None
        
        full_response = ""
        async for token in self.llm.generate_streaming_async(
            user_message, 
            system_prompt=self._system_prompt,
            conversation_history=history
        ):
            full_response += token
            yield token
        
        # Update history after streaming completes
        self.conversation_history.append({
            "user": user_message,
            "assistant": full_response
        })
        if len(self.conversation_history) > self.max_history_turns:
            self.conversation_history = self.conversation_history[-self.max_history_turns:]
    
    def reset(self) -> None:
        """Reset conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")


class PerplexityLLM:
    """
    Perplexity API-based LLM with real-time web search built-in.
    
    This replaces the local Qwen model with Perplexity's API for:
    - Real-time web search (no separate tool needed)
    - Faster responses via API
    - Up-to-date knowledge
    """
    
    def __init__(self, api_key: str, model: str = "sonar"):
        """
        Initialize Perplexity LLM.
        
        Args:
            api_key: Perplexity API key
            model: Model to use (sonar, sonar-pro, sonar-reasoning)
        
        Pricing (as of Dec 2024):
        - sonar: $1/M input tokens, $1/M output tokens (cheapest, fastest)
        - sonar-pro: $3/M input, $15/M output (better quality)
        - sonar-reasoning: $5/M input, $5/M output (best for complex reasoning)
        
        With $2 budget using 'sonar':
        - ~2M tokens total = ~10,000+ conversations (avg 200 tokens each)
        """
        import os
        self.api_key = api_key or os.environ.get("PERPLEXITY_API_KEY", "")
        self.model = model
        self.api_url = "https://api.perplexity.ai/chat/completions"
        self.conversation_history = []
        self.max_history_turns = 20  # Increased for viva examiner context retention
        self._system_prompt = None
        self.max_retries = 3
        self.retry_delay = 0.5  # seconds
        
        logger.info(f"PerplexityLLM initialized: model={model}")
    
    def initialize(self) -> None:
        """No initialization needed for API-based LLM"""
        if not self.api_key:
            logger.warning("PERPLEXITY_API_KEY not set - LLM will not work")
        else:
            logger.info("PerplexityLLM ready")
    
    def set_system_prompt(self, prompt: str) -> None:
        """Set a custom system prompt"""
        self._system_prompt = prompt
        logger.info(f"System prompt updated ({len(prompt)} chars)")
    
    def get_system_prompt(self) -> str:
        """Get current system prompt with voice-specific instructions"""
        base_prompt = self._system_prompt or "You are VEMI AI, a helpful voice assistant."
        
        # Add voice-specific instructions to prevent citations and markdown
        voice_instructions = """

CRITICAL VOICE OUTPUT RULES (ALWAYS FOLLOW):
- This is a VOICE conversation - your response will be spoken aloud by TTS
- NEVER use citations like [1], [2], [3] - they will be read aloud awkwardly
- NEVER use markdown formatting like **bold** or *italic* - speak naturally
- NEVER use bullet points, numbered lists, or special characters
- Keep responses SHORT (2-3 sentences max) - this is voice, not text
- Speak conversationally as if talking to a friend
- Do NOT reference sources or say "according to" - just give the answer naturally"""
        
        return base_prompt + voice_instructions
    
    def _clean_response_for_voice(self, text: str) -> str:
        """Clean response for voice output - remove citations, markdown, etc."""
        import re
        
        # Remove citation brackets [1], [2][3], etc.
        text = re.sub(r'\[\d+\]', '', text)
        
        # Remove markdown bold **text** and *text*
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        
        # Remove markdown headers # ## ###
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
        
        # Remove markdown links [text](url)
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        
        # Remove bullet points and numbered lists at start of lines
        text = re.sub(r'^\s*[-•*]\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s*', '', text, flags=re.MULTILINE)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove multiple newlines
        text = re.sub(r'\n+', ' ', text)
        
        return text.strip()
    
    def respond(
        self,
        user_message: str,
        include_history: bool = True
    ) -> LLMResponse:
        """
        Generate a response using Perplexity API with retry logic.
        """
        import requests
        start_time = time.time()
        
        if not self.api_key:
            return LLMResponse(
                text="I'm sorry, I'm not configured properly. Please set up the API key.",
                tokens_generated=0,
                generation_time_ms=0,
                tokens_per_second=0,
                is_complete=True
            )
        
        # Build messages
        messages = [{"role": "system", "content": self.get_system_prompt()}]
        
        # Add conversation history
        if include_history:
            for turn in self.conversation_history[-self.max_history_turns:]:
                messages.append({"role": "user", "content": turn["user"]})
                messages.append({"role": "assistant", "content": turn["assistant"]})
        
        # Add current message
        messages.append({"role": "user", "content": user_message})
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Payload for voice responses
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 400,  # Increased for complete scenario questions
            "temperature": 0.7
        }
        
        # Retry logic for network resilience
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=12  # Slightly reduced timeout
                )
                
                if response.status_code == 429:  # Rate limited
                    logger.warning(f"Perplexity rate limited, attempt {attempt + 1}/{self.max_retries}")
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                tokens = data.get("usage", {}).get("completion_tokens", 0)
                
                # Clean response for voice output
                answer = self._clean_response_for_voice(answer)
                
                gen_time = (time.time() - start_time) * 1000
                
                # Update history
                self.conversation_history.append({
                    "user": user_message,
                    "assistant": answer
                })
                if len(self.conversation_history) > self.max_history_turns:
                    self.conversation_history = self.conversation_history[-self.max_history_turns:]
                
                return LLMResponse(
                    text=answer,
                    tokens_generated=tokens,
                    generation_time_ms=gen_time,
                    tokens_per_second=tokens / (gen_time / 1000) if gen_time > 0 else 0,
                    is_complete=True
                )
                
            except requests.exceptions.Timeout:
                last_error = "Request timed out"
                logger.warning(f"Perplexity timeout, attempt {attempt + 1}/{self.max_retries}")
                continue
            except requests.exceptions.ConnectionError:
                last_error = "Connection error"
                logger.warning(f"Perplexity connection error, attempt {attempt + 1}/{self.max_retries}")
                time.sleep(self.retry_delay)
                continue
            except Exception as e:
                last_error = str(e)
                logger.error(f"Perplexity API error: {e}")
                break
        
        # All retries failed
        gen_time = (time.time() - start_time) * 1000
        return LLMResponse(
            text="I'm having trouble connecting right now. Please try again.",
            tokens_generated=0,
            generation_time_ms=gen_time,
            tokens_per_second=0,
            is_complete=False
        )
    
    def respond_streaming(
        self,
        user_message: str,
        include_history: bool = True
    ) -> Generator[str, None, None]:
        """
        Stream response (Perplexity doesn't support streaming, so we yield full response).
        """
        response = self.respond(user_message, include_history)
        yield response.text
    
    async def respond_streaming_async(
        self,
        user_message: str,
        include_history: bool = True
    ) -> AsyncGenerator[str, None]:
        """
        Async response - uses non-streaming for reliable voice output cleaning.
        """
        import aiohttp
        
        if not self.api_key:
            yield "I'm sorry, I'm not configured properly."
            return
        
        try:
            # Build messages
            messages = [{"role": "system", "content": self.get_system_prompt()}]
            
            if include_history:
                for turn in self.conversation_history[-self.max_history_turns:]:
                    messages.append({"role": "user", "content": turn["user"]})
                    messages.append({"role": "assistant", "content": turn["assistant"]})
            
            messages.append({"role": "user", "content": user_message})
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 400,  # Increased for complete scenario questions
                "temperature": 0.7
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=20)
                ) as response:
                    response.raise_for_status()
                    
                    import json
                    data = await response.json()
                    answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    
                    # Clean response for voice output
                    answer = self._clean_response_for_voice(answer)
                    
                    # Update history
                    self.conversation_history.append({
                        "user": user_message,
                        "assistant": answer
                    })
                    if len(self.conversation_history) > self.max_history_turns:
                        self.conversation_history = self.conversation_history[-self.max_history_turns:]
                    
                    # Yield the cleaned response
                    yield answer
                        
        except Exception as e:
            logger.error(f"Perplexity streaming error: {e}")
            yield "I'm having trouble connecting. Please try again."
    
    def reset(self) -> None:
        """Reset conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")


# Convenience function
def create_llm(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    device: str = "cuda",
    dtype: str = "bfloat16",
    max_tokens: int = 400
) -> StreamingLLM:
    """Create a configured streaming LLM instance"""
    config = LLMConfig(
        model_name=model_name,
        device=device,
        dtype=dtype,
        max_new_tokens=max_tokens
    )
    llm = StreamingLLM(config)
    llm.initialize()
    return llm


def create_perplexity_llm(
    api_key: str = None,
    model: str = "sonar"
) -> PerplexityLLM:
    """Create a Perplexity API-based LLM instance"""
    llm = PerplexityLLM(api_key=api_key, model=model)
    llm.initialize()
    return llm
