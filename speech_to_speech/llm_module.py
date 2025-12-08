"""
LLM Module using Qwen2.5-1.5B-Instruct for Speech-to-Speech Pipeline
=====================================================================

Apache 2.0 License - Commercially free, no attribution required.
Target Latency: <300ms for short responses.

Features:
- Streaming token generation
- Optimized for conversational AI
- BFloat16/INT8 quantization support
- Configurable response length and style
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
    max_new_tokens: int = 64  # Short responses for real-time
    min_new_tokens: int = 1
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    
    # Streaming settings
    do_sample: bool = True
    
    # System prompt for conversational AI
    system_prompt: str = """You are a helpful, friendly AI assistant engaged in real-time voice conversation.
Keep your responses concise and natural - typically 1-2 sentences.
Respond conversationally as if speaking, not writing.
Avoid lists, bullet points, or long explanations unless specifically asked."""

    # Early stop settings
    stop_strings: List[str] = None
    
    def __post_init__(self):
        if self.stop_strings is None:
            self.stop_strings = ["\n\n", "User:", "Human:", "Assistant:"]


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
    ) -> LLMResponse:
        """
        Generate a complete response.
        
        Args:
            user_message: User's input message
            system_prompt: Override system prompt
            max_tokens: Override max tokens
            temperature: Override temperature
            
        Returns:
            LLMResponse with generated text
        """
        if not self._model_loaded:
            raise RuntimeError("LLM not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        # Build messages
        messages = self._build_messages(user_message, system_prompt)
        
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
    ) -> Generator[str, None, None]:
        """
        Stream tokens as they are generated.
        
        Args:
            user_message: User's input message
            system_prompt: Override system prompt
            max_tokens: Override max tokens
            
        Yields:
            Token strings as they are generated
        """
        if not self._model_loaded:
            raise RuntimeError("LLM not loaded. Call load_model() first.")
        
        from transformers import TextIteratorStreamer
        
        # Build messages
        messages = self._build_messages(user_message, system_prompt)
        
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
    ) -> AsyncGenerator[str, None]:
        """
        Async streaming generation for WebSocket integration.
        """
        # Run sync generator in thread pool
        loop = asyncio.get_event_loop()
        
        gen = self.generate_streaming(user_message, system_prompt, max_tokens)
        
        for token in gen:
            yield token
            await asyncio.sleep(0)  # Yield control
    
    def _build_messages(
        self,
        user_message: str,
        system_prompt: Optional[str] = None
    ) -> List[dict]:
        """Build chat messages format"""
        messages = [
            {
                "role": "system",
                "content": system_prompt or self.config.system_prompt
            },
            {
                "role": "user", 
                "content": user_message
            }
        ]
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
        
    def initialize(self) -> None:
        """Initialize the LLM model"""
        self.llm.load_model()
        
    def respond(
        self,
        user_message: str,
        include_history: bool = False
    ) -> LLMResponse:
        """
        Generate a response to user message.
        
        Args:
            user_message: Transcribed user speech
            include_history: Whether to include conversation history
            
        Returns:
            LLM response
        """
        # Add context if using history
        if include_history and self.conversation_history:
            context = self._build_context(user_message)
            response = self.llm.generate(context)
        else:
            response = self.llm.generate(user_message)
        
        # Update history
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
        user_message: str
    ) -> Generator[str, None, None]:
        """
        Stream response tokens.
        """
        yield from self.llm.generate_streaming(user_message)
    
    async def respond_streaming_async(
        self,
        user_message: str
    ) -> AsyncGenerator[str, None]:
        """
        Async stream response tokens.
        """
        async for token in self.llm.generate_streaming_async(user_message):
            yield token
    
    def _build_context(self, user_message: str) -> str:
        """Build context from conversation history"""
        context_parts = []
        for turn in self.conversation_history[-3:]:  # Last 3 turns
            context_parts.append(f"User: {turn['user']}")
            context_parts.append(f"Assistant: {turn['assistant']}")
        context_parts.append(f"User: {user_message}")
        return "\n".join(context_parts)
    
    def reset(self) -> None:
        """Reset conversation history"""
        self.conversation_history = []


# Convenience function
def create_llm(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    device: str = "cuda",
    dtype: str = "bfloat16",
    max_tokens: int = 64
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
