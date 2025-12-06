#!/usr/bin/env python3
"""
LLM Integration Module - Llama-3.2-3B Streaming Generation
Optimized for real-time inference on RTX PRO 6000 Ada
"""

import os
import sys
import logging
import torch
from typing import AsyncIterator, Optional
from dataclasses import dataclass

# Setup paths
os.environ['HF_HOME'] = '/workspace/models/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/models/huggingface/transformers'

from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class LLMConfig:
    """LLM Configuration"""
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    model_path: str = "/workspace/models/huggingface/llama-3.2-3b"
    dtype: torch.dtype = torch.float16  # FP16 for speed
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # System prompt for conversational context
    system_prompt: str = """You are a helpful, concise AI assistant. 
Respond naturally and conversationally.
Keep responses brief (1-2 sentences max) for real-time voice interaction."""


# =============================================================================
# LLM STREAM WRAPPER
# =============================================================================

class LlamaStreamer:
    """Llama-3.2-3B streaming text generator"""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.model = None
        self.tokenizer = None
        self.device = torch.device(self.config.device)
        self.logger = logging.getLogger(__name__)
        
    def load_model(self):
        """Load Llama model and tokenizer"""
        try:
            self.logger.info(f"Loading {self.config.model_name}...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path or self.config.model_name,
                trust_remote_code=True,
                use_auth_token=os.getenv('HF_TOKEN')  # Auto-use env token
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path or self.config.model_name,
                torch_dtype=self.config.dtype,
                device_map="auto",  # Auto distribute across devices
                trust_remote_code=True,
                use_auth_token=os.getenv('HF_TOKEN'),
                load_in_8bit=False,  # Disable 8bit for speed (have VRAM)
                attn_implementation="sdpa"  # Scaled-dot product attention
            )
            
            self.model.eval()
            self.logger.info("✅ Llama model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def generate_streaming(
        self,
        text: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> AsyncIterator[str]:
        """
        Generate response token-by-token (streaming)
        
        Args:
            text: Input prompt
            system_prompt: System context (optional)
            max_tokens: Max generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling
        """
        
        if self.model is None:
            self.load_model()
        
        try:
            # Build prompt with system context
            if system_prompt is None:
                system_prompt = self.config.system_prompt
            
            prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate with streaming
            with torch.no_grad():
                output_tokens = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=False,
                    output_scores=False
                )
            
            # Decode response
            response = self.tokenizer.decode(
                output_tokens[0][inputs['input_ids'].shape[-1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Stream tokens one by one
            words = response.split()
            accumulated = ""
            
            for word in words:
                accumulated += word + " "
                yield accumulated.strip()
                
        except Exception as e:
            self.logger.error(f"❌ Generation error: {e}")
            raise
    
    def generate_simple(
        self,
        text: str,
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> str:
        """
        Simple non-streaming generation (faster for short responses)
        """
        
        if self.model is None:
            self.load_model()
        
        try:
            prompt = f"<|im_start|>system\n{self.config.system_prompt}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                output_tokens = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(
                output_tokens[0][inputs['input_ids'].shape[-1]:],
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"❌ Generation error: {e}")
            raise


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_llm(model_path: str = "/workspace/models/huggingface/llama-3.2-3b") -> LlamaStreamer:
    """Factory function to create LLM instance"""
    config = LLMConfig(model_path=model_path)
    llm = LlamaStreamer(config)
    llm.load_model()
    return llm


if __name__ == "__main__":
    # Test standalone
    logging.basicConfig(level=logging.INFO)
    
    llm = create_llm()
    
    test_prompt = "What is the capital of France?"
    print(f"\n📝 Prompt: {test_prompt}")
    print("\n🤖 Response:")
    
    response = llm.generate_simple(test_prompt)
    print(response)
