from typing import Callable, List, Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.auto import AutoModel, AutoModelForCausalLM

from transformers.modeling_outputs import CausalLMOutput, BaseModelOutputWithPast
from transformers import modeling_utils
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.generation import GenerationMixin

from .modular_vibevoice_tokenizer import (
    VibeVoiceTokenizerStreamingCache, 
    VibeVoiceTokenizerEncoderOutput
)

from .configuration_vibevoice import VibeVoiceASRConfig
from .modeling_vibevoice import (
    VibeVoiceCausalLMOutputWithPast,    
    SpeechConnector
)

logger = logging.get_logger(__name__)

if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]

# @auto_docstring
class VibeVoiceASRPreTrainedModel(PreTrainedModel):
    config_class = VibeVoiceASRConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True
    _supports_flash_attn = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):

        # Use the language model's initializer_range if available
        if hasattr(self.config, 'language_model_config') and hasattr(self.config.language_model_config, 'initializer_range'):
            std = self.config.language_model_config.initializer_range
        elif hasattr(self.config, 'decoder_config') and hasattr(self.config.decoder_config, 'initializer_range'):
            std = self.config.decoder_config.initializer_range
        else:
            std = 0.02  # Default value
            
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

# @auto_docstring
class VibeVoiceASRModel(VibeVoiceASRPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        
        if hasattr(config, 'torch_dtype') and config.torch_dtype is not None:
            if isinstance(config.torch_dtype, str):
                dtype = getattr(torch, config.torch_dtype)
            else:
                dtype = config.torch_dtype
        else:
            dtype = torch.float32
        
        # Initialize Qwen2 model for language modeling
        lm_config = config.decoder_config 
        self.language_model = AutoModel.from_config(lm_config)
        
        # Initialize speech components if needed
        self.acoustic_tokenizer = AutoModel.from_config(config.acoustic_tokenizer_config).to(dtype)
        self.semantic_tokenizer = AutoModel.from_config(config.semantic_tokenizer_config).to(dtype)

        self.acoustic_connector = SpeechConnector(config.acoustic_vae_dim, lm_config.hidden_size).to(dtype)
        self.semantic_connector = SpeechConnector(config.semantic_vae_dim, lm_config.hidden_size).to(dtype)
    
    def get_input_embeddings(self):
        if hasattr(self.language_model, 'embed_tokens'):
            # If the language model has an embed_tokens attribute, return it
            return self.language_model.embed_tokens
        
        for name, attr in self.language_model.fullmap.items(): # parallel by nnscaler, the name is changed
            if attr.orig_name == 'embed_tokens.weight':
                return getattr(self.language_model, name)
        assert False, 'should not arrive here'

    def set_input_embeddings(self, value):
        self.language_model.embed_tokens = value
    
    def set_speech_tokenizers(self, acoustic_tokenizer=None, semantic_tokenizer=None):
        """Set the speech tokenizers used for encoding and decoding speech."""
        self.acoustic_tokenizer = acoustic_tokenizer
        self.semantic_tokenizer = semantic_tokenizer
        
        # Reset the encoder to evaluation mode
        if self.acoustic_tokenizer is not None:
            self.acoustic_tokenizer.eval()
            
        if self.semantic_tokenizer is not None:
            self.semantic_tokenizer.eval()
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Forward through language model
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )
        
        if not return_dict:
            return outputs
            
        return BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class VibeVoiceASRForConditionalGeneration(VibeVoiceASRPreTrainedModel, GenerationMixin):
    """
    VibeVoice model for Automatic Speech Recognition (ASR) with language modeling head for conditional generation.
    This class is designed for inference and generation tasks.
    """
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}

    def __init__(self, config):
        super().__init__(config)
        self.model = VibeVoiceASRModel(config)
        self.vocab_size = config.decoder_config.vocab_size
        
        # Determine the dtype to use
        if hasattr(config, 'torch_dtype') and config.torch_dtype is not None:
            if isinstance(config.torch_dtype, str):
                dtype = getattr(torch, config.torch_dtype)
            else:
                dtype = config.torch_dtype
        else:
            dtype = torch.float32
        
        # Initialize lm_head with the correct dtype
        self.lm_head = nn.Linear(config.decoder_config.hidden_size, self.vocab_size, bias=False).to(dtype)
        
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.language_model = decoder

    def get_decoder(self):
        return self.model.language_model

    def tie_weights(self):
        """Tie the weights between the input embeddings and the output embeddings."""
        if getattr(self.config.decoder_config, 'tie_word_embeddings', False):
            output_embeddings = self.get_output_embeddings()
            input_embeddings = self.get_input_embeddings()
            if hasattr(input_embeddings, 'weight'):
                output_embeddings.weight = input_embeddings.weight
            else:
                output_embeddings.weight = input_embeddings

    def encode_speech(
        self,
        speech_tensors: torch.FloatTensor,
        speech_masks: Optional[torch.BoolTensor] = None,
        speech_semantic_tensors: Optional[torch.FloatTensor] = None,
        streaming_segment_duration: float = 60.0,  # seconds
    ):
        """
        Encode speech input into features that can be used by the language model.
        This method is called once before generation to process the speech input.
        
        For long audio (>600s by default), uses streaming processing to avoid conv overflow (>2^32).
        Segments are processed independently, then concatenated before final sampling.
        
        Args:
            speech_tensors: Input audio tensor [batch_size, samples]
            speech_masks: Optional mask for speech features
            speech_semantic_tensors: Optional pre-computed semantic tokens
            streaming_segment_duration: Segment duration in seconds for streaming processing (default: 60s)
        """
        if hasattr(self.config, 'torch_dtype') and self.config.torch_dtype is not None:
            if isinstance(self.config.torch_dtype, str):
                dtype = getattr(torch, self.config.torch_dtype)
            else:
                dtype = self.config.torch_dtype
        else:
            dtype = torch.float32
            
        speech_tensors = speech_tensors.to(dtype)
        
        # Ensure proper shape: (batch, samples)
        if speech_tensors.ndim == 1:
            speech_tensors = speech_tensors.unsqueeze(0)
        
        batch_size, total_samples = speech_tensors.shape
        sample_rate = 24000  # fix 24kHz sample rate
        
        # Calculate segment size in samples
        segment_samples = int(streaming_segment_duration * sample_rate)
        
        # Decide whether to use streaming based on audio length
        use_streaming = total_samples > segment_samples
        
        with torch.no_grad():
            if not use_streaming:
                # Short audio: direct processing (original behavior)
                encoder_output = self.model.acoustic_tokenizer.encode(speech_tensors.unsqueeze(1))
                audio_tokens = encoder_output.sample(dist_type=self.model.acoustic_tokenizer.std_dist_type)[0]
                acoustic_features = self.model.acoustic_connector(audio_tokens)
                
                # Encode semantic features
                if speech_semantic_tensors is not None:
                    semantic_features = self.model.semantic_connector(speech_semantic_tensors)
                else:
                    semantic_tokens = self.model.semantic_tokenizer.encode(speech_tensors.unsqueeze(1)).mean
                    semantic_features = self.model.semantic_connector(semantic_tokens)
            else:
                # Long audio: streaming processing
                # print(f"Using streaming processing for long audio: {total_samples/sample_rate:.1f}s "
                #       f"(segment size: {streaming_segment_duration}s)")
                
                # Initialize caches for both tokenizers
                acoustic_encoder_cache = VibeVoiceTokenizerStreamingCache()
                semantic_encoder_cache = VibeVoiceTokenizerStreamingCache()
                acoustic_mean_segments = []
                semantic_mean_segments = []
                sample_indices = torch.arange(batch_size, device=speech_tensors.device)
                
                # Helper function from batch_asr_sft_cache.py
                def _iter_segments(total_length: int, segment_length: int):
                    """Iterate over audio segments with a given segment length."""
                    if segment_length <= 0:
                        raise ValueError("segment_length must be positive")
                    for start in range(0, total_length, segment_length):
                        end = min(start + segment_length, total_length)
                        if end > start:
                            yield start, end
                
                # Process each segment for both acoustic and semantic tokenizers
                segments = list(_iter_segments(total_samples, segment_samples))
                num_segments = len(segments)
                for seg_idx, (start, end) in enumerate(segments):
                    chunk = speech_tensors[:, start:end].contiguous()
                    if chunk.numel() == 0:
                        continue
                    
                    # Check if this is the final segment
                    is_final = (seg_idx == num_segments - 1)
                    
                    # Encode chunk for acoustic tokenizer (don't sample yet)
                    acoustic_encoder_output = self.model.acoustic_tokenizer.encode(
                        chunk.unsqueeze(1),
                        cache=acoustic_encoder_cache,
                        sample_indices=sample_indices,
                        use_cache=True,
                        is_final_chunk=is_final,
                    )
                    acoustic_mean_segments.append(acoustic_encoder_output.mean)
                    
                    # Encode chunk for semantic tokenizer (take mean directly)
                    semantic_encoder_output = self.model.semantic_tokenizer.encode(
                        chunk.unsqueeze(1),
                        cache=semantic_encoder_cache,
                        sample_indices=sample_indices,
                        use_cache=True,
                        is_final_chunk=is_final,
                    )
                    semantic_mean_segments.append(semantic_encoder_output.mean)
                
                # print(f"Processed {len(acoustic_mean_segments)} segments.")
                # Concatenate all acoustic means and sample once
                acoustic_mean_full = torch.cat(acoustic_mean_segments, dim=1).contiguous()
                acoustic_encoder_output = VibeVoiceTokenizerEncoderOutput(
                    mean=acoustic_mean_full,
                    std=self.model.acoustic_tokenizer.fix_std
                )
                audio_tokens = acoustic_encoder_output.sample(
                    dist_type=self.model.acoustic_tokenizer.std_dist_type
                )[0]
                acoustic_features = self.model.acoustic_connector(audio_tokens)
                
                # Concatenate all semantic means
                semantic_tokens = torch.cat(semantic_mean_segments, dim=1).contiguous()
                semantic_features = self.model.semantic_connector(semantic_tokens)
            
            # Combine acoustic and semantic features
            if speech_masks is not None:
                # Under device_map="auto", acoustic_connector/semantic_connector can be
                # dispatched to a different device than speech_masks (e.g. multi-GPU
                # pipeline parallelism), and boolean-indexing requires the mask to be
                # on the same device as the indexed tensor.
                speech_masks = speech_masks.to(acoustic_features.device)
                combined_features = acoustic_features[speech_masks] + semantic_features[speech_masks]
            else:
                combined_features = acoustic_features + semantic_features
                
        return combined_features

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # Speech-specific arguments
        speech_tensors: Optional[torch.FloatTensor] = None,
        speech_masks: Optional[torch.BoolTensor] = None,
        speech_semantic_tensors: Optional[torch.FloatTensor] = None,
        acoustic_input_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutput]:
        """
        Forward pass for the model. Handles both training and generation scenarios.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else getattr(self.config, 'use_cache', False)

        # Process inputs
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            
        # If we have speech input and acoustic_input_mask, encode and insert speech features
        if speech_tensors is not None and acoustic_input_mask is not None:
            speech_features = self.encode_speech(
                speech_tensors=speech_tensors,
                speech_masks=speech_masks,
                speech_semantic_tensors=speech_semantic_tensors,
            )
            # Clone to avoid in-place operation on leaf variable during training
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[acoustic_input_mask] = speech_features

        # Forward through the model
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return VibeVoiceCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def streaming_generate(
        self,
        audio_tensor: torch.FloatTensor,
        tokenizer,
        prompt_text: str = None,
        chunk_duration: float = 2.0,
        text_audio_delay: float = None,
        sample_rate: int = 24000,
        max_new_tokens_per_chunk: int = 256,
        temperature: float = 0.0,
        context_info: str = None,
        encode_mode: str = "split_then_encode",
        repetition_penalty: float = 1.0,
        pad_last_chunk: bool = True,
    ):
        """Streaming model: chunked ASR inference over a full audio tensor."""
        device = next(self.parameters()).device

        sp_start_id = tokenizer.speech_start_id
        sp_end_id = tokenizer.speech_end_id
        text_chunk_end_id = tokenizer.text_chunk_end_id
        eos_id = tokenizer.eos_token_id

        embed_tokens = self.get_input_embeddings()

        if prompt_text is None:
            keys_str = "speaker, content"
            if context_info:
                prompt_text = (
                    "You are a helpful assistant that transcribes audio input into text output. "
                    f"Please transcribe the following audios streamingly with these keys: {keys_str} "
                    f"and extra info: {context_info}\n"
                )
            else:
                prompt_text = (
                    "You are a helpful assistant that transcribes audio input into text output. "
                    f"Please transcribe the following audios streamingly with these keys: {keys_str}\n"
                )

        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        prompt_embeds = embed_tokens(prompt_tensor)

        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        audio_tensor = audio_tensor.to(device)

        if encode_mode == "encode_then_split":
            all_features = self.encode_speech(audio_tensor)
            total_tokens = all_features.shape[1]
            audio_duration = audio_tensor.shape[1] / sample_rate
            tps = total_tokens / audio_duration if audio_duration > 0 else 7.5
            tokens_per_chunk = max(1, math.ceil(chunk_duration * tps))

            _frame_dur = 3200 / sample_rate
            _delay = text_audio_delay if text_audio_delay is not None else 0.5
            lookahead_sec = round(_delay / _frame_dur) * _frame_dur
            lookahead_tokens = int(round(lookahead_sec * tps))

            feature_chunks = []
            text_start = 0
            while text_start < total_tokens:
                audio_end = min(text_start + tokens_per_chunk + lookahead_tokens, total_tokens)
                if audio_end > text_start:
                    feature_chunks.append(all_features[:, text_start:audio_end, :])
                text_start = min(text_start + tokens_per_chunk, total_tokens)
        else:
            total_samples = audio_tensor.shape[1]
            chunk_samples = int(chunk_duration * sample_rate)

            _frame_dur = 3200 / sample_rate
            _delay = text_audio_delay if text_audio_delay is not None else 0.5
            lookahead_sec = round(_delay / _frame_dur) * _frame_dur
            lookahead_samples = int(lookahead_sec * sample_rate)

            feature_chunks = []
            text_start_sample = 0
            _target_samples = chunk_samples + lookahead_samples
            while text_start_sample < total_samples:
                audio_end_sample = min(text_start_sample + chunk_samples + lookahead_samples, total_samples)
                if audio_end_sample > text_start_sample:
                    seg = audio_tensor[:, text_start_sample:audio_end_sample]
                    if pad_last_chunk and seg.shape[1] < _target_samples:
                        seg = F.pad(seg, (0, _target_samples - seg.shape[1]))
                    chunk_features = self.encode_speech(seg)
                    feature_chunks.append(chunk_features)
                text_start_sample = min(text_start_sample + chunk_samples, total_samples)

        total_chunks = len(feature_chunks)

        outputs = self(
            inputs_embeds=prompt_embeds,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values

        sp_start_embed = embed_tokens(torch.tensor([[sp_start_id]], device=device))
        sp_end_embed = embed_tokens(torch.tensor([[sp_end_id]], device=device))

        for chunk_idx, feat_chunk in enumerate(feature_chunks):
            audio_embeds = torch.cat([sp_start_embed, feat_chunk, sp_end_embed], dim=1)

            outputs = self(
                inputs_embeds=audio_embeds,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            next_logits = outputs.logits[:, -1:, :]

            chunk_tokens = []
            for _ in range(max_new_tokens_per_chunk):
                logits = next_logits[:, -1, :]

                if repetition_penalty != 1.0 and chunk_tokens:
                    prev_ids = torch.tensor(chunk_tokens, device=logits.device)
                    prev_logits = logits[:, prev_ids]
                    prev_logits = torch.where(
                        prev_logits > 0,
                        prev_logits / repetition_penalty,
                        prev_logits * repetition_penalty,
                    )
                    logits[:, prev_ids] = prev_logits

                if temperature <= 0:
                    next_token_id = torch.argmax(logits, dim=-1).item()
                else:
                    probs = F.softmax(logits / temperature, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1).squeeze(-1).item()

                if next_token_id == text_chunk_end_id or next_token_id == eos_id:
                    break

                chunk_tokens.append(next_token_id)

                next_embed = embed_tokens(torch.tensor([[next_token_id]], device=device))
                outputs = self(
                    inputs_embeds=next_embed,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = outputs.past_key_values
                next_logits = outputs.logits

            tce_embed = embed_tokens(torch.tensor([[text_chunk_end_id]], device=device))
            outputs = self(
                inputs_embeds=tce_embed,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values

            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            for _st in ['<|text_chunk_end|>', '<|object_ref_start|>', '<|object_ref_end|>',
                        '<|box_start|>', '<|speech_start|>', '<|speech_end|>', '<|speech_pad|>']:
                chunk_text = chunk_text.replace(_st, '')
            yield chunk_idx, total_chunks, chunk_text

    @torch.no_grad()
    def init_streaming_state(self, tokenizer, context_info: str = None):
        """Streaming model: build the initial KV cache and cached embeddings."""
        device = next(self.parameters()).device
        embed_tokens = self.get_input_embeddings()

        keys_str = "speaker, content"
        if context_info:
            prompt_text = (
                "You are a helpful assistant that transcribes audio input into text output. "
                f"Please transcribe the following audios streamingly with these keys: {keys_str} "
                f"and extra info: {context_info}\n"
            )
        else:
            prompt_text = (
                "You are a helpful assistant that transcribes audio input into text output. "
                f"Please transcribe the following audios streamingly with these keys: {keys_str}\n"
            )

        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        prompt_embeds = embed_tokens(prompt_tensor)

        outputs = self(
            inputs_embeds=prompt_embeds,
            use_cache=True,
            return_dict=True,
        )

        return {
            "past_key_values": outputs.past_key_values,
            "sp_start_embed": embed_tokens(torch.tensor([[tokenizer.speech_start_id]], device=device)),
            "sp_end_embed": embed_tokens(torch.tensor([[tokenizer.speech_end_id]], device=device)),
            "text_chunk_end_id": tokenizer.text_chunk_end_id,
            "eos_id": tokenizer.eos_token_id,
            "embed_tokens": embed_tokens,
        }

    @torch.no_grad()
    def streaming_generate_step(
        self,
        audio_features: torch.FloatTensor,
        streaming_state: dict,
        tokenizer,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        on_first_token: Optional[Callable[[int], None]] = None,
    ) -> Tuple[str, dict]:
        """Streaming model: advance one audio chunk against the running KV cache."""
        device = next(self.parameters()).device
        past_key_values = streaming_state["past_key_values"]
        sp_start_embed = streaming_state["sp_start_embed"]
        sp_end_embed = streaming_state["sp_end_embed"]
        text_chunk_end_id = streaming_state["text_chunk_end_id"]
        eos_id = streaming_state["eos_id"]
        embed_tokens = streaming_state["embed_tokens"]

        audio_embeds = torch.cat([sp_start_embed, audio_features, sp_end_embed], dim=1)

        outputs = self(
            inputs_embeds=audio_embeds,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        next_logits = outputs.logits[:, -1:, :]

        chunk_tokens = []
        for _ in range(max_new_tokens):
            if temperature <= 0:
                next_token_id = torch.argmax(next_logits[:, -1, :], dim=-1).item()
            else:
                probs = F.softmax(next_logits[:, -1, :] / temperature, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).squeeze(-1).item()

            if next_token_id == text_chunk_end_id or next_token_id == eos_id:
                break

            chunk_tokens.append(next_token_id)
            if len(chunk_tokens) == 1 and on_first_token is not None:
                on_first_token(next_token_id)

            next_embed = embed_tokens(torch.tensor([[next_token_id]], device=device))
            outputs = self(
                inputs_embeds=next_embed,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            next_logits = outputs.logits

        tce_embed = embed_tokens(torch.tensor([[text_chunk_end_id]], device=device))
        outputs = self(
            inputs_embeds=tce_embed,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values

        streaming_state["past_key_values"] = past_key_values

        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        for _st in ['<|text_chunk_end|>', '<|object_ref_start|>', '<|object_ref_end|>',
                    '<|box_start|>', '<|speech_start|>', '<|speech_end|>', '<|speech_pad|>']:
            chunk_text = chunk_text.replace(_st, '')

        return chunk_text, streaming_state

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        speech_tensors=None,
        speech_masks=None,
        speech_semantic_tensors=None,
        acoustic_input_mask=None,
        **kwargs,
    ):
        """
        Prepare inputs for generation step. This method is called by generate() 
        for each token generation step.
        
        Following Qwen2-VL's approach: speech inputs are only forwarded on the first pass
        (when cache_position[0] == 0), and are excluded in subsequent generation steps.
        """
        # If we have past key values, we only need to process the new tokens
        if past_key_values is not None:
            if isinstance(past_key_values, tuple):
                past_length = past_key_values[0][0].shape[2]
            else:
                past_length = past_key_values.get_seq_length()
            
            # Keep only the new tokens
            if input_ids is not None and input_ids.shape[1] > past_length:
                input_ids = input_ids[:, past_length:]

        # Prepare position ids
        if position_ids is None and attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None and input_ids is not None:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        # Prepare cache position
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, 
                past_seen_tokens + (input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]), 
                device=input_ids.device if input_ids is not None else inputs_embeds.device
            )

        # Prepare model inputs
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        
        # Following Qwen2-VL pattern: only include speech inputs on the first forward pass
        # (when cache_position[0] == 0), exclude them in subsequent generation steps
        if cache_position is not None and len(cache_position) > 0 and cache_position[0] == 0:
            # First forward pass - include speech inputs if provided
            model_inputs.update({
                "speech_tensors": speech_tensors,
                "speech_masks": speech_masks,
                "speech_semantic_tensors": speech_semantic_tensors,
                "acoustic_input_mask": acoustic_input_mask,
            })
        else:
            # Subsequent generation steps - exclude speech inputs
            model_inputs.update({
                "speech_tensors": None,
                "speech_masks": None, 
                "speech_semantic_tensors": None,
                "acoustic_input_mask": None,
            })
        
        # Include any remaining kwargs that might be needed
        model_inputs.update(kwargs)
        
        return model_inputs

AutoModel.register(VibeVoiceASRConfig, VibeVoiceASRModel)
AutoModelForCausalLM.register(VibeVoiceASRConfig, VibeVoiceASRForConditionalGeneration)

__all__ = [
    "VibeVoiceASRPreTrainedModel",
    "VibeVoiceASRModel",
    "VibeVoiceASRForConditionalGeneration",
]