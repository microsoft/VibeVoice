from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from tqdm import tqdm
import torch
import torch.nn as nn

from transformers.models.auto import AutoModel, AutoModelForCausalLM

from transformers.generation import GenerationMixin, GenerationConfig, LogitsProcessor, LogitsProcessorList, StoppingCriteriaList
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers import modeling_utils
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.utils import logging

from .modular_vibevoice_tokenizer import VibeVoiceTokenizerStreamingCache
from .modular_vibevoice_diffusion_head import VibeVoiceDiffusionHead
from vibevoice.schedule.dpm_solver import DPMSolverMultistepScheduler
from .configuration_vibevoice_streaming import VibeVoiceStreamingConfig
from .modular_vibevoice_text_tokenizer import VibeVoiceTextTokenizer, VibeVoiceTextTokenizerFast
from .modeling_vibevoice_streaming import VibeVoiceStreamingPreTrainedModel, VibeVoiceStreamingModel, BinaryClassifier
from .streamer import AudioStreamer, AsyncAudioStreamer

logger = logging.get_logger(__name__)

if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]

TTS_TEXT_WINDOW_SIZE = 5
TTS_SPEECH_WINDOW_SIZE = 6


def _update_model_kwargs_for_generation(
    outputs: ModelOutput,
    model_kwargs: Dict[str, Any],
    num_new_tokens: int = 1,
) -> Dict[str, Any]:
    """
    Update model_kwargs after adding new tokens.

    Mainly for the case num_new_tokens > 1 (e.g. a whole text window):
      - past_key_values: take from current outputs
      - attention_mask: append num_new_tokens ones
      - cache_position: advance by creating a range for all new positions
    """

    # update past_key_values keeping its naming used in model code
    model_kwargs["past_key_values"] = getattr(outputs, "past_key_values")

    attention_mask = model_kwargs["attention_mask"]
    model_kwargs["attention_mask"] = torch.cat(
        [attention_mask, attention_mask.new_ones((attention_mask.shape[0], num_new_tokens))], dim=-1
    )

    model_kwargs["cache_position"] = torch.arange(model_kwargs["cache_position"][-1] + 1, model_kwargs["cache_position"][-1] + num_new_tokens + 1).to(model_kwargs["cache_position"].device)
    
    return model_kwargs


@dataclass
class VibeVoiceCausalLMOutputWithPast(BaseModelOutputWithPast):
    logits: Optional[torch.FloatTensor] = None


@dataclass
class VibeVoiceGenerationOutput(ModelOutput):
    """
    Output type for VibeVoice generation.
    
    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. 
        speech_outputs (`List[torch.FloatTensor]`, *optional*):
            List of generated speech waveforms or latents for each speech segment.
    """
    sequences: torch.LongTensor = None
    speech_outputs: Optional[List[torch.FloatTensor]] = None
    reach_max_step_sample: Optional[torch.BoolTensor] = None


class VibeVoiceStreamingForConditionalGenerationInference(VibeVoiceStreamingPreTrainedModel, GenerationMixin):

    def __init__(self, config):
        super().__init__(config)
        
        # Initialize the base model
        self.model = VibeVoiceStreamingModel(config)

        # TTS generation EOS classifier
        self.tts_eos_classifier = BinaryClassifier(config.decoder_config.hidden_size)
        
        # inference configuration
        self.ddpm_inference_steps = config.diffusion_head_config.ddpm_num_inference_steps

        # Initialize weights and apply final processing
        self.post_init()

    @property
    def noise_scheduler(self):
        return self.model.noise_scheduler

    @property
    def prediction_head(self):
        return self.model.prediction_head
    
    @property
    def speech_scaling_factor(self):
        return self.model.speech_scaling_factor

    @property
    def speech_bias_factor(self):
        return self.model.speech_bias_factor

    @property
    def acoustic_tokenizer(self):
        return self.model.acoustic_tokenizer
    
    @property
    def acoustic_connector(self):
        return self.model.acoustic_connector
        
    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.
        """
        # Tie lm_head.weight to language_model.embed_tokens.weight
        if not getattr(self.config, 'tie_word_embeddings', False):
            return
         
        if hasattr(self, 'lm_head') and hasattr(self.model.language_model, 'embed_tokens'):
            self.lm_head.weight = self.model.language_model.embed_tokens.weight
        
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)
    
    def get_output_embeddings(self):
        """
        This model does not define an `lm_head` (vocabulary projection).
        """
        return None
    
    def set_output_embeddings(self, new_embeddings):
        """
        No-op because there is no `lm_head`. Provided only to satisfy optional API calls.
        To enable, first create `self.lm_head` then allow assignment.
        """
        raise RuntimeError("Output embeddings (lm_head) are not defined for this model. "
                           "Create one before calling set_output_embeddings if needed.")
    
    def set_speech_tokenizers(self, acoustic_tokenizer=None):
        """Set the speech tokenizers used for encoding and decoding speech."""
        self.model.set_speech_tokenizers(acoustic_tokenizer)
    
    def set_ddpm_inference_steps(self, num_steps=None):
        self.ddpm_inference_steps = num_steps or self.config.diffusion_head_config.ddpm_num_inference_steps

    # @can_return_tuple
    def forward_lm(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Single pass of the base text LM.

        - Builds embeddings if `inputs_embeds` not provided.
        - Uses (and returns) `past_key_values` when `use_cache=True`.
        - No loss / no lm_head / no speech logic.

        Args:
            input_ids: (B, S) token ids.
            attention_mask: (B, S) mask.
            past_key_values: cache from previous steps.
            cache_position: positions for cached tokens.
            labels: unsupported (will raise).

        Returns:
            BaseModelOutputWithPast with `last_hidden_state` and `past_key_values`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state
                
        if labels is not None:
            raise NotImplementedError("Loss computation is not implemented in this version.")

        return BaseModelOutputWithPast(
            past_key_values=outputs.past_key_values,
            last_hidden_state=hidden_states,
            attentions=outputs.attentions,
        )

    # @can_return_tuple
    def forward_tts_lm(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        lm_last_hidden_state: Optional[torch.FloatTensor] = None,
        tts_text_masks: Optional[torch.BoolTensor] = None,
        **kwargs,
    ) -> Union[Tuple, VibeVoiceCausalLMOutputWithPast]:
        """
        Single pass of the TTS LM.

        - Overwrites tail embeddings with `lm_last_hidden_state`.
        - Adds type embedding via `tts_text_masks` (1=text, 0=speech).
        - Predicts EOS from last hidden state (binary classifier).
        - No loss / no full acoustic decoding here.

        Args:
            input_ids: (B, S) token ids.
            attention_mask: (B, S) mask.
            lm_last_hidden_state: (B, K, H) hidden states to splice into the tail.
            tts_text_masks: (B, 1) mask marking current position as text(1)/speech(0).
            past_key_values: cache from previous TTS steps.
            cache_position: positions for cached tokens.
            labels: unsupported (will raise).

        Returns:
            VibeVoiceCausalLMOutputWithPast with `logits` (EOS), `last_hidden_state`, `past_key_values`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get embeddings
        if inputs_embeds is None:
            # Will be replaced with lm_last_hidden_state
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
        
        # Replace the last part of inputs_embeds with lm_last_hidden_state
        start_idx = inputs_embeds.shape[1] - lm_last_hidden_state.shape[1]
        inputs_embeds[:, start_idx:, :] = lm_last_hidden_state
        
        # Adds type embedding via `tts_text_masks`.
        inputs_embeds = inputs_embeds + self.model.tts_input_types(tts_text_masks.long())

        outputs = self.model.tts_language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state
        logits = self.tts_eos_classifier(hidden_states[:, -1, :])
                
        if labels is not None:
            raise NotImplementedError("Loss computation is not implemented in this version.")

        return VibeVoiceCausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            last_hidden_state=hidden_states,
            attentions=outputs.attentions,
        )

    def forward(self, *args, **kwargs):
        """
        Unified forward is intentionally disabled.

        Reasons:
          1. The inference pipeline is staged: base text LM, then TTS LM, plus streaming & diffusion handled in `generate`.
          2. A monolithic call would hide required sequencing (prefill, window stepping, speech diffusion sampling).

        Use instead:
          - self.forward_lm(...)       for a base text LM step (prefill or incremental).
          - self.forward_tts_lm(...)   for a single TTS LM step (needs LM hidden states).
          - self.generate(...)         for full streaming (text + speech + diffusion + audio assembly).

        Raises:
            RuntimeError: Always (by design).
        """
        raise RuntimeError(
            "Unified forward is disabled. Use `forward_lm`, `forward_tts_lm`, or `generate` instead."
        )

    def _build_generate_config_model_kwargs(self, generation_config, inputs, tokenizer, return_processors=False, **kwargs):
        if generation_config is None:
            generation_config = GenerationConfig(
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id = tokenizer.pad_token_id
            )
        else:
            generation_config = GenerationConfig(
                **generation_config,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id = tokenizer.pad_token_id
            )

        generation_config, model_kwargs = self._prepare_generation_config(
            generation_config, 
            True, 
            speech_start_id=tokenizer.speech_start_id, 
            speech_end_id=tokenizer.speech_end_id, 
            speech_diffusion_id=tokenizer.speech_diffusion_id, 
            **kwargs
        )
        generation_config.speech_start_id = tokenizer.speech_start_id
        generation_config.speech_end_id = tokenizer.speech_end_id
        generation_config.speech_diffusion_id = tokenizer.speech_diffusion_id

        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(inputs, generation_config.bos_token_id, model_kwargs)
        batch_size = inputs_tensor.shape[0]
        device = self.device
        
        self._prepare_special_tokens(generation_config, True, device=device)
        generation_config.use_cache = True
        model_kwargs["use_cache"] = generation_config.use_cache
        input_ids = inputs_tensor.to(self.device)

        input_ids_length = input_ids.shape[1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        max_cache_length = generation_config.max_length - 1
        self._prepare_cache_for_generation(generation_config, model_kwargs, None, batch_size, max_cache_length, device)
        model_kwargs['cache_position'] = torch.arange(input_ids_length, device=device, dtype=torch.long)
        for k, v in model_kwargs.items():
            if isinstance(v, torch.Tensor):
                model_kwargs[k] = v.to(device=device)
        
        if return_processors:
            logits_processor = self._get_logits_processor(
                generation_config=generation_config,
                input_ids_seq_length=input_ids_length,
                encoder_input_ids=inputs_tensor,
                prefix_allowed_tokens_fn=None,
                logits_processor=LogitsProcessorList(),
                device=inputs_tensor.device,
                model_kwargs=model_kwargs,
            )

            stopping_criteria = self._get_stopping_criteria(generation_config=generation_config, stopping_criteria=StoppingCriteriaList())
        
            return generation_config, model_kwargs, input_ids, logits_processor, stopping_criteria
        else:
            return generation_config, model_kwargs, input_ids

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        audio_streamer: Optional[Union[AudioStreamer, AsyncAudioStreamer]] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        speech_tensors: Optional[torch.FloatTensor] = None,
        speech_masks: Optional[torch.BoolTensor] = None,
        speech_input_mask: Optional[torch.BoolTensor] = None,
        tts_text_ids: Optional[torch.LongTensor] = None,
        return_speech: bool = True,
        cfg_scale: float = 1.0,
        stop_check_fn: Optional[Callable[[], bool]] = None,
        **kwargs,
    ) -> Union[torch.LongTensor, VibeVoiceGenerationOutput]:
        """
        Text is fed in small windows (dynamic slicing of `tts_text_ids`), which enables streaming text input: you don’t need the full text upfront. After each text window, a loop samples several speech latents (diffusion). The interleaved text encoding + speech generation enables streaming text input and realtime speech output.
        The function only supports batch size = 1 currently.

        - Windowed text prefill → incremental LM + TTS LM updates.
        - Interleave speech token diffusion sampling (`sample_speech_tokens`).
        - Stops on EOS (binary classifier) or max length / external `stop_check_fn`.
        - Returns final token `sequences` and (optionally) concatenated speech audio.

        Args (selected):
            tts_text_ids: Full text tokens to stream in windows.
            audio_streamer: If provided, emits audio chunks during generation.
            cfg_scale: Classifier-free guidance scale for speech diffusion.
            return_speech: If False, skips audio decode concatenation.
            stop_check_fn: External early-stop hook (returns True to halt).

        Returns:
            VibeVoiceGenerationOutput with:
              - sequences: final token ids
              - speech_outputs: list of concatenated audio tensors (or None)
              - reach_max_step_sample: flags for samples stopped by max length
        """
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        tokenizer = kwargs.pop("tokenizer", None)
        neg_text_input_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        
        tts_lm_input_ids = kwargs.pop("tts_lm_input_ids", None)
        tts_lm_attention_mask = kwargs.pop("tts_lm_attention_mask", None)
        # all_prefilled_outputs: cached prefilled prompt outputs for lm, tts_lm, neg_lm, neg_tts_lm
        all_prefilled_outputs = kwargs.pop("all_prefilled_outputs", None)
        tts_text_ids = tts_text_ids.to(self.device)

        if kwargs.get('max_new_tokens', None) is None:
            kwargs['max_new_tokens'] = self.config.decoder_config.max_position_embeddings - tts_lm_input_ids.shape[-1]

        generation_config, model_kwargs, input_ids, logits_processor, stopping_criteria = self._build_generate_config_model_kwargs(
            generation_config, inputs, tokenizer, return_processors=True, **kwargs
        )
        
        negative_kwargs = {
            'input_ids': torch.full((kwargs['input_ids'].shape[0], 1), neg_text_input_id, dtype=torch.long, device=kwargs['input_ids'].device),
            'attention_mask':  torch.ones((kwargs['input_ids'].shape[0], 1), dtype=torch.long, device=kwargs['input_ids'].device),
            'max_new_tokens': kwargs.get('max_new_tokens', 100) 
        }
        negative_generation_config, negative_model_kwargs, negative_input_ids = self._build_generate_config_model_kwargs(
            None, None, tokenizer, return_processors=False, **negative_kwargs
        )

        tts_lm_kwargs = {
            'input_ids': tts_lm_input_ids,
            'attention_mask': tts_lm_attention_mask,
            'max_new_tokens': kwargs.get('max_new_tokens', 100) 
        }
        tts_lm_generation_config, tts_lm_model_kwargs, tts_lm_input_ids = self._build_generate_config_model_kwargs(
            None, None, tokenizer, return_processors=False, **tts_lm_kwargs
        )

        tts_lm_negative_kwargs = {
            'input_ids': torch.full((kwargs['input_ids'].shape[0], 1), neg_text_input_id, dtype=torch.long, device=kwargs['input_ids'].device),
            'attention_mask':  torch.ones((kwargs['input_ids'].shape[0], 1), dtype=torch.long, device=kwargs['input_ids'].device),
            'max_new_tokens': kwargs.get('max_new_tokens', 100) 
        }
        tts_lm_negative_generation_config, tts_lm_negative_model_kwargs, tts_lm_negative_input_ids = self._build_generate_config_model_kwargs(
            None, None, tokenizer, return_processors=False, **tts_lm_negative_kwargs
        )

        acoustic_cache = VibeVoiceTokenizerStreamingCache()
        batch_size = input_ids.shape[0]
        assert batch_size == 1, "Currently only supports batch size == 1"
        device = input_ids.device
        finished_tags = torch.zeros(batch_size, dtype=torch.bool, device=device)
        verbose = kwargs.get("verbose", False)

        # Initialize audio chunks storage for each sample
        audio_chunks = [[] for _ in range(batch_size)]
        tts_text_window_index = 0
        reach_max_step_sample = torch.zeros(batch_size, dtype=torch.bool, device=device)
        first_text_window_size = TTS_TEXT_WINDOW_SIZE if tts_text_ids.shape[1] >= TTS_TEXT_WINDOW_SIZE else tts_text_ids.shape[1]

        outputs = all_prefilled_outputs["lm"]
        tts_lm_outputs = all_prefilled_outputs["tts_lm"]
        negative_outputs = all_prefilled_outputs["neg_lm"]
        tts_lm_negative_outputs = all_prefilled_outputs["neg_tts_lm"]

        model_kwargs = _update_model_kwargs_for_generation(
            outputs, model_kwargs, num_new_tokens=first_text_window_size,
        )
        tts_lm_model_kwargs = _update_model_kwargs_for_generation(
            tts_lm_outputs, tts_lm_model_kwargs, num_new_tokens=first_text_window_size,
        )
        negative_model_kwargs = self._update_model_kwargs_for_generation(
            negative_outputs, negative_model_kwargs, is_encoder_decoder=False,
        )
        tts_lm_negative_model_kwargs = self._update_model_kwargs_for_generation(
            tts_lm_negative_outputs, tts_lm_negative_model_kwargs, is_encoder_decoder=False,
        )

        step = tts_lm_input_ids.shape[1]
        total_generated_speech_tokens = 0
        total_prefilled_text_tokens = 0
        if kwargs.get("show_progress_bar", True):
            progress_bar = tqdm(
                total=tts_lm_generation_config.max_length,
                desc=f"Prefilled {step} tokens, current step ({step} / {tts_lm_generation_config.max_length})",
                initial=step,
                leave=False
            )
        else:
            progress_bar = None

        while True:
            # Check for external stop signal
            if stop_check_fn is not None and stop_check_fn():
                if verbose:
                    print(f"Generation stopped externally at step {step + 1}")
                # End the audio streamer if it exists
                if audio_streamer is not None:
                    audio_streamer.end()
                break
            
            # # Check if audio_streamer has been ended (stopped externally)
            # if audio_streamer is not None and hasattr(audio_streamer, 'finished_flags'):
            #     if any(audio_streamer.finished_flags):
            #         if verbose:
            #             print(f"Audio generation stopped externally at step {step + 1}")
            #         break
            
            if finished_tags.all():
                if hasattr(progress_bar, 'set_description'):
                    progress_bar.set_description("Generation complete")
                break

            cur_input_tts_text_ids = tts_text_ids[:, tts_text_window_index*TTS_TEXT_WINDOW_SIZE:(tts_text_window_index+1)*TTS_TEXT_WINDOW_SIZE]
            next_text_window_size = tts_text_ids[:, (tts_text_window_index+1)*TTS_TEXT_WINDOW_SIZE:(tts_text_window_index+2)*TTS_TEXT_WINDOW_SIZE].shape[1]
            tts_text_window_index += 1

            if cur_input_tts_text_ids.shape[1] > 0:
                input_ids = torch.cat([input_ids, cur_input_tts_text_ids], dim=-1)
                tts_lm_input_ids = torch.cat([tts_lm_input_ids, cur_input_tts_text_ids], dim=-1)

                if tts_lm_input_ids.shape[1] > tts_lm_generation_config.max_length:
                    if verbose:
                        print(f"Reached maximum generation length {generation_config.max_length}, stopped it.")
                    reached_samples = torch.arange(batch_size, device=device)[~finished_tags]
                    if reached_samples.numel() > 0:
                        reach_max_step_sample[reached_samples] = True
                    break
                
                step += cur_input_tts_text_ids.shape[1]
                total_prefilled_text_tokens += cur_input_tts_text_ids.shape[1]
                if progress_bar is not None:
                    progress_bar.update(cur_input_tts_text_ids.shape[1])
                    progress_bar.set_description(f"Prefilled {total_prefilled_text_tokens} text tokens, generated {total_generated_speech_tokens} speech tokens, current step ({step} / {tts_lm_generation_config.max_length})")

                model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
                # Forward pass through the model
                outputs = self.forward_lm(
                    **model_inputs, return_dict=True, output_attentions=False, output_hidden_states=False,
                )
                model_kwargs = _update_model_kwargs_for_generation(
                    outputs, model_kwargs, num_new_tokens=next_text_window_size,
                )

                tts_lm_model_inputs = self.prepare_inputs_for_generation(tts_lm_input_ids, **tts_lm_model_kwargs)
                tts_lm_additional_inputs = {
                    "tts_text_masks": torch.ones_like(tts_lm_input_ids[:, -1:]),
                    "lm_last_hidden_state": outputs.last_hidden_state,
                }
                # Forward pass through the model
                tts_lm_outputs = self.forward_tts_lm(
                    **tts_lm_model_inputs, **tts_lm_additional_inputs, return_dict=True, output_attentions=False, output_hidden_states=False,
                )
                tts_lm_model_kwargs = self._update_model_kwargs_for_generation(
                    tts_lm_outputs, tts_lm_model_kwargs, is_encoder_decoder=False,
                )

            diffusion_indices = torch.LongTensor([0])
            for cur_speech_index in range(TTS_SPEECH_WINDOW_SIZE):
                positive_condition = tts_lm_outputs.last_hidden_state[diffusion_indices, -1, :]
                negative_condition = tts_lm_negative_outputs.last_hidden_state[diffusion_indices, -1, :]
                
                speech_latent = self.sample_speech_tokens(
                    positive_condition,
                    negative_condition,
                    cfg_scale=cfg_scale,
                ).unsqueeze(1)
                                
                # Decode acoustic latent to audio using acoustic streaming cache
                scaled_latent = speech_latent / self.model.speech_scaling_factor.to(speech_latent.device) - self.model.speech_bias_factor.to(speech_latent.device)
                audio_chunk = self.model.acoustic_tokenizer.decode(
                    scaled_latent.to(self.model.acoustic_tokenizer.device),
                    cache=acoustic_cache,  # Use acoustic-specific cache
                    sample_indices=diffusion_indices.to(self.model.acoustic_tokenizer.device),
                    use_cache=True,
                    debug=False
                )
                
                # Store audio chunks for each sample
                for i, sample_idx in enumerate(diffusion_indices):
                    idx = sample_idx.item()
                    # Only append audio chunk if the sample is not finished
                    if not finished_tags[idx]:
                        audio_chunks[idx].append(audio_chunk[i])

                 # Add streaming support here
                if audio_streamer is not None:
                    # Stream the audio chunks immediately
                    audio_streamer.put(audio_chunk, diffusion_indices)

                acoustic_embed = self.model.acoustic_connector(speech_latent)
                tts_lm_input_ids = torch.cat([tts_lm_input_ids, torch.ones_like(tts_lm_input_ids[:, -1:])], dim=-1)

                if tts_lm_input_ids.shape[1] > tts_lm_generation_config.max_length:
                    break
                
                step += 1
                total_generated_speech_tokens += 1
                if progress_bar is not None:
                    progress_bar.update(1)
                    progress_bar.set_description(f"Prefilled {total_prefilled_text_tokens} text tokens, generated {total_generated_speech_tokens} speech tokens, current step ({step} / {tts_lm_generation_config.max_length})")

                tts_lm_model_inputs = self.prepare_inputs_for_generation(tts_lm_input_ids, **tts_lm_model_kwargs)
                tts_lm_additional_inputs = {
                    "tts_text_masks": torch.zeros_like(tts_lm_input_ids[:, -1:]),
                    "lm_last_hidden_state": acoustic_embed,
                }
                # Forward pass through the model
                tts_lm_outputs = self.forward_tts_lm(
                    **tts_lm_model_inputs, **tts_lm_additional_inputs, return_dict=True, output_attentions=False, output_hidden_states=False,
                )
                if cur_speech_index == TTS_SPEECH_WINDOW_SIZE - 1 and next_text_window_size > 0:
                    tts_lm_model_kwargs = _update_model_kwargs_for_generation(
                        tts_lm_outputs, tts_lm_model_kwargs, num_new_tokens=next_text_window_size,
                    )
                else:
                    tts_lm_model_kwargs = self._update_model_kwargs_for_generation(
                        tts_lm_outputs, tts_lm_model_kwargs, is_encoder_decoder=False,
                    )

                tts_lm_negative_input_ids = torch.cat([tts_lm_negative_input_ids, torch.ones_like(tts_lm_input_ids[:, -1:])], dim=-1)
                tts_lm_negative_model_inputs = self.prepare_inputs_for_generation(tts_lm_negative_input_ids, **tts_lm_negative_model_kwargs)
                # Forward negative pass through the model
                tts_lm_negative_additional_inputs = {
                    "tts_text_masks": torch.zeros_like(tts_lm_negative_input_ids[:, -1:]),
                    "lm_last_hidden_state": acoustic_embed,
                }
                tts_lm_negative_outputs = self.forward_tts_lm(
                    **tts_lm_negative_model_inputs, **tts_lm_negative_additional_inputs, return_dict=True, output_attentions=False, output_hidden_states=False,
                )
                tts_lm_negative_model_kwargs = self._update_model_kwargs_for_generation(
                    tts_lm_negative_outputs, tts_lm_negative_model_kwargs, is_encoder_decoder=False,
                )

                tts_eos_logits = torch.sigmoid(self.tts_eos_classifier(tts_lm_outputs.last_hidden_state[diffusion_indices, -1, :]))
                if tts_eos_logits[0].item() > 0.5:
                    # If EOS token is predicted, we can stop generation for this sample
                    finished_tags[diffusion_indices] = True
                    if audio_streamer is not None:
                        audio_streamer.end(diffusion_indices)

            if tts_lm_input_ids.shape[1] > tts_lm_generation_config.max_length:
                if verbose:
                    print(f"Reached maximum generation length {tts_lm_generation_config.max_length}, stopped it.")
                reached_samples = torch.arange(batch_size, device=device)[~finished_tags]
                if reached_samples.numel() > 0:
                    reach_max_step_sample[reached_samples] = True
                break

        if audio_streamer is not None:
            audio_streamer.end()

        # Concatenate audio chunks for each sample
        final_audio_outputs = []
        for sample_chunks in audio_chunks:
            if sample_chunks:
                # Concatenate all chunks along the time dimension (assumed to be the last dimension)
                concatenated_audio = torch.cat(sample_chunks, dim=-1)
                final_audio_outputs.append(concatenated_audio)
            else:
                # If no audio was generated for this sample, append None
                final_audio_outputs.append(None)
        
        if reach_max_step_sample is not None and reach_max_step_sample.any():
            print(f"Reached maximum generation length {tts_lm_generation_config.max_length}, stopped it.")

        return VibeVoiceGenerationOutput(
            sequences=tts_lm_input_ids,
            speech_outputs=final_audio_outputs if return_speech else None,
            reach_max_step_sample=reach_max_step_sample,
        )

    @torch.no_grad()
    def sample_speech_tokens(self, condition, neg_condition, cfg_scale=3.0):
        self.model.noise_scheduler.set_timesteps(self.ddpm_inference_steps)
        condition = torch.cat([condition, neg_condition], dim=0).to(self.model.prediction_head.device)
        speech = torch.randn(condition.shape[0], self.config.acoustic_vae_dim).to(condition)
        for t in self.model.noise_scheduler.timesteps:
            half = speech[: len(speech) // 2]
            combined = torch.cat([half, half], dim=0)
            eps = self.model.prediction_head(combined, t.repeat(combined.shape[0]).to(combined), condition=condition)
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            speech = self.model.noise_scheduler.step(eps, t, speech).prev_sample
        return speech[: len(speech) // 2]
    

AutoModelForCausalLM.register(VibeVoiceStreamingConfig, VibeVoiceStreamingForConditionalGenerationInference)

__all__ = [
    "VibeVoiceStreamingForConditionalGenerationInference",
]
