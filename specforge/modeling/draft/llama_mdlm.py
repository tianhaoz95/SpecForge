"""
Llama MDLM draft model implementation.

This module implements a Llama-based MDLM draft model for diffusion training
and speculative decoding.
"""
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
)

from .mdlm_draft import MDLMDraftModel


class LlamaMDLMDraftModel(MDLMDraftModel):
    """Llama-based MDLM draft model.

    This model implements a simplified Llama architecture for MDLM training
    with bidirectional attention support.
    """

    def __init__(self, config: LlamaConfig, vocab_mapping=None):
        super().__init__(config, vocab_mapping)

        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = getattr(config, 'num_hidden_layers', 1)
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
        self.intermediate_size = config.intermediate_size
        self.max_position_embeddings = config.max_position_embeddings
        self.rms_norm_eps = config.rms_norm_eps
        self.rope_theta = getattr(config, 'rope_theta', 10000.0)

        # Model components
        self.embed_tokens = nn.Embedding(config.vocab_size, self.hidden_size, config.pad_token_id)

        # Decoder layers
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, layer_idx=i)
            for i in range(self.num_hidden_layers)
        ])

        # Final layer norm
        self.norm = LlamaRMSNorm(self.hidden_size, eps=self.rms_norm_eps)

        # Language modeling head
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

        # RoPE embeddings - simplified for MDLM
        # Note: RoPE will be handled by the decoder layers

        # Vocab mapping buffers for draft-target compatibility
        self.register_buffer("t2d", torch.arange(config.vocab_size), persistent=False)
        self.register_buffer("d2t", torch.arange(config.vocab_size), persistent=False)
        self.vocab_mapping_loaded = False

        # Initialize weights
        self.post_init()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed the input ids."""
        return self.embed_tokens(input_ids)

    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project concatenated hidden states.

        For MDLM, we don't use concatenated hidden states like Eagle3,
        so this is a no-op that returns the input.
        """
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute logits from hidden states."""
        return self.lm_head(hidden_states)

    def backbone(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """The backbone of the MDLM draft model.

        For MDLM, we ignore the Eagle3-specific hidden_states and cache_hidden
        and just process input_embeds through the transformer layers.
        """
        # Use input_embeds as the main input
        hidden_states = input_embeds

        # Process through decoder layers
        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]

        # Apply final layer norm
        hidden_states = self.norm(hidden_states)

        return hidden_states

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass for MDLM model.

        This is a simplified forward pass that supports bidirectional attention
        for diffusion training.
        """
        batch_size, seq_length = input_ids.shape if input_ids is not None else inputs_embeds.shape[:2]

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Generate position IDs if not provided
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                0, seq_length, dtype=torch.long, device=device
            ).unsqueeze(0).expand(batch_size, -1)

        # Create attention mask for bidirectional attention
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length),
                dtype=torch.bool,
                device=inputs_embeds.device
            )

        # Convert attention mask to 4D for transformer layers
        # For bidirectional attention, we don't use causal masking
        if attention_mask.dim() == 2:
            # Expand to [batch_size, 1, seq_length, seq_length]
            expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            expanded_mask = expanded_mask.expand(batch_size, 1, seq_length, seq_length)
            # Convert to float and apply masking
            attention_mask = expanded_mask.to(dtype=inputs_embeds.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(inputs_embeds.dtype).min

        # Process through transformer
        hidden_states = inputs_embeds

        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]

        # Apply final layer norm
        hidden_states = self.norm(hidden_states)

        # Compute logits
        logits = self.lm_head(hidden_states)

        # Create simple output object
        class ModelOutput:
            def __init__(self, logits):
                self.logits = logits

        return ModelOutput(logits)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs
    ):
        """Prepare inputs for generation."""
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # Create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        # If `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def _reorder_cache(self, past_key_values, beam_idx):
        """Reorder cache for beam search."""
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past