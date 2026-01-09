"""
Qwen3 MDLM draft model implementation.

This module implements a Qwen3-based MDLM draft model for diffusion training
and speculative decoding.
"""
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3Config
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Model,
    Qwen3RMSNorm,
)
from transformers import AutoConfig, AutoModelForCausalLM

from .mdlm_draft import MDLMDraftModel


class Qwen3MDLMDraftModel(MDLMDraftModel):
    """Qwen3-based MDLM draft model.

    This model implements a simplified Qwen3 architecture for MDLM training
    with bidirectional attention support.
    """

    config_class = Qwen3Config

    def __init__(self, config: Qwen3Config, vocab_mapping=None):
        super().__init__(config, vocab_mapping)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Use the full Qwen3 model to ensure proper initialization
        self.model = Qwen3Model(config)

        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights
        self.post_init()

    @property
    def embed_tokens(self):
        """Property to access embeddings for compatibility with base class methods."""
        return self.model.embed_tokens

    @embed_tokens.setter
    def embed_tokens(self, value):
        """Setter for embeddings."""
        self.model.embed_tokens = value

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass for MDLM model using full Qwen3Model."""

        # For MDLM, use the attention mask as-is (like Eagle3)
        # The bidirectional attention will be handled by the model internally

        # Use the full Qwen3Model which handles all the complex initialization
        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = model_outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        return type('ModelOutput', (), {
            'logits': logits,
            'hidden_states': hidden_states,
        })()

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """Prepare inputs for generation (used in iterative unmasking)."""
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
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
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

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """Reorder cache for beam search."""
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


# Model registration is handled by AutoMDLMDraftModel factory in auto.py