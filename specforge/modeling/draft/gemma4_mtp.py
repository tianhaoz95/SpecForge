from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    from transformers import Gemma4AssistantConfig
    from transformers.models.gemma4_assistant.modeling_gemma4_assistant import (
        Gemma4AssistantForCausalLM,
    )

    _GEMMA4_ASSISTANT_AVAILABLE = True
except ImportError:
    _GEMMA4_ASSISTANT_AVAILABLE = False

from .base import Eagle3DraftModel


class Gemma4MTPDraftModel(Eagle3DraftModel):
    """
    EAGLE-3-compatible wrapper around Gemma4AssistantForCausalLM with a standalone lm_head.

    Architecture:
      - embed_tokens: frozen, loaded from target (backbone_H size)
      - 4 Gemma4 decoder layers, all num_kv_shared_layers → cross-attention to shared_kv_states
      - pre_projection: Linear(2 * backbone_H → draft_H)  [inside HF model]
      - post_projection: Linear(draft_H → backbone_H)     [inside HF model]
      - lm_head: standalone nn.Linear(draft_H → vocab_size) with tie_word_embeddings=False
    """

    config_class = Gemma4AssistantConfig if _GEMMA4_ASSISTANT_AVAILABLE else None

    def __init__(self, config: "Gemma4AssistantConfig", **kwargs) -> None:
        if not _GEMMA4_ASSISTANT_AVAILABLE:
            raise ImportError(
                "Gemma4MTPDraftModel requires transformers>=5.8.0 with Gemma4 assistant. "
                "Install: pip install 'transformers>=5.8.0'"
            )
        super().__init__(config)
        self.config = config

        text_config = config.get_text_config()
        self.vocab_size = text_config.vocab_size
        self.draft_vocab_size = text_config.vocab_size

        backbone_hidden_size = config.backbone_hidden_size
        draft_hidden_size = text_config.hidden_size

        # Wrap the full Gemma4AssistantForCausalLM (which has pre_projection + backbone + post_projection + lm_head)
        self._assistant_model = Gemma4AssistantForCausalLM(config)

        # embed_tokens for base class load_embedding() — backbone_H size, frozen from target
        self.embed_tokens = nn.Embedding(self.vocab_size, backbone_hidden_size)

        # Identity vocab mapping buffers (no draft vocab compression)
        t2d = torch.ones(self.vocab_size, dtype=torch.bool)
        d2t = torch.arange(self.vocab_size, dtype=torch.int64)
        self.register_buffer("t2d", t2d)
        self.register_buffer("d2t", d2t)
        self.vocab_mapping_loaded = True

        self.backbone_hidden_size = backbone_hidden_size
        self.draft_hidden_size = draft_hidden_size

        self.post_init()

    # ── Eagle3DraftModel interface ──────────────────────────────────────

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.size(-1) == self.backbone_hidden_size, (
            f"Gemma4MTPDraftModel.project_hidden_states(): expected {self.backbone_hidden_size}, "
            f"got {hidden_states.size(-1)}. Use --target-model-backend gemma4_mtp."
        )
        return hidden_states

    def backbone(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_hidden,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        past_key_values=None,
        use_cache: bool = False,
        shared_kv_states: Optional[dict] = None,
        **kwargs,
    ):
        if shared_kv_states is None:
            raise ValueError(
                "Gemma4MTPDraftModel.backbone() requires shared_kv_states. "
                "Use --target-model-backend gemma4_mtp."
            )

        # Concatenate token embedding and target hidden state
        # (B, T, backbone_H) + (B, T, backbone_H) → (B, T, 2*backbone_H)
        inputs_embeds = torch.cat([input_embeds, hidden_states], dim=-1)

        out = self._assistant_model(
            inputs_embeds=inputs_embeds,
            attention_mask=None,
            position_ids=position_ids,
            shared_kv_states=shared_kv_states,
            use_cache=False,
        )

        # out.last_hidden_state is post_projection output: (B, T, backbone_H)
        projected_hidden = out.last_hidden_state
        logits = out.logits

        assert projected_hidden.shape[-1] == self.backbone_hidden_size, (
            f"post_projection output dim {projected_hidden.shape[-1]} != backbone_hidden_size {self.backbone_hidden_size}"
        )

        return projected_hidden, logits

    def compute_logits(self, backbone_output) -> torch.Tensor:
        if isinstance(backbone_output, tuple):
            return backbone_output[1]
        raise TypeError(
            "compute_logits received non-tuple; Gemma4MTPDraftModel.backbone() returns (projected_hidden, logits)."
        )

    def freeze_embedding(self) -> None:
        self.embed_tokens.weight.requires_grad_(False)
