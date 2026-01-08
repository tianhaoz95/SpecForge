"""
MDLM draft model implementation.

This module implements MDLM-based draft models for speculative decoding,
extending SpecForge's Eagle3DraftModel base class.
"""
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache

from .base import Eagle3DraftModel
from ...core.schedulers import BaseAlphaScheduler


class MDLMDraftModel(Eagle3DraftModel):
    """MDLM-based draft model for speculative decoding.

    This model extends the Eagle3DraftModel base class to support diffusion-based
    training and generation. It implements masked diffusion language modeling
    where tokens are iteratively unmasked during generation.
    """

    def __init__(self, config, vocab_mapping=None):
        super().__init__(config)

        # MDLM-specific configuration
        self.mask_token_id = getattr(config, 'mask_token_id', config.vocab_size)
        self.time_epsilon = getattr(config, 'time_epsilon', 1e-3)
        self.alpha_scheduler_name = getattr(config, 'alpha_scheduler', 'linear')

        # Initialize alpha scheduler for inference
        self._alpha_scheduler = None

        # Vocab mapping for draft-target model compatibility
        if vocab_mapping is not None:
            self.load_vocab_mapping(vocab_mapping)

    @property
    def alpha_scheduler(self) -> Optional[BaseAlphaScheduler]:
        """Get the alpha scheduler instance."""
        if self._alpha_scheduler is None and hasattr(self, 'alpha_scheduler_name'):
            from ...core.schedulers import make_alpha_scheduler
            self._alpha_scheduler = make_alpha_scheduler(self.alpha_scheduler_name)
        return self._alpha_scheduler

    @alpha_scheduler.setter
    def alpha_scheduler(self, scheduler: BaseAlphaScheduler):
        """Set the alpha scheduler instance."""
        self._alpha_scheduler = scheduler

    def forward_diffusion_step(
        self,
        input_ids: torch.Tensor,
        timestep: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        maskable_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Perform a single diffusion forward pass.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            timestep: Timestep for this batch [batch_size] or scalar
            attention_mask: Attention mask [batch_size, seq_len]
            maskable_mask: Mask indicating which tokens can be masked [batch_size, seq_len]

        Returns:
            Logits for masked positions [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape

        # Ensure timestep is a tensor
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor(timestep, device=input_ids.device)
        if timestep.dim() == 0:
            timestep = timestep.expand(batch_size)

        # Create default maskable mask if not provided
        if maskable_mask is None:
            maskable_mask = torch.ones_like(input_ids, dtype=torch.bool)

        # Apply masking based on timestep
        if self.alpha_scheduler is not None:
            alpha_vals = self.alpha_scheduler.alpha(timestep)  # [batch_size]
            p_mask = 1.0 - alpha_vals.unsqueeze(1).expand(batch_size, seq_len)  # [batch_size, seq_len]

            # Sample which positions to mask
            random_vals = torch.rand_like(p_mask, device=input_ids.device)
            masked_positions = (random_vals < p_mask) & maskable_mask

            # Apply masking
            noised_input_ids = torch.where(
                masked_positions,
                self.mask_token_id,
                input_ids
            )
        else:
            noised_input_ids = input_ids

        # Create default attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Forward pass through the model
        outputs = self.forward(
            input_ids=noised_input_ids,
            attention_mask=attention_mask,
        )

        return outputs.logits

    def sample_timestep(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample timesteps for training.

        Args:
            batch_size: Number of timesteps to sample
            device: Device to place tensors on

        Returns:
            Timesteps in [epsilon, 1) with shape (batch_size,)
        """
        eps = self.time_epsilon
        return eps + (1 - eps) * torch.rand(batch_size, device=device)

    def generate_iterative_unmasking(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 128,
        steps: int = 128,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        alpha_scheduler: Optional[BaseAlphaScheduler] = None,
    ) -> torch.Tensor:
        """Generate text using iterative unmasking.

        Args:
            prompt_ids: Prompt token IDs [1, prompt_len]
            max_new_tokens: Maximum number of new tokens to generate
            steps: Number of denoising steps
            temperature: Sampling temperature (0.0 for deterministic)
            top_k: Top-k sampling parameter
            alpha_scheduler: Alpha scheduler for generation (uses model's if None)

        Returns:
            Generated sequence [1, prompt_len + generated_len]
        """
        if alpha_scheduler is None:
            alpha_scheduler = self.alpha_scheduler
        if alpha_scheduler is None:
            raise ValueError("No alpha scheduler provided for generation")

        device = prompt_ids.device
        batch_size = prompt_ids.size(0)
        prompt_len = prompt_ids.size(1)
        total_len = prompt_len + max_new_tokens

        # Initialize canvas with prompt + mask tokens
        canvas = torch.cat([
            prompt_ids,
            torch.full((batch_size, max_new_tokens), self.mask_token_id, device=device)
        ], dim=1)

        # Create mask indicating which positions can be modified (not prompt)
        modifiable_mask = torch.cat([
            torch.zeros(batch_size, prompt_len, dtype=torch.bool, device=device),
            torch.ones(batch_size, max_new_tokens, dtype=torch.bool, device=device)
        ], dim=1)

        # Iterative unmasking
        for step in range(steps):
            # Current timestep (from 1 to epsilon)
            t = 1.0 - (step / steps) * (1.0 - self.time_epsilon)
            timestep = torch.full((batch_size,), t, device=device)

            # Forward pass
            with torch.no_grad():
                logits = self.forward_diffusion_step(
                    canvas,
                    timestep,
                    maskable_mask=modifiable_mask
                )  # [batch_size, total_len, vocab_size]

            # Apply temperature and top-k sampling
            if temperature > 0.0:
                logits = logits / temperature
                if top_k is not None:
                    # Apply top-k filtering
                    top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(-1, top_k_indices, top_k_logits)

                # Sample from distribution
                probs = F.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1)
                next_tokens = next_tokens.view(batch_size, total_len)
            else:
                # Deterministic (argmax)
                next_tokens = logits.argmax(dim=-1)

            # Determine how many tokens to unmask at this step
            if step == steps - 1:
                # Last step: unmask all remaining
                num_unmask = (canvas == self.mask_token_id).sum(dim=1)
            else:
                # Compute number to unmask based on schedule
                next_t = 1.0 - ((step + 1) / steps) * (1.0 - self.time_epsilon)
                current_mask_rate = 1.0 - alpha_scheduler.alpha(t)
                next_mask_rate = 1.0 - alpha_scheduler.alpha(next_t)

                total_masked = (canvas == self.mask_token_id).sum(dim=1).float()
                unmask_rate = (current_mask_rate - next_mask_rate) / current_mask_rate.clamp_min(1e-8)
                num_unmask = (total_masked * unmask_rate).round().long()

            # For each sequence in batch, unmask top-confidence positions
            for b in range(batch_size):
                if num_unmask[b] <= 0:
                    continue

                # Find masked positions
                masked_positions = (canvas[b] == self.mask_token_id) & modifiable_mask[b]
                masked_indices = torch.where(masked_positions)[0]

                if len(masked_indices) == 0:
                    continue

                # Compute confidence (max probability)
                masked_logits = logits[b, masked_indices]  # [num_masked, vocab_size]
                confidence = F.softmax(masked_logits, dim=-1).max(dim=-1)[0]  # [num_masked]

                # Select top-k positions by confidence
                k = min(num_unmask[b].item(), len(masked_indices))
                _, top_indices = torch.topk(confidence, k)
                positions_to_unmask = masked_indices[top_indices]

                # Commit predictions
                canvas[b, positions_to_unmask] = next_tokens[b, positions_to_unmask]

        return canvas

    # Abstract methods from Eagle3DraftModel - these need to be implemented by concrete model classes
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed the input ids. Must be implemented by concrete model."""
        raise NotImplementedError("Must be implemented by concrete model class")

    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project concatenated hidden states. Must be implemented by concrete model."""
        raise NotImplementedError("Must be implemented by concrete model class")

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute logits from hidden states. Must be implemented by concrete model."""
        raise NotImplementedError("Must be implemented by concrete model class")

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
        """The backbone of the draft model. Must be implemented by concrete model."""
        raise NotImplementedError("Must be implemented by concrete model class")