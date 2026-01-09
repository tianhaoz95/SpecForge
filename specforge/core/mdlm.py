"""
MDLM (Masked Diffusion Language Model) training infrastructure.

This module implements the core MDLM training logic adapted to SpecForge architecture,
porting from the dLLM framework while leveraging SpecForge's existing infrastructure.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from .schedulers import BaseAlphaScheduler


@dataclass
class MDLMConfig:
    """Configuration for MDLM training."""

    # Core MDLM parameters
    mask_token_id: int
    alpha_scheduler: str = "linear"  # "linear" or "cosine"
    time_epsilon: float = 1e-3  # Minimum timestep to avoid degenerate values
    loss_weight_type: str = "scheduler"  # "scheduler" or "uniform"
    loss_norm_type: str = "token"  # "batch", "sequence", or "token"
    right_shift_logits: bool = False

    # Training parameters
    max_length: int = 2048
    vocab_size: Optional[int] = None


class MDLMTrainer:
    """MDLM trainer for diffusion-based language model training.

    This trainer implements masked diffusion training by:
    1. Sampling timesteps for each batch
    2. Applying stochastic masking based on alpha scheduler
    3. Computing weighted cross-entropy loss
    4. Supporting various loss normalization strategies
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer,
        scheduler,
        alpha_scheduler: BaseAlphaScheduler,
        config: MDLMConfig,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.alpha_scheduler = alpha_scheduler
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Training state
        self.step = 0
        self.epoch = 0

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """Sample diffusion timesteps for training.

        Args:
            batch_size: Number of timesteps to sample

        Returns:
            Timesteps in [epsilon, 1) with shape (batch_size,)
        """
        eps = self.config.time_epsilon
        return eps + (1 - eps) * torch.rand(batch_size, device=self.device)

    def apply_stochastic_masking(
        self,
        input_ids: torch.Tensor,
        timesteps: torch.Tensor,
        maskable_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply stochastic masking based on timesteps.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            timesteps: Timesteps for each example [batch_size]
            maskable_mask: Mask indicating which tokens can be masked [batch_size, seq_len]

        Returns:
            Tuple of (noised_input_ids, masked_positions)
        """
        batch_size, seq_len = input_ids.shape

        # Compute masking probability from alpha scheduler
        # p_mask = 1 - α(t)
        alpha_vals = self.alpha_scheduler.alpha(timesteps)  # [batch_size]
        p_mask = 1.0 - alpha_vals.unsqueeze(1).expand(batch_size, seq_len)  # [batch_size, seq_len]

        # Sample which positions to mask
        random_vals = torch.rand_like(p_mask, device=self.device)
        masked_positions = (random_vals < p_mask) & maskable_mask  # [batch_size, seq_len]

        # Apply masking
        noised_input_ids = torch.where(
            masked_positions,
            self.config.mask_token_id,
            input_ids
        )

        return noised_input_ids, masked_positions

    def compute_loss_weights(self, timesteps: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Compute loss weights based on timesteps.

        Args:
            timesteps: Timesteps for each example [batch_size]
            seq_len: Sequence length

        Returns:
            Loss weights [batch_size, seq_len]
        """
        if self.config.loss_weight_type == "scheduler":
            # Use scheduler-based weighting: w(t) = -α'(t) / (1 - α(t))
            weights = self.alpha_scheduler.weight(timesteps)  # [batch_size]
            return weights.unsqueeze(1).expand(-1, seq_len)  # [batch_size, seq_len]
        else:
            # Uniform weighting
            batch_size = timesteps.size(0)
            return torch.ones(batch_size, seq_len, device=self.device)

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute MDLM loss for a batch.

        Args:
            batch: Batch dictionary with keys:
                - input_ids: [batch_size, seq_len]
                - attention_mask: [batch_size, seq_len]
                - loss_mask: [batch_size, seq_len] (1 for positions to include in loss, 0 to ignore)

        Returns:
            Tuple of (loss, metrics_dict)
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        loss_mask = batch["loss_mask"]

        batch_size, seq_len = input_ids.shape

        # 1. Sample timesteps
        timesteps = self.sample_timesteps(batch_size)  # [batch_size]

        # 2. Create maskable mask (where loss_mask == 1)
        maskable_mask = (loss_mask == 1)  # [batch_size, seq_len]

        # 3. Apply stochastic masking
        noised_input_ids, masked_positions = self.apply_stochastic_masking(
            input_ids, timesteps, maskable_mask
        )

        # 4. Forward pass through model
        # Generate position IDs for proper rotary embeddings
        device = input_ids.device
        position_ids = torch.arange(
            0, seq_len, dtype=torch.long, device=device
        ).unsqueeze(0).expand(batch_size, -1)

        outputs = self.model(
            input_ids=noised_input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]

        # 5. For MDLM, target is the original input_ids (no shifting needed)
        target_ids = input_ids  # [batch_size, seq_len]

        # 6. Compute cross-entropy loss
        logits_flat = logits.view(-1, logits.size(-1))  # [batch_size * seq_len, vocab_size]
        target_flat = target_ids.view(-1)  # [batch_size * seq_len]

        token_nll = F.cross_entropy(
            logits_flat,
            target_flat,
            reduction="none"
        )  # [batch_size * seq_len]

        token_nll = token_nll.view(batch_size, seq_len)  # [batch_size, seq_len]

        # 7. Apply loss weights and masking
        loss_weights = self.compute_loss_weights(timesteps, seq_len)  # [batch_size, seq_len]
        weighted_loss = token_nll * loss_weights * masked_positions.float()  # [batch_size, seq_len]

        # 8. Normalize loss
        if self.config.loss_norm_type == "token":
            # Normalize by total number of masked tokens
            total_masked = masked_positions.sum().clamp_min(1)
            loss = weighted_loss.sum() / total_masked
        elif self.config.loss_norm_type == "sequence":
            # Normalize per sequence then average over batch
            seq_masked = masked_positions.sum(dim=1, keepdim=True).clamp_min(1)  # [batch_size, 1]
            seq_loss = weighted_loss.sum(dim=1, keepdim=True) / seq_masked  # [batch_size, 1]
            loss = seq_loss.mean()
        else:  # "batch"
            # Normalize by batch size
            loss = weighted_loss.sum() / batch_size

        # 9. Compute metrics
        with torch.no_grad():
            # Basic metrics
            total_tokens = maskable_mask.sum().item()
            masked_tokens = masked_positions.sum().item()
            mask_ratio = masked_tokens / max(total_tokens, 1)
            avg_timestep = timesteps.mean().item()

            # Accuracy on masked positions
            if masked_tokens > 0:
                masked_logits = logits[masked_positions]  # [masked_tokens, vocab_size]
                masked_targets = target_ids[masked_positions]  # [masked_tokens]
                masked_preds = masked_logits.argmax(dim=-1)  # [masked_tokens]
                accuracy = (masked_preds == masked_targets).float().mean().item()
            else:
                accuracy = 0.0

            metrics = {
                "loss": loss.item(),
                "nll": token_nll[masked_positions].mean().item() if masked_tokens > 0 else 0.0,
                "ppl": torch.exp(token_nll[masked_positions]).mean().item() if masked_tokens > 0 else 1.0,
                "mask_ratio": mask_ratio,
                "avg_timestep": avg_timestep,
                "masked_tokens": masked_tokens,
                "total_tokens": total_tokens,
                "accuracy": accuracy,
            }

        return loss, metrics

    def training_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Perform a single training step.

        Args:
            batch: Training batch

        Returns:
            Tuple of (loss, metrics)
        """
        self.model.train()

        # Forward pass and loss computation
        loss, metrics = self.compute_loss(batch)

        # Update step counter
        self.step += 1

        return loss, metrics

    def validation_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Perform a single validation step.

        Args:
            batch: Validation batch

        Returns:
            Tuple of (loss, metrics)
        """
        self.model.eval()

        with torch.no_grad():
            loss, metrics = self.compute_loss(batch)

        return loss, metrics


class MDLMMetrics:
    """Metrics tracker for MDLM training."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.total_loss = 0.0
        self.total_nll = 0.0
        self.total_ppl = 0.0
        self.total_mask_ratio = 0.0
        self.total_timestep = 0.0
        self.total_accuracy = 0.0
        self.num_batches = 0

    def update(self, metrics: Dict[str, float]):
        """Update metrics with batch results."""
        self.total_loss += metrics["loss"]
        self.total_nll += metrics["nll"]
        self.total_ppl += metrics["ppl"]
        self.total_mask_ratio += metrics["mask_ratio"]
        self.total_timestep += metrics["avg_timestep"]
        self.total_accuracy += metrics["accuracy"]
        self.num_batches += 1

    def compute(self) -> Dict[str, float]:
        """Compute average metrics."""
        if self.num_batches == 0:
            return {}

        return {
            "loss": self.total_loss / self.num_batches,
            "nll": self.total_nll / self.num_batches,
            "ppl": self.total_ppl / self.num_batches,
            "mask_ratio": self.total_mask_ratio / self.num_batches,
            "avg_timestep": self.total_timestep / self.num_batches,
            "accuracy": self.total_accuracy / self.num_batches,
        }