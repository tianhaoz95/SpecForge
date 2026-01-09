#!/usr/bin/env python3
"""
MDLM (Masked Diffusion Language Model) training script.

This script implements MDLM training by maximally reusing SpecForge's existing
infrastructure while adapting for diffusion-based training.
"""
import argparse
import contextlib
import math
import os
import time
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from specforge import (
    AutoDraftModelConfig,
)
from specforge.args import SGLangBackendArgs, TrackerArgs
from specforge.data import (
    build_eagle3_dataset,
    generate_vocab_mapping_file,
    prepare_dp_dataloaders,
)
from specforge.distributed import (
    destroy_distributed,
    get_dp_group,
    get_draft_dp_group,
    get_tp_group,
    init_distributed,
)
from specforge.modeling.auto import AutoMDLMDraftModel
from specforge.modeling.draft.qwen3_mdlm import Qwen3MDLMDraftModel  # Import to register the model
from specforge.modeling.target import (
    Eagle3TargetModel,
    get_eagle3_target_model,
)
from specforge.optimizer import BF16Optimizer
from specforge.tracker import Tracker, create_tracker, get_tracker_class
from specforge.utils import (
    create_mdlm_config_from_target,
    get_last_checkpoint,
    print_args_with_dots,
    print_on_rank0,
    print_with_rank,
)
from specforge.core.schedulers import make_alpha_scheduler


def parse_args():
    """Parse command line arguments."""
    parser = ArgumentParser(description="MDLM Training Script")

    # Model arguments
    model_group = parser.add_argument_group("Model Arguments")
    model_group.add_argument(
        "--target-model-path",
        type=str,
        required=True,
        help="Path to target model for hidden state extraction"
    )
    model_group.add_argument(
        "--draft-model-config",
        type=str,
        default=None,
        help="Path to draft model config file (auto-generated if not provided)"
    )
    model_group.add_argument(
        "--embedding-key",
        type=str,
        default="model.embed_tokens.weight",
        help="Key for embedding layer in target model"
    )
    model_group.add_argument(
        "--target-model-backend",
        type=str,
        choices=["sglang", "hf", "custom"],
        default="hf",
        help="Backend for target model"
    )
    model_group.add_argument(
        "--model-download-dir",
        type=str,
        default=None,
        help="Directory to download models to"
    )
    model_group.add_argument(
        "--cache-dir",
        type=str,
        default="./cache",
        help="Cache directory for model downloads"
    )
    model_group.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Whether to trust remote code when loading models and tokenizers"
    )

    # MDLM-specific arguments
    mdlm_group = parser.add_argument_group("MDLM Arguments")
    mdlm_group.add_argument(
        "--mask-token-id",
        type=int,
        required=True,
        help="Token ID for masking (e.g., 32000 for <mask>)"
    )
    mdlm_group.add_argument(
        "--alpha-scheduler",
        type=str,
        choices=["linear", "cosine"],
        default="linear",
        help="Alpha scheduler for masking rate"
    )
    mdlm_group.add_argument(
        "--time-epsilon",
        type=float,
        default=1e-3,
        help="Minimum timestep to avoid degenerate values"
    )
    mdlm_group.add_argument(
        "--loss-weight-type",
        type=str,
        choices=["scheduler", "uniform"],
        default="scheduler",
        help="Loss weighting strategy"
    )
    mdlm_group.add_argument(
        "--loss-norm-type",
        type=str,
        choices=["batch", "sequence", "token"],
        default="token",
        help="Loss normalization strategy"
    )

    # Dataset arguments
    dataset_group = parser.add_argument_group("Dataset Arguments")
    dataset_group.add_argument(
        "--train-data-path",
        type=str,
        required=True,
        help="Path to training data (JSON file)"
    )
    dataset_group.add_argument(
        "--eval-data-path",
        type=str,
        default=None,
        help="Path to evaluation data (JSON file)"
    )
    dataset_group.add_argument(
        "--chat-template",
        type=str,
        default="llama3",
        help="Chat template to use"
    )
    dataset_group.add_argument(
        "--is-preformatted",
        action="store_true",
        help="Whether data is already formatted"
    )
    dataset_group.add_argument(
        "--build-dataset-num-proc",
        type=int,
        default=8,
        help="Number of processes for dataset building"
    )
    dataset_group.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )

    # Training arguments
    training_group = parser.add_argument_group("Training Arguments")
    training_group.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for checkpoints and logs"
    )
    training_group.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    training_group.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size per device"
    )
    training_group.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    training_group.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio"
    )
    training_group.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping"
    )
    training_group.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps"
    )
    training_group.add_argument(
        "--eval-interval",
        type=int,
        default=500,
        help="Evaluation interval (steps)"
    )
    training_group.add_argument(
        "--save-interval",
        type=int,
        default=1000,
        help="Save interval (steps)"
    )
    training_group.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Log interval (steps)"
    )
    training_group.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint"
    )
    training_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    training_group.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    # Optimization arguments
    opt_group = parser.add_argument_group("Optimization Arguments")
    opt_group.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel size"
    )
    opt_group.add_argument(
        "--sp-ulysses-size",
        type=int,
        default=1,
        help="Sequence parallel ulysses size"
    )
    opt_group.add_argument(
        "--sp-ring-size",
        type=int,
        default=1,
        help="Sequence parallel ring size"
    )
    opt_group.add_argument(
        "--attention-backend",
        type=str,
        choices=["flex_attention", "usp", "sdpa"],
        default="sdpa",
        help="Attention backend"
    )

    # Tracker arguments
    TrackerArgs.add_args(parser)

    # SGLang backend arguments (if using sglang backend)
    SGLangBackendArgs.add_args(parser)

    return parser.parse_args()


def build_mdlm_draft_model(args: Namespace) -> Tuple[AutoDraftModelConfig, nn.Module]:
    """Build MDLM draft model by reusing SpecForge's proven model building infrastructure.

    This function follows the exact same pattern as build_draft_model() in train_eagle3.py
    but uses MDLM-specific model factory and configuration.
    """
    # 1. Handle draft model config (REUSE: same logic as Eagle3)
    if args.draft_model_config is None:
        # Auto-generate MDLM config from target model
        auto_config_path = create_mdlm_config_from_target(
            target_model_path=args.target_model_path,
            mask_token_id=args.mask_token_id,
            alpha_scheduler=args.alpha_scheduler,
            time_epsilon=args.time_epsilon,
            cache_dir=args.cache_dir,
            trust_remote_code=args.trust_remote_code,
        )
        draft_model_config = AutoDraftModelConfig.from_file(auto_config_path)
    else:
        # Use provided config file (REUSE: same as Eagle3)
        draft_model_config = AutoDraftModelConfig.from_file(args.draft_model_config)

    # 2. Handle checkpoints (REUSE: identical to Eagle3)
    draft_model_last_checkpoint = None

    # Detect last checkpoint for resume training (REUSE: same logic)
    if args.resume and os.path.isdir(args.output_dir):
        draft_model_last_checkpoint = get_last_checkpoint(args.output_dir)
        if draft_model_last_checkpoint:
            print_on_rank0(f"Last checkpoint detected: {draft_model_last_checkpoint}")

    # 3. Create MDLM model (NEW: use MDLM factory instead of Eagle3)
    if draft_model_last_checkpoint:
        draft_model = AutoMDLMDraftModel.from_pretrained(
            draft_model_last_checkpoint,
            torch_dtype=torch.bfloat16,  # REUSE: same dtype
        ).cuda()
    else:
        draft_model = AutoMDLMDraftModel.from_config(
            draft_model_config,
            torch_dtype=torch.bfloat16,
        ).cuda()

    # 4. Load embeddings (REUSE: identical to Eagle3)
    draft_model.load_embedding(args.target_model_path, embedding_key=args.embedding_key)
    draft_model.freeze_embedding()

    return draft_model_config, draft_model


def build_target_model(args: Namespace) -> Eagle3TargetModel:
    """Build target model (REUSE: identical to Eagle3)."""
    if args.target_model_backend == "sglang":
        target_model_kwargs = SGLangBackendArgs.from_args(args).to_kwargs()
    else:
        target_model_kwargs = {}
    target_model = get_eagle3_target_model(
        pretrained_model_name_or_path=args.target_model_path,
        backend=args.target_model_backend,
        torch_dtype=torch.bfloat16,
        device="cuda",
        cache_dir=args.cache_dir,
        **target_model_kwargs,
    )
    return target_model


def build_dataloaders(args: Namespace) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Build data loaders (REUSE: mostly same as Eagle3 with MDLM data collator)."""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_path,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build training dataset (REUSE: same pattern as Eagle3)
    print_on_rank0("Building training dataset...")
    import hashlib
    from specforge.utils import rank_0_priority

    cache_params_string = (
        f"{args.train_data_path}-"
        f"{args.max_length}-"
        f"{args.chat_template}-"
        f"{args.target_model_path}"
    )
    cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()
    train_raw_dataset = load_dataset("json", data_files=args.train_data_path)["train"]

    with rank_0_priority():
        train_dataset = build_eagle3_dataset(
            dataset=train_raw_dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
            cache_key=cache_key,
            is_preformatted=args.is_preformatted,
            num_proc=args.build_dataset_num_proc,
        )

    # Build evaluation dataset if provided
    eval_dataset = None
    if args.eval_data_path:
        print_on_rank0("Building evaluation dataset...")
        eval_cache_params_string = (
            f"{args.eval_data_path}-"
            f"{args.max_length}-"
            f"{args.chat_template}-"
            f"{args.target_model_path}"
        )
        eval_cache_key = hashlib.md5(eval_cache_params_string.encode()).hexdigest()
        eval_raw_dataset = load_dataset("json", data_files=args.eval_data_path)["train"]

        with rank_0_priority():
            eval_dataset = build_eagle3_dataset(
                dataset=eval_raw_dataset,
                tokenizer=tokenizer,
                chat_template=args.chat_template,
                max_length=args.max_length,
                cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
                cache_key=eval_cache_key,
                is_preformatted=args.is_preformatted,
                num_proc=args.build_dataset_num_proc,
            )

    # Generate vocab mapping file (REUSE: same pattern as Eagle3)
    vocab_mapping_path = generate_vocab_mapping_file(
        dataset=train_dataset,
        target_vocab_size=tokenizer.vocab_size,
        draft_vocab_size=tokenizer.vocab_size,  # For MDLM, same vocab as target
        cache_dir=os.path.join(args.cache_dir, "vocab_mapping"),
        cache_key=cache_key,
    )

    # Create distributed dataloaders (REUSE: same pattern as Eagle3)
    train_dataloader = prepare_dp_dataloaders(
        train_dataset,
        args.batch_size,
        num_workers=4,  # Default from Eagle3
        shuffle=True,
        process_group=(
            get_draft_dp_group() if args.attention_backend == "usp" else get_dp_group()
        ),
    )

    eval_dataloader = None
    if eval_dataset is not None:
        eval_dataloader = prepare_dp_dataloaders(
            eval_dataset,
            args.batch_size,
            num_workers=4,
            shuffle=False,
            process_group=(
                get_draft_dp_group() if args.attention_backend == "usp" else get_dp_group()
            ),
        )

    return train_dataloader, eval_dataloader


def sample_timesteps(batch_size: int, time_epsilon: float, device: torch.device) -> torch.Tensor:
    """Sample diffusion timesteps for training."""
    eps = time_epsilon
    return eps + (1 - eps) * torch.rand(batch_size, device=device)


def apply_stochastic_masking(
    input_ids: torch.Tensor,
    timesteps: torch.Tensor,
    maskable_mask: torch.Tensor,
    alpha_scheduler,
    mask_token_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply stochastic masking based on timesteps."""
    batch_size, seq_len = input_ids.shape

    # Compute masking probability from alpha scheduler
    alpha_vals = alpha_scheduler.alpha(timesteps)  # [batch_size]
    p_mask = 1.0 - alpha_vals.unsqueeze(1).expand(batch_size, seq_len)  # [batch_size, seq_len]

    # Sample which positions to mask
    random_vals = torch.rand_like(p_mask, device=device)
    masked_positions = (random_vals < p_mask) & maskable_mask  # [batch_size, seq_len]

    # Apply masking
    noised_input_ids = torch.where(
        masked_positions,
        mask_token_id,
        input_ids
    )

    return noised_input_ids, masked_positions


def compute_loss_weights(timesteps: torch.Tensor, seq_len: int, alpha_scheduler, loss_weight_type: str, device: torch.device) -> torch.Tensor:
    """Compute loss weights based on timesteps."""
    if loss_weight_type == "scheduler":
        # Use scheduler-based weighting: w(t) = -α'(t) / (1 - α(t))
        weights = alpha_scheduler.weight(timesteps)  # [batch_size]
        return weights.unsqueeze(1).expand(-1, seq_len)  # [batch_size, seq_len]
    else:
        # Uniform weighting
        batch_size = timesteps.size(0)
        return torch.ones(batch_size, seq_len, device=device)


def run_mdlm_forward(
    args: Namespace,
    mdlm_model: nn.Module,
    batch: Dict[str, torch.Tensor],
    alpha_scheduler,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Run MDLM forward pass directly (following Eagle3 pattern)."""
    # Extract inputs
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    loss_mask = batch["loss_mask"]

    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    # 1. Sample timesteps
    timesteps = sample_timesteps(batch_size, args.time_epsilon, device)

    # 2. Create maskable mask (where loss_mask == 1)
    maskable_mask = (loss_mask == 1)

    # 3. Apply stochastic masking
    noised_input_ids, masked_positions = apply_stochastic_masking(
        input_ids, timesteps, maskable_mask, alpha_scheduler, args.mask_token_id, device
    )

    # 4. Forward pass through FSDP model directly
    position_ids = torch.arange(
        0, seq_len, dtype=torch.long, device=device
    ).unsqueeze(0).expand(batch_size, -1)

    outputs = mdlm_model(
        input_ids=noised_input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )
    logits = outputs.logits

    # 5. Compute loss
    target_ids = input_ids
    logits_flat = logits.view(-1, logits.size(-1))
    target_flat = target_ids.view(-1)

    token_nll = torch.nn.functional.cross_entropy(
        logits_flat, target_flat, reduction="none"
    )
    token_nll = token_nll.view(batch_size, seq_len)

    # 6. Apply loss weights and masking
    loss_weights = compute_loss_weights(timesteps, seq_len, alpha_scheduler, args.loss_weight_type, device)
    weighted_loss = token_nll * loss_weights * masked_positions.float()

    # 7. Normalize loss
    if args.loss_norm_type == "token":
        total_masked = masked_positions.sum().clamp_min(1)
        loss = weighted_loss.sum() / total_masked
    elif args.loss_norm_type == "sequence":
        seq_masked = masked_positions.sum(dim=1, keepdim=True).clamp_min(1)
        seq_loss = weighted_loss.sum(dim=1, keepdim=True) / seq_masked
        loss = seq_loss.mean()
    else:  # "batch"
        loss = weighted_loss.sum() / batch_size

    # 8. Compute metrics
    with torch.no_grad():
        total_tokens = maskable_mask.sum().item()
        masked_tokens = masked_positions.sum().item()
        mask_ratio = masked_tokens / max(total_tokens, 1)
        avg_timestep = timesteps.mean().item()

        if masked_tokens > 0:
            masked_logits = logits[masked_positions]
            masked_targets = target_ids[masked_positions]
            masked_preds = masked_logits.argmax(dim=-1)
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


def run_mdlm_backward_and_update(
    args: Namespace,
    loss: torch.Tensor,
    optimizer: BF16Optimizer,
    global_step: int
) -> None:
    """Run backward pass and optimizer update (following Eagle3 pattern exactly)."""
    # Scale loss for gradient accumulation
    loss = loss / args.gradient_accumulation_steps

    # Backward pass
    loss.backward()

    # Update optimizer on accumulation boundary
    if global_step % args.gradient_accumulation_steps == 0:
        optimizer.step()


class MDLMMetrics:
    """Simple metrics tracker for MDLM training."""

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


def train_epoch(
    args: Namespace,
    mdlm_model: nn.Module,
    optimizer: BF16Optimizer,
    alpha_scheduler,
    dataloader: DataLoader,
    epoch: int,
    tracker: Optional[Tracker] = None,
    global_step_counter: Optional[List[int]] = None,
) -> float:
    """Train one epoch."""
    mdlm_model.train()

    metrics = MDLMMetrics()
    metrics.reset()
    total_loss = 0.0

    if global_step_counter is None:
        global_step_counter = [0]

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch}",
        disable=dist.get_rank() != 0,
    )

    for step, batch in enumerate(progress_bar):
        # Move batch to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].cuda()

        # Forward pass (following Eagle3 pattern)
        loss, step_metrics = run_mdlm_forward(args, mdlm_model, batch, alpha_scheduler)

        # Backward pass and optimization (following Eagle3 pattern)
        run_mdlm_backward_and_update(args, loss, optimizer, global_step_counter[0] + 1)

        # Update counters
        global_step_counter[0] += 1

        # Update metrics
        metrics.update(step_metrics)
        total_loss += loss.item()

        # Logging
        if step % args.log_interval == 0:
            current_metrics = metrics.compute()
            lr = optimizer.scheduler.get_last_lr()[0]

            progress_bar.set_postfix({
                'loss': f"{current_metrics['loss']:.4f}",
                'ppl': f"{current_metrics['ppl']:.2f}",
                'mask_ratio': f"{current_metrics['mask_ratio']:.3f}",
                'lr': f"{lr:.2e}",
            })

            if tracker and dist.get_rank() == 0:
                log_data = {
                    'train/loss': current_metrics['loss'],
                    'train/nll': current_metrics['nll'],
                    'train/ppl': current_metrics['ppl'],
                    'train/mask_ratio': current_metrics['mask_ratio'],
                    'train/avg_timestep': current_metrics['avg_timestep'],
                    'train/accuracy': current_metrics['accuracy'],
                    'train/lr': lr,
                    'train/step': global_step_counter[0],
                    'train/epoch': epoch,
                }
                tracker.log(log_data)

        # Save checkpoint
        if step % args.save_interval == 0 and step > 0:
            save_checkpoint(mdlm_model, optimizer, epoch, global_step_counter[0], args)

    return total_loss / len(dataloader)


def evaluate(
    args: Namespace,
    mdlm_model: nn.Module,
    alpha_scheduler,
    dataloader: DataLoader,
    epoch: int,
    tracker: Optional[Tracker] = None,
) -> Dict[str, float]:
    """Evaluate the model."""
    mdlm_model.eval()

    metrics = MDLMMetrics()
    metrics.reset()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", disable=dist.get_rank() != 0):
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].cuda()

            # Validation step
            loss, step_metrics = run_mdlm_forward(args, mdlm_model, batch, alpha_scheduler)
            metrics.update(step_metrics)

    # Compute final metrics
    eval_metrics = metrics.compute()

    if dist.get_rank() == 0:
        print(f"Evaluation - Epoch {epoch}:")
        for key, value in eval_metrics.items():
            print(f"  {key}: {value:.4f}")

    if tracker and dist.get_rank() == 0:
        log_data = {f'eval/{key}': value for key, value in eval_metrics.items()}
        log_data['eval/epoch'] = epoch
        tracker.log(log_data)

    return eval_metrics


def save_checkpoint(model: nn.Module, optimizer: BF16Optimizer, epoch: int, step: int, args: Namespace):
    """Save model checkpoint."""
    if dist.get_rank() == 0:
        checkpoint_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save model
        model.save_pretrained(checkpoint_dir)

        # Save training state
        torch.save({
            'epoch': epoch,
            'step': step,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': optimizer.scheduler.state_dict(),
        }, os.path.join(checkpoint_dir, 'training_state.pt'))

        print_on_rank0(f"Checkpoint saved to {checkpoint_dir}")

    dist.barrier()


def main():
    """Main training function."""
    args = parse_args()

    # Initialize distributed training
    init_distributed(
        tp_size=args.tp_size,
        sp_ulysses_size=args.sp_ulysses_size,
        sp_ring_size=args.sp_ring_size,
    )

    # Set seed
    set_seed(args.seed)

    # Print arguments
    if dist.get_rank() == 0:
        print_args_with_dots(args)

    # Build models (REUSE: same pattern as Eagle3)
    print_on_rank0("Building MDLM draft model...")
    draft_model_config, draft_model = build_mdlm_draft_model(args)

    print_on_rank0("Building target model...")
    target_model = build_target_model(args)

    # Build dataloaders (REUSE: with MDLM data collator extension)
    print_on_rank0("Building dataloaders...")
    train_dataloader, eval_dataloader = build_dataloaders(args)

    # Calculate total training steps
    total_steps = len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps

    # Create optimizer with underlying model (before FSDP wrapping, following Eagle3 pattern)
    print_on_rank0("Creating optimizer...")
    optimizer = BF16Optimizer(
        draft_model,
        lr=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        total_steps=total_steps,
    )

    # Wrap model with FSDP (following Eagle3 configuration)
    mdlm_model = FSDP(
        draft_model,
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        process_group=dist.group.WORLD,
    )

    # Create alpha scheduler for MDLM
    print_on_rank0("Creating alpha scheduler...")
    alpha_scheduler = make_alpha_scheduler(args.alpha_scheduler)

    # Create tracker (REUSE: existing tracking infrastructure)
    tracker = None
    if args.report_to and dist.get_rank() == 0:
        tracker = create_tracker(args, args.output_dir)

    # Training loop (following Eagle3 exact pattern)
    print_on_rank0("Starting training...")
    global_step = 0

    # Metrics tracking
    metrics = MDLMMetrics()

    for epoch in range(args.num_epochs):
        print_on_rank0(f"Epoch {epoch + 1}/{args.num_epochs}")

        # Set model to training mode
        mdlm_model.train()
        metrics.reset()

        # Create progress bar
        if dist.get_rank() == 0:
            progress_bar = tqdm(
                train_dataloader, desc=f"Training Epoch {epoch}", leave=True
            )
        else:
            progress_bar = train_dataloader

        # Main training loop (exactly like Eagle3)
        for data in progress_bar:
            global_step += 1

            # Move batch to device
            for key in data:
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].cuda()

            # ================================================
            # Training Step (following Eagle3 pattern exactly)
            # ================================================
            loss, step_metrics = run_mdlm_forward(args, mdlm_model, data, alpha_scheduler)
            run_mdlm_backward_and_update(args, loss, optimizer, global_step)

            # Update metrics
            metrics.update(step_metrics)

            # Logging
            if global_step % (args.log_interval * args.gradient_accumulation_steps) == 0:
                current_metrics = metrics.compute()
                lr = optimizer.scheduler.get_last_lr()[0]

                if dist.get_rank() == 0:
                    progress_bar.set_postfix({
                        'loss': f"{current_metrics['loss']:.4f}",
                        'ppl': f"{current_metrics['ppl']:.2f}",
                        'mask_ratio': f"{current_metrics['mask_ratio']:.3f}",
                        'lr': f"{lr:.2e}",
                    })

                if tracker and dist.get_rank() == 0:
                    log_data = {
                        'train/loss': current_metrics['loss'],
                        'train/nll': current_metrics['nll'],
                        'train/ppl': current_metrics['ppl'],
                        'train/mask_ratio': current_metrics['mask_ratio'],
                        'train/avg_timestep': current_metrics['avg_timestep'],
                        'train/accuracy': current_metrics['accuracy'],
                        'train/lr': lr,
                        'train/step': global_step,
                        'train/epoch': epoch,
                    }
                    tracker.log(log_data)

            # Save checkpoint
            if global_step % (args.save_interval * args.gradient_accumulation_steps) == 0:
                save_checkpoint(mdlm_model, optimizer, epoch, global_step, args)

        # Evaluate if eval dataloader exists
        if eval_dataloader is not None:
            eval_metrics = evaluate(args, mdlm_model, alpha_scheduler, eval_dataloader, epoch, tracker)

        # Save checkpoint at end of epoch
        save_checkpoint(mdlm_model, optimizer, epoch, global_step, args)

        # Compute epoch metrics
        epoch_metrics = metrics.compute()
        print_on_rank0(f"Epoch {epoch + 1} completed. Train loss: {epoch_metrics['loss']:.4f}")

    print_on_rank0("Training completed!")

    # Cleanup
    if tracker:
        tracker.finish()
    destroy_distributed()


if __name__ == "__main__":
    main()