#!/usr/bin/env python3
"""
MDLM (Masked Diffusion Language Model) training script.

This script implements MDLM training by maximally reusing SpecForge's existing
infrastructure while adapting for diffusion-based training.
"""
import argparse
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
from specforge.core.mdlm import MDLMTrainer, MDLMConfig, MDLMMetrics
from specforge.core.schedulers import make_alpha_scheduler
from specforge.lr_scheduler import CosineAnnealingWarmupLR


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
            cache_dir=args.model_download_dir,
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
    target_model = get_eagle3_target_model(
        target_model_path=args.target_model_path,
        backend=args.target_model_backend,
        cache_dir=args.model_download_dir,
        sglang_args=args,  # Pass SGLang args if using sglang backend
    )
    return target_model


def build_dataloaders(args: Namespace) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Build data loaders (REUSE: mostly same as Eagle3 with MDLM data collator)."""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_path,
        cache_dir=args.model_download_dir,
        trust_remote_code=True,
    )

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build training dataset
    print_on_rank0("Building training dataset...")
    train_dataset = build_eagle3_dataset(
        data_path=args.train_data_path,
        tokenizer=tokenizer,
        chat_template=args.chat_template,
        is_preformatted=args.is_preformatted,
        max_length=args.max_length,
        num_proc=args.build_dataset_num_proc,
    )

    # Build evaluation dataset if provided
    eval_dataset = None
    if args.eval_data_path:
        print_on_rank0("Building evaluation dataset...")
        eval_dataset = build_eagle3_dataset(
            data_path=args.eval_data_path,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            is_preformatted=args.is_preformatted,
            max_length=args.max_length,
            num_proc=args.build_dataset_num_proc,
        )

    # Generate vocab mapping file
    vocab_mapping_path = os.path.join(args.output_dir, "vocab_mapping.pt")
    if dist.get_rank() == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        generate_vocab_mapping_file(
            tokenizer=tokenizer,
            draft_vocab_size=tokenizer.vocab_size,  # For MDLM, same vocab as target
            save_path=vocab_mapping_path,
        )
    dist.barrier()

    # Create distributed dataloaders
    train_dataloader, eval_dataloader = prepare_dp_dataloaders(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        batch_size=args.batch_size,
        dp_group=get_dp_group(),
        draft_dp_group=get_draft_dp_group(),
    )

    return train_dataloader, eval_dataloader


def create_mdlm_trainer(
    model: nn.Module,
    args: Namespace,
    total_steps: int,
) -> MDLMTrainer:
    """Create MDLM trainer with SpecForge's optimization infrastructure."""
    # Create MDLM config
    mdlm_config = MDLMConfig(
        mask_token_id=args.mask_token_id,
        alpha_scheduler=args.alpha_scheduler,
        time_epsilon=args.time_epsilon,
        loss_weight_type=args.loss_weight_type,
        loss_norm_type=args.loss_norm_type,
        max_length=args.max_length,
    )

    # Create optimizer (REUSE: existing BF16Optimizer)
    optimizer = BF16Optimizer(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    # Create learning rate scheduler (REUSE: existing scheduler)
    num_warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = CosineAnnealingWarmupLR(
        optimizer=optimizer,
        warmup_steps=num_warmup_steps,
        total_steps=total_steps,
        eta_min=args.learning_rate * 0.1,
    )

    # Create alpha scheduler for MDLM
    alpha_scheduler = make_alpha_scheduler(args.alpha_scheduler)

    # Create MDLM trainer
    trainer = MDLMTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        alpha_scheduler=alpha_scheduler,
        config=mdlm_config,
    )

    return trainer


def train_epoch(
    trainer: MDLMTrainer,
    dataloader: DataLoader,
    epoch: int,
    args: Namespace,
    tracker: Optional[Tracker] = None,
    metrics: Optional[MDLMMetrics] = None,
) -> float:
    """Train one epoch."""
    trainer.model.train()

    if metrics is None:
        metrics = MDLMMetrics()

    metrics.reset()
    total_loss = 0.0

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

        # Training step
        loss, step_metrics = trainer.training_step(batch)

        # Scale loss for gradient accumulation
        loss = loss / args.gradient_accumulation_steps

        # Update metrics
        metrics.update(step_metrics)
        total_loss += loss.item()

        # Gradient accumulation and optimization
        if (step + 1) % args.gradient_accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), args.max_grad_norm)

            # Optimizer step
            trainer.optimizer.step()
            trainer.scheduler.step()
            trainer.optimizer.zero_grad()

        # Logging
        if step % args.log_interval == 0:
            current_metrics = metrics.compute()
            lr = trainer.scheduler.get_last_lr()[0]

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
                    'train/step': trainer.step,
                    'train/epoch': epoch,
                }
                tracker.log(log_data)

        # Save checkpoint
        if step % args.save_interval == 0 and step > 0:
            save_checkpoint(trainer, epoch, step, args)

    return total_loss / len(dataloader)


def evaluate(
    trainer: MDLMTrainer,
    dataloader: DataLoader,
    epoch: int,
    tracker: Optional[Tracker] = None,
) -> Dict[str, float]:
    """Evaluate the model."""
    trainer.model.eval()

    metrics = MDLMMetrics()
    metrics.reset()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", disable=dist.get_rank() != 0):
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].cuda()

            # Validation step
            loss, step_metrics = trainer.validation_step(batch)
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


def save_checkpoint(trainer: MDLMTrainer, epoch: int, step: int, args: Namespace):
    """Save model checkpoint."""
    if dist.get_rank() == 0:
        checkpoint_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save model
        trainer.model.save_pretrained(checkpoint_dir)

        # Save training state
        torch.save({
            'epoch': epoch,
            'step': step,
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': trainer.scheduler.state_dict(),
            'alpha_scheduler_name': trainer.alpha_scheduler.__class__.__name__,
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

    # Wrap model with FSDP (REUSE: existing FSDP setup)
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    draft_model = FSDP(
        draft_model,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
    )

    # Calculate total training steps
    total_steps = len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps

    # Create MDLM trainer (NEW: MDLM-specific trainer but reusing optimization infrastructure)
    print_on_rank0("Creating MDLM trainer...")
    trainer = create_mdlm_trainer(draft_model, args, total_steps)

    # Create tracker (REUSE: existing tracking infrastructure)
    tracker = None
    if args.report_to and dist.get_rank() == 0:
        tracker = create_tracker(args)

    # Training loop
    print_on_rank0("Starting training...")
    for epoch in range(args.num_epochs):
        print_on_rank0(f"Epoch {epoch + 1}/{args.num_epochs}")

        # Train epoch
        train_loss = train_epoch(trainer, train_dataloader, epoch, args, tracker)

        # Evaluate if eval dataloader exists
        if eval_dataloader is not None:
            eval_metrics = evaluate(trainer, eval_dataloader, epoch, tracker)

        # Save checkpoint at end of epoch
        save_checkpoint(trainer, epoch, trainer.step, args)

        print_on_rank0(f"Epoch {epoch + 1} completed. Train loss: {train_loss:.4f}")

    print_on_rank0("Training completed!")

    # Cleanup
    if tracker:
        tracker.finish()
    destroy_distributed()


if __name__ == "__main__":
    main()