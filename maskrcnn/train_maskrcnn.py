"""
Modernized main.py with enhanced MLOps capabilities and YAML configuration support.
"""

import argparse
import json
import os
import yaml
from pathlib import Path
from datetime import datetime
import torch
import torchvision
from torch.utils.data import DataLoader

from core import (
    CocoSegmentationDataset,
    detection_collate,
    seed_everything,
    build_model,
    get_parameter_groups,
    fit,
)

# Import MLOps module
try:
    from mlops import (
        ExperimentTracker,
        create_experiment_config,
        log_training_metrics,
    )

    MLOPS_AVAILABLE = True
except ImportError:
    MLOPS_AVAILABLE = False
    print("[WARNING] MLOps module not available. Using basic TensorBoard only.")

from torch.utils.tensorboard import SummaryWriter


ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path):
    """Load configuration from YAML file."""
    if config_path is None:
        return {}
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    # First pass to get config file
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None, help="Path to experiment config YAML")
    args, remaining_argv = pre_parser.parse_known_args()

    # Load config
    config = load_config(args.config)
    
    # Defaults from config or fallback
    dataset_cfg = config.get("dataset", {})
    train_cfg = config.get("training", {})
    optim_cfg = config.get("optimization", {})
    out_cfg = config.get("output", {})
    mlops_cfg = config.get("mlops", {})

    # Helper to resolve paths relative to ROOT
    def resolve_path(path_str, default=None):
        if path_str:
            return str(ROOT / path_str)
        return default

    default_train_images = resolve_path(dataset_cfg.get("train_images"))
    default_train_anno = resolve_path(dataset_cfg.get("train_anno"))
    default_val_images = resolve_path(dataset_cfg.get("val_images"))
    default_val_anno = resolve_path(dataset_cfg.get("val_anno"))
    
    default_checkpoint_dir = resolve_path(out_cfg.get("checkpoint_dir", "checkpoints"))
    default_output_dir = resolve_path(out_cfg.get("output_dir", "visualizations"))
    default_log_dir = out_cfg.get("log_dir", "runs/maskrcnn")

    p = argparse.ArgumentParser(
        description="Train Mask R-CNN on COCO-format data (MLOps Enhanced)"
    )
    
    # Config arg (already parsed but included for help)
    p.add_argument("--config", type=str, default=None, help="Path to experiment config YAML")

    # Dataset
    p.add_argument("--train-images", type=str, default=default_train_images, required=default_train_images is None)
    p.add_argument("--train-anno", type=str, default=default_train_anno, required=default_train_anno is None)
    p.add_argument("--val-images", type=str, default=default_val_images)
    p.add_argument("--val-anno", type=str, default=default_val_anno)

    # Training
    p.add_argument("--epochs", type=int, default=train_cfg.get("epochs", 300))
    p.add_argument("--batch-size", type=int, default=train_cfg.get("batch_size", 2))
    p.add_argument("--workers", type=int, default=train_cfg.get("workers", max(2, os.cpu_count())))
    p.add_argument("--lr", type=float, default=train_cfg.get("lr", 1e-3))
    p.add_argument("--momentum", type=float, default=train_cfg.get("momentum", 0.9))
    p.add_argument("--weight-decay", type=float, default=train_cfg.get("weight_decay", 1e-4))
    p.add_argument(
        "--box-head-lr-multiplier",
        type=float,
        default=optim_cfg.get("box_head_lr_multiplier", 10.0),
        help="Learning rate multiplier for box heads",
    )
    p.add_argument(
        "--mask-head-lr-multiplier",
        type=float,
        default=optim_cfg.get("mask_head_lr_multiplier", 10.0),
        help="Learning rate multiplier for mask heads",
    )

    # Output
    p.add_argument("--out", type=str, default=default_checkpoint_dir)
    p.add_argument("--seed", type=int, default=train_cfg.get("seed", 42))
    p.add_argument("--logdir", type=str, default=default_log_dir)

    # Features
    # Handle boolean args safely
    def str2bool(v):
        if isinstance(v, bool): return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
        else: raise argparse.ArgumentTypeError('Boolean value expected.')

    p.add_argument("--amp", type=str2bool, default=optim_cfg.get("amp", True))
    p.add_argument("--bf16", type=str2bool, default=optim_cfg.get("bf16", True))
    p.add_argument("--compile", type=str2bool, default=optim_cfg.get("compile", False))
    
    p.add_argument("--patience", type=int, default=optim_cfg.get("patience", 10))
    p.add_argument("--warmup-epochs", type=int, default=optim_cfg.get("warmup_epochs", 1))
    p.add_argument("--cosine-epochs", type=int, default=optim_cfg.get("cosine_epochs", 0))
    p.add_argument(
        "--resume",
        type=str,
        nargs="?",
        const="auto",
        default=None,
        help="Resume training from checkpoint. Usage: --resume (auto-find latest), --resume <file>, or --resume <dir>",
    )

    # MLOps
    p.add_argument("--use-wandb", action="store_true", default=mlops_cfg.get("use_wandb", False), help="Enable Weights & Biases tracking")
    p.add_argument("--use-mlflow", action="store_true", default=mlops_cfg.get("use_mlflow", False), help="Enable MLflow tracking")
    p.add_argument(
        "--experiment-name",
        type=str,
        default=out_cfg.get("experiment_name", None),
        help="Experiment name (auto-generated if not provided)",
    )

    return p.parse_args(remaining_argv)


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    from torchvision.transforms import ToTensor
    from torch.utils.data import DataLoader

    # convert a PIL image to a PyTorch tensor
    def get_transform():
        return ToTensor()

    # Datasets
    print("\n" + "=" * 80)
    print("LOADING DATASETS")
    print("=" * 80)
    print(f"Train Images: {args.train_images}")
    print(f"Train Annotations: {args.train_anno}")
    
    train_ds = CocoSegmentationDataset(
        args.train_images,
        args.train_anno,
        # is_train=True
        transforms=get_transform(),
    )
    val_ds = CocoSegmentationDataset(
        args.val_images,
        args.val_anno,
        # is_train=False,
        # catid2contig=train_ds.catid2contig,
        # contig2catid=train_ds.contig2catid,
        transforms=get_transform(),
    )

    num_classes = len(train_ds.coco.getCatIds()) + 1
    print(f"Training dataset: {len(train_ds):,} images")
    print(f"Validation dataset: {len(val_ds):,} images")
    print(f"Number of classes: {num_classes} (including background)")

    # DataLoaders
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin,
        persistent_workers=args.workers > 0,
        collate_fn=detection_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin,
        persistent_workers=args.workers > 0,
        collate_fn=detection_collate,
    )

    print(f"Training batches: {len(train_loader):,} (batch_size={args.batch_size})")
    print(f"Validation batches: {len(val_loader):,} (batch_size=1)")

    # Model
    model = build_model(num_classes)
    model.to(device)

    if args.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"torch.compile failed: {e}. Continuing without compile.")

    # Optimizer and Schedulers with differential learning rates
    # Backbone uses base LR, heads use higher LR (they're randomly initialized)
    param_groups = get_parameter_groups(
        model,
        lr=args.lr,
        box_head_lr_multiplier=args.box_head_lr_multiplier,
        mask_head_lr_multiplier=args.mask_head_lr_multiplier,
    )
    optimizer = torch.optim.SGD(
        param_groups, momentum=args.momentum, weight_decay=args.weight_decay
    )
    # optimizer = torch.optim.AdamW(param_groups, lr=args.lr)

    # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer, max_lr=args.lr, total_steps=args.epochs * len(train_loader)
    # )

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint_path = None

        # Handle "auto" or empty string to find latest in --out directory
        if args.resume.lower() in ("auto", ""):
            checkpoint_dir = Path(args.out)
            print(f"\n[SEARCH] Auto-detecting latest checkpoint in {checkpoint_dir}...")
        else:
            checkpoint_dir = Path(args.resume)

        # If resume is a directory (or auto), find latest checkpoint
        if checkpoint_dir.is_dir():
            checkpoint_files = list(checkpoint_dir.glob("epoch_*.pth"))

            if checkpoint_files:
                # Sort by modification time (newest first)
                checkpoint_path = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
                print(f"[FOUND] Found latest checkpoint: {checkpoint_path}")
            else:
                print(f"[WARNING] No checkpoints found in {checkpoint_dir}")
                args.resume = None
        elif checkpoint_dir.exists() and checkpoint_dir.is_file():
            # Resume is a specific file path
            checkpoint_path = checkpoint_dir
        else:
            print(f"[WARNING] Checkpoint path not found: {checkpoint_dir}")
            print("   Starting training from scratch...\n")
            args.resume = None

        if checkpoint_path and checkpoint_path.exists():
            print(f"[RESUMING] Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint.get("epoch", 0)
            print(f"[SUCCESS] Resumed from epoch {start_epoch}\n")

    # MLOps: Enhanced Experiment Tracking
    print("\n" + "=" * 80)
    print("INITIALIZING EXPERIMENT TRACKING")
    print("=" * 80)

    # Generate experiment name
    if args.experiment_name is None:
        dataset_name = Path(args.train_images).parent.parent.name
        args.experiment_name = f"maskrcnn_{dataset_name}"

    # Create experiment config
    if MLOPS_AVAILABLE:
        config = create_experiment_config(args)
        tracker = ExperimentTracker(
            experiment_name=args.experiment_name,
            project_name="maskrcnn",
            use_wandb=args.use_wandb,
            use_mlflow=args.use_mlflow,
            log_dir=args.logdir,
            config=config,
            tags=[Path(args.train_images).parent.parent.name, "segmentation"],
        )
        writer = tracker.tb_writer  # For backward compatibility
    else:
        # Fallback to basic TensorBoard
        writer = SummaryWriter(log_dir=args.logdir)
        tracker = None
        print("Using basic TensorBoard (MLOps module not available)")

    # Log model architecture
    if tracker:
        # Log model graph to TensorBoard
        try:
            dummy_input = [torch.randn(3, 512, 512).to(device)]
            writer.add_graph(model, dummy_input)
        except Exception as e:
            print(f"Could not log model graph: {e}")

    # Train
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        dataset=val_ds,
        device=device,
        optimizer=optimizer,
        # lr_scheduler=lr_scheduler,
        epochs=args.epochs,
        out_dir=args.out,
        writer=writer,
        use_amp=args.amp,
        use_bf16=args.bf16,
        early_stop_patience=args.patience,
        start_epoch=start_epoch,
        tracker=tracker,  # Pass tracker for enhanced logging
    )

    # Finish tracking
    if tracker:
        tracker.finish()
    else:
        writer.close()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Checkpoints: {args.out}")
    print(f"TensorBoard: tensorboard --logdir {args.logdir}")


if __name__ == "__main__":
    main()
