"""
Modernized main.py with enhanced MLOps capabilities.

This is an example of how to integrate the MLOps module.
You can either:
1. Use this as a replacement for main.py
2. Gradually migrate features from this to main.py
"""

import argparse
import json
import os
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
    print("⚠️  MLOps module not available. Using basic TensorBoard only.")

from torch.utils.tensorboard import SummaryWriter


ROOT = Path(__file__).resolve().parent

DEF = {
    "train_images": ROOT / "Fuji-Apple-Segmentation/trainingset/JPEGImages",
    "train_anno": ROOT / "Fuji-Apple-Segmentation/trainingset/annotations.json",
    "val_images": ROOT / "Fuji-Apple-Segmentation/testset/JPEGImages",
    "val_anno": ROOT / "Fuji-Apple-Segmentation/testset/annotations.json",
    "checkpoint": ROOT / "checkpoints/best_bbox_ap.pth",
    "checkpoint_dir": ROOT / "checkpoints",
    "output_dir": ROOT / "visualizations",
    "dataset_name": "Fuji-Apple-Segmentation",
}

# DEF = {
#     "train_images": ROOT / "Car-Parts-Segmentation/trainingset/JPEGImages",
#     "train_anno": ROOT / "Car-Parts-Segmentation/trainingset/annotations.json",
#     "val_images": ROOT / "Car-Parts-Segmentation/testset/JPEGImages",
#     "val_anno": ROOT / "Car-Parts-Segmentation/testset/annotations.json",
#     "checkpoint": ROOT / "checkpoints/best_bbox_ap.pth",
#     "output_dir": ROOT / "visualizations",
#     "dataset_name": "Car-Parts-Segmentation",
# }


def parse_args():
    p = argparse.ArgumentParser(
        description="Train Mask R-CNN on COCO-format data (MLOps Enhanced)"
    )

    # Dataset
    p.add_argument(
        "--train-images",
        type=str,
        default=DEF["train_images"],
    )
    p.add_argument(
        "--train-anno",
        type=str,
        default=DEF["train_anno"],
    )
    p.add_argument("--val-images", type=str, default=DEF["val_images"])
    p.add_argument(
        "--val-anno",
        type=str,
        default=DEF["val_anno"],
    )

    # Training
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--workers", type=int, default=max(2, os.cpu_count()))
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument(
        "--box-head-lr-multiplier",
        type=float,
        default=10.0,
        help="Learning rate multiplier for box heads (default: 10x)",
    )
    p.add_argument(
        "--mask-head-lr-multiplier",
        type=float,
        default=10.0,
        help="Learning rate multiplier for mask heads (default: 10x)",
    )

    # Output
    p.add_argument("--out", type=str, default=DEF["checkpoint_dir"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--logdir", type=str, default="runs/maskrcnn")

    # Features
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--warmup-epochs", type=int, default=1)
    p.add_argument("--cosine-epochs", type=int, default=0)
    p.add_argument(
        "--resume",
        type=str,
        nargs="?",
        const="auto",
        default=None,
        help="Resume training from checkpoint. Usage: --resume (auto-find latest), --resume <file>, or --resume <dir>",
    )

    # MLOps
    p.add_argument(
        "--use-wandb", action="store_true", help="Enable Weights & Biases tracking"
    )
    p.add_argument("--use-mlflow", action="store_true", help="Enable MLflow tracking")
    p.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name (auto-generated if not provided)",
    )

    return p.parse_args()


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

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * len(train_loader)
    )

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint_path = None

        # Handle "auto" or empty string to find latest in --out directory
        if args.resume.lower() in ("auto", ""):
            checkpoint_dir = Path(args.out)
            print(f"\n🔍 Auto-detecting latest checkpoint in {checkpoint_dir}...")
        else:
            checkpoint_dir = Path(args.resume)

        # If resume is a directory (or auto), find latest checkpoint
        if checkpoint_dir.is_dir():
            checkpoint_files = list(checkpoint_dir.glob("epoch_*.pth"))

            if checkpoint_files:
                # Sort by modification time (newest first)
                checkpoint_path = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
                print(f"📂 Found latest checkpoint: {checkpoint_path}")
            else:
                print(f"⚠️  No checkpoints found in {checkpoint_dir}")
                args.resume = None
        elif checkpoint_dir.exists() and checkpoint_dir.is_file():
            # Resume is a specific file path
            checkpoint_path = checkpoint_dir
        else:
            print(f"⚠️  Checkpoint path not found: {checkpoint_dir}")
            print("   Starting training from scratch...\n")
            args.resume = None

        if checkpoint_path and checkpoint_path.exists():
            print(f"🔄 Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint.get("epoch", 0)
            print(f"✅ Resumed from epoch {start_epoch}\n")

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
        lr_scheduler=lr_scheduler,
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
