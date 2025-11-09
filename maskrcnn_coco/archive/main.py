import argparse
import json
import os
from pathlib import Path
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import (
    CocoSegmentationDataset,
    detection_collate,
    seed_everything,
    build_model,
    fit,
    build_schedulers,
)


def parse_args():
    import argparse, os

    p = argparse.ArgumentParser(description="Train Mask R-CNN on COCO-format data")

    # Defaults match your example invocation
    p.add_argument(
        "--train-images",
        type=str,
        # default="Car-Parts-Segmentation/trainingset/JPEGImages",
        default="Fuji-Apple-Segmentation/trainingset/JPEGImages",
    )
    p.add_argument(
        "--train-anno",
        type=str,
        # default="Car-Parts-Segmentation/trainingset/annotations.json",
        default="Fuji-Apple-Segmentation/trainingset/annotations.json",
    )
    p.add_argument(
        "--val-images",
        type=str,
        # default="Car-Parts-Segmentation/testset/JPEGImages"
        default="Fuji-Apple-Segmentation/testset/JPEGImages",
    )
    p.add_argument(
        "--val-anno",
        type=str,
        # default="Car-Parts-Segmentation/testset/annotations.json"
        default="Fuji-Apple-Segmentation/testset/annotations.json",
    )

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--workers", type=int, default=max(2, os.cpu_count() // 2))
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--momentum", type=float, default=0.9)

    p.add_argument("--out", type=str, default="checkpoints")
    p.add_argument("--seed", type=int, default=42)

    # Default-on feature toggles. Disable with --no-amp / --no-bf16 / --no-compile
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=False
    )  # Disabled by default due to warnings

    p.add_argument("--patience", type=int, default=10)  # early stop patience
    p.add_argument("--warmup-epochs", type=int, default=1)
    p.add_argument("--cosine-epochs", type=int, default=0)  # 0 => epochs - warmup
    p.add_argument("--logdir", type=str, default="runs/maskrcnn")
    p.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets
    print("\n" + "=" * 80)
    print("LOADING DATASETS")
    print("=" * 80)
    train_ds = CocoSegmentationDataset(
        args.train_images, args.train_anno, is_train=True
    )
    val_ds = CocoSegmentationDataset(
        args.val_images,
        args.val_anno,
        is_train=False,
        catid2contig=train_ds.catid2contig,
        contig2catid=train_ds.contig2catid,
    )

    print(f"Training dataset: {len(train_ds):,} images")
    print(f"Validation dataset: {len(val_ds):,} images")
    print(f"Number of classes: {train_ds.num_classes} (including background)")
    print("=" * 80 + "\n")

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
    print(
        f"Total images per epoch: {len(train_ds):,} training, {len(val_ds):,} validation\n"
    )

    # Model
    num_classes = train_ds.num_classes  # includes background
    model = build_model(num_classes)
    model.to(device)

    if args.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # PyTorch 2.x
        except Exception as e:
            print(f"torch.compile failed: {e}. Continuing without compile.")

    # Optimizer and Schedulers
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    cosine_epochs = (
        args.cosine_epochs
        if args.cosine_epochs > 0
        else max(args.epochs - args.warmup_epochs, 1)
    )
    sched_warmup, sched_cosine = build_schedulers(
        optimizer, args.warmup_epochs, cosine_epochs
    )

    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint.get("epoch", 0)
        print(f"Resumed from epoch {start_epoch}\n")

    # Logging
    Path(args.out).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.logdir)

    # Train
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        dataset=val_ds,
        device=device,
        optimizer=optimizer,
        schedulers=(sched_warmup, sched_cosine),
        epochs=args.epochs,
        out_dir=args.out,
        writer=writer,
        use_amp=args.amp,
        use_bf16=args.bf16,
        early_stop_patience=args.patience,
        start_epoch=start_epoch,
    )

    writer.close()


if __name__ == "__main__":
    main()
