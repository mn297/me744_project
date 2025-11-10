from __future__ import annotations
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask_utils
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from tqdm import tqdm

# ----------------------------
# Reproducibility
# ----------------------------


def seed_everything(seed: int = 42) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN: prefer deterministic where possible
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ----------------------------
# Dataset with COCO id mapping
# ----------------------------


class CocoSegmentationDataset(Dataset):
    """
    COCO-format dataset that maps arbitrary category_ids to contiguous labels [1..K]
    required by TorchVision detection heads. Keeps reverse map for evaluation.
    """

    def __init__(
        self,
        root_dir: str,
        annotation_file: str,
        is_train: bool = True,
        catid2contig: Dict[int, int] | None = None,
        contig2catid: Dict[int, int] | None = None,
    ):
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.is_train = is_train

        # Build or reuse mapping between COCO category ids and contiguous labels
        if catid2contig is None or contig2catid is None:
            cat_ids = self.coco.getCatIds()
            self.catid2contig = {cid: i + 1 for i, cid in enumerate(sorted(cat_ids))}
            self.contig2catid = {v: k for k, v in self.catid2contig.items()}
        else:
            self.catid2contig = catid2contig
            self.contig2catid = contig2catid

        self.num_classes = len(self.contig2catid) + 1  # + background

    def __len__(self):
        return len(self.image_ids)

    def _load_image(self, image_id: int) -> Image.Image:
        info = self.coco.loadImgs(image_id)[0]
        path = os.path.join(self.root_dir, info["file_name"])
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        image = self._load_image(image_id)

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels, masks, area, iscrowd = [], [], [], [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(self.catid2contig[ann["category_id"]])
            masks.append(self.coco.annToMask(ann))
            area.append(ann["area"])
            iscrowd.append(ann.get("iscrowd", 0))

        # Convert masks list to numpy array first to avoid slow tensor creation warning
        if masks:
            masks_array = (
                np.stack(masks, axis=0)
                if len(masks) > 1
                else np.expand_dims(masks[0], axis=0)
            )
        else:
            # Empty masks - get image size for correct shape
            img_info = self.coco.loadImgs(image_id)[0]
            h, w = img_info["height"], img_info["width"]
            masks_array = np.zeros((0, h, w), dtype=np.uint8)

        # Ensure empty boxes have correct shape [0, 4] instead of [0]
        if boxes:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)

        target = {
            "boxes": boxes_tensor,
            "labels": (
                torch.as_tensor(labels, dtype=torch.int64)
                if labels
                else torch.zeros((0,), dtype=torch.int64)
            ),
            "masks": torch.as_tensor(masks_array, dtype=torch.uint8),
            "image_id": torch.as_tensor(image_id, dtype=torch.int64),
            "area": (
                torch.as_tensor(area, dtype=torch.float32)
                if area
                else torch.zeros((0,), dtype=torch.float32)
            ),
            "iscrowd": (
                torch.as_tensor(iscrowd, dtype=torch.int64)
                if iscrowd
                else torch.zeros((0,), dtype=torch.int64)
            ),
        }

        # Minimal transforms to tensor in [0,1]
        image = F.pil_to_tensor(image).float() / 255.0

        return image, target


# ----------------------------
# Data utils
# ----------------------------


def detection_collate(batch: List[Tuple[torch.Tensor, dict]]):
    images, targets = list(zip(*batch))
    return list(images), list(targets)


# ----------------------------
# Model factory (TorchVision 0.24+ API)
# ----------------------------


def build_model(num_classes: int) -> torch.nn.Module:
    from torchvision.models.detection import (
        maskrcnn_resnet50_fpn,
        MaskRCNN_ResNet50_FPN_Weights,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    # Load pretrained model with COCO weights (91 classes)
    # We'll replace the heads afterward to support our number of classes
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)

    # Replace the box predictor head for our number of classes
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

    # Replace the mask predictor head for our number of classes
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


# TODO check LR
def get_parameter_groups(
    model: torch.nn.Module,
    lr: float,
    box_head_lr_multiplier: float = 10.0,
    mask_head_lr_multiplier: float = 10.0,
):
    """
    Create parameter groups with differential learning rates for finetuning.

    Args:
        model: Mask R-CNN model
        lr: Base learning rate for pretrained backbone
        head_lr_multiplier: Multiplier for learning rate of new heads (default: 10x)

    Returns:
        List of parameter groups with different learning rates
    """
    # Collect parameters for different parts of the model
    backbone_params = []
    box_head_params = []
    mask_head_params = []
    other_params = []

    # Separate backbone parameters
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "backbone" in name:
            backbone_params.append(param)
        elif "box_predictor" in name:
            box_head_params.append(param)
        elif "mask_predictor" in name:
            mask_head_params.append(param)
        else:
            other_params.append(param)

    # Create parameter groups with different learning rates
    param_groups = [
        {"params": backbone_params, "lr": lr, "name": "backbone"},
        {
            "params": box_head_params,
            "lr": lr * box_head_lr_multiplier,
            "name": "box_head",
        },
        {
            "params": mask_head_params,
            "lr": lr * mask_head_lr_multiplier,
            "name": "mask_head",
        },
    ]

    # Add other params (RPN, etc.) with intermediate LR
    if other_params:
        param_groups.append(
            {
                "params": other_params,
                "lr": lr * 5.0,  # Intermediate LR for RPN
                "name": "other",
            }
        )

    # Log parameter counts
    print(f"\n📊 Parameter Groups:")
    print(
        f"  Backbone: {sum(p.numel() for p in backbone_params):,} params @ lr={lr:.2e}"
    )
    print(
        f"  Box Head: {sum(p.numel() for p in box_head_params):,} params @ lr={lr * box_head_lr_multiplier:.2e}"
    )
    print(
        f"  Mask Head: {sum(p.numel() for p in mask_head_params):,} params @ lr={lr * mask_head_lr_multiplier:.2e}"
    )
    if other_params:
        print(
            f"  Other (RPN): {sum(p.numel() for p in other_params):,} params @ lr={lr * 5.0:.2e}"
        )

    return param_groups


# ----------------------------
# Training, validation, evaluation
# ----------------------------


def _to_device(images, targets, device: torch.device):
    images = [img.to(device, non_blocking=True) for img in images]
    new_targets = []
    for t in targets:
        nt = {
            k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in t.items()
        }
        new_targets.append(nt)
    return images, new_targets


def _xyxy_to_xywh(x: np.ndarray) -> np.ndarray:
    y = x.copy()
    y[:, 2] = y[:, 2] - y[:, 0]
    y[:, 3] = y[:, 3] - y[:, 1]
    return y


@torch.no_grad()
def evaluate_coco(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    dataset: CocoSegmentationDataset,
    device: torch.device,
    mask_thresh: float = 0.5,
    max_dets: int = 100,
) -> dict:
    model.eval()
    bbox_res, segm_res = [], []

    for images, targets in tqdm(loader, desc="eval", leave=False):
        img_ids = [int(t["image_id"]) for t in targets]
        images = [img.to(device, non_blocking=True) for img in images]
        outputs = model(images)

        for img_id, out in zip(img_ids, outputs):
            if len(out["boxes"]) == 0:
                continue
            boxes = out["boxes"].detach().cpu().numpy()
            scores = out["scores"].detach().cpu().numpy()
            labels = out["labels"].detach().cpu().numpy()

            # Use a positive-stride ordering to avoid negative-stride issues
            # TODO why are we sorting by negative scores?
            order_np = np.argsort(-scores)[
                :max_dets
            ]  # no [::-1] view => no negative stride
            boxes, scores, labels = boxes[order_np], scores[order_np], labels[order_np]

            # map contiguous label => original COCO category_id
            labels_coco = [dataset.contig2catid[int(c)] for c in labels]

            for b, s, c in zip(boxes, scores, labels_coco):
                bbox_res.append(
                    {
                        "image_id": img_id,
                        "category_id": int(c),
                        "bbox": _xyxy_to_xywh(b[None, :])[0].tolist(),
                        "score": float(s),
                    }
                )

            if "masks" in out and len(out["masks"]) > 0:
                # Fix: properly extract masks - shape is [N, 1, H, W], so squeeze the channel dimension
                # Use torch LongTensor indexing to avoid negative-stride issues from numpy views (TODO why?)
                order_t = torch.as_tensor(
                    order_np, dtype=torch.long, device=out["masks"].device
                )
                masks_t = out["masks"].detach()[order_t].cpu()  # [N, 1, H, W] on CPU
                masks = masks_t.numpy()
                masks = (
                    masks.squeeze(1) >= mask_thresh
                )  # Now shape is [N, H, W], boolean (TODO why?)
                for m, s, c in zip(masks, scores, labels_coco):
                    rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
                    rle["counts"] = rle["counts"].decode("utf-8")
                    segm_res.append(
                        {
                            "image_id": img_id,
                            "category_id": int(c),
                            "segmentation": rle,
                            "score": float(s),
                        }
                    )

    def _eval(iou_type, results):
        if not results:
            return {"AP": float("nan"), "AP50": float("nan"), "AP75": float("nan")}
        # Ensure the COCO dataset has required 'info' field for loadRes
        # Some COCO files don't have it, so we need to add a minimal one
        if "info" not in dataset.coco.dataset:
            dataset.coco.dataset["info"] = {
                "description": "Car Parts Segmentation Dataset",
                "version": "1.0",
                "year": 2024,
            }
        coco_dt = dataset.coco.loadRes(results)
        ce = COCOeval(dataset.coco, coco_dt, iouType=iou_type)
        ce.evaluate()
        ce.accumulate()
        ce.summarize()
        s = ce.stats  # 0=AP, 1=AP50, 2=AP75
        return {"AP": float(s[0]), "AP50": float(s[1]), "AP75": float(s[2])}

    bbox = _eval("bbox", bbox_res)
    segm = _eval("segm", segm_res)
    return {"bbox": bbox, "segm": segm}


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None = None,
    log_every: int = 50,
    max_grad_norm: float = 20.0,
) -> Dict[str, float]:
    model.train()

    # Use a dictionary to track all losses
    running_losses = {}
    seen = 0

    pbar = tqdm(loader, desc="train", leave=False)

    for i, (images, targets) in enumerate(pbar, 1):
        images, targets = _to_device(images, targets, device)

        if scaler is not None:
            with torch.autocast(
                device_type=device.type,
                dtype=(
                    scaler._scale.dtype if hasattr(scaler, "_scale") else torch.float16
                ),
            ):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            # Gradient clipping (unscale first for accurate clipping)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

        bs = len(images)
        seen += bs

        # Update running losses
        if i == 1:
            # Initialize the dict on the first batch
            running_losses = {k: 0.0 for k in loss_dict.keys()}

        for k, v in loss_dict.items():
            running_losses[k] += v.item() * bs

        if i % log_every == 0:
            # Postfix the *total* loss
            total_loss_val = sum(running_losses.values())
            pbar.set_postfix(loss=total_loss_val / max(seen, 1))

    # Return the dictionary of mean losses
    mean_losses = {k: v / max(seen, 1) for k, v in running_losses.items()}
    return mean_losses


@torch.no_grad()
def validate_loss(model, loader, device):
    # For loss calculation, model needs to be in train mode (even though we're evaluating)
    # This is because detection models only return losses when in train mode
    was_training = model.training
    model.train()

    total, n = 0.0, 0
    for images, targets in tqdm(loader, desc="val_loss", leave=False):
        images, targets = _to_device(images, targets, device)
        # Model returns dict of losses when given targets in train mode
        loss_dict = model(images, targets)
        if isinstance(loss_dict, dict):
            loss = sum(loss_dict.values())
        else:
            raise ValueError(f"Expected loss dict, got {type(loss_dict)}")
        bs = len(images)
        total += float(loss.item()) * bs
        n += bs

    # Restore original training state
    if not was_training:
        model.eval()

    return total / max(n, 1)


def build_schedulers(optimizer, warmup_epochs: int, cosine_epochs: int):
    # Linear warmup then cosine annealing
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, total_iters=max(warmup_epochs, 1)
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cosine_epochs, 1)
    )
    return warmup, cosine


def fit(
    model,
    train_loader,
    val_loader,
    dataset,
    device,
    optimizer,
    schedulers,
    epochs: int,
    out_dir: str,
    writer=None,
    use_amp: bool = True,
    use_bf16: bool = False,
    early_stop_patience: int = 10,
    start_epoch: int = 0,
    tracker=None,  # Enhanced MLOps tracker (optional)
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    best_bbox_ap = -1.0
    epochs_no_improve = 0

    # AMP setup
    scaler = None
    if use_amp and device.type == "cuda":
        if use_bf16 and torch.cuda.is_bf16_supported():
            torch.set_float32_matmul_precision("high")  # allow faster kernels
            scaler = None  # bf16 does not need GradScaler
        else:
            scaler = torch.cuda.amp.GradScaler()

    sched_warmup, sched_cosine = schedulers

    for ep in range(start_epoch + 1, epochs + 1):
        print(f"\nEpoch {ep}/{epochs}")
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, scaler, max_grad_norm=10.0
        )

        # Print individual training losses for debugging
        print(f"📊 Training Losses:")
        total = sum(train_loss.values())
        print(f"   Total: {total:.4f}")
        for loss_name, loss_value in sorted(train_loss.items()):
            print(f"   {loss_name}: {loss_value:.4f}")

        # Step warmup then cosine
        if ep <= sched_warmup.total_iters:
            sched_warmup.step()
        else:
            sched_cosine.step()

        val_loss = validate_loss(model, val_loader, device)
        metrics = evaluate_coco(model, val_loader, dataset, device)

        # Print metrics summary
        print(f"📈 Validation Loss: {val_loss:.4f}")
        print(
            f"📦 Box mAP: {metrics['bbox']['AP']:.4f} (AP50: {metrics['bbox']['AP50']:.4f})"
        )
        print(
            f"🎭 Mask mAP: {metrics['segm']['AP']:.4f} (AP50: {metrics['segm']['AP50']:.4f})"
        )

        # Enhanced logging with MLOps tracker or fallback to basic writer
        if tracker is not None:
            # Use enhanced tracker (supports W&B, MLflow, etc.)
            from mlops_modernization import log_training_metrics

            # TODO clean up
            # Flatten everything into a single-level dict
            all_metrics = {"epoch": ep, "val_loss": val_loss}

            # Add train losses with prefix
            for loss_name, loss_value in train_loss.items():
                all_metrics[f"train_{loss_name}"] = loss_value

            # Add bbox metrics with prefix
            for metric_name, metric_value in metrics["bbox"].items():
                all_metrics[f"bbox_{metric_name}"] = metric_value

            # Add segm metrics with prefix
            for metric_name, metric_value in metrics["segm"].items():
                all_metrics[f"segm_{metric_name}"] = metric_value

            # Add learning rates with prefix
            for i, group in enumerate(optimizer.param_groups):
                group_name = group.get("name", f"group_{i}")
                all_metrics[f"lr_{group_name}"] = group["lr"]

            log_training_metrics(tracker, all_metrics)

        # Checkpoints
        checkpoint_path = out / f"epoch_{ep:03d}.pth"
        torch.save(
            {
                "epoch": ep,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            checkpoint_path,
        )
        print(f"💾 Saved checkpoint: {checkpoint_path}")

        # Track best by bbox AP
        curr_ap = metrics["bbox"]["AP"]
        if np.isfinite(curr_ap) and curr_ap > best_bbox_ap:
            best_bbox_ap = curr_ap
            epochs_no_improve = 0
            best_path = out / "best_bbox_ap.pth"
            torch.save(model.state_dict(), best_path)
            print(f"🏆 Saved new best model: {best_path} (bbox AP: {best_bbox_ap:.4f})")
        else:
            epochs_no_improve += 1

        # Early stopping
        if early_stop_patience > 0 and epochs_no_improve >= early_stop_patience:
            print(
                f"Early stopping: no bbox AP improvement for {early_stop_patience} epochs"
            )
            break

    # Save final metrics summary
    summary = {
        "best_bbox_AP": float(best_bbox_ap),
        "epochs_trained": ep,
    }
    with open(out / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
