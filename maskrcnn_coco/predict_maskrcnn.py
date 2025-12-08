"""
Visualize Mask R-CNN predictions on images with bounding boxes and masks overlay.
Supports both matplotlib and FiftyOne visualization.
Supports YAML configuration.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.transforms import ToTensor
import os
import yaml
import pickle

try:
    import fiftyone as fo

    FIFTYONE_AVAILABLE = True
except ImportError:
    FIFTYONE_AVAILABLE = False
    print("Warning: FiftyOne not available. Install with: pip install fiftyone")

from core import CocoSegmentationDataset, build_model, detection_collate
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path):
    """Load configuration from YAML file."""
    if config_path is None:
        return {}

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_transform():
    """Get standard transforms for inference."""
    return ToTensor()


def load_model(checkpoint_path, num_classes, device):
    """Load trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    model = build_model(num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        epoch = checkpoint.get("epoch", "unknown")
        print(f"Loaded checkpoint from epoch {epoch}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model state dict")

    model.to(device)
    model.eval()
    return model


def predict_single_image(model, image_tensor, device) -> dict:
    """
    Run inference on a single image tensor.
    Args:
        model: Loaded model
        image_tensor: Tensor [C, H, W]
        device: torch device
    Returns:
        output dictionary from model
    """
    with torch.no_grad():
        images_tensor = [image_tensor.to(device)]
        outputs = model(images_tensor)
        return outputs[0]


def compute_metrics(pred_masks, target_masks):
    """
    Compute IoU and Dice treating all instances as one global mask.
    Assumes 1 mask per category per picture (foreground vs background).

    Args:
        pred_masks: boolean array [N, H, W] or [H, W]
        target_masks: boolean array [M, H, W] or [H, W]

    Returns:
        iou, dice
    """
    if len(target_masks) == 0:
        return -1.0, -1.0

    if len(pred_masks) == 0:
        return 0.0, 0.0

    # Collapse all masks into one binary mask per set (Union) if they are stacked
    if pred_masks.ndim == 3:
        global_pred = np.any(pred_masks, axis=0)
    else:
        global_pred = pred_masks

    if target_masks.ndim == 3:
        global_target = np.any(target_masks, axis=0)
    else:
        global_target = target_masks

    intersection = np.logical_and(global_pred, global_target).sum()
    union = np.logical_or(global_pred, global_target).sum()

    if union == 0:
        return 1.0, 1.0  # Both empty

    iou = intersection / union
    dice = (2 * intersection) / (global_pred.sum() + global_target.sum())

    return iou, dice


def draw_prediction(
    image_np, prediction, target=None, score_threshold=0.5, category_mapping=None
):
    """
    Draw predictions (and optionally ground truth) on an image using matplotlib.
    Returns a matplotlib Figure object.
    """
    if category_mapping is None:
        category_mapping = {}

    # Filter predictions
    keep = prediction["scores"] >= score_threshold
    boxes = prediction["boxes"][keep].cpu().numpy()
    labels = prediction["labels"][keep].cpu().numpy()
    scores = prediction["scores"][keep].cpu().numpy()
    masks = prediction["masks"][keep].cpu().numpy() > 0.5

    # Setup plot
    cols = 3 if target else 2  # Always show Original, Predicted. Add GT if available.
    if not target:
        # Fallback to original behavior if no target: Orig (inferred) + Pred
        # But actually the original code did: Pred (ax0) if no target.
        # Let's make it: [Original, Predicted] or [Original, Ground Truth, Predicted]
        cols = 2

    fig, axes = plt.subplots(1, cols, figsize=(6 * cols, 6))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    # 0. Original Image (Clean)
    ax_orig = axes[0]
    ax_orig.imshow(image_np)
    ax_orig.set_title("Original Image", fontsize=14, fontweight="bold")
    ax_orig.axis("off")

    if target:
        # 1. Ground Truth
        ax_gt = axes[1]
        ax_gt.imshow(image_np)
        ax_gt.set_title("Ground Truth", fontsize=14, fontweight="bold")
        ax_gt.axis("off")

        gt_boxes = target["boxes"].cpu().numpy()
        gt_labels = target["labels"].cpu().numpy()
        gt_masks = target["masks"].cpu().numpy()
        gt_overlay = image_np.copy().astype(np.float32)

        for box, label, mask in zip(gt_boxes, gt_labels, gt_masks):
            cat_name = category_mapping.get(int(label), f"Class {label}")
            x1, y1, x2, y2 = box

            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="green",
                facecolor="none",
            )
            ax_gt.add_patch(rect)

            mask_binary = np.squeeze(mask).astype(bool)
            color = np.array([0, 255, 0])  # Green
            gt_overlay[mask_binary] = gt_overlay[mask_binary] * 0.5 + color * 0.5

            ax_gt.text(
                x1,
                y1 - 5,
                cat_name,
                fontsize=9,
                color="green",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )

        if len(gt_masks) > 0:
            ax_gt.imshow(gt_overlay.astype(np.uint8), alpha=1.0)

        # 2. Predictions
        ax_pred = axes[2]
    else:
        # No target provided, so col 2 is predictions
        ax_pred = axes[1]

    # Draw Predictions (on ax_pred)
    ax_pred.imshow(image_np)
    ax_pred.set_title(
        f"Predictions (score >= {score_threshold})", fontsize=14, fontweight="bold"
    )
    ax_pred.axis("off")

    pred_overlay = image_np.copy().astype(np.float32)

    for box, label, score, mask in zip(boxes, labels, scores, masks):
        cat_name = category_mapping.get(int(label), f"Class {label}")
        x1, y1, x2, y2 = box

        # Box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="red", facecolor="none"
        )
        ax_pred.add_patch(rect)

        # Mask
        mask_binary = np.squeeze(mask).astype(bool)
        color = np.array([255, 105, 180])  # Pink
        pred_overlay[mask_binary] = pred_overlay[mask_binary] * 0.5 + color * 0.5

        # Label
        ax_pred.text(
            x1,
            y1 - 5,
            f"{cat_name} {score:.2f}",
            fontsize=9,
            color="red",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

    if len(masks) > 0:
        ax_pred.imshow(pred_overlay.astype(np.uint8), alpha=1.0)

    # Add Stats
    stats_text = f"Pred Objects: {len(boxes)}"
    if target:
        stats_text = f"GT Objects: {len(target['boxes'])}\n" + stats_text

        # Compute Metrics if we have predictions and ground truth
        if len(masks) > 0 and len(target["masks"]) > 0:
            # Ensure correct shapes and types
            # masks is [N, 1, H, W] (bool) from earlier processing
            pm = masks.copy()
            if pm.ndim == 4:
                pm = pm.squeeze(1)

            # target masks is [M, H, W] (uint8 or bool)
            tm = target["masks"].cpu().numpy()
            tm = tm > 0  # Ensure boolean

            iou, dice = compute_metrics(pm, tm)

            stats_text += f"\nGlobal IoU: {iou:.3f}"
            stats_text += f"\nGlobal Dice: {dice:.3f}"

    # Adjust layout to make room for text at the bottom
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)

    fig.text(
        0.5,
        0.02,
        stats_text,
        ha="center",
        va="bottom",
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
    )

    return fig


import scipy.ndimage

# RGBA of custom mask
# [216, 147, 236, 255]
APPLE_MASK_COLOR = [[226, 134, 234], [207, 125, 248]]


def construct_ground_truth(image_np, mask_color=APPLE_MASK_COLOR):
    """
    Construct ground truth dict from an image and a specific color mask.
    Args:
        image_np: Label image as numpy array [H, W, C] (RGB or RGBA)
        mask_color: List of [R, G, B] colors or single [R, G, B] to match
    Returns:
        target dict with boxes, labels, masks
    """
    # Ensure image is numpy
    img = np.array(image_np)

    # Handle RGB vs RGBA: just check first 3 channels
    if img.shape[-1] >= 3:
        img_rgb = img[:, :, :3]
    else:
        img_rgb = img

    # Normalize mask_color to be a list of colors
    # If mask_color is [R, G, B] (1D array/list), wrap it.
    # If it's [[R, G, B], ...] (2D), keep it.
    if len(np.array(mask_color).shape) == 1:
        target_colors = [mask_color]
    else:
        target_colors = mask_color

    # Create binary mask where ANY of the target colors match
    # Using small tolerance for potential compression artifacts
    # Cast to int16 to avoid uint8 overflow during subtraction
    img_int16 = img_rgb.astype(np.int16)

    # Initialize empty mask
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=bool)

    for color in target_colors:
        diff = np.abs(img_int16 - color)
        current_match = np.all(diff < 10, axis=-1)  # Tolerance of 10
        mask = np.logical_or(mask, current_match)

    if not np.any(mask):
        return {
            "boxes": torch.zeros((0, 4)),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "masks": torch.zeros((0, img.shape[0], img.shape[1]), dtype=torch.uint8),
        }

    # Connected components to separate instances
    labeled_mask, num_features = scipy.ndimage.label(mask)

    boxes = []
    masks = []
    labels = []

    for i in range(1, num_features + 1):
        # boolean mask for this instance
        instance_mask = labeled_mask == i

        # Find bounding box
        rows = np.any(instance_mask, axis=1)
        cols = np.any(instance_mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            continue

        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        # Box format: x1, y1, x2, y2
        boxes.append([x1, y1, x2 + 1, y2 + 1])
        masks.append(instance_mask)
        labels.append(1)  # Default class 1

    if not boxes:
        return {
            "boxes": torch.zeros((0, 4)),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "masks": torch.zeros((0, img.shape[0], img.shape[1]), dtype=torch.uint8),
        }

    # Convert to tensors
    target = {
        "boxes": torch.as_tensor(boxes, dtype=torch.float32),
        "labels": torch.as_tensor(labels, dtype=torch.int64),
        "masks": torch.as_tensor(np.stack(masks), dtype=torch.uint8),  # [N, H, W]
    }

    return target


def visualize_custom_folder(
    model,
    folder_path,
    device,
    output_dir,
    score_threshold=0.5,
    category_mapping=None,
    num_images=None,
):
    """Visualize predictions on raw images from a folder."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    transform = get_transform()

    folder = Path(folder_path)
    image_files = sorted(
        [
            f
            for f in folder.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        ]
    )

    if num_images:
        import random

        image_files = random.sample(image_files, min(num_images, len(image_files)))

    print(f"Processing {len(image_files)} images from {folder_path}...")

    for img_path in tqdm(image_files):
        # Skip label files if they are in the same folder and end with _label
        if "_label" in img_path.name:
            continue

        # Load and transform
        original_img = Image.open(img_path).convert("RGB")
        image_tensor = transform(original_img)

        # Check for corresponding label file
        label_path = img_path.parent / f"{img_path.stem}_label.png"

        target = None
        if label_path.exists():
            label_img = np.array(Image.open(label_path).convert("RGB"))
            target = construct_ground_truth(label_img)

        # Inference
        output = predict_single_image(model, image_tensor, device)

        # Prepare for vis
        image_np = np.array(original_img)

        # Draw
        fig = draw_prediction(
            image_np,
            output,
            target=target,
            score_threshold=score_threshold,
            category_mapping=category_mapping,
        )

        # Save
        save_name = output_path / f"pred_{img_path.name}"
        fig.savefig(save_name, dpi=100, bbox_inches="tight")
        plt.close(fig)

        # Prepare combined output data
        output["boxes"] = output["boxes"].cpu().detach().numpy()
        output["labels"] = output["labels"].cpu().detach().numpy()
        output["scores"] = output["scores"].cpu().detach().numpy()
        output["masks"] = output["masks"].cpu().detach().numpy()

        file_path = output_path / f"pred_{img_path.name}.pkl"
        with open(file_path, "wb") as f:
            pickle.dump((output), f)

    print(f"[SUCCESS] Visualizations saved to {output_dir}")


def visualize_coco_dataset(
    model,
    dataset,
    device,
    output_dir,
    num_images=6,
    score_threshold=0.5,
    category_mapping=None,
    metadata=None,
):
    """Visualize predictions on a COCO dataset (with ground truth) and save metrics."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if num_images >= len(dataset):
        indices = range(len(dataset))
    else:
        indices = np.random.choice(len(dataset), num_images, replace=False)

    print(f"Visualizing and evaluating {len(indices)} images from dataset...")

    per_file_metrics = {}
    total_iou = 0.0
    total_dice = 0.0
    valid_count = 0

    for idx in tqdm(indices):
        image_tensor, target = dataset[idx]
        image_id = dataset.image_ids[idx]

        # Inference
        output = predict_single_image(model, image_tensor, device)

        # Prepare image np [0, 255]
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        image_np = np.clip(image_np, 0, 1)
        image_np = (image_np * 255).astype(np.uint8)

        # Get Filename
        try:
            file_name = dataset.coco.loadImgs(image_id)[0]["file_name"]
            file_stem = Path(file_name).stem
        except:
            file_name = f"image_{image_id}.png"
            file_stem = f"{image_id}"

        # Calculate Metrics
        # Filter predictions by score
        keep = output["scores"] >= score_threshold
        pred_masks = output["masks"][keep].cpu().numpy() > 0.5
        # Squeeze channel dim if present [N, 1, H, W] -> [N, H, W]
        if pred_masks.ndim == 4:
            pred_masks = pred_masks.squeeze(1)

        target_masks = target["masks"].cpu().numpy() > 0

        iou, dice = compute_metrics(pred_masks, target_masks)

        per_file_metrics[file_name] = {
            "apple": {"iou": float(iou), "dice": float(dice)}
        }

        if iou != -1.0:
            total_iou += iou
            total_dice += dice
            valid_count += 1

        # Draw
        fig = draw_prediction(
            image_np,
            output,
            target=target,
            score_threshold=score_threshold,
            category_mapping=category_mapping,
        )

        save_name = output_path / f"pred_{file_stem}.png"
        fig.savefig(save_name, dpi=100, bbox_inches="tight")
        plt.close(fig)

        # Prepare combined output data
        output["boxes"] = output["boxes"].cpu().detach().numpy()
        output["labels"] = output["labels"].cpu().detach().numpy()
        output["scores"] = output["scores"].cpu().detach().numpy()
        output["masks"] = output["masks"].cpu().detach().numpy()

        file_path = output_path / f"pred_{file_stem}.pkl"
        with open(file_path, "wb") as f:
            pickle.dump((output), f)

    # Save Metrics JSON
    avg_iou = total_iou / valid_count if valid_count > 0 else 0
    avg_dice = total_dice / valid_count if valid_count > 0 else 0

    metrics_summary = {
        "metadata": metadata if metadata else {},
        "average": {
            "apple": {
                "average_iou": float(avg_iou),
                "average_dice": float(avg_dice),
                "valid_images": valid_count,
            }
        },
        "per_file": per_file_metrics,
    }

    with open(output_path / "metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=4)

    print(f"[SUCCESS] Visualizations and metrics saved to {output_dir}")


def visualize_predictions_fiftyone(
    model,
    images_dir,
    annotations_path,
    device,
    score_threshold=0.5,
    dataset_name="maskrcnn_predictions",
    pytorch_dataset=None,
):
    # ... (FiftyOne code remains largely the same, just wrapped) ...
    # Kept existing logic for FiftyOne as it's tightly coupled to COCO format
    if not FIFTYONE_AVAILABLE:
        raise ImportError("FiftyOne is not installed.")

    model.eval()
    print(f"Loading COCO dataset from {images_dir}...")

    try:
        if fo.dataset_exists(dataset_name):
            print(f"[WARNING] Deleting existing dataset '{dataset_name}'...")
            fo.delete_dataset(dataset_name)
    except:
        pass

    coco_dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=images_dir,
        labels_path=annotations_path,
        include_id=True,
        name=dataset_name,
    )

    # Load category mapping
    with open(annotations_path) as f:
        anno_data = json.load(f)
    cat_id_to_name = {cat["id"]: cat["name"] for cat in anno_data.get("categories", [])}

    if pytorch_dataset is None:
        pytorch_dataset = CocoSegmentationDataset(images_dir, annotations_path)

    print(f"\nRunning inference on {len(coco_dataset)} images...")

    for sample in tqdm(coco_dataset, desc="Processing"):
        image_id = sample.coco_id
        try:
            dataset_idx = pytorch_dataset.image_ids.index(image_id)
            image, _ = pytorch_dataset[dataset_idx]
        except (ValueError, IndexError):
            continue

        output = predict_single_image(model, image, device)

        keep = output["scores"] >= score_threshold
        boxes = output["boxes"][keep].cpu().numpy()
        labels = output["labels"][keep].cpu().numpy()
        scores = output["scores"][keep].cpu().numpy()
        masks = output["masks"][keep].cpu().numpy() > 0.5

        img_width = sample.metadata.width
        img_height = sample.metadata.height

        detections = []
        for box, label, score, mask in zip(boxes, labels, scores, masks):
            cat_id = pytorch_dataset.contig2catid.get(int(label), int(label))
            cat_name = cat_id_to_name.get(cat_id, f"class_{cat_id}")

            x1, y1, x2, y2 = box
            width, height = x2 - x1, y2 - y1

            # Mask resizing logic (same as before)
            mask_tensor = (
                torch.from_numpy(mask[0].astype(np.float32)).unsqueeze(0)
                if len(mask.shape) == 3
                else torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
            )
            resized_mask = (
                F.resize(
                    mask_tensor,
                    [img_height, img_width],
                    interpolation=F.InterpolationMode.NEAREST,
                )
                .squeeze(0)
                .numpy()
                .astype(bool)
            )

            x1_i, y1_i, x2_i, y2_i = map(int, box)
            x1_i, y1_i = max(0, x1_i), max(0, y1_i)
            x2_i, y2_i = min(img_width, x2_i), min(img_height, y2_i)
            instance_mask = resized_mask[y1_i:y2_i, x1_i:x2_i]

            detections.append(
                fo.Detection(
                    label=cat_name,
                    bounding_box=[
                        x1 / img_width,
                        y1 / img_height,
                        width / img_width,
                        height / img_height,
                    ],
                    confidence=float(score),
                    mask=instance_mask,
                )
            )

        sample["predictions"] = fo.Detections(detections=detections)
        sample.save()

    session = fo.launch_app(coco_dataset)
    print("\nFiftyOne app launched! Press Ctrl+C to exit.")
    try:
        session.wait()
    except KeyboardInterrupt:
        session.close()


def main():
    current_dir = Path(__file__).resolve().parent
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config", type=str, default=current_dir / "experiment_fuji.yaml"
    )
    args, remaining_argv = pre_parser.parse_known_args()

    config = load_config(args.config)
    dataset_cfg = config.get("dataset", {})
    out_cfg = config.get("output", {})

    def resolve_path(path_str, default=None):
        if path_str:
            return str(PROJECT_ROOT / path_str)
        return default

    # Defaults
    default_val_images = resolve_path(dataset_cfg.get("val_images"))
    default_val_anno = resolve_path(dataset_cfg.get("val_anno"))
    default_checkpoint = resolve_path(out_cfg.get("checkpoint_dir", "checkpoints"))
    if default_checkpoint and Path(default_checkpoint).is_dir():
        default_checkpoint = str(Path(default_checkpoint) / "best_val_loss.pth")
    default_out_dir = resolve_path(out_cfg.get("output_dir", "predictions"))

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--checkpoint", type=str, default=default_checkpoint)

    # Modes
    parser.add_argument(
        "--custom-folder",
        type=str,
        help="Path to folder of images for manual inference",
    )
    parser.add_argument(
        "--images", type=str, default=default_val_images, help="COCO images dir"
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default=default_val_anno,
        help="COCO annotations json",
    )

    parser.add_argument("--num-images", type=int, default=6)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=str, default=default_out_dir)
    parser.add_argument("--use-fiftyone", action="store_true")
    parser.add_argument(
        "--dataset-name", type=str, default=out_cfg.get("experiment_name")
    )

    # Training args needed for mapping
    parser.add_argument(
        "--train-images",
        type=str,
        default=resolve_path(dataset_cfg.get("train_images")),
    )
    parser.add_argument(
        "--train-anno", type=str, default=resolve_path(dataset_cfg.get("train_anno"))
    )

    args = parser.parse_args(remaining_argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Category Mapping (try to load from training annotations if available)
    category_mapping = {}
    anno_path = args.train_anno or args.annotations
    if anno_path and os.path.exists(anno_path):
        with open(anno_path) as f:
            data = json.load(f)
            # Map category ID to name
            category_mapping = {
                cat["id"]: cat["name"] for cat in data.get("categories", [])
            }
            # If contiguous mapping is needed, we might need the dataset logic,
            # but for custom images we assume the model output labels match COCO ids or we map them.
            # NOTE: MaskRCNN typically predicts labels 1..N.
            # CocoSegmentationDataset re-maps them to 1..N contiguous.
            # We'll assume for custom viz that we just want names if possible.

    # Determine num_classes (hacky if we don't have dataset loaded)
    # We can infer from category mapping len + 1 (background)
    num_classes = len(category_mapping) + 1 if category_mapping else 91  # Default COCO

    # Better: Load training dataset to get exact config if we are not in custom mode or if we want to be safe
    if args.train_anno:
        # We load train dataset just to get num_classes and mappings exactly right
        train_ds = CocoSegmentationDataset(args.train_images, args.train_anno)
        num_classes = train_ds.num_classes
        # Update mapping to handle contiguous ids
        # contig_id -> cat_id -> name
        real_mapping = {}
        for contig_id, cat_id in train_ds.contig2catid.items():
            real_mapping[contig_id] = category_mapping.get(cat_id, str(cat_id))
        category_mapping = real_mapping

    # 2. Load Model
    model = load_model(args.checkpoint, num_classes, device)

    # 3. Execution Mode
    if args.custom_folder:
        # Custom Folder Mode
        print(f"Running on custom folder: {args.custom_folder}")
        output_dir = args.output_dir / Path("custom_folder")
        output_dir.mkdir(parents=True, exist_ok=True)
        visualize_custom_folder(
            model,
            args.custom_folder,
            device,
            str(output_dir),
            args.score_threshold,
            category_mapping,
            args.num_images,
        )
    elif args.use_fiftyone:
        # FiftyOne Mode
        visualize_predictions_fiftyone(
            model,
            args.images,
            args.annotations,
            device,
            args.score_threshold,
            args.dataset_name,
        )
    else:
        # COCO Dataset Mode (Default)
        print("Running on COCO dataset validation set...")
        # We reuse the train_ds logic to ensure consistency if possible, but we need val_ds
        val_ds = CocoSegmentationDataset(
            args.images,
            args.annotations,
            # Reuse mappings if we loaded train_ds
            catid2contig=train_ds.catid2contig if "train_ds" in locals() else None,
            contig2catid=train_ds.contig2catid if "train_ds" in locals() else None,
        )

        metadata = {
            "model": "Mask R-CNN",
            "checkpoint": args.checkpoint,
            "score_threshold": args.score_threshold,
            "images_dir": args.images,
            "config": str(args.config) if args.config else None,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }

        visualize_coco_dataset(
            model,
            val_ds,
            device,
            args.output_dir,
            args.num_images,
            args.score_threshold,
            category_mapping,
            metadata=metadata,
        )


if __name__ == "__main__":
    main()
