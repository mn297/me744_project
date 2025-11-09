"""
Visualize Mask R-CNN predictions on images with bounding boxes and masks overlay.
Supports both matplotlib and FiftyOne visualization.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from PIL import Image
import torchvision.transforms.functional as F

try:
    import fiftyone as fo

    FIFTYONE_AVAILABLE = True
except ImportError:
    FIFTYONE_AVAILABLE = False
    print("Warning: FiftyOne not available. Install with: pip install fiftyone")

from utils import CocoSegmentationDataset, build_model, detection_collate
from torch.utils.data import DataLoader
from tqdm import tqdm


def visualize_predictions_fiftyone(
    model: torch.nn.Module,
    images_dir: str,
    annotations_path: str,
    device: torch.device,
    score_threshold: float = 0.5,
    dataset_name: str = "maskrcnn_predictions",
):
    """
    Visualize model predictions using FiftyOne.

    Args:
        model: Trained Mask R-CNN model
        images_dir: Path to images directory
        annotations_path: Path to COCO annotations JSON
        device: torch.device
        score_threshold: Minimum confidence score to display
        dataset_name: Name for the FiftyOne dataset
    """
    if not FIFTYONE_AVAILABLE:
        raise ImportError(
            "FiftyOne is not installed. Install with: pip install fiftyone"
        )

    model.eval()
    print(f"Loading COCO dataset from {images_dir}...")

    # Load or create FiftyOne dataset
    # Always create fresh dataset to avoid loading old cached data
    # Delete existing dataset if it exists
    try:
        existing_dataset = fo.load_dataset(dataset_name)
        print(
            f"⚠️  Found existing dataset '{dataset_name}' - deleting it to load fresh data..."
        )
        fo.delete_dataset(dataset_name)
    except:
        pass  # Dataset doesn't exist, which is fine

    # Create new dataset from COCO format
    coco_dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=images_dir,
        labels_path=annotations_path,
        include_id=True,
        name=dataset_name,
    )
    print(f"✅ Created new dataset: {dataset_name} from {images_dir}")

    print(f"Loaded {len(coco_dataset)} images")
    print(f"Categories: {coco_dataset.info['categories']}")

    # Check if ground truth is loaded (COCO datasets load GT automatically)
    sample = coco_dataset.first()
    gt_field_lst = []
    for field_name in ["ground_truth", "detections", "segmentations", "coco"]:
        if hasattr(sample, field_name) and sample[field_name] is not None:
            gt_field_lst.append(field_name)
    if len(gt_field_lst) > 0:
        print(f"✅ Ground truth loaded in fields: {gt_field_lst}")
        print(
            f"   You can toggle between ground truth and predictions in the FiftyOne app!"
        )
    else:
        print("⚠️  Warning: Ground truth field not found. Check dataset loading.")

    # Load category mapping from annotations
    with open(annotations_path) as f:
        anno_data = json.load(f)
    cat_id_to_name = {cat["id"]: cat["name"] for cat in anno_data.get("categories", [])}

    # Create a mapping from image file path to image_id for dataset lookup
    # We'll need to load the PyTorch dataset to get the same preprocessing
    pytorch_dataset = CocoSegmentationDataset(
        images_dir,
        annotations_path,
        is_train=False,
    )

    print(f"\nRunning inference on {len(coco_dataset)} images...")

    # Process each sample and add predictions
    for sample in tqdm(coco_dataset, desc="Processing images"):
        # Find corresponding image in PyTorch dataset
        image_id = sample.coco_id
        try:
            # Find index in dataset by image_id
            dataset_idx = pytorch_dataset.image_ids.index(image_id)
            image, target = pytorch_dataset[dataset_idx]
        except (ValueError, IndexError):
            print(f"Warning: Could not find image_id {image_id} in dataset")
            continue

        # Run inference
        with torch.no_grad():
            images_tensor = [image.to(device)]
            outputs = model(images_tensor)
            output = outputs[0]

        # Filter predictions by score
        keep = output["scores"] >= score_threshold
        boxes = output["boxes"][keep].cpu().numpy()
        labels = output["labels"][keep].cpu().numpy()
        scores = output["scores"][keep].cpu().numpy()
        masks = output["masks"][keep].cpu().numpy() > 0.5

        # Get image dimensions
        img_width = sample.metadata.width
        img_height = sample.metadata.height

        # Convert to FiftyOne format
        detections = []
        for box, label, score, mask in zip(boxes, labels, scores, masks):
            # Get category ID (map from contiguous to COCO category_id)
            cat_id = pytorch_dataset.contig2catid.get(int(label), int(label))
            cat_name = cat_id_to_name.get(cat_id, f"class_{cat_id}")

            # Convert box from [x1, y1, x2, y2] to normalized [x, y, width, height]
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1

            # Normalize to [0, 1] range for FiftyOne
            x_norm = x1 / img_width
            y_norm = y1 / img_height
            w_norm = width / img_width
            h_norm = height / img_height

            # Handle mask shape - FiftyOne expects 2D boolean array
            mask_2d = mask[0] if len(mask.shape) == 3 else mask
            mask_bool = mask_2d.astype(bool)

            # Create detection with mask
            detection = fo.Detection(
                label=cat_name,
                bounding_box=[x_norm, y_norm, w_norm, h_norm],
                confidence=float(score),
                mask=mask_bool,
            )
            detections.append(detection)

        # Add predictions to sample
        sample["predictions"] = fo.Detections(detections=detections)
        sample.save()

    print(f"\n✅ Predictions added to dataset!")
    print(f"\n📊 Dataset fields:")
    print(f"   - Ground truth: '{gt_field_lst}' (if available)")
    print(f"   - Predictions: 'predictions'")
    print(f"\n💡 In FiftyOne app:")
    print(
        f"   - Use the field selector to toggle between 'ground_truth' and 'predictions'"
    )
    print(f"   - You can view both simultaneously or switch between them")
    print(
        f"   - Ground truth boxes are typically shown in one color, predictions in another"
    )
    print(f"\nLaunching FiftyOne app...")

    # Launch FiftyOne app
    session = fo.launch_app(coco_dataset)
    print(f"\nFiftyOne app launched! Press Ctrl+C to exit.")

    # Keep the session alive
    try:
        session.wait()
    except KeyboardInterrupt:
        print("\nClosing FiftyOne session...")
        session.close()

    return coco_dataset


def visualize_predictions(
    model: torch.nn.Module,
    dataset: CocoSegmentationDataset,
    device: torch.device,
    num_images: int = 6,
    score_threshold: float = 0.5,
    output_dir: str = "visualizations",
):
    """
    Visualize model predictions on images using matplotlib.

    Args:
        model: Trained Mask R-CNN model
        dataset: CocoSegmentationDataset
        device: torch.device
        num_images: Number of images to visualize
        score_threshold: Minimum confidence score to display
        output_dir: Directory to save visualizations
    """
    model.eval()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get random image indices
    indices = np.random.choice(
        len(dataset), min(num_images, len(dataset)), replace=False
    )

    # Load category names if available
    cat_names = {}
    try:
        # Try to get annotation file path from dataset or use default
        anno_file = (
            getattr(dataset, "annotation_file", None)
            or getattr(dataset, "_annotation_file", None)
            or str(
                Path(__file__).parent
                / "Fuji-Apple-Segmentation/testset/annotations.json"
            )
        )
        with open(anno_file) as f:
            data = json.load(f)
            cat_names = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
    except Exception as e:
        print(f"Warning: Could not load category names: {e}")
        pass

    for idx, image_idx in enumerate(indices):
        # Get image and ground truth
        image, target = dataset[image_idx]
        image_id = dataset.image_ids[image_idx]

        # Get prediction
        with torch.no_grad():
            images_tensor = [image.to(device)]
            outputs = model(images_tensor)
            output = outputs[0]

        # Convert image to numpy for visualization
        # Image is in [0, 1] range from dataset, convert to [0, 255]
        image_np = image.permute(1, 2, 0).cpu().numpy()
        # Clip to [0, 1] range first, then convert to uint8
        image_np = np.clip(image_np, 0, 1)
        image_np = (image_np * 255).astype(np.uint8)

        # Filter predictions by score
        keep = output["scores"] >= score_threshold
        boxes = output["boxes"][keep].cpu().numpy()
        labels = output["labels"][keep].cpu().numpy()
        scores = output["scores"][keep].cpu().numpy()
        masks = output["masks"][keep].cpu().numpy() > 0.5

        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        # Left: Ground truth
        ax1 = axes[0]
        ax1.imshow(image_np)
        ax1.set_title("Ground Truth", fontsize=16, fontweight="bold")
        ax1.axis("off")

        # Draw ground truth boxes and masks
        gt_boxes = target["boxes"].cpu().numpy()
        gt_labels = target["labels"].cpu().numpy()
        gt_masks = target["masks"].cpu().numpy()

        # Create combined mask overlay for all objects
        gt_overlay = image_np.copy().astype(np.float32)

        for i, (box, label, mask) in enumerate(zip(gt_boxes, gt_labels, gt_masks)):
            # Get category name
            cat_id = dataset.contig2catid.get(int(label), int(label))
            cat_name = cat_names.get(cat_id, f"Class {cat_id}")

            # Draw bounding box
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="green",
                facecolor="none",
            )
            ax1.add_patch(rect)

            # Add mask to combined overlay
            mask_binary = mask.astype(bool)
            color = np.array(plt.cm.tab20(i % 20)[:3]) * 255
            for c in range(3):
                gt_overlay[:, :, c] = np.where(
                    mask_binary,
                    gt_overlay[:, :, c] * 0.6 + color[c] * 0.4,  # Blend with color
                    gt_overlay[:, :, c],
                )

            # Add label
            ax1.text(
                x1,
                y1 - 5,
                cat_name,
                fontsize=10,
                color="green",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        # Draw combined overlay once
        if len(gt_masks) > 0:
            ax1.imshow(gt_overlay.astype(np.uint8), alpha=0.7)

        # Right: Predictions
        ax2 = axes[1]
        ax2.imshow(image_np)
        ax2.set_title(
            f"Predictions (score ≥ {score_threshold})", fontsize=16, fontweight="bold"
        )
        ax2.axis("off")

        # Create combined mask overlay for all predictions
        pred_overlay = image_np.copy().astype(np.float32)

        # Draw predicted boxes and masks
        for i, (box, label, score, mask) in enumerate(
            zip(boxes, labels, scores, masks)
        ):
            # Get category name
            cat_id = dataset.contig2catid.get(int(label), int(label))
            cat_name = cat_names.get(cat_id, f"Class {cat_id}")

            # Draw bounding box
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            ax2.add_patch(rect)

            # Add mask to combined overlay
            mask_binary = mask[0].astype(bool)
            color = np.array(plt.cm.tab20(i % 20)[:3]) * 255
            for c in range(3):
                pred_overlay[:, :, c] = np.where(
                    mask_binary,
                    pred_overlay[:, :, c] * 0.6 + color[c] * 0.4,  # Blend with color
                    pred_overlay[:, :, c],
                )

            # Add label with score
            ax2.text(
                x1,
                y1 - 5,
                f"{cat_name} {score:.2f}",
                fontsize=10,
                color="red",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        # Draw combined overlay once
        if len(masks) > 0:
            ax2.imshow(pred_overlay.astype(np.uint8), alpha=0.7)

        # Add statistics
        stats_text = (
            f"Image ID: {image_id}\n"
            f"GT Objects: {len(gt_boxes)}\n"
            f"Pred Objects: {len(boxes)}"
        )
        fig.text(
            0.5,
            0.02,
            stats_text,
            ha="center",
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
        )

        plt.tight_layout()

        # Save figure
        save_path = output_path / f"prediction_{image_id:04d}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
        plt.close()


import os


ROOT = Path(__file__).resolve().parent
DEF = {
    "train_images": ROOT / "Fuji-Apple-Segmentation/trainingset/JPEGImages",
    "train_anno": ROOT / "Fuji-Apple-Segmentation/trainingset/annotations.json",
    "val_images": ROOT / "Fuji-Apple-Segmentation/testset/JPEGImages",
    "val_anno": ROOT / "Fuji-Apple-Segmentation/testset/annotations.json",
    "checkpoint": ROOT / "checkpoints/best_bbox_ap.pth",
    "output_dir": ROOT / "visualizations",
    "dataset_name": "Fuji-Apple-Segmentation",
}


def main():
    parser = argparse.ArgumentParser(description="Visualize Mask R-CNN predictions")
    parser.add_argument(
        "--checkpoint",
        type=str,
        # required=True,
        default=DEF["checkpoint"],
        help=f"Path to model checkpoint (e.g., {DEF['checkpoint']})",
    )
    parser.add_argument(
        "--images",
        type=str,
        default=DEF["val_images"],
        help=f"Path to images directory (e.g., {DEF['val_images']})",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default=DEF["val_anno"],
        help=f"Path to annotations JSON file (e.g., {DEF['val_anno']})",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=6,
        help=f"Number of images to visualize",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Minimum confidence score to display",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEF["output_dir"],
        help=f"Output directory for visualizations",
    )
    parser.add_argument(
        "--use-fiftyone",
        action="store_true",
        help="Use FiftyOne for interactive visualization instead of matplotlib",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,  # Will be auto-generated from dataset path
        help="Name for the FiftyOne dataset (only used with --use-fiftyone). Default: auto-generated",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    print("Loading dataset...")
    # We need to load training dataset first to get category mappings
    train_ds = CocoSegmentationDataset(
        DEF["train_images"],
        DEF["train_anno"],
        is_train=True,
    )
    val_ds = CocoSegmentationDataset(
        args.images,
        args.annotations,
        is_train=False,
        catid2contig=train_ds.catid2contig,
        contig2catid=train_ds.contig2catid,
    )
    # Store annotation file path for category name loading
    val_ds._annotation_file = args.annotations

    print(f"Dataset loaded: {len(val_ds)} images")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    num_classes = val_ds.num_classes
    model = build_model(num_classes)

    # Load checkpoint - handle both formats:
    # 1. Full checkpoint: {"epoch": ..., "model": ..., "optimizer": ...}
    # 2. Model state dict only: {model weights...}
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        # Full checkpoint format
        model.load_state_dict(checkpoint["model"])
        epoch = checkpoint.get("epoch", "unknown")
        print(f"Loaded checkpoint from epoch {epoch}")
    else:
        # Model state dict only (e.g., best_bbox_ap.pth)
        model.load_state_dict(checkpoint)
        print("Loaded model state dict")

    model.to(device)
    model.eval()
    print("Model loaded!")

    # Visualize
    if args.use_fiftyone:
        if not FIFTYONE_AVAILABLE:
            print(
                "Error: FiftyOne is not installed. Install with: pip install fiftyone"
            )
            return

        print(f"\nUsing FiftyOne for visualization...")
        # Auto-generate dataset name from path if not provided
        if args.dataset_name is None:
            dataset_path = Path(args.images).parent.name
            args.dataset_name = f"maskrcnn_{dataset_path}"
            print(f"Auto-generated dataset name: {args.dataset_name}")

        visualize_predictions_fiftyone(
            model,
            args.images,
            args.annotations,
            device,
            score_threshold=args.score_threshold,
            dataset_name=(
                "Fuji-Apple-Segmentation"
                if args.dataset_name is None
                else args.dataset_name
            ),
        )
    else:
        print(f"\nVisualizing {args.num_images} images with matplotlib...")
        visualize_predictions(
            model,
            val_ds,
            device,
            num_images=args.num_images,
            score_threshold=args.score_threshold,
            output_dir=args.output_dir,
        )
        print(f"\n✅ Visualizations saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
