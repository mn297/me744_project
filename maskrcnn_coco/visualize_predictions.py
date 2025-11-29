"""
Visualize Mask R-CNN predictions on images with bounding boxes and masks overlay.
Supports both matplotlib and FiftyOne visualization.
Supports YAML configuration.
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
import os
import yaml

try:
    import fiftyone as fo

    FIFTYONE_AVAILABLE = True
except ImportError:
    FIFTYONE_AVAILABLE = False
    print("Warning: FiftyOne not available. Install with: pip install fiftyone")

from core import CocoSegmentationDataset, build_model, detection_collate
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path):
    """Load configuration from YAML file."""
    if config_path is None:
        return {}

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def visualize_predictions_fiftyone(
    model: torch.nn.Module,
    images_dir: str,
    annotations_path: str,
    device: torch.device,
    score_threshold: float = 0.5,
    dataset_name: str = "maskrcnn_predictions",
    pytorch_dataset=None,  # Pass dataset to reuse
):
    """
    Visualize model predictions using FiftyOne.
    """
    if not FIFTYONE_AVAILABLE:
        raise ImportError(
            "FiftyOne is not installed. Install with: pip install fiftyone"
        )

    model.eval()
    print(f"Loading COCO dataset from {images_dir}...")

    # Load or create FiftyOne dataset
    try:
        existing_dataset = fo.load_dataset(dataset_name)
        print(
            f"[WARNING] Found existing dataset '{dataset_name}' - deleting it to load fresh data..."
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
    print(f"[SUCCESS] Created new dataset: {dataset_name} from {images_dir}")

    print(f"Loaded {len(coco_dataset)} images")
    print(f"Categories: {coco_dataset.info['categories']}")

    # Check if ground truth is loaded
    sample = coco_dataset.first()
    gt_field_lst = []
    for field_name in ["ground_truth", "detections", "segmentations", "coco"]:
        if hasattr(sample, field_name) and sample[field_name] is not None:
            gt_field_lst.append(field_name)
    if len(gt_field_lst) > 0:
        print(f"[SUCCESS] Ground truth loaded in fields: {gt_field_lst}")
        print(
            f"   You can toggle between ground truth and predictions in the FiftyOne app!"
        )
    else:
        print("[WARNING] Ground truth field not found. Check dataset loading.")

    # Load category mapping from annotations
    with open(annotations_path) as f:
        anno_data = json.load(f)
    cat_id_to_name = {cat["id"]: cat["name"] for cat in anno_data.get("categories", [])}

    # Use passed pytorch dataset or create new one
    if pytorch_dataset is None:
        pytorch_dataset = CocoSegmentationDataset(
            images_dir,
            annotations_path,
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
            full_mask_bool = mask_2d.astype(bool)

            # Resize mask logic
            mask_tensor = torch.from_numpy(full_mask_bool.astype(np.float32)).unsqueeze(
                0
            )  # Shape: [1, H, W]
            # Resize tensor to original image dimensions (NEAREST interpolation)
            resized_mask_tensor = F.resize(
                mask_tensor,
                [img_height, img_width],
                interpolation=F.InterpolationMode.NEAREST,
            )
            resized_full_mask = resized_mask_tensor.squeeze(0).numpy().astype(bool)

            # Crop instance mask
            x1, y1, x2, y2 = box.astype(int)
            # Clip to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_width, x2)
            y2 = min(img_height, y2)

            instance_mask = resized_full_mask[y1:y2, x1:x2]

            # Create detection with the instance mask
            detection = fo.Detection(
                label=cat_name,
                bounding_box=[x_norm, y_norm, w_norm, h_norm],
                confidence=float(score),
                mask=instance_mask,
            )
            detections.append(detection)

        # Add predictions to sample
        sample["predictions"] = fo.Detections(detections=detections)
        sample.save()

    print(f"\n[SUCCESS] Predictions added to dataset!")
    print(f"\n[INFO] Dataset fields:")
    print(f"   - Ground truth: '{gt_field_lst}' (if available)")
    print(f"   - Predictions: 'predictions'")
    print(f"\n[INFO] In FiftyOne app:")
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
        anno_file = getattr(dataset, "annotation_file", None) or getattr(
            dataset, "_annotation_file", None
        )
        if anno_file:
            with open(anno_file) as f:
                data = json.load(f)
                cat_names = {
                    cat["id"]: cat["name"] for cat in data.get("categories", [])
                }
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
            mask_binary = np.squeeze(mask).astype(bool)
            color = np.array([0, 255, 0])  # Bright green for GT masks
            gt_overlay[mask_binary] = gt_overlay[mask_binary] * 0.5 + color * 0.5

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
            ax1.imshow(gt_overlay.astype(np.uint8), alpha=1.0)

        # Right: Predictions
        ax2 = axes[1]
        ax2.imshow(image_np)
        ax2.set_title(
            f"Predictions (score >= {score_threshold})", fontsize=16, fontweight="bold"
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
            mask_binary = np.squeeze(mask).astype(bool)
            color = np.array([255, 105, 180])  # Bright pink for prediction masks
            pred_overlay[mask_binary] = pred_overlay[mask_binary] * 0.5 + color * 0.5

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
            ax2.imshow(pred_overlay.astype(np.uint8), alpha=1.0)

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


def main():
    # Pre-parse to get config
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config", type=str, default=None, help="Path to experiment config YAML"
    )
    args, remaining_argv = pre_parser.parse_known_args()

    # Load config
    config = load_config(args.config)
    dataset_cfg = config.get("dataset", {})
    out_cfg = config.get("output", {})

    # Helper to resolve paths relative to ROOT
    def resolve_path(path_str, default=None):
        if path_str:
            return str(ROOT / path_str)
        return default

    default_train_images = resolve_path(dataset_cfg.get("train_images"))
    default_train_anno = resolve_path(dataset_cfg.get("train_anno"))
    default_val_images = resolve_path(dataset_cfg.get("val_images"))
    default_val_anno = resolve_path(dataset_cfg.get("val_anno"))

    default_checkpoint_path = resolve_path(out_cfg.get("checkpoint_dir", "checkpoints"))
    if default_checkpoint_path:
        # Try to find best model if directory given
        if Path(default_checkpoint_path).is_dir():
            default_checkpoint_path = str(
                Path(default_checkpoint_path) / "best_bbox_ap.pth"
            )

    default_output_dir = resolve_path(out_cfg.get("output_dir", "visualizations"))
    default_experiment_name = out_cfg.get("experiment_name")

    parser = argparse.ArgumentParser(description="Visualize Mask R-CNN predictions")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to experiment config YAML"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=default_checkpoint_path,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--images",
        type=str,
        default=default_val_images,
        help="Path to images directory",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default=default_val_anno,
        help="Path to annotations JSON file",
    )
    parser.add_argument(
        "--train-images",
        type=str,
        default=default_train_images,
        help="Path to training images (for category mapping)",
    )
    parser.add_argument(
        "--train-anno",
        type=str,
        default=default_train_anno,
        help="Path to training annotations (for category mapping)",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=6,
        help="Number of images to visualize",
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
        default=default_output_dir,
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--use-fiftyone",
        action="store_true",
        help="Use FiftyOne for interactive visualization instead of matplotlib",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=default_experiment_name,
        help="Name for the FiftyOne dataset (only used with --use-fiftyone)",
    )
    args = parser.parse_args(remaining_argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    print("Loading dataset...")
    # We need to load training dataset first to get category mappings
    train_ds = CocoSegmentationDataset(
        args.train_images,
        args.train_anno,
    )
    val_ds = CocoSegmentationDataset(
        args.images,
        args.annotations,
        catid2contig=train_ds.catid2contig,
        contig2catid=train_ds.contig2catid,
    )
    # Store annotation file path for category name loading
    val_ds._annotation_file = args.annotations

    print(f"Dataset loaded: {len(val_ds)} images")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    if not args.checkpoint or not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return

    num_classes = val_ds.num_classes
    model = build_model(num_classes)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        epoch = checkpoint.get("epoch", "unknown")
        print(f"Loaded checkpoint from epoch {epoch}")
    else:
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
            dataset_name=args.dataset_name,
            pytorch_dataset=val_ds,  # Pass prepared dataset
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
        print(f"\n[SUCCESS] Visualizations saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
