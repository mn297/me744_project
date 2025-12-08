import argparse
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add unet to sys.path
# Assuming maskrcnn_coco is current dir, unet is ../unet/Pytorch-UNet
PROJECT_ROOT = Path(__file__).resolve().parent.parent
UNET_ROOT = PROJECT_ROOT / "unet" / "Pytorch-UNet"
sys.path.append(str(UNET_ROOT))

try:
    from unet import UNet
    from utils.data_loading import BasicDataset
except ImportError as e:
    print(f"Error importing UNet modules: {e}")
    print(f"Make sure {UNET_ROOT} exists and contains the unet package.")
    sys.exit(1)


def find_latest_checkpoint(checkpoint_dir: Path) -> str:
    """Find the latest trained checkpoint in the provided directory."""
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    # Look for .pth or .pt files
    checkpoints = list(checkpoint_dir.glob("*.pth")) + list(checkpoint_dir.glob("*.pt"))

    if not checkpoints:
        raise FileNotFoundError(f"No .pth or .pt files found in {checkpoint_dir}")

    # Sort by modification time (latest first)
    checkpoints.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return str(checkpoints[0])


def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray):
    """
    Compute IoU and Dice for binary masks.
    pred_mask, gt_mask: boolean or 0/1 arrays
    """
    # Ensure binary
    pred = pred_mask > 0
    gt = gt_mask > 0

    if not np.any(gt):
        return -1.0, -1.0

    if not np.any(pred):
        return 0.0, 0.0

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return 1.0, 1.0

    iou = intersection / union
    dice = (2 * intersection) / (pred.sum() + gt.sum())

    return float(iou), float(dice)


def predict_single_image(net, full_img, device, scale_factor=1.0, out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(
        BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False)
    )
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(
            output, (full_img.size[1], full_img.size[0]), mode="bilinear"
        )
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def save_visualization(img_np, pred_mask, gt_mask, output_path, iou=None, dice=None):
    """
    Save visualization of Image, GT, Pred.
    """
    # Normalize if needed for display
    if img_np.dtype != np.uint8:
        if img_np.max() <= 1.0:
            display_img = (img_np * 255).astype(np.uint8)
        else:
            display_img = img_np.astype(np.uint8)
    else:
        display_img = img_np

    cols = 3 if gt_mask is not None else 2
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 5))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    # 1. Original
    axes[0].imshow(display_img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # 2. GT
    if gt_mask is not None:
        axes[1].imshow(display_img)
        # Overlay GT in Green
        overlay_gt = np.zeros((*gt_mask.shape, 4))
        overlay_gt[gt_mask == 2] = [0, 1, 0, 0.4]  # Green, alpha 0.4
        axes[1].imshow(overlay_gt)
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")
        idx_pred = 2
    else:
        idx_pred = 1

    # 3. Pred
    axes[idx_pred].imshow(display_img)
    # Overlay Pred in Red
    overlay_pred = np.zeros((*pred_mask.shape, 4))
    overlay_pred[pred_mask == 1] = [1, 0, 0, 0.4]  # Red, alpha 0.4
    axes[idx_pred].imshow(overlay_pred)

    title = "Prediction"
    if iou is not None and dice is not None:
        title += f"\nIoU: {iou:.3f}, Dice: {dice:.3f}"

    axes[idx_pred].set_title(title)
    axes[idx_pred].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run UNet inference and evaluation")
    parser.add_argument(
        "--source", type=str, default=None, help="Path to dataset root or images dir"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--scale", type=float, default=0.5, help="Scale factor for input images"
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Mask threshold")
    parser.add_argument(
        "--classes",
        type=int,
        default=1,
        help="Number of classes (default 1 for binary)",
    )
    parser.add_argument(
        "--save-npy",
        action="store_true",
        help="Save predicted masks as .npy files",
    )
    args = parser.parse_args()

    # 1. Resolve Paths
    if args.source:
        source_dir = Path(args.source)
        # If source is provided, assume it's either a dataset root or an images directory
        if (source_dir / "np_imgs_val").exists():
            img_dir = source_dir / "np_imgs_val"
            mask_dir = source_dir / "np_segs_val"
        else:
            # Assume source_dir IS the images dir
            img_dir = source_dir
            # Try to deduce mask dir (only if it follows the pattern)
            if "np_imgs" in str(img_dir):
                mask_dir = Path(str(img_dir).replace("np_imgs", "np_segs"))
            elif "images" in str(img_dir):
                mask_dir = Path(
                    str(img_dir).replace("images", "labels")
                )  # YOLO/Coco style
            else:
                mask_dir = None  # No GT or custom folder
    else:
        # Default to the dataset mentioned
        source_dir = (
            PROJECT_ROOT / "datasets" / "Fuji-Apple-Segmentation_with_envy_mask_coco"
        )
        img_dir = source_dir / "np_imgs_val"
        mask_dir = source_dir / "np_segs_val"

    if not img_dir.exists():
        print(f"Error: Images directory not found: {img_dir}")
        return

    print(f"Images directory: {img_dir}")
    print(f"Masks directory: {mask_dir}")

    # 2. Resolve Checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        # Default checkpoint dir
        checkpoint_dir = PROJECT_ROOT / "datasets" / "checkpoints" / "unet"
        try:
            checkpoint_path = Path(find_latest_checkpoint(checkpoint_dir))
            print(f"Resolved latest checkpoint: {checkpoint_path}")
        except FileNotFoundError as e:
            print(f"Error resolving checkpoint: {e}")
            print(
                "Please specify --checkpoint or ensure checkpoints exist in datasets/checkpoints/unet"
            )
            return

    # 3. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {checkpoint_path} on {device}...")

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=False)

    state_dict = torch.load(checkpoint_path, map_location=device)
    if "mask_values" in state_dict:
        del state_dict["mask_values"]

    # Detect number of classes from state_dict if possible
    if "outc.conv.weight" in state_dict:
        n_classes_checkpoint = state_dict["outc.conv.weight"].shape[0]
        if n_classes_checkpoint != args.classes:
            print(
                f"Warning: Checkpoint has {n_classes_checkpoint} classes, but args.classes is {args.classes}."
            )
            print(f"Adjusting model to use {n_classes_checkpoint} classes.")
            net = UNet(n_channels=3, n_classes=n_classes_checkpoint, bilinear=False)
            net.to(device)

    # Handle state dict mismatch if possible
    try:
        net.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading state dict: {e}")
        print("Trying with strict=False...")
        net.load_state_dict(state_dict, strict=False)

    net.to(device)
    print("Model loaded.")

    # 4. Prepare Output Directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    weights_name = checkpoint_path.stem
    output_dir = (
        PROJECT_ROOT
        / "datasets"
        / "checkpoints"
        / "unet"
        / f"inference_{timestamp}_{weights_name}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to {output_dir}")

    # 5. Run Inference
    img_files = sorted(list(img_dir.glob("*.npy")))
    if not img_files:
        # Try finding images if not npy
        img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))

    print(f"Found {len(img_files)} images.")

    per_file_metrics = {}
    total_iou = 0.0
    total_dice = 0.0
    valid_count = 0

    for img_path in tqdm(img_files, desc="Processing images"):
        # Load Image
        if img_path.suffix == ".npy":
            img_arr = np.load(img_path)
            pil_img = Image.fromarray(img_arr)
        else:
            pil_img = Image.open(img_path).convert("RGB")
            img_arr = np.array(pil_img)

        # Inference
        pred_mask = predict_single_image(
            net, pil_img, device, scale_factor=args.scale, out_threshold=args.threshold
        )

        # Load GT if available
        gt_mask = None
        iou = -1.0
        dice = -1.0

        if mask_dir and mask_dir.exists():
            gt_path = mask_dir / img_path.name
            if gt_path.exists():
                if gt_path.suffix == ".npy":
                    gt_mask = np.load(gt_path)
                else:
                    gt_mask = np.array(Image.open(gt_path))

                # Compute Metrics
                iou, dice = compute_metrics(pred_mask, gt_mask)

                per_file_metrics[img_path.name] = {
                    "branches": {"iou": iou, "dice": dice}
                }

                if iou != -1.0:
                    total_iou += iou
                    total_dice += dice
                    valid_count += 1

        # Save Visualization
        save_path = output_dir / f"pred_{img_path.stem}.png"
        save_visualization(
            img_arr,
            pred_mask,
            gt_mask,
            save_path,
            iou if iou != -1.0 else None,
            dice if dice != -1.0 else None,
        )

        if args.save_npy:
            np.save(output_dir / f"pred_{img_path.stem}.npy", pred_mask)

    # Save Metrics JSON
    avg_iou = total_iou / valid_count if valid_count > 0 else 0
    avg_dice = total_dice / valid_count if valid_count > 0 else 0

    metadata = {
        "model": "UNet",
        "checkpoint": str(checkpoint_path),
        "source": str(source_dir),
        "scale": args.scale,
        "timestamp": timestamp,
    }

    metrics_summary = {
        "metadata": metadata,
        "average": {
            "branches": {
                "average_iou": avg_iou,
                "average_dice": avg_dice,
                "valid_images": valid_count,
            }
        },
        "per_file": per_file_metrics,
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=4)

    print(f"[SUCCESS] Metrics and visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
