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
import cv2

# uv run maskrcnn_coco/predict_fusion.py \
#   --source "datasets/Fuji-Apple-Segmentation_with_envy_mask_coco/np_imgs_val" \
#   --unet-checkpoint "datasets/checkpoints/unet/baseline.pth" \
#   --rcnn-checkpoint "datasets/checkpoints/Fuji-Apple-Segmentation/best_val_loss.pth" \
#   --output-dir "datasets/predictions/fusion_final" \
#   --save-npy

# Add paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
UNET_ROOT = PROJECT_ROOT / "unet" / "Pytorch-UNet"
sys.path.append(str(UNET_ROOT))

try:
    from unet import UNet
    from utils.data_loading import BasicDataset
    from core import build_model, CocoSegmentationDataset
    from torchvision.transforms import ToTensor
    import torchvision.transforms.functional as TF
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


def find_latest_checkpoint(checkpoint_dir: Path, extension=".pth") -> str:
    """Find the latest trained checkpoint in the provided directory."""
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    checkpoints = list(checkpoint_dir.glob(f"*{extension}"))

    if not checkpoints:
        raise FileNotFoundError(f"No *{extension} files found in {checkpoint_dir}")

    checkpoints.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return str(checkpoints[0])


def load_unet_model(checkpoint_path, device, n_classes=1):
    print(f"Loading UNet from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    if "mask_values" in state_dict:
        del state_dict["mask_values"]

    # Auto-detect classes
    if "outc.conv.weight" in state_dict:
        ckpt_classes = state_dict["outc.conv.weight"].shape[0]
        if ckpt_classes != n_classes:
            print(
                f"[UNet] Adjusting classes from {n_classes} to {ckpt_classes} based on checkpoint."
            )
            n_classes = ckpt_classes

    net = UNet(n_channels=3, n_classes=n_classes, bilinear=False)
    net.load_state_dict(state_dict, strict=False)
    net.to(device)
    net.eval()
    return net, n_classes


def load_rcnn_model(checkpoint_path, device, num_classes=2):
    print(f"Loading Mask R-CNN from {checkpoint_path}...")
    model = build_model(num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def predict_unet(net, img_pil, device, scale=0.5, threshold=0.5, n_classes=1):
    img_tensor = (
        torch.from_numpy(BasicDataset.preprocess(None, img_pil, scale, is_mask=False))
        .unsqueeze(0)
        .to(device, dtype=torch.float32)
    )

    with torch.no_grad():
        output = net(img_tensor).cpu()
        output = F.interpolate(
            output, (img_pil.size[1], img_pil.size[0]), mode="bilinear"
        )

        if n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > threshold

    return mask[0].long().squeeze().numpy()


def predict_rcnn(model, img_pil, device, threshold=0.5):
    transform = ToTensor()
    img_tensor = transform(img_pil).to(device).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)[0]

    # Extract masks
    if len(output["masks"]) > 0:
        keep = output["scores"] >= threshold
        masks = output["masks"][keep].detach().cpu().numpy()

        if len(masks) > 0:
            # Combine masks [N, 1, H, W] -> [H, W]
            masks = masks.squeeze(1)
            combined_mask = np.any(masks > 0.5, axis=0)
            return combined_mask

    return np.zeros((img_pil.size[1], img_pil.size[0]), dtype=bool)


def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray, num_classes=3):
    """
    Compute IoU and Dice per class.
    0: Background
    1: Apple (R-CNN)
    2: Branch (UNet)
    """
    ious = []
    dices = []

    for cls in range(num_classes):
        p = pred_mask == cls
        g = gt_mask == cls

        intersection = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()

        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union

        if p.sum() + g.sum() == 0:
            dice = 1.0
        else:
            dice = 2 * intersection / (p.sum() + g.sum())

        ious.append(iou)
        dices.append(dice)

    return ious, dices


def save_visualization(img_np, pred_mask, gt_mask, output_path, ious=None, dices=None):
    cols = 3 if gt_mask is not None else 2
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 5))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    # 1. Original
    axes[0].imshow(img_np)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # 2. GT
    if gt_mask is not None:
        axes[1].imshow(img_np)
        # Overlay: 1=Apple(Green), 2=Branch(Red)
        overlay = np.zeros((*gt_mask.shape, 4))
        overlay[gt_mask == 2] = [1, 0, 0, 0.4]  # Branch Red
        overlay[gt_mask == 1] = [0, 1, 0, 0.4]  # Apple Green
        axes[1].imshow(overlay)
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")
        idx_pred = 2
    else:
        idx_pred = 1

    # 3. Prediction
    axes[idx_pred].imshow(img_np)
    overlay = np.zeros((*pred_mask.shape, 4))
    overlay[pred_mask == 2] = [1, 0, 0, 0.4]  # Branch Red
    overlay[pred_mask == 1] = [0, 1, 0, 0.4]  # Apple Green
    axes[idx_pred].imshow(overlay)

    title = "Fused Prediction"
    if ious:
        title += f"\nIoU (App/Br): {ious[1]:.2f}/{ious[2]:.2f}"

    axes[idx_pred].set_title(title)
    axes[idx_pred].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Online Fusion Inference (UNet + Mask R-CNN)"
    )
    parser.add_argument("--source", type=str, default=None, help="Input directory")

    # UNet args
    parser.add_argument("--unet-checkpoint", type=str, default=None)
    parser.add_argument("--unet-scale", type=float, default=0.5)
    parser.add_argument("--unet-thresh", type=float, default=0.5)

    # R-CNN args
    parser.add_argument("--rcnn-checkpoint", type=str, default=None)
    parser.add_argument("--rcnn-thresh", type=float, default=0.5)

    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--save-npy", action="store_true")
    args = parser.parse_args()

    # 1. Resolve Paths
    if args.source:
        source_dir = Path(args.source)
        if (source_dir / "np_imgs_val").exists():
            img_dir = source_dir / "np_imgs_val"
            mask_dir = source_dir / "np_segs_val"  # Assuming multi-class GT here?
            # Actually UNet dataset usually has binary masks per class or 1-channel multi-class
            # Let's assume we want to eval against something.
            # If just raw images folder, mask_dir will be None
        else:
            img_dir = source_dir
            if "np_imgs" in str(img_dir):
                mask_dir = Path(str(img_dir).replace("np_imgs", "np_segs"))
            else:
                mask_dir = None  # Try finding labels folder? Not standard.
    else:
        # Default
        source_dir = (
            PROJECT_ROOT / "datasets" / "Fuji-Apple-Segmentation_with_envy_mask_coco"
        )
        img_dir = source_dir / "np_imgs_val"
        mask_dir = source_dir / "np_segs_val"

    if not img_dir.exists():
        print(f"Error: {img_dir} not found.")
        return

    # 2. Resolve Checkpoints
    # UNet
    if args.unet_checkpoint:
        unet_ckpt = Path(args.unet_checkpoint)
    else:
        unet_ckpt_dir = PROJECT_ROOT / "datasets" / "checkpoints" / "unet"
        unet_ckpt = Path(find_latest_checkpoint(unet_ckpt_dir))

    # R-CNN
    if args.rcnn_checkpoint:
        rcnn_ckpt = Path(args.rcnn_checkpoint)
    else:
        # Default location?
        rcnn_ckpt_dir = (
            PROJECT_ROOT / "datasets" / "checkpoints"
        )  # Mask RCNN usually here or in 'checkpoints'
        try:
            rcnn_ckpt = Path(find_latest_checkpoint(rcnn_ckpt_dir, extension=".pth"))
        except:
            # Fallback to specific path if known or error
            print(
                "Could not auto-resolve Mask R-CNN checkpoint. Please specify --rcnn-checkpoint."
            )
            return

    # 3. Output Dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = PROJECT_ROOT / "datasets" / "predictions" / f"fusion_{timestamp}"

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {out_dir}")

    # 4. Load Models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet_model, unet_n_classes = load_unet_model(unet_ckpt, device)
    # Assuming Mask R-CNN trained with default 91 classes or custom.
    # Need to know num_classes for build_model.
    # predict_maskrcnn.py usually infers it from dataset or defaults.
    # We'll assume 2 classes (Background + Apple) for Apple-specific RCNN?
    # Or 91 if COCO pretrained not finetuned?
    # User context: "apple inference". Likely 2 classes (0=bg, 1=apple) or (1=apple in 1-based index).
    # Let's assume num_classes=2 (bg + apple).
    rcnn_model = load_rcnn_model(rcnn_ckpt, device, num_classes=2)

    # 5. Infer
    img_files = sorted(
        list(img_dir.glob("*.npy"))
        + list(img_dir.glob("*.jpg"))
        + list(img_dir.glob("*.png"))
    )
    print(f"Found {len(img_files)} images")

    per_file_metrics = {}
    class_names = {0: "Background", 1: "Apple", 2: "Branch"}
    ious_all = []
    dices_all = []

    for img_path in tqdm(img_files, desc="Fusion Inference"):
        # Load Image
        if img_path.suffix == ".npy":
            img_arr = np.load(img_path)
            img_pil = Image.fromarray(img_arr)
        else:
            img_pil = Image.open(img_path).convert("RGB")
            img_arr = np.array(img_pil)

        # UNet Prediction (Branch)
        unet_mask = predict_unet(
            unet_model,
            img_pil,
            device,
            args.unet_scale,
            args.unet_thresh,
            unet_n_classes,
        )
        # unet_mask is likely 0/1. We want it to be Class 2 (Branch).

        # R-CNN Prediction (Apple)
        rcnn_mask = predict_rcnn(rcnn_model, img_pil, device, args.rcnn_thresh)
        # rcnn_mask is boolean. Treat True as Class 1 (Apple).

        # Merge
        # Initialize with 0
        final_mask = np.zeros_like(unet_mask, dtype=np.uint8)

        # Set Branches (2)
        final_mask[unet_mask == 1] = 2

        # Set Apples (1) - Overwrite branches
        final_mask[rcnn_mask] = 1

        # Load GT
        gt_mask = None
        if mask_dir and mask_dir.exists():
            gt_path = mask_dir / img_path.name
            if gt_path.exists():
                if gt_path.suffix == ".npy":
                    gt_mask = np.load(gt_path)
                else:
                    gt_mask = np.array(Image.open(gt_path))

                # Compute Metrics
                # Note: Check if GT matches our class mapping.
                # If GT is standard semantic mask: 0=bg, 1=branch, 2=apple?
                # Or separate files? Assuming single file with multi-class for now.
                ious, dices = compute_metrics(final_mask, gt_mask)
                ious_all.append(ious)
                dices_all.append(dices)

                per_file_metrics[img_path.name] = {
                    "iou": {class_names[i]: v for i, v in enumerate(ious)},
                    "dice": {class_names[i]: v for i, v in enumerate(dices)},
                }

        # Save
        save_visualization(
            img_arr,
            final_mask,
            gt_mask,
            out_dir / f"vis_{img_path.stem}.png",
            ious if gt_mask is not None else None,
            dices if gt_mask is not None else None,
        )

        if args.save_npy:
            np.save(out_dir / f"pred_{img_path.stem}.npy", final_mask)

    # Summary
    if ious_all:
        ious_all = np.array(ious_all)
        dices_all = np.array(dices_all)

        mean_ious = np.mean(ious_all, axis=0)
        mean_dices = np.mean(dices_all, axis=0)

        summary = {
            "metadata": {
                "unet_checkpoint": str(unet_ckpt),
                "rcnn_checkpoint": str(rcnn_ckpt),
                "source": str(source_dir),
                "timestamp": timestamp,
            },
            "per_class_mean": {
                "iou": {class_names[i]: mean_ious[i] for i in range(3)},
                "dice": {class_names[i]: mean_dices[i] for i in range(3)},
            },
            "overall_mean": {
                "iou": np.mean(mean_ious[1:]),
                "dice": np.mean(mean_dices[1:]),
            },
            "per_file": per_file_metrics,
        }

        with open(out_dir / "metrics.json", "w") as f:
            json.dump(summary, f, indent=4)

        print("\nResults:")
        print(f"Apple  IoU: {mean_ious[1]:.4f}, Dice: {mean_dices[1]:.4f}")
        print(f"Branch IoU: {mean_ious[2]:.4f}, Dice: {mean_dices[2]:.4f}")
        print(f"Overall IoU: {np.mean(mean_ious[1:]):.4f}")


if __name__ == "__main__":
    main()
