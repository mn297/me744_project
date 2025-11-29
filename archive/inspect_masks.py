"""
Simple script to inspect masks in CocoSegmentationDataset.
Checks mask shapes, values, and visualizes them.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils import CocoSegmentationDataset


def inspect_dataset_masks(images_dir: str, anno_file: str, num_samples: int = 5):
    """Inspect masks in dataset."""

    print("=" * 80)
    print("LOADING DATASET")
    print("=" * 80)

    dataset = CocoSegmentationDataset(images_dir, anno_file, is_train=False)
    print(f"Dataset size: {len(dataset)} images")
    print(f"Number of classes: {dataset.num_classes}")
    print()

    # Inspect first few samples
    print("=" * 80)
    print(f"INSPECTING {num_samples} SAMPLES")
    print("=" * 80)

    samples_to_visualize = []

    for i in range(min(num_samples, len(dataset))):
        print(f"\n📸 Sample {i + 1}/{num_samples}")
        print("-" * 80)

        image, target = dataset[i]
        image_id = int(target["image_id"])

        # Image info
        print(f"Image ID: {image_id}")
        print(f"Image shape: {image.shape}")  # [C, H, W]

        # Masks info
        masks = target["masks"]
        boxes = target["boxes"]
        labels = target["labels"]

        print(f"Number of objects: {len(masks)}")

        if len(masks) == 0:
            print("⚠️  No masks in this image!")
            continue

        print(f"Masks tensor shape: {masks.shape}")  # [N, H, W]

        # Check each mask
        for j in range(len(masks)):
            mask = masks[j].numpy()
            box = boxes[j].numpy()
            label = labels[j].item()

            num_pixels = np.sum(mask > 0)
            total_pixels = mask.size
            percentage = (num_pixels / total_pixels) * 100

            print(f"\n  Object {j + 1}:")
            print(f"    Label: {label}")
            print(f"    Mask shape: {mask.shape}")
            print(f"    Mask dtype: {mask.dtype}")
            print(f"    Mask values: min={mask.min()}, max={mask.max()}")
            print(
                f"    Non-zero pixels: {num_pixels:,} / {total_pixels:,} ({percentage:.2f}%)"
            )
            print(
                f"    Bounding box: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]"
            )

            # Check if mask is all zeros
            if num_pixels == 0:
                print("    ❌ WARNING: Mask is all zeros!")
            else:
                print("    ✅ Mask has non-zero pixels")

        # Save for visualization
        samples_to_visualize.append((image, target))

    return samples_to_visualize


def visualize_masks(samples, save_dir: str = "mask_inspections"):
    """Visualize masks overlaid on images."""

    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    print("\n" + "=" * 80)
    print("VISUALIZING MASKS")
    print("=" * 80)

    for idx, (image, target) in enumerate(samples):
        image_id = int(target["image_id"])
        masks = target["masks"].numpy()
        boxes = target["boxes"].numpy()
        labels = target["labels"].numpy()

        if len(masks) == 0:
            continue

        # Convert image from [C, H, W] to [H, W, C] for plotting
        img_np = image.permute(1, 2, 0).numpy()

        # Create figure with subplots
        num_objects = len(masks)
        fig, axes = plt.subplots(1, num_objects + 1, figsize=(4 * (num_objects + 1), 4))

        if num_objects == 1:
            axes = [axes[0], axes[1]]

        # Plot original image
        axes[0].imshow(img_np)
        axes[0].set_title(f"Original Image (ID: {image_id})")
        axes[0].axis("off")

        # Plot each mask
        for i in range(num_objects):
            mask = masks[i]
            box = boxes[i]
            label = labels[i]

            # Show mask
            axes[i + 1].imshow(img_np)
            axes[i + 1].imshow(mask, alpha=0.5, cmap="jet")

            # Draw bounding box
            x1, y1, x2, y2 = box
            # rect = plt.Rectangle(
            #     (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=2
            # )
            # axes[i + 1].add_patch(rect)

            axes[i + 1].set_title(f"Object {i + 1} (Label: {label})")
            axes[i + 1].axis("off")

        plt.tight_layout()

        # Save figure
        output_file = save_path / f"sample_{idx + 1}_image_{image_id}.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"✅ Saved: {output_file}")
        plt.close()

    print(f"\n📁 All visualizations saved to: {save_path}/")


def check_dataset_statistics(images_dir: str, anno_file: str):
    """Check overall dataset statistics."""

    print("\n" + "=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)

    dataset = CocoSegmentationDataset(images_dir, anno_file, is_train=False)

    total_objects = 0
    total_masks = 0
    zero_masks = 0
    mask_sizes = []

    for i in range(len(dataset)):
        _, target = dataset[i]
        masks = target["masks"]

        total_objects += len(masks)

        for mask in masks:
            total_masks += 1
            mask_np = mask.numpy()
            num_pixels = np.sum(mask_np > 0)

            if num_pixels == 0:
                zero_masks += 1

            mask_sizes.append(num_pixels)

    print(f"\nTotal images: {len(dataset)}")
    print(f"Total objects: {total_objects}")
    print(f"Total masks: {total_masks}")
    print(f"Zero masks: {zero_masks}")

    if zero_masks > 0:
        print(
            f"⚠️  WARNING: {zero_masks} masks are all zeros ({zero_masks/total_masks*100:.2f}%)"
        )
    else:
        print("✅ All masks have non-zero pixels")

    if mask_sizes:
        mask_sizes = np.array(mask_sizes)
        print(f"\nMask size statistics (non-zero pixels):")
        print(f"  Mean: {mask_sizes.mean():.1f}")
        print(f"  Median: {np.median(mask_sizes):.1f}")
        print(f"  Min: {mask_sizes.min()}")
        print(f"  Max: {mask_sizes.max()}")
        print(f"  Std: {mask_sizes.std():.1f}")


def main():
    parser = argparse.ArgumentParser(description="Inspect masks in COCO dataset")
    parser.add_argument(
        "--images-dir",
        type=str,
        default="Fuji-Apple-Segmentation/testset/JPEGImages",
        help="Path to images directory",
    )
    parser.add_argument(
        "--anno-file",
        type=str,
        default="Fuji-Apple-Segmentation/testset/annotations.json",
        help="Path to annotations JSON file",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to inspect in detail",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize masks (saves to disk)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Compute full dataset statistics",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="mask_inspections",
        help="Directory to save visualizations",
    )

    args = parser.parse_args()

    # Inspect samples
    samples = inspect_dataset_masks(args.images_dir, args.anno_file, args.num_samples)

    # Visualize if requested
    if args.visualize:
        visualize_masks(samples, args.output_dir)

    # Compute statistics if requested
    if args.stats:
        check_dataset_statistics(args.images_dir, args.anno_file)

    print("\n" + "=" * 80)
    print("✅ INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
