import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random


def visualize_samples(base_dir, split="train", num_samples=5):
    """
    Visualize random samples from the converted U-Net dataset.

    Args:
        base_dir (Path): Path to the U-Net dataset root.
        split (str): 'train' or 'val'.
        num_samples (int): Number of samples to visualize.
    """
    img_dir = base_dir / f"np_imgs_{split}"
    seg_dir = base_dir / f"np_segs_{split}"

    if not img_dir.exists() or not seg_dir.exists():
        print(f"Directories for {split} split not found.")
        return

    # Get all .npy files
    img_files = sorted(list(img_dir.glob("*.npy")))

    if not img_files:
        print(f"No files found in {img_dir}")
        return

    # Select random samples
    samples = random.sample(img_files, min(num_samples, len(img_files)))

    print(f"Visualizing {len(samples)} samples from {split} set...")

    for img_path in samples:
        # Construct corresponding seg path
        seg_path = seg_dir / img_path.name

        if not seg_path.exists():
            print(f"Segmentation file missing for {img_path.name}")
            continue

        # Load data
        img = np.load(img_path)
        mask = np.load(seg_path)

        # Create plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Sample: {img_path.stem}")

        # 1. Original Image
        axes[0].imshow(img)
        axes[0].set_title("Image")
        axes[0].axis("off")

        # 2. Mask (Binary)
        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title(f"Mask (Unique values: {np.unique(mask)})")
        axes[1].axis("off")

        # 3. Overlay
        axes[2].imshow(img)
        # Create a colored mask for overlay (e.g., red)
        colored_mask = np.zeros_like(img)
        colored_mask[:, :, 0] = 255  # Red channel

        # Create alpha channel based on mask
        # Mask is 0 or 1. We want 0 -> transparent, 1 -> semi-transparent
        overlay = np.zeros((*mask.shape, 4))
        overlay[mask == 1] = [1, 0, 0, 0.5]  # Red with 0.5 alpha

        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()


def main():
    base_dir = (
        # Path(__file__).parent.parent / "datasets" / "Fuji-Apple-Segmentation_unet"
        Path(__file__).parent.parent
        / "datasets"
        / "Fuji-Apple-Segmentation_with_envy_mask_coco"
    )

    print(f"Looking for data in: {base_dir}")

    # Visualize Training Samples
    visualize_samples(base_dir, split="train", num_samples=3)

    # Visualize Validation Samples
    visualize_samples(base_dir, split="val", num_samples=2)


if __name__ == "__main__":
    main()
