import json
import shutil
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2
from scipy.ndimage import label as scipy_label

# Paths
BASE_DIR = Path(__file__).parent.parent
DATASETS_DIR = BASE_DIR / "datasets"

# Source Datasets
# We use Fuji U-Net NPY files for easy apple extraction
FUJI_UNET_DIR = DATASETS_DIR / "Fuji-Apple-Segmentation_unet"
FUJI_IMGS_DIR = FUJI_UNET_DIR / "np_imgs_train"
FUJI_SEGS_DIR = FUJI_UNET_DIR / "np_segs_train"

ENVY_RAW_DIR = DATASETS_DIR / "image_envy_5000"

# Output Dataset
OUTPUT_DIR = DATASETS_DIR / "Fuji-Apple-Segmentation_with_envy_mask_coco"

# Configuration
NUM_ENVY_IMAGES = 100
TRAIN_SPLIT = 0.8
MIN_FRUITS = 0
MAX_FRUITS = 12


def setup_output_dir():
    if OUTPUT_DIR.exists():
        print(f"Removing existing output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    print(f"Copying Fuji dataset to {OUTPUT_DIR}...")
    # We copy the original Fuji COCO dataset as a base
    shutil.copytree(DATASETS_DIR / "Fuji-Apple-Segmentation_coco", OUTPUT_DIR)


def get_envy_images():
    all_files = list(ENVY_RAW_DIR.glob("*_rgb_*.png"))
    rgb_files = [f for f in all_files if "_label_" not in f.name]
    return sorted(rgb_files)


def get_fuji_samples() -> list[tuple[Path, Path]]:
    """Return list of (img_path, seg_path) tuples."""
    imgs = sorted(list(FUJI_IMGS_DIR.glob("*.npy")))
    segs = sorted(list(FUJI_SEGS_DIR.glob("*.npy")))
    return list(zip(imgs, segs))


def mask_to_polygons(mask: np.ndarray) -> list[list[float]]:
    """Convert binary mask to COCO polygons."""
    # cv2.RETR_EXTERNAL: retrieves only the extreme outer contours
    # cv2.CHAIN_APPROX_SIMPLE: compresses horizontal, vertical, and diagonal segments
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = []
    for contour in contours:
        if contour.size >= 6:
            polygons.append(contour.flatten().tolist())
    return polygons


def augment_dataset(
    json_path: Path,
    img_dir: Path,
    envy_files: list[Path],
    fuji_samples: list[tuple[Path, Path]],
):
    """
    Augment dataset by pasting Fuji apples onto Envy backgrounds.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    max_id = 0
    if data["images"]:
        max_id = max(img["id"] for img in data["images"])

    current_id = max_id + 1

    # Track annotation ID
    max_ann_id = 0
    if data["annotations"]:
        max_ann_id = max(ann["id"] for ann in data["annotations"])
    current_ann_id = max_ann_id + 1

    for envy_path in tqdm(envy_files, desc=f"Augmenting {json_path.parent.name}"):
        # 1. Load Envy Background
        # Convert to numpy RGB
        with Image.open(envy_path) as bg_pil:
            bg_img = np.array(bg_pil.convert("RGB"))

        bg_h, bg_w = bg_img.shape[:2]

        # 2. Determine number of apples to paste
        num_apples = random.randint(MIN_FRUITS, MAX_FRUITS)

        # Keep track of annotations for this image
        new_annotations = []

        for _ in range(num_apples):
            # 3. Pick random Fuji sample
            f_img_path, f_seg_path = random.choice(fuji_samples)
            f_img = np.load(f_img_path)
            f_seg = np.load(f_seg_path)

            # 4. Extract Apples
            # Find connected components (instances)
            labeled_array, num_features = scipy_label(f_seg > 0)

            if num_features == 0:
                continue

            # Pick random instance
            inst_idx = random.randint(1, num_features)
            inst_mask = labeled_array == inst_idx

            # Get bounding box of the apple
            ys, xs = np.where(inst_mask)
            if len(ys) == 0:
                continue

            y1, y2 = np.min(ys), np.max(ys)
            x1, x2 = np.min(xs), np.max(xs)
            h, w = y2 - y1 + 1, x2 - x1 + 1

            # Extract apple pixels
            apple_crop = f_img[y1 : y2 + 1, x1 : x2 + 1]
            mask_crop = inst_mask[y1 : y2 + 1, x1 : x2 + 1]

            # 5. Paste onto Envy
            # Random position
            # Ensure it fits
            if bg_h < h or bg_w < w:
                continue

            pos_y = random.randint(0, bg_h - h)
            pos_x = random.randint(0, bg_w - w)

            # Paste logic
            # Only update pixels where mask is True
            region_bg = bg_img[pos_y : pos_y + h, pos_x : pos_x + w]

            # Apply mask
            # Expand mask to 3 channels for RGB
            mask_3c = np.stack([mask_crop] * 3, axis=-1)

            # Blend (simple replacement)
            region_bg[mask_crop] = apple_crop[mask_crop]

            # Update background image
            bg_img[pos_y : pos_y + h, pos_x : pos_x + w] = region_bg

            # 6. Create Annotation
            # Create a full-sized mask for this new object to extract polygon
            full_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
            full_mask[pos_y : pos_y + h, pos_x : pos_x + w] = mask_crop.astype(np.uint8)

            polygons = mask_to_polygons(full_mask)
            if not polygons:
                continue

            # BBox [x, y, w, h]
            bbox = [float(pos_x), float(pos_y), float(w), float(h)]
            area = float(np.sum(mask_crop))

            new_annotations.append(
                {
                    "id": current_ann_id,
                    "image_id": current_id,
                    "category_id": 1,  # Fuji dataset uses 1 for apple
                    "segmentation": polygons,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                }
            )
            current_ann_id += 1

        # 7. Save augmented image
        new_filename = f"aug_{envy_path.stem}.jpg"
        save_path = img_dir / new_filename

        # Save as JPEG
        Image.fromarray(bg_img).save(save_path, quality=95)

        # 8. Add Image entry to JSON
        data["images"].append(
            {"id": current_id, "file_name": new_filename, "width": bg_w, "height": bg_h}
        )

        # Add annotations
        data["annotations"].extend(new_annotations)

        current_id += 1

    # Save JSON
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    print("=== Augmenting Fuji Dataset (Copy-Paste) ===")

    if not FUJI_UNET_DIR.exists():
        print(
            "Error: Fuji U-Net directory not found. Run convert_fuji_to_unet.py first?"
        )
        return

    setup_output_dir()

    envy_images = get_envy_images()
    fuji_samples = get_fuji_samples()

    print(f"Found {len(envy_images)} Envy images")
    print(f"Found {len(fuji_samples)} Fuji training samples for extraction")

    # Selection
    if len(envy_images) < NUM_ENVY_IMAGES:
        selected_envy = envy_images
    else:
        random.seed(42)
        selected_envy = random.sample(envy_images, NUM_ENVY_IMAGES)

    num_train = int(len(selected_envy) * TRAIN_SPLIT)
    train_envy = selected_envy[:num_train]
    test_envy = selected_envy[num_train:]

    print(f"Augmenting with {len(train_envy)} train and {len(test_envy)} test images")

    # Train
    train_json = OUTPUT_DIR / "trainingset" / "annotations.json"
    train_img_dir = OUTPUT_DIR / "trainingset" / "JPEGImages"
    augment_dataset(train_json, train_img_dir, train_envy, fuji_samples)

    # Test
    test_json = OUTPUT_DIR / "testset" / "annotations.json"
    test_img_dir = OUTPUT_DIR / "testset" / "JPEGImages"
    augment_dataset(test_json, test_img_dir, test_envy, fuji_samples)

    print("\n=== Done! ===")


if __name__ == "__main__":
    main()
