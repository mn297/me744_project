import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from tqdm import tqdm
import shutil
import os

# Define paths
BASE_DIR = Path(__file__).parent.parent  # Assuming script is in maskrcnn_coco/
DATASET_DIR = BASE_DIR / "datasets" / "Fuji-Apple-Segmentation_coco"
OUTPUT_DIR = BASE_DIR / "datasets" / "Fuji-Apple-Segmentation_unet"

# Input paths
TRAIN_JSON = DATASET_DIR / "trainingset" / "annotations.json"
TRAIN_IMG_DIR = DATASET_DIR / "trainingset" / "JPEGImages"
VAL_JSON = DATASET_DIR / "testset" / "annotations.json"
VAL_IMG_DIR = DATASET_DIR / "testset" / "JPEGImages"

# Output paths
# Following the structure of convert_envy_to_unet.py but adapted for pre-split data
NP_IMGDIR_TRAIN = OUTPUT_DIR / "np_imgs_train"
NP_SEGDIR_TRAIN = OUTPUT_DIR / "np_segs_train"
NP_IMGDIR_VAL = OUTPUT_DIR / "np_imgs_val"
NP_SEGDIR_VAL = OUTPUT_DIR / "np_segs_val"


def create_mask_from_polygons(image_shape, annotations):
    """
    Create a binary mask from a list of COCO annotations.

    Args:
        image_shape: (height, width)
        annotations: List of annotation dicts, each having a 'segmentation' field.
                     'segmentation' is a list of polygons [x1, y1, x2, y2, ...].

    Returns:
        mask: Numpy array of shape (height, width) with values 0 (bg) and 1 (fruit).
    """
    height, width = image_shape[:2]
    # Create an empty mask (0=background)
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    for ann in annotations:
        # COCO segmentation is a list of polygons (usually just one, but can be multiple for occluded objects)
        for polygon in ann["segmentation"]:
            # Polygon is [x1, y1, x2, y2, ...]
            # Convert to [(x1, y1), (x2, y2), ...]
            xy_poly = []
            for i in range(0, len(polygon), 2):
                xy_poly.append((polygon[i], polygon[i + 1]))

            if len(xy_poly) < 3:
                continue

            # Draw the polygon with value 1
            draw.polygon(xy_poly, outline=1, fill=1)

    return np.array(mask_img, dtype=np.uint8)


def process_split(json_path, img_dir, out_img_dir, out_seg_dir, desc="Processing"):
    """
    Process a dataset split (train or val).
    """
    # Ensure output directories exist
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_seg_dir.mkdir(parents=True, exist_ok=True)

    # Load annotations
    print(f"Loading annotations from {json_path}...")
    with open(json_path, "r") as f:
        coco_data = json.load(f)

    # Map image_id to annotations
    img_to_anns = {}
    for ann in coco_data["annotations"]:
        image_id = ann["image_id"]
        if image_id not in img_to_anns:
            img_to_anns[image_id] = []
        img_to_anns[image_id].append(ann)

    # Process images
    images = coco_data["images"]
    print(f"Found {len(images)} images in {json_path.name}")

    for img_info in tqdm(images, desc=desc):
        file_name = img_info["file_name"]
        image_id = img_info["id"]

        # Full path to source image
        src_img_path = img_dir / file_name

        if not src_img_path.exists():
            print(f"Warning: Image {src_img_path} not found. Skipping.")
            continue

        # Load image
        try:
            img = Image.open(src_img_path).convert("RGB")
            img_np = np.asarray(img)
        except Exception as e:
            print(f"Error reading image {src_img_path}: {e}")
            continue

        # Get annotations for this image
        anns = img_to_anns.get(image_id, [])

        # Create mask
        # Check if we have dimensions in img_info, otherwise use loaded image size
        height = img_info.get("height", img_np.shape[0])
        width = img_info.get("width", img_np.shape[1])

        # Double check against actual loaded image
        if height != img_np.shape[0] or width != img_np.shape[1]:
            # Use actual image dimensions
            height, width = img_np.shape[:2]

        mask_np = create_mask_from_polygons((height, width), anns)

        # Save as .npy
        # Base filename without extension
        stem = Path(file_name).stem

        img_out_path = out_img_dir / f"{stem}.npy"
        seg_out_path = out_seg_dir / f"{stem}.npy"

        np.save(img_out_path, img_np)
        np.save(seg_out_path, mask_np)


def main():
    print("Starting conversion from COCO to U-Net format...")

    if not TRAIN_JSON.exists():
        print(f"Error: Training annotations not found at {TRAIN_JSON}")
        return
    if not VAL_JSON.exists():
        print(f"Error: Validation annotations not found at {VAL_JSON}")
        return

    # Process Training Set
    process_split(
        TRAIN_JSON,
        TRAIN_IMG_DIR,
        NP_IMGDIR_TRAIN,
        NP_SEGDIR_TRAIN,
        desc="Converting Training Set",
    )

    # Process Validation Set (using testset as validation)
    process_split(
        VAL_JSON,
        VAL_IMG_DIR,
        NP_IMGDIR_VAL,
        NP_SEGDIR_VAL,
        desc="Converting Validation Set",
    )

    print("\nConversion complete!")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
