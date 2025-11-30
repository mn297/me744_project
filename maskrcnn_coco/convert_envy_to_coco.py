# %%
import os
import json
import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage.measure import label as cc_label
from pycocotools import mask as maskUtils
import matplotlib.pyplot as plt
import cv2

# ---------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
RAW_DIR = (
    BASE_DIR / "datasets" / "image_envy_5000"
)  # where all *_rgb_*, *_label_rgb_* live
OUT_DIR = (
    BASE_DIR / "datasets" / "image_envy_5000_coco"
)  # will contain JPEGImages/ and annotations.json
TRAIN_DIR = OUT_DIR / "trainingset"
TEST_DIR = OUT_DIR / "testset"

# ---------------------------------------------------------------------
# label colors (RGBA)  ---> 2 categories: trunk (id=1), branches (id=2)
# ---------------------------------------------------------------------
color_dict = {
    "trunk": [255, 0, 0, 255],
    "trunk_2": [255, 1, 1, 255],
    # "trunk_3": [0, 137, 137, 255],
    # "trunk_4": [1, 137, 137, 255],
    "branches": [255, 255, 0, 255],
    "branches_2": [255, 255, 1, 255],
    "branches_3": [0, 255, 0, 255],
    "branches_4": [1, 255, 1, 255],
}

CATEGORY_DEFS = [
    # ("trunk", 1, [color_dict["trunk"], color_dict["trunk_2"]]),
    # ("branches", 2, [color_dict["branches"], color_dict["branches_2"]]),
    (
        "branches",
        2,
        [
            color_dict["trunk"],
            color_dict["trunk_2"],
            color_dict["branches"],
            color_dict["branches_2"],
            color_dict["branches_3"],
            color_dict["branches_4"],
        ],
    ),
]


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def build_mask(label_arr, colors):
    """
    label_arr: H x W x 4 uint8 (RGBA)
    colors: list of [R,G,B,A]
    returns: H x W uint8 mask (0 or 1)
    """
    h, w, _ = label_arr.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for c in colors:
        c_arr = np.array(c, dtype=np.uint8)
        match = np.all(label_arr == c_arr, axis=-1)
        mask |= match.astype(np.uint8)
    return mask


def process_dataset(image_paths, output_dir, set_name):
    output_img_dir = output_dir / "JPEGImages"
    output_img_dir.mkdir(parents=True, exist_ok=True)
    annotation_file = output_dir / "annotations.json"

    images = []
    annotations = []
    ann_id = 1

    print(f"Processing {set_name} set: {len(image_paths)} images...")

    for img_id, label_path in enumerate(tqdm(image_paths), start=1):
        # derive matching rgb filename:  0073_label_rgb_0001.png -> 0073_rgb_0001.png
        rgb_name = label_path.name.replace("_label", "")
        rgb_path = RAW_DIR / rgb_name
        if not rgb_path.exists():
            print(f"Warning: RGB file not found for {label_path.name}")
            continue

        # load RGB image and save as JPEG
        # Using CV2 for speed
        rgb_img = cv2.imread(str(rgb_path))
        if rgb_img is None:
            print(f"Warning: Could not read image {rgb_path}")
            continue
        height, width = rgb_img.shape[:2]

        jpg_name = Path(rgb_name).stem + ".jpg"
        jpg_path = output_img_dir / jpg_name

        # Write image
        cv2.imwrite(str(jpg_path), rgb_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        images.append(
            {
                "id": img_id,
                "file_name": jpg_name,
                "width": width,
                "height": height,
            }
        )

        # load label image as RGBA
        label_img = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
        if label_img is None:
            print(f"Warning: Could not read label {label_path}")
            continue

        # Convert BGR(A) to RGB(A) for color matching
        if label_img.shape[2] == 4:
            label_img = cv2.cvtColor(label_img, cv2.COLOR_BGRA2RGBA)
        elif label_img.shape[2] == 3:
            label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB)
            # Add dummy alpha if needed, but logic below expects 4 channels
            alpha = np.full((height, width), 255, dtype=np.uint8)
            label_img = np.dstack((label_img, alpha))

        # exactly one instance per category (if present)
        for cat_name, cat_id, colors in CATEGORY_DEFS:
            mask = build_mask(label_img, colors).astype(np.uint8)

            if mask.sum() == 0:
                continue  # this class not present in this image

            # RLE encoding of the whole class mask
            # Convert RLE to Polygon for YOLO compatibility
            # Ultralytics YOLO converter expects polygons in "segmentation", not RLE
            # We need to extract contours from the mask

            # contours, _ = cv2.findContours(
            #     mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            # )

            # segmentation = []
            # for contour in contours:
            #     if contour.size >= 6:  # Need at least 3 points (6 coords)
            #         segmentation.append(contour.flatten().tolist())

            # # If no valid contours found, skip
            # if not segmentation:
            #     continue

            # Recalculate area and bbox from mask
            rle = maskUtils.encode(np.asfortranarray(mask))

            # Decode bytes to string for JSON serialization
            if isinstance(rle["counts"], bytes):
                rle["counts"] = rle["counts"].decode("ascii")

            area = float(maskUtils.area(rle))
            bbox = [float(x) for x in maskUtils.toBbox(rle)]

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat_id,
                    # "segmentation": segmentation,  # Use polygon list
                    "segmentation": rle,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    coco = {
        "info": {
            "description": f"Tree trunk and branches segmentation ({set_name})",
            "version": "1.0",
            "year": datetime.datetime.now().year,
            "date_created": datetime.datetime.now().isoformat(),
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": 2, "name": "branches", "supercategory": "tree"},
        ],
    }

    with open(annotation_file, "w") as f:
        json.dump(coco, f, indent=4)

    print(
        f"Saved {len(images)} images and {len(annotations)} annotations to {annotation_file}"
    )


# ---------------------------------------------------------------------
# main conversion
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Get all label files first
    all_label_paths = sorted(RAW_DIR.glob("*_label_rgb_*.png"))

    # Filter to only valid indices if needed, assuming filenames start with index or just taking first 400
    # Since files are sorted, we can slice the list.

    # Slice for training (first 300)
    train_labels = all_label_paths[:800]

    # Slice for testing (next 100: 301-400)
    test_labels = all_label_paths[800:1000]

    if not train_labels:
        print("No labels found for training set!")
    else:
        process_dataset(train_labels, TRAIN_DIR, "training")

    if not test_labels:
        print("No labels found for test set!")
    else:
        process_dataset(test_labels, TEST_DIR, "testing")
