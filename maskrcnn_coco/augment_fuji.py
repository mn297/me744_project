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


ENVY_DIR = DATASETS_DIR / "image_envy_5000"

# Output Dataset
OUTPUT_DIR = DATASETS_DIR / "Fuji-Apple-Segmentation_with_envy_mask_coco"

# Configuration
NUM_ENVY_IMAGES = 500
TRAIN_SPLIT = 0.8
MIN_FRUITS = 0
MAX_FRUITS = 20

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


def setup_output_dir():
    if OUTPUT_DIR.exists():
        print(f"Removing existing output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True)

    # Create empty COCO structure
    for split in ["trainingset", "testset"]:
        (OUTPUT_DIR / split / "JPEGImages").mkdir(parents=True)
        # Initialize empty annotations file
        with open(OUTPUT_DIR / split / "annotations.json", "w") as f:
            json.dump(
                {
                    "images": [],
                    "annotations": [],
                    "categories": [
                        {"id": 1, "name": "apple", "supercategory": "fruit"}
                    ],
                },
                f,
                indent=2,
            )


def get_envy_images():
    all_files = list(ENVY_DIR.glob("*_rgb_*.png"))
    rgb_files = [f for f in all_files if "_label_" not in f.name]
    seg_files = [
        f for f in all_files if "_label_" in f.name and "_depth_" not in f.name
    ]
    # exclude first 1000 files
    rgb_files = sorted(rgb_files)[2000:5000]
    seg_files = sorted(seg_files)[2000:5000]
    return rgb_files, seg_files


def get_fuji_samples(img_dir: Path, seg_dir: Path) -> list[tuple[Path, Path]]:
    """Return list of (img_path, seg_path) tuples."""
    imgs = sorted(list(img_dir.glob("*.npy")))
    segs = sorted(list(seg_dir.glob("*.npy")))
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
    envy_img_lst: list[Path],
    envy_seg_lst: list[Path],
    fuji_samples: list[tuple[Path, Path]],
    unet_img_dir: Path,
    unet_seg_dir: Path,
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

    for envy_img, envy_seg in tqdm(
        zip(envy_img_lst, envy_seg_lst),
        desc=f"Augmenting {json_path.parent.name}",
        total=len(envy_img_lst),
    ):
        # 1. Load Envy Background
        # Convert to numpy RGB
        try:
            with Image.open(envy_img) as bg_pil:
                bg_img = np.array(bg_pil.convert("RGB"))
        except (OSError, SyntaxError) as e:
            print(f"Skipping corrupt image: {envy_img} ({e})")
            continue

        bg_h, bg_w = bg_img.shape[:2]

        # 2. Determine number of apples to paste
        num_apples = random.randint(MIN_FRUITS, MAX_FRUITS)

        # Keep track of annotations for this image
        new_annotations = []
        full_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)

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
            apple_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
            apple_mask[pos_y : pos_y + h, pos_x : pos_x + w] = mask_crop.astype(
                np.uint8
            )
            full_mask[pos_y : pos_y + h, pos_x : pos_x + w] = mask_crop.astype(np.uint8)

            polygons = mask_to_polygons(apple_mask)
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

        # save envy seg to unet format
        # construct unet format for testset
        gt_key = envy_seg.stem.split("_")[0]
        im_key = envy_img.stem.split("_")[0]
        assert (
            gt_key == im_key
        ), f"Label/image mismatch: {envy_seg.name} vs {envy_img.name}"

        annot = np.asarray(Image.open(envy_seg))
        assert np.all(annot[:, :, 3] == 255)
        annot = annot[:, :, :3]  # ignore alpha
        im_cur = np.asarray(Image.open(envy_img))
        assert np.all(im_cur[:, :, 3] == 255)
        im_cur = im_cur[:, :, :3]  # ignore alpha
        # all_labels = np.unique(annot.reshape(annot.shape[0] * annot.shape[1], 3), axis=0, return_counts=True)
        # print(annotation_filename)
        # print(annot.shape)
        # print(all_labels)
        # print(annot.shape)
        masks = []
        matches = []
        for col in color_dict.values():
            matches.append(np.all(annot == col[:3], axis=-1))
        masks.append(np.logical_or.reduce(matches))
        masks = np.stack(masks, axis=0).astype(np.uint8)
        assert np.max(np.sum(masks, axis=0)) == 1
        label_map = np.zeros((annot.shape[0], annot.shape[1]), dtype=np.uint8)
        for cur_label in range(masks.shape[0]):
            label_map[masks[cur_label] > 0] = 1

        # zero the label map where the full_mask is 1
        # This treats pasted Fuji apples as background (class 0) in the Envy mask
        # label_map[full_mask > 0] = 0

        # However, for the U-Net training, we probably want the pasted apples to be class 1 (apple)
        # The original Fuji apples are class 1.
        # The Envy background (trunk/branches) is class 2.
        # The Envy background (sky/ground) is class 0.
        # So we should actually set the pasted apple regions to 0 to consider it as background in the context of U-Net training
        label_map[full_mask > 0] = 2

        seg_out = unet_seg_dir / f"{envy_img.stem}.npy"
        img_out = unet_img_dir / f"{envy_img.stem}.npy"

        np.save(seg_out, label_map.astype(np.uint8))
        np.save(
            img_out, bg_img
        )  # Save the AUGMENTED image (bg_img), not the original Envy image (im_cur)

        # 7. Save augmented image (JPEG for COCO)
        new_filename = f"aug_{envy_img.stem}.jpg"
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

    envy_images, envy_labels = get_envy_images()

    fuji_train_imgs = FUJI_UNET_DIR / "np_imgs_train"
    fuji_train_segs = FUJI_UNET_DIR / "np_segs_train"
    fuji_val_imgs = FUJI_UNET_DIR / "np_imgs_val"
    fuji_val_segs = FUJI_UNET_DIR / "np_segs_val"

    fuji_train_samples = get_fuji_samples(fuji_train_imgs, fuji_train_segs)
    fuji_val_samples = get_fuji_samples(fuji_val_imgs, fuji_val_segs)

    print(f"Found {len(envy_images)} Envy images")
    print(f"Found {len(fuji_train_samples)} Fuji training samples")
    print(f"Found {len(fuji_val_samples)} Fuji validation samples")

    # Selection
    if len(envy_images) < NUM_ENVY_IMAGES:
        selected_envy = envy_images
    else:
        random.seed(69)
        selected_envy = random.sample(envy_images, NUM_ENVY_IMAGES)

    num_train = int(len(selected_envy) * TRAIN_SPLIT)
    train_envy_img_lst = selected_envy[:num_train]
    train_envy_seg_lst = [
        ENVY_DIR / img.name.replace("_rgb_", "_label_rgb_")
        for img in train_envy_img_lst
    ]
    test_envy_img_lst = selected_envy[num_train:]
    test_envy_seg_lst = [
        ENVY_DIR / img.name.replace("_rgb_", "_label_rgb_") for img in test_envy_img_lst
    ]
    print(
        f"Augmenting with {len(train_envy_img_lst)} train and {len(test_envy_img_lst)} test images"
    )

    # Train
    unet_train_img_dir = OUTPUT_DIR / "np_imgs_train"
    unet_train_seg_dir = OUTPUT_DIR / "np_segs_train"
    unet_test_seg_dir = OUTPUT_DIR / "np_segs_val"
    train_json = OUTPUT_DIR / "trainingset" / "annotations.json"
    train_img_dir = OUTPUT_DIR / "trainingset" / "JPEGImages"

    unet_test_img_dir = OUTPUT_DIR / "np_imgs_val"
    unet_test_seg_dir = OUTPUT_DIR / "np_segs_val"
    test_json = OUTPUT_DIR / "testset" / "annotations.json"
    test_img_dir = OUTPUT_DIR / "testset" / "JPEGImages"

    if not unet_train_seg_dir.exists():
        unet_train_img_dir.mkdir(parents=True, exist_ok=True)
    if not unet_train_seg_dir.exists():
        unet_train_seg_dir.mkdir(parents=True, exist_ok=True)
    if not unet_test_seg_dir.exists():
        unet_test_img_dir.mkdir(parents=True, exist_ok=True)
    if not unet_test_seg_dir.exists():
        unet_test_seg_dir.mkdir(parents=True, exist_ok=True)

    augment_dataset(
        train_json,
        train_img_dir,
        train_envy_img_lst,
        train_envy_seg_lst,
        fuji_train_samples,
        unet_train_img_dir,
        unet_train_seg_dir,
    )

    augment_dataset(
        test_json,
        test_img_dir,
        test_envy_img_lst,
        test_envy_seg_lst,
        fuji_val_samples,
        unet_test_img_dir,
        unet_test_seg_dir,
    )

    print("\n=== Done! ===")


if __name__ == "__main__":
    main()
