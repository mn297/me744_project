from ultralytics.data.converter import convert_coco
from pathlib import Path
import shutil
import yaml
import json
import cv2
import numpy as np
from pycocotools import mask as maskUtils
from skimage.measure import find_contours
from tqdm import tqdm


def rle_to_polygons(in_json_path: Path, out_json_path: Path):
    with open(in_json_path, "r") as f:
        coco = json.load(f)

    new_annotations = []
    next_ann_id = max([a.get("id", 0) for a in coco["annotations"]], default=0) + 1
    print("Processing and splitting disjoint masks...")

    for ann in tqdm(coco["annotations"]):
        seg = ann.get("segmentation", None)
        if seg is None:
            continue

        # Already polygons: keep but ensure unique IDs
        if isinstance(seg, list):
            new_ann = ann.copy()
            new_ann["id"] = next_ann_id
            new_ann["iscrowd"] = 0
            new_annotations.append(new_ann)
            next_ann_id += 1
            continue

        # Decode RLE
        mask = maskUtils.decode(seg)
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # Fix potential inversion (background mostly ones)
        if mask.mean() > 0.5:
            mask = 1 - mask

        mask = (mask * 255).astype(np.uint8)

        # Pad to avoid contour wrapping at borders
        mask_padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

        contours, _ = cv2.findContours(
            mask_padded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # allow for multiple polygons per annotation
        for c in contours:
            epsilon = 0.001 * cv2.arcLength(c, True)
            c_smooth = cv2.approxPolyDP(c, epsilon, True)
            c_final = c_smooth - 1  # remove padding offset
            poly = c_final.flatten().tolist()

            if len(poly) >= 6:
                pts = np.array(poly).reshape(-1, 2)
                x_min, y_min = pts.min(axis=0)
                x_max, y_max = pts.max(axis=0)
                w_box, h_box = x_max - x_min, y_max - y_min

                new_ann = ann.copy()
                new_ann["segmentation"] = [poly]  # one polygon per annotation
                new_ann["bbox"] = [
                    float(x_min),
                    float(y_min),
                    float(w_box),
                    float(h_box),
                ]
                new_ann["id"] = next_ann_id
                new_ann["iscrowd"] = 0
                new_annotations.append(new_ann)
                next_ann_id += 1

    coco["annotations"] = new_annotations

    with open(out_json_path, "w") as f:
        json.dump(coco, f)
    return out_json_path


def create_yaml(yolo_root: Path, class_names: list):
    """Generate dataset.yaml for YOLO training."""
    yaml_data = {
        "path": str(yolo_root.absolute()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(class_names)},
    }

    yaml_path = yolo_root / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, sort_keys=False)
    print(f"\nCreated dataset config: {yaml_path}")


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 0. Paths
    # ------------------------------------------------------------------
    current_dir = Path(__file__).parent
    dataset_dir = current_dir.parent / "datasets"
    # root_coco = dataset_dir / "image_envy_5000_coco"
    # root_yolo = dataset_dir / "image_envy_5000_yolo"
    root_coco = dataset_dir / "Fuji-Apple-Segmentation_coco"
    root_yolo = dataset_dir / "Fuji-Apple-Segmentation_yolo"
    root_coco = dataset_dir / "Fuji-Apple-Segmentation_with_envy_mask_coco"
    root_yolo = dataset_dir / "Fuji-Apple-Segmentation_with_envy_mask_yolo"

    # 'trainingset' -> train, 'testset' -> val
    splits = {
        "train": root_coco / "trainingset",
        "val": root_coco / "testset",
    }

    # ------------------------------------------------------------------
    # 1. Convert RLE COCO -> polygon COCO for each split
    # ------------------------------------------------------------------
    polygon_jsons = {}

    for split_name, src_dir in splits.items():
        if not src_dir.exists():
            print(f"Warning: {src_dir} not found. Skipping {split_name}.")
            continue

        in_json = src_dir / "annotations.json"
        if not in_json.exists():
            print(f"Warning: {in_json} not found. Skipping {split_name}.")
            continue

        out_json = src_dir / "annotations_polygons.json"
        polygon_jsons[split_name] = rle_to_polygons(in_json, out_json)

    # ------------------------------------------------------------------
    # 2. Prepare temp COCO jsons for ultralytics.convert_coco
    # ------------------------------------------------------------------
    temp_json_dir = current_dir / "temp_coco_jsons"
    if temp_json_dir.exists():
        shutil.rmtree(temp_json_dir)
    temp_json_dir.mkdir()

    CLASSES = []

    print("Preparing annotations for YOLO conversion...")
    for split_name, poly_json in polygon_jsons.items():
        # copy and rename to instances_{split}.json
        dst_json = temp_json_dir / f"instances_{split_name}.json"
        shutil.copy(poly_json, dst_json)

        if not CLASSES:
            with open(poly_json, "r") as f:
                data = json.load(f)
                cats = sorted(data["categories"], key=lambda x: x["id"])
                CLASSES = [x["name"] for x in cats]
                print(f"Extracted classes: {CLASSES}")

    # ------------------------------------------------------------------
    # 3. Run COCO -> YOLO conversion (segments)
    # ------------------------------------------------------------------
    if root_yolo.exists():
        print(f"Removing existing output directory: {root_yolo}")
        shutil.rmtree(root_yolo)

    print(f"Converting annotations to YOLO at {root_yolo}...")
    convert_coco(
        labels_dir=str(temp_json_dir),
        save_dir=str(root_yolo),
        use_segments=True,
        cls91to80=False,
    )

    # ------------------------------------------------------------------
    # 4. Copy images into YOLO structure
    # ------------------------------------------------------------------
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    for split_name, src_dir in splits.items():
        if not src_dir.exists():
            continue

        dest_img_dir = root_yolo / "images" / split_name
        dest_img_dir.mkdir(parents=True, exist_ok=True)

        img_src = src_dir / "JPEGImages"
        if not img_src.exists():
            img_src = src_dir

        print(f"Copying images for {split_name} from {img_src}...")
        count = 0
        for file_path in img_src.glob("*"):
            if file_path.suffix.lower() in valid_exts:
                shutil.copy2(file_path, dest_img_dir / file_path.name)
                count += 1
        print(f"Copied {count} images to {dest_img_dir}")

    # ------------------------------------------------------------------
    # 5. Create dataset.yaml
    # ------------------------------------------------------------------
    create_yaml(root_yolo, CLASSES)

    # ------------------------------------------------------------------
    # 6. Cleanup
    # ------------------------------------------------------------------
    shutil.rmtree(temp_json_dir)
    print(f"\nDone! Dataset ready at: {root_yolo}")
