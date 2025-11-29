from ultralytics.data.converter import convert_coco
from pathlib import Path
import shutil
import yaml
import json

import numpy as np
from pycocotools import mask as maskUtils
from skimage.measure import find_contours
from tqdm import tqdm

def rle_to_polygons(in_json_path: Path, out_json_path: Path):
    """Convert COCO RLE segmentations to polygon segmentations."""
    with open(in_json_path, "r") as f:
        coco = json.load(f)

    for ann in tqdm(coco["annotations"]):
        seg = ann.get("segmentation", None)

        # already polygon format
        if isinstance(seg, list):
            continue

        # seg is RLE dict: {"size":[h,w], "counts": "..."}
        rle = seg
        mask = maskUtils.decode(rle)  # H x W or H x W x 1
        # print("mask.shape:", mask.shape)

        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = mask.astype(np.uint8)

        contours = find_contours(mask, 0.5)

        polygons = []
        for c in contours:
            # c is (N, 2) in (row, col) = (y, x)
            c = np.flip(c, axis=1)  # to (x, y)
            poly = c.ravel().tolist()
            if len(poly) >= 6:  # at least 3 points
                polygons.append(poly)

        ann["segmentation"] = polygons
        ann["iscrowd"] = 0

    with open(out_json_path, "w") as f:
        json.dump(coco, f)

    print(f"Saved polygon COCO to {out_json_path}")
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
    root_coco = current_dir / "image_envy_5000_coco"
    root_yolo = current_dir / "image_envy_5000_yolo"

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
