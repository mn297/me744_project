from ultralytics.data.converter import convert_coco
from pathlib import Path
import shutil
import yaml
import os


def create_yaml(yolo_root: Path, class_names: list):
    """Generates the data.yaml file required for training."""
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
    # Setup Paths
    current_dir = Path(__file__).parent
    root_coco = current_dir / "Fuji-Apple-Segmentation"
    root_yolo = current_dir / "Fuji-Apple-Segmentation_yolo"

    # Cleanup existing YOLO directory to avoid "folder_2", "folder_3" creation
    if root_yolo.exists():
        print(f"Removing existing output directory: {root_yolo}")
        shutil.rmtree(root_yolo)

    # Define Splits
    # 'trainingset' -> 'train', 'testset' -> 'val'
    splits = {"train": root_coco / "trainingset", "val": root_coco / "testset"}

    # 1. Prepare JSONs in a temporary directory
    # convert_coco requires files named instances_{split}.json to automatically determine split name
    temp_json_dir = current_dir / "temp_coco_jsons"
    if temp_json_dir.exists():
        shutil.rmtree(temp_json_dir)
    temp_json_dir.mkdir()

    print("Preparing annotations...")
    for split_name, src_dir in splits.items():
        if not src_dir.exists():
            print(f"Warning: {src_dir} not found. Skipping {split_name}.")
            continue

        json_files = list(src_dir.glob("*.json"))
        if not json_files:
            print(f"Warning: No JSON found in {src_dir}. Skipping {split_name}.")
            continue

        # Copy and rename JSON
        src_json = json_files[0]
        dst_json = temp_json_dir / f"instances_{split_name}.json"
        shutil.copy(src_json, dst_json)

    # 2. Run Conversion (Calls convert_coco ONCE to handle all splits)
    # This creates root_yolo and labels/{split} folders
    print(f"Converting annotations to {root_yolo}...")
    convert_coco(
        labels_dir=temp_json_dir,
        save_dir=root_yolo,
        use_segments=True,
        cls91to80=False,
    )

    # 3. Copy Images
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    for split_name, src_dir in splits.items():
        if not src_dir.exists():
            continue

        dest_img_dir = root_yolo / "images" / split_name
        dest_img_dir.mkdir(parents=True, exist_ok=True)

        # Check for JPEGImages subfolder
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

    # 4. Generate YAML
    CLASSES = ["apple"]
    create_yaml(root_yolo, CLASSES)

    # Cleanup temp dir
    shutil.rmtree(temp_json_dir)

    print(f"\nDone! Dataset ready at: {root_yolo}")
