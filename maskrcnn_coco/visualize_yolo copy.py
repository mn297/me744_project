import fiftyone as fo
import os
import yaml
from pathlib import Path
from glob import glob


def load_yolo_segmentation(dataset_dir, name):
    """
    Manually loads a YOLO segmentation dataset into FiftyOne.
    Standard fo.types.YOLOv5Dataset treats data as bounding boxes.
    This loader interprets the .txt files as polygons.
    """
    dataset_dir = Path(dataset_dir)
    yaml_path = dataset_dir / "dataset.yaml"

    # 1. Read Class Names from YAML
    with open(yaml_path, "r") as f:
        data_config = yaml.safe_load(f)

    # Handle 'names' which can be a list or dict
    names = data_config.get("names", {})
    if isinstance(names, dict):
        # Convert {0: 'apple'} to list ['apple'] assuming 0-indexed continuous
        class_names = [names[i] for i in sorted(names.keys())]
    else:
        class_names = names

    # 2. Setup Dataset
    if fo.dataset_exists(name):
        print(f"Deleting existing dataset '{name}'...")
        fo.delete_dataset(name)

    dataset = fo.Dataset(name)

    # 3. Iterate Splits
    splits = ["train", "val"]

    samples = []
    for split in splits:
        # Handle paths from yaml (e.g., "images/train")
        # yaml paths might be absolute or relative to yaml location
        split_rel_path = data_config.get(split)
        if not split_rel_path:
            continue

        img_dir = dataset_dir / split_rel_path
        if not img_dir.exists():
            # Try standard structure if yaml path is just "images/train" but dir is absolute
            img_dir = dataset_dir / "images" / split

        if not img_dir.exists():
            print(f"Warning: Image directory {img_dir} not found. Skipping {split}.")
            continue

        print(f"Processing {split} from {img_dir}...")

        # Supported extensions
        image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            image_paths.extend(list(img_dir.glob(ext)))

        for img_path in image_paths:
            # Find corresponding label file
            # ..images/train/img.jpg -> ..labels/train/img.txt
            label_dir = img_path.parent.parent.parent / "labels" / split
            label_path = label_dir / f"{img_path.stem}.txt"

            ground_truth = []

            if label_path.exists():
                with open(label_path, "r") as f:
                    lines = f.readlines()

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    cls_idx = int(parts[0])
                    # Parse coordinates [x1, y1, x2, y2, ...]
                    coords = [float(x) for x in parts[1:]]

                    # Group into (x,y) tuples
                    points = []
                    for i in range(0, len(coords), 2):
                        if i + 1 < len(coords):
                            points.append((coords[i], coords[i + 1]))

                    # Create Polyline (Polygon)
                    # points argument expects a list of list of points (list of shapes)
                    # closed=True makes it a polygon
                    label = (
                        class_names[cls_idx]
                        if 0 <= cls_idx < len(class_names)
                        else str(cls_idx)
                    )

                    poly = fo.Polyline(
                        label=label,
                        points=[points],
                        closed=True,
                        filled=True,
                    )
                    ground_truth.append(poly)

            # Create Sample
            sample = fo.Sample(filepath=str(img_path))
            if ground_truth:
                sample["ground_truth"] = fo.Polylines(polylines=ground_truth)

            sample["tags"] = [split]
            samples.append(sample)

    # 4. Add samples to dataset
    print(f"Adding {len(samples)} samples to dataset...")
    dataset.add_samples(samples)
    dataset.compute_metadata()  # Calculates width/height for samples
    return dataset


if __name__ == "__main__":
    # Configuration
    DATASET_DIR = "Fuji-Apple-Segmentation_yolo"
    DATASET_NAME = "fuji-apple-yolo-seg"

    try:
        dataset = load_yolo_segmentation(DATASET_DIR, DATASET_NAME)

        print("\nDataset Statistics:")
        print(dataset)

        print("\nLaunching FiftyOne App...")
        session = fo.launch_app(dataset)
        session.wait()

    except Exception as e:
        print(f"\nError: {e}")
