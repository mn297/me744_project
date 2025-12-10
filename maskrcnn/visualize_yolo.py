import fiftyone as fo
import os
from pathlib import Path

# Path to the YOLO dataset directory containing data.yaml
# Assuming this script is run from maskrcnn_coco/
dataset_dir = Path(__file__).parent.parent / "datasets"
dataset_name = "image_envy_5000_yolo"
dataset_path = dataset_dir / dataset_name
yaml_path = dataset_path / "data.yaml"

try:
    dataset = fo.Dataset.from_dir(
        dataset_dir=dataset_path,
        dataset_type=fo.types.YOLOv5Dataset,
        name=dataset_name,
        label_type="polylines",
        # yaml_path is not needed if data.yaml is directly in dataset_dir
        # or if provided, it should be relative if dataset_dir is used, but simpler to omit if standard
    )

    print("\nDataset Statistics:")
    print(dataset)

    # Launch the App
    print("\nLaunching FiftyOne App...")
    session = fo.launch_app(dataset)

    # Block execution so the server stays alive (if running as script)
    session.wait()

except Exception as e:
    print(f"Error loading dataset: {e}")
