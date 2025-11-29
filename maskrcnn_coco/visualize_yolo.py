import fiftyone as fo
import os

# Path to the YOLO dataset directory containing data.yaml
# Assuming this script is run from maskrcnn_coco/
dataset_dir = "Fuji-Apple-Segmentation_yolo"
yaml_path = os.path.join(dataset_dir, "data.yaml")

# Name for the dataset in FiftyOne
dataset_name = "fuji-apple-yolo"

# Check if dataset exists and delete it to ensure fresh load
if fo.dataset_exists(dataset_name):
    print(f"Dataset '{dataset_name}' already exists. Deleting and reloading...")
    fo.delete_dataset(dataset_name)

# Load the dataset
# dataset_type=fo.types.YOLOv5Dataset reads the data.yaml to find images/labels
print(f"Loading YOLO dataset from {dataset_dir}...")
try:
    dataset = fo.Dataset.from_dir(
        dataset_dir=dataset_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        name=dataset_name,
        label_type="polylines"
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
