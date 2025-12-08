from pathlib import Path
from ultralytics import YOLO


def main() -> None:
    # 1. Setup Paths
    current_dir = Path(__file__).parent.resolve()
    project_root = current_dir.parent

    # Define dataset path
    # Update this to point to your desired dataset
    dataset_dir = (
        project_root / "datasets" / "Fuji-Apple-Segmentation_with_envy_mask_yolo"
    )
    dataset_yaml = dataset_dir / "dataset.yaml"
    checkpoint_dir = project_root / "datasets" / "checkpoints" / "yolo"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_yaml.exists():
        print(f"Error: Dataset config not found at {dataset_yaml}")
        return

    print(f"Training with dataset: {dataset_yaml}")

    # 2. Initialize Model
    # Load a pretrained YOLOv8/11 segmentation model
    model = YOLO("yolo11s-seg.pt")

    # 3. Train
    print("Starting training...")
    results = model.train(
        data=str(dataset_yaml),
        epochs=5,
        imgsz=1024,
        batch=8,
        device=0,
        workers=2,
        save=True,  # Save checkpoints
        save_period=1,  # Save checkpoint every epoch (epoch1.pt, epoch2.pt...)
        project=str(checkpoint_dir),  # Save checkpoints to the explicit directory
        name="train",  # Default name, will auto-increment (train, train2...)
    )
    print("Training complete.")

    # 4. Validate
    print("Running validation on best model...")
    metrics = model.val()
    print("Validation metrics:", metrics)


if __name__ == "__main__":
    # Required on Windows so torch Dataloaders can spawn workers safely.
    from multiprocessing import freeze_support

    freeze_support()
    main()
