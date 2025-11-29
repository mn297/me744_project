from pathlib import Path
import os
import random

from ultralytics import YOLO


def run_prediction(model: YOLO, val_dir: Path) -> None:
    """Run a quick prediction on a random validation image if available."""
    if not val_dir.exists():
        print(f"Validation directory not found at: {val_dir}")
        print("Skipping prediction on a test image.")
        return

    # 10 random images
    test_image_names = random.sample(os.listdir(val_dir), 10)
    for test_image_name in test_image_names:
        test_image_path = val_dir / test_image_name
        print(f"\nRunning prediction on a random test image: {test_image_path}")
        model.predict(str(test_image_path), conf=0.5, save=True)
        print(
            "\nPrediction results saved. Check the latest 'runs/segment/predict' dir."
        )


def main() -> None:
    """Train, validate, and briefly test a YOLOv8 segmentation model."""
    model = YOLO("yolo11s-seg.pt")
    dataset_yaml = Path("image_envy_5000_yolo/dataset.yaml")
    val_dir = Path("image_envy_5000_yolo/images/val")

    results = model.train(
        data=str(dataset_yaml),
        epochs=5,
        imgsz=1024,
        device=0,
        # name="yolov8m-seg-fuji-apple",
    )
    print("Training complete:", results)

    print("Validation metrics:")
    metrics = model.val()
    print(metrics)

    run_prediction(model, val_dir)


if __name__ == "__main__":
    # Required on Windows so torch Dataloaders can spawn workers safely.
    from multiprocessing import freeze_support

    freeze_support()
    main()
