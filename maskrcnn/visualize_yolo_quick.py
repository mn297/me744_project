import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
dataset_dir = Path(__file__).parent.parent / "datasets"
dataset_name = "Fuji-Apple-Segmentation_with_envy_mask_yolo"
YOLO_ROOT = dataset_dir / dataset_name
SPLIT = "train"  # 'train' or 'val'
NUM_IMAGES = 5


def visualize_yolo():
    # 1. Load dataset.yaml to get paths and class names
    yaml_path = YOLO_ROOT / "dataset.yaml"
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    class_names = data.get("names", {})

    # 2. Resolve image directory
    # YAML paths can be absolute or relative to the YAML location
    img_dir_path = data.get(SPLIT)
    if not Path(img_dir_path).is_absolute():
        img_dir_path = YOLO_ROOT / img_dir_path

    img_dir = Path(img_dir_path)

    # 3. Get list of images
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = [p for p in img_dir.glob("*") if p.suffix.lower() in valid_exts]
    images = sorted(images)[:NUM_IMAGES]

    if not images:
        print(f"No images found in {img_dir}")
        return

    # 4. Loop and Visualize
    plt.figure(figsize=(15, 10))

    for i, img_path in enumerate(images):
        # Read Image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # Construct Label Path:
        # usually .../images/train/1.jpg -> .../labels/train/1.txt
        label_path = (
            list(img_path.parents)[2]
            / "labels"
            / SPLIT
            / img_path.with_suffix(".txt").name
        )

        if label_path.exists():
            with open(label_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = list(map(float, line.strip().split()))
                cls_id = int(parts[0])
                coords = parts[1:]

                # Reshape to (N, 2)
                points = np.array(coords).reshape(-1, 2)

                # Denormalize (0-1 -> pixel coords)
                points[:, 0] *= w
                points[:, 1] *= h
                points = points.astype(np.int32)

                # Draw Polygon
                # Color based on class id (arbitrary math for distinct colors)
                color = (
                    (cls_id * 50) % 255,
                    (cls_id * 100 + 100) % 255,
                    (cls_id * 150 + 50) % 255,
                )

                # Draw outline
                cv2.polylines(img, [points], isClosed=True, color=color, thickness=2)

                # Optional: Fill with transparency
                overlay = img.copy()
                cv2.fillPoly(overlay, [points], color=color)
                alpha = 0.4
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Draw Label Text
                label_text = class_names.get(cls_id, str(cls_id))
                cv2.putText(
                    img,
                    label_text,
                    (points[0][0], points[0][1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        else:
            print(f"No label file for {img_path.name}")

        # Plot
        plt.subplot(1, NUM_IMAGES, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(img_path.name, fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_yolo()
