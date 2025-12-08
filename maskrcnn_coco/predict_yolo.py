import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


def find_latest_checkpoint(project: Path) -> str:
    """Find the latest trained checkpoint in the provided project directory."""
    project_path = Path(project)
    if not project_path.exists():
        raise FileNotFoundError(f"No training runs found in {project}")

    # Find all train directories (train, train2, train3, ...)
    train_dirs = [
        d for d in project_path.iterdir() if d.is_dir() and d.name.startswith("train")
    ]

    if not train_dirs:
        raise FileNotFoundError(f"No 'train' directories found in {project}")

    # Sort by modification time (latest first)
    train_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)

    latest_dir = train_dirs[0]
    weights_dir = latest_dir / "weights"

    # Prefer best.pt, fallback to last.pt
    best_pt = weights_dir / "best.pt"
    last_pt = weights_dir / "last.pt"

    if best_pt.exists():
        return str(best_pt)
    elif last_pt.exists():
        return str(last_pt)
    else:
        raise FileNotFoundError(
            f"No checkpoints (best.pt/last.pt) found in {weights_dir}"
        )


def compute_metrics(
    pred_masks: np.ndarray, target_masks: np.ndarray
) -> Tuple[float, float]:
    """
    Compute IoU and Dice treating all instances as one global mask.
    Returns (-1, -1) if there is no target mask.
    """
    if len(target_masks) == 0:
        return -1.0, -1.0

    if len(pred_masks) == 0:
        return 0.0, 0.0

    # Collapse to single mask
    global_pred = np.any(pred_masks, axis=0) if pred_masks.ndim == 3 else pred_masks
    global_target = (
        np.any(target_masks, axis=0) if target_masks.ndim == 3 else target_masks
    )

    intersection = np.logical_and(global_pred, global_target).sum()
    union = np.logical_or(global_pred, global_target).sum()

    if union == 0:
        return 1.0, 1.0  # Both empty

    iou = intersection / union
    dice = (2 * intersection) / (global_pred.sum() + global_target.sum())
    return iou, dice


def load_gt_masks(
    label_path: Path, image_shape: Tuple[int, int]
) -> Dict[int, List[np.ndarray]]:
    """
    Load YOLO-seg labels from a txt file and return masks grouped by class id.
    """
    h, w = image_shape
    masks_by_cls: Dict[int, List[np.ndarray]] = {}

    if not label_path.exists():
        return masks_by_cls

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = list(map(float, line.strip().split()))
        if len(parts) < 7:  # need at least cls + 3 points (6 coords)
            continue
        cls_id = int(parts[0])
        coords = np.array(parts[1:], dtype=float).reshape(-1, 2)
        coords[:, 0] *= w
        coords[:, 1] *= h
        coords = coords.astype(np.int32)
        if len(coords) < 3:
            continue

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [coords], 1)
        masks_by_cls.setdefault(cls_id, []).append(mask.astype(bool))

    return masks_by_cls


def evaluate_directory(
    model: YOLO,
    source_dir: Path,
    names: Dict[int, str],
    conf: float,
    metadata: Dict = None,
) -> None:
    """
    Run inference on a directory and compute per-class IoU/Dice metrics against YOLO labels.
    """
    if metadata is None:
        metadata = {}

    if not source_dir.is_dir():
        print(f"[WARN] Metrics require a directory with labels. Got file: {source_dir}")
        return

    dataset_dir = source_dir.parent.parent  # .../dataset/images/val -> dataset
    labels_dir = dataset_dir / "labels" / source_dir.name

    image_files = sorted(
        [
            p
            for p in source_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
    )

    per_file: Dict[str, Dict[str, Dict[str, float]]] = {}
    totals: Dict[int, Dict[str, float]] = {}

    for cls_id in names.keys():
        totals[cls_id] = {"iou": 0.0, "dice": 0.0, "count": 0}

    for img_path in image_files:
        label_path = labels_dir / f"{img_path.stem}.txt"
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        gt_masks = load_gt_masks(label_path, (h, w))

        # Run prediction
        results = model.predict(
            source=str(img_path), conf=conf, save=False, verbose=False
        )
        if not results:
            continue
        res = results[0]
        if res.masks is None:
            continue

        pred_masks_all = res.masks.data.cpu().numpy()
        # Resize predicted masks back to original image size
        resized_pred_masks = []
        for pmask in pred_masks_all:
            pm_resized = cv2.resize(
                pmask, (w, h), interpolation=cv2.INTER_LINEAR
            ) > 0.5
            resized_pred_masks.append(pm_resized)
        pred_masks_all = (
            np.stack(resized_pred_masks, axis=0)
            if resized_pred_masks
            else np.zeros((0, h, w), dtype=bool)
        )
        pred_classes = res.boxes.cls.cpu().numpy().astype(int)
        pred_scores = res.boxes.conf.cpu().numpy()

        pred_by_cls: Dict[int, List[np.ndarray]] = {}
        for pmask, cls_id, score in zip(pred_masks_all, pred_classes, pred_scores):
            if score < conf:
                continue
            pred_by_cls.setdefault(cls_id, []).append(pmask.astype(bool))

        file_metrics: Dict[str, Dict[str, float]] = {}
        class_ids = set(pred_by_cls.keys()) | set(gt_masks.keys())

        for cls_id in class_ids:
            cls_name = names.get(cls_id, str(cls_id))
            pm = (
                np.stack(pred_by_cls.get(cls_id, []))
                if pred_by_cls.get(cls_id)
                else np.zeros((0, h, w), dtype=bool)
            )
            gm = (
                np.stack(gt_masks.get(cls_id, []))
                if gt_masks.get(cls_id)
                else np.zeros((0, h, w), dtype=bool)
            )

            iou, dice = compute_metrics(pm, gm)
            file_metrics[cls_name] = {"iou": float(iou), "dice": float(dice)}

            if iou != -1.0:
                totals.setdefault(cls_id, {"iou": 0.0, "dice": 0.0, "count": 0})
                totals[cls_id]["iou"] += iou
                totals[cls_id]["dice"] += dice
                totals[cls_id]["count"] += 1

        per_file[img_path.name] = file_metrics

    averages = {}
    for cls_id, agg in totals.items():
        cnt = agg.get("count", 0)
        if cnt == 0:
            averages[names.get(cls_id, str(cls_id))] = {
                "average_iou": 0.0,
                "average_dice": 0.0,
                "valid_images": 0,
            }
        else:
            averages[names.get(cls_id, str(cls_id))] = {
                "average_iou": agg["iou"] / cnt,
                "average_dice": agg["dice"] / cnt,
                "valid_images": cnt,
            }

    metrics_summary = {"metadata": metadata, "average": averages, "per_file": per_file}

    metrics_path = source_dir / "yolo_metrics.json"
    with open(metrics_path, "w") as f:
        import json

        json.dump(metrics_summary, f, indent=4)

    print(f"[SUCCESS] Metrics saved to {metrics_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLO inference on images or folders."
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to image file or folder for inference. Defaults to validation images.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="latest",
        help=(
            "Path to weights file. Use 'latest' to load the most recent training run "
            "(default: latest)."
        ),
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--calc-metrics",
        action="store_true",
        default=True,
        help="If set, compute per-class IoU/Dice using YOLO labels (dir sources only).",
    )
    args = parser.parse_args()

    # Resolve default source (validation images) if not provided
    source_path = args.source
    if source_path is None:
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        source_path = (
            project_root
            / "datasets"
            / "Fuji-Apple-Segmentation_with_envy_mask_yolo"
            / "images"
            / "val"
        )
        print(f"No --source provided. Using default val set: {source_path}")
    else:
        source_path = Path(source_path)

    # Resolve weights
    weights_path = args.weights
    if weights_path == "latest":
        try:
            # Default checkpoint directory matches train_yolo.py
            script_dir = Path(__file__).resolve().parent
            default_runs = script_dir.parent / "datasets" / "checkpoints" / "yolo"

            weights_path = find_latest_checkpoint(default_runs)
            print(f"Resolved 'latest' weights to: {weights_path}")
        except Exception as e:
            print(f"Error resolving latest weights: {e}")
            return

    print(f"Loading model from {weights_path}...")
    model = YOLO(weights_path)

    print(f"Running inference on {source_path}...")

    # Create systematic output directory
    script_dir = Path(__file__).resolve().parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    weights_name = Path(weights_path).stem
    project_dir = script_dir.parent / "datasets" / "checkpoints" / "yolo"
    run_name = f"inference_{timestamp}_{weights_name}"

    # save=True saves to project/name
    model.predict(
        source=source_path,
        conf=args.conf,
        save=True,
        project=str(project_dir),
        name=run_name,
    )
    print(f"[INFO] Inference results saved to {project_dir / run_name}")

    if args.calc_metrics:
        # Attempt to load class names from dataset.yaml next to the source
        names: Dict[int, str] = {}
        dataset_dir = (
            source_path.parent.parent
            if source_path.is_dir()
            else source_path.parent.parent.parent
        )
        yaml_path = dataset_dir / "dataset.yaml"
        if yaml_path.exists():
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
                names = (
                    {int(k): v for k, v in data.get("names", {}).items()}
                    if isinstance(data.get("names"), dict)
                    else {i: n for i, n in enumerate(data.get("names", []))}
                )
        else:
            # fallback to numeric ids
            names = {0: "class_0", 1: "class_1"}

        metadata = {
            "model": "YOLO",
            "weights": str(weights_path),
            "conf": args.conf,
            "source": str(source_path),
            "timestamp": timestamp,
            "run_name": run_name,
        }

        evaluate_directory(
            model, Path(source_path), names, args.conf, metadata=metadata
        )


if __name__ == "__main__":
    main()
