"""
Convert Fuji-SfM Apple Dataset from VIA CSV format to COCO format.

This script:
1. Reads CSV annotation files with polygon coordinates
2. Converts to COCO format JSON
3. Copies images to organized structure
4. Creates COCO-compliant annotations.json files
"""

import json
import csv
import shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_polygon_from_csv_row(row):
    """Parse polygon coordinates from CSV row."""
    try:
        shape_attrs = json.loads(row["region_shape_attributes"])
        x_coords = shape_attrs["all_points_x"]
        y_coords = shape_attrs["all_points_y"]

        # Convert to COCO format: flatten to [x1,y1,x2,y2,...]
        polygon = []
        for x, y in zip(x_coords, y_coords):
            polygon.extend([float(x), float(y)])

        return polygon
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Failed to parse polygon: {e}")
        return None


def calculate_bbox_from_polygon(polygon):
    """Calculate bounding box from polygon coordinates."""
    # Polygon is [x1,y1,x2,y2,...]
    x_coords = polygon[::2]  # Every even index
    y_coords = polygon[1::2]  # Every odd index

    x_min = min(x_coords)
    y_min = min(y_coords)
    x_max = max(x_coords)
    y_max = max(y_coords)

    width = x_max - x_min
    height = y_max - y_min

    return [x_min, y_min, width, height]


def calculate_area_from_polygon(polygon):
    """Calculate area using shoelace formula."""
    # Polygon is [x1,y1,x2,y2,...]
    x_coords = polygon[::2]
    y_coords = polygon[1::2]

    # Shoelace formula
    area = 0.0
    n = len(x_coords)
    for i in range(n):
        j = (i + 1) % n
        area += x_coords[i] * y_coords[j]
        area -= x_coords[j] * y_coords[i]

    return abs(area) / 2.0


def process_split(
    source_dir: Path,
    output_dir: Path,
    split_name: str,
    category_id: int = 1,  # Single category: apple
):
    """
    Process a split (train/val) and convert to COCO format.

    Args:
        source_dir: Source directory with images and CSV files
        output_dir: Output directory for COCO format
        split_name: Name of split (train/val)
        category_id: Category ID for apples (default 1)
    """
    print(f"\n{'='*80}")
    print(f"Processing {split_name} split")
    print(f"{'='*80}")

    # Create output directories
    images_dir = output_dir / "JPEGImages"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Find all CSV files
    csv_files = sorted(source_dir.glob("mask_*.csv"))
    print(f"Found {len(csv_files)} CSV annotation files")

    # COCO data structures
    images = []
    annotations = []

    image_id = 1
    annotation_id = 1

    # Process each CSV file
    for csv_file in tqdm(csv_files, desc=f"Processing {split_name}"):
        # Get corresponding image filename
        # CSV: mask__MG_8065_01.csv -> Image: _MG_8065_01.jpg
        image_filename = csv_file.name.replace("mask_", "")
        if not image_filename.endswith((".jpg", ".jpeg", ".png")):
            # Add .jpg extension if not present
            image_filename = image_filename.rsplit(".", 1)[0] + ".jpg"
        image_path = source_dir / image_filename

        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue

        # Get image dimensions
        try:
            img = Image.open(image_path)
            width, height = img.size
        except Exception as e:
            print(f"Warning: Could not read image {image_path}: {e}")
            continue

        # Add image entry
        images.append(
            {
                "id": image_id,
                "file_name": image_filename,
                "width": width,
                "height": height,
            }
        )

        # Read CSV annotations
        try:
            with open(csv_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Parse polygon
                    polygon = parse_polygon_from_csv_row(row)
                    if polygon is None or len(polygon) < 6:  # Need at least 3 points
                        continue

                    # Calculate bbox and area
                    bbox = calculate_bbox_from_polygon(polygon)
                    area = calculate_area_from_polygon(polygon)

                    # Add annotation
                    annotations.append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category_id,
                            "segmentation": [polygon],  # COCO format: list of polygons
                            "area": area,
                            "bbox": bbox,
                            "iscrowd": 0,
                        }
                    )

                    annotation_id += 1
        except Exception as e:
            print(f"Warning: Could not read CSV {csv_file}: {e}")
            continue

        # Copy image to output directory
        output_image_path = images_dir / image_filename
        shutil.copy2(image_path, output_image_path)

        image_id += 1

    # Create categories (single category: apple)
    categories = [
        {
            "id": category_id,
            "name": "apple",
            "supercategory": "fruit",
        }
    ]

    # Create COCO JSON structure
    coco_data = {
        "info": {
            "description": "Fuji-SfM Apple Segmentation Dataset",
            "version": "1.0",
            "year": 2024,
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    # Save annotations JSON
    annotations_file = output_dir / "annotations.json"
    with open(annotations_file, "w") as f:
        json.dump(coco_data, f, indent=2)

    print(f"\n✅ {split_name} split converted:")
    print(f"   Images: {len(images)}")
    print(f"   Annotations: {len(annotations)}")
    print(f"   Categories: {len(categories)}")
    print(f"   Output: {output_dir}")

    return coco_data


def main():
    """Main conversion function."""
    # Paths
    base_dir = Path(__file__).parent.parent
    fuji_dir = base_dir / "datasets" / "Fuji-SfM_dataset" / "1-Mask-set"
    output_base = (
        base_dir / "datasets" / "Fuji-Apple-Segmentation_coco"
    )  # Next to Car-Parts-Segmentation

    # Source directories
    train_source = fuji_dir / "training_images_and_annotations"
    val_source = fuji_dir / "validation_images_and_annotations"

    # Output directories
    train_output = output_base / "trainingset"
    val_output = output_base / "testset"

    print("=" * 80)
    print("FUJI-SFM TO COCO FORMAT CONVERTER")
    print("=" * 80)
    print(f"Source: {fuji_dir}")
    print(f"Output: {output_base}")
    print("=" * 80)

    # Verify source directories exist
    if not train_source.exists():
        raise FileNotFoundError(f"Training directory not found: {train_source}")
    if not val_source.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_source}")

    # Process training set
    train_data = process_split(train_source, train_output, "training")

    # Process validation set
    val_data = process_split(val_source, val_output, "validation")

    # Summary
    print(f"\n{'='*80}")
    print("CONVERSION COMPLETE")
    print(f"{'='*80}")
    print(f"Training set:")
    print(f"  Images: {len(train_data['images'])}")
    print(f"  Annotations: {len(train_data['annotations'])}")
    print(f"\nValidation set:")
    print(f"  Images: {len(val_data['images'])}")
    print(f"  Annotations: {len(val_data['annotations'])}")
    print(f"\nOutput directory: {output_base}")
    print(f"{'='*80}\n")

    print("✅ Dataset is now COCO-compliant!")
    print(f"\nYou can now use it with:")
    print(f"  --train-images Fuji-Apple-Segmentation_coco/trainingset/JPEGImages")
    print(f"  --train-anno Fuji-Apple-Segmentation_coco/trainingset/annotations.json")
    print(f"  --val-images Fuji-Apple-Segmentation_coco/testset/JPEGImages")
    print(f"  --val-anno Fuji-Apple-Segmentation_coco/testset/annotations.json")


if __name__ == "__main__":
    main()
