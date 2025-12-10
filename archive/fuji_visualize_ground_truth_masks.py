# %%
"""
Visualize Ground Truth Masks from Fuji-SfM Dataset
Loads and displays polygon annotations from CSV files
"""
import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

# For WSL2 display
os.environ["XDG_SESSION_TYPE"] = "x11"

# %%
# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def parse_polygon_from_csv_row(row):
    """
    Parse polygon coordinates from a CSV row.

    Args:
        row: Dictionary from CSV reader

    Returns:
        Dictionary with parsed polygon data
    """
    # Parse the JSON strings in region_shape_attributes
    shape_attrs = json.loads(row["region_shape_attributes"])

    x_coords = shape_attrs["all_points_x"]
    y_coords = shape_attrs["all_points_y"]

    # Combine into polygon points
    polygon_points = np.array(list(zip(x_coords, y_coords)))

    return {
        "filename": row["#filename"],
        "region_id": int(row["region_id"]),
        "region_count": int(row["region_count"]),
        "polygon": polygon_points,
        "name": shape_attrs["name"],
    }


def load_masks_from_csv(csv_path):
    """
    Load all mask polygons from a CSV file.

    Args:
        csv_path: Path to CSV file

    Returns:
        List of polygon dictionaries
    """
    polygons = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            poly_data = parse_polygon_from_csv_row(row)
            polygons.append(poly_data)

    return polygons


def draw_polygons_on_image(image, polygons, show_ids=True, alpha=0.4):
    """
    Draw polygon annotations on an image.

    Args:
        image: Image array (H, W, 3)
        polygons: List of polygon dictionaries
        show_ids: Whether to show apple IDs
        alpha: Transparency of masks

    Returns:
        Image with drawn polygons
    """
    fig, ax = plt.subplots(1, figsize=(15, 15))
    ax.imshow(image)

    # Create colormap for different apples
    colors = plt.cm.rainbow(np.linspace(0, 1, len(polygons)))

    patches = []
    for i, poly_data in enumerate(polygons):
        polygon = poly_data["polygon"]

        # Create matplotlib polygon
        patch = MplPolygon(
            polygon,
            closed=True,
            alpha=alpha,
            facecolor=colors[i],
            edgecolor="white",
            linewidth=2,
        )
        patches.append(patch)

        # Add apple ID text at centroid
        if show_ids:
            centroid = polygon.mean(axis=0)
            ax.text(
                centroid[0],
                centroid[1],
                str(poly_data["region_id"]),
                fontsize=12,
                fontweight="bold",
                color="white",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
            )

    # Add patches to axis
    for patch in patches:
        ax.add_patch(patch)

    ax.axis("off")
    ax.set_title(
        f"Ground Truth Annotations - {len(polygons)} apples",
        fontsize=16,
        fontweight="bold",
    )

    return fig, ax


def create_binary_masks(image_shape, polygons):
    """
    Create binary masks from polygons.

    Args:
        image_shape: (height, width) of image
        polygons: List of polygon dictionaries

    Returns:
        List of binary masks, one per apple
    """
    masks = []

    for poly_data in polygons:
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        polygon = poly_data["polygon"].astype(np.int32)
        cv2.fillPoly(mask, [polygon], 1)
        masks.append(mask)

    return masks


def visualize_masks_grid(image, masks, polygons, max_display=16):
    """
    Visualize individual masks in a grid.

    Args:
        image: Original image
        masks: List of binary masks
        polygons: List of polygon dictionaries
        max_display: Maximum number of masks to display
    """
    num_masks = min(len(masks), max_display)
    cols = 4
    rows = (num_masks + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten() if num_masks > 1 else [axes]

    for i in range(num_masks):
        ax = axes[i]

        # Show image with mask overlay
        ax.imshow(image)

        # Create colored overlay for this mask
        mask_colored = np.zeros_like(image, dtype=np.float32)
        color = plt.cm.rainbow(i / num_masks)[:3]
        for c in range(3):
            mask_colored[:, :, c] = masks[i] * color[c] * 255

        ax.imshow(mask_colored.astype(np.uint8), alpha=0.5)

        # Draw polygon boundary
        polygon = polygons[i]["polygon"]
        polygon_closed = np.vstack([polygon, polygon[0]])
        ax.plot(polygon_closed[:, 0], polygon_closed[:, 1], "w-", linewidth=2)

        # Get mask statistics
        area = masks[i].sum()
        bbox = cv2.boundingRect(masks[i])

        ax.set_title(
            f'Apple {polygons[i]["region_id"]}\n'
            f"Area: {area} px\n"
            f"BBox: {bbox[2]}x{bbox[3]}",
            fontsize=10,
        )
        ax.axis("off")

    # Hide unused subplots
    for i in range(num_masks, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    return fig


# %%
# ============================================================================
# PATHS CONFIGURATION
# ============================================================================

DATASET_PATH = Path("fuji_sfm_data/Fuji-SfM_dataset/")
TRAINING_PATH = DATASET_PATH / "1-Mask-set" / "training_images_and_annotations"
VALIDATION_PATH = DATASET_PATH / "1-Mask-set" / "validation_images_and_annotations"
OUTPUT_PATH = DATASET_PATH / "ground_truth_visualizations"
OUTPUT_PATH.mkdir(exist_ok=True)

print(f"Dataset path: {DATASET_PATH}")
print(f"Training path: {TRAINING_PATH}")
print(f"Output path: {OUTPUT_PATH}")

# %%
# ============================================================================
# LOAD AND VISUALIZE SINGLE IMAGE
# ============================================================================

print("\n=== Visualizing Single Image with Ground Truth ===")

# Get list of mask files
mask_files = sorted(list(TRAINING_PATH.glob("mask_*.csv")))
print(f"Found {len(mask_files)} annotated images")

# Load first example with valid annotations
mask_file = None
polygons = []
image = None
image_name = None

for mask_file_candidate in mask_files:
    # Parse filename to get image name
    image_name_candidate = mask_file_candidate.name.replace("mask_", "").replace(
        ".csv", ".jpg"
    )
    image_path_candidate = TRAINING_PATH / image_name_candidate

    if not image_path_candidate.exists():
        continue

    # Try to load polygons
    polygons_candidate = load_masks_from_csv(mask_file_candidate)

    if len(polygons_candidate) > 0:
        # Found valid file
        mask_file = mask_file_candidate
        image_name = image_name_candidate
        image_path = image_path_candidate
        polygons = polygons_candidate
        break

if mask_file is None:
    print("Error: No valid annotated images found!")
    import sys

    sys.exit(1)

print(f"\nProcessing: {mask_file.name}")

# Load image
image = cv2.imread(str(image_path))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(f"Image shape: {image.shape}")
print(f"Number of apples annotated: {len(polygons)}")

# Display image with annotations
fig, ax = draw_polygons_on_image(image, polygons, show_ids=True)
plt.savefig(
    OUTPUT_PATH / f"{Path(image_name).stem}_annotated.png", dpi=150, bbox_inches="tight"
)
plt.show()

print(f"Saved to: {OUTPUT_PATH / f'{Path(image_name).stem}_annotated.png'}")

# %%
# ============================================================================
# VISUALIZE INDIVIDUAL MASKS
# ============================================================================

print("\n=== Visualizing Individual Masks ===")

# Create binary masks
masks = create_binary_masks(image.shape, polygons)

# Show statistics
areas = [m.sum() for m in masks]
print(f"\nMask statistics:")
print(f"  Total apples: {len(masks)}")
print(
    f"  Areas (pixels): min={min(areas)}, max={max(areas)}, mean={np.mean(areas):.0f}"
)

# Visualize in grid
fig = visualize_masks_grid(image, masks, polygons, max_display=len(polygons))
plt.savefig(
    OUTPUT_PATH / f"{Path(image_name).stem}_individual_masks.png",
    dpi=150,
    bbox_inches="tight",
)
plt.show()

print(f"Saved to: {OUTPUT_PATH / f'{Path(image_name).stem}_individual_masks.png'}")

# %%
# ============================================================================
# COMPARISON: MULTIPLE VISUALIZATION STYLES
# ============================================================================

print("\n=== Creating Multiple Visualization Styles ===")

fig, axes = plt.subplots(2, 2, figsize=(20, 20))

# 1. Original image
axes[0, 0].imshow(image)
axes[0, 0].set_title("Original Image", fontsize=14, fontweight="bold")
axes[0, 0].axis("off")

# 2. Polygons with IDs
axes[0, 1].imshow(image)
colors = plt.cm.rainbow(np.linspace(0, 1, len(polygons)))
for i, poly_data in enumerate(polygons):
    polygon = poly_data["polygon"]
    patch = MplPolygon(
        polygon,
        closed=True,
        alpha=0.4,
        facecolor=colors[i],
        edgecolor="white",
        linewidth=2,
    )
    axes[0, 1].add_patch(patch)
    centroid = polygon.mean(axis=0)
    axes[0, 1].text(
        centroid[0],
        centroid[1],
        str(poly_data["region_id"]),
        fontsize=10,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
    )
axes[0, 1].set_title(
    f"Polygon Annotations ({len(polygons)} apples)", fontsize=14, fontweight="bold"
)
axes[0, 1].axis("off")

# 3. Binary masks (all combined)
combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
for mask in masks:
    combined_mask = np.maximum(combined_mask, mask)
axes[1, 0].imshow(image)
axes[1, 0].imshow(combined_mask, alpha=0.5, cmap="jet")
axes[1, 0].set_title("Combined Binary Masks", fontsize=14, fontweight="bold")
axes[1, 0].axis("off")

# 4. Colored instance segmentation
instance_mask = np.zeros(image.shape[:2], dtype=np.int32)
for i, mask in enumerate(masks, start=1):
    instance_mask[mask > 0] = i
axes[1, 1].imshow(image)
axes[1, 1].imshow(instance_mask, alpha=0.5, cmap="tab20")
axes[1, 1].set_title("Instance Segmentation", fontsize=14, fontweight="bold")
axes[1, 1].axis("off")

plt.tight_layout()
plt.savefig(
    OUTPUT_PATH / f"{Path(image_name).stem}_comparison.png",
    dpi=150,
    bbox_inches="tight",
)
plt.show()

print(f"Saved to: {OUTPUT_PATH / f'{Path(image_name).stem}_comparison.png'}")

# %%
# ============================================================================
# BATCH PROCESSING: VISUALIZE MULTIPLE IMAGES
# ============================================================================

print("\n=== Batch Processing Multiple Images ===")

num_images_to_process = min(10, len(mask_files))
print(f"Processing {num_images_to_process} images...")

batch_stats = []

for i, mask_file in enumerate(mask_files[:num_images_to_process]):
    print(f"\n[{i+1}/{num_images_to_process}] Processing: {mask_file.name}")

    # Get image path
    image_name = mask_file.name.replace("mask_", "").replace(".csv", ".jpg")
    image_path = TRAINING_PATH / image_name

    if not image_path.exists():
        print(f"  Warning: Image not found: {image_path}")
        continue

    # Load image and annotations
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    polygons = load_masks_from_csv(mask_file)

    # Skip if no polygons found
    if len(polygons) == 0:
        print(f"  Warning: No polygons found, skipping...")
        continue

    masks = create_binary_masks(image.shape, polygons)

    # Calculate statistics
    areas = [m.sum() for m in masks]
    batch_stats.append(
        {
            "filename": image_name,
            "num_apples": len(polygons),
            "total_area": sum(areas),
            "mean_area": np.mean(areas) if areas else 0,
            "min_area": min(areas) if areas else 0,
            "max_area": max(areas) if areas else 0,
        }
    )

    print(f"  Apples: {len(polygons)}, Mean area: {np.mean(areas):.0f} px")

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis("off")

    # Annotated image
    axes[1].imshow(image)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(polygons)))
    for j, poly_data in enumerate(polygons):
        polygon = poly_data["polygon"]
        patch = MplPolygon(
            polygon,
            closed=True,
            alpha=0.4,
            facecolor=colors[j],
            edgecolor="white",
            linewidth=2,
        )
        axes[1].add_patch(patch)
    axes[1].set_title(f"Annotations ({len(polygons)} apples)", fontsize=14)
    axes[1].axis("off")

    plt.tight_layout()
    save_path = OUTPUT_PATH / f"{Path(image_name).stem}_batch.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {save_path}")

print(f"\n✓ Batch processing complete!")

# %%
# ============================================================================
# DATASET STATISTICS
# ============================================================================

print("\n=== Dataset Statistics ===")

if batch_stats and len(batch_stats) > 0:
    total_apples = sum(s["num_apples"] for s in batch_stats)
    all_mean_areas = [s["mean_area"] for s in batch_stats]
    all_num_apples = [s["num_apples"] for s in batch_stats]

    print(f"\nOverall statistics ({len(batch_stats)} images):")
    print(f"  Total apples annotated: {total_apples}")
    print(
        f"  Apples per image: min={min(all_num_apples)}, max={max(all_num_apples)}, mean={np.mean(all_num_apples):.1f}"
    )
    print(f"  Mean apple area: {np.mean(all_mean_areas):.0f} pixels")
    print(f"  Std apple area: {np.std(all_mean_areas):.0f} pixels")

    # Create statistics visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Histogram of apples per image
    axes[0].hist(
        all_num_apples,
        bins=range(min(all_num_apples), max(all_num_apples) + 2),
        edgecolor="black",
        alpha=0.7,
    )
    axes[0].set_xlabel("Number of Apples per Image", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].set_title(
        "Distribution of Apples per Image", fontsize=14, fontweight="bold"
    )
    axes[0].grid(axis="y", alpha=0.3)

    # Histogram of mean apple areas
    axes[1].hist(all_mean_areas, bins=20, edgecolor="black", alpha=0.7, color="orange")
    axes[1].set_xlabel("Mean Apple Area (pixels)", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_title(
        "Distribution of Mean Apple Areas", fontsize=14, fontweight="bold"
    )
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "dataset_statistics.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\nSaved statistics to: {OUTPUT_PATH / 'dataset_statistics.png'}")
else:
    print("No statistics to display - no valid images were processed.")

# %%
# ============================================================================
# ADVANCED: POLYGON COMPLEXITY ANALYSIS
# ============================================================================

print("\n=== Polygon Complexity Analysis ===")

# Analyze first image
mask_file = mask_files[0]
image_name = mask_file.name.replace("mask_", "").replace(".csv", ".jpg")
image_path = TRAINING_PATH / image_name
image = cv2.imread(str(image_path))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
polygons = load_masks_from_csv(mask_file)

# Calculate polygon statistics
polygon_stats = []
for poly_data in polygons:
    polygon = poly_data["polygon"]
    num_vertices = len(polygon)

    # Calculate perimeter
    perimeter = 0
    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]
        perimeter += np.linalg.norm(p2 - p1)

    # Calculate area using shoelace formula
    x, y = polygon[:, 0], polygon[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    # Calculate bounding box
    bbox = cv2.boundingRect(polygon.astype(np.int32))
    bbox_area = bbox[2] * bbox[3]
    fill_ratio = area / bbox_area if bbox_area > 0 else 0

    polygon_stats.append(
        {
            "id": poly_data["region_id"],
            "vertices": num_vertices,
            "perimeter": perimeter,
            "area": area,
            "bbox_area": bbox_area,
            "fill_ratio": fill_ratio,
        }
    )

print(f"\nPolygon complexity for {image_name}:")
print(f"  Number of apples: {len(polygon_stats)}")
print(
    f"  Vertices per polygon: min={min(s['vertices'] for s in polygon_stats)}, "
    f"max={max(s['vertices'] for s in polygon_stats)}, "
    f"mean={np.mean([s['vertices'] for s in polygon_stats]):.1f}"
)
print(
    f"  Fill ratio: min={min(s['fill_ratio'] for s in polygon_stats):.3f}, "
    f"max={max(s['fill_ratio'] for s in polygon_stats):.3f}, "
    f"mean={np.mean([s['fill_ratio'] for s in polygon_stats]):.3f}"
)

# Visualize complexity
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Original with annotations
axes[0].imshow(image)
for poly_data in polygons:
    polygon = poly_data["polygon"]
    patch = MplPolygon(
        polygon, closed=True, alpha=0.4, facecolor="red", edgecolor="white", linewidth=2
    )
    axes[0].add_patch(patch)
axes[0].set_title("Annotated Image", fontsize=14, fontweight="bold")
axes[0].axis("off")

# Vertices count
vertices_counts = [s["vertices"] for s in polygon_stats]
axes[1].bar(range(len(vertices_counts)), vertices_counts, alpha=0.7)
axes[1].set_xlabel("Apple ID", fontsize=12)
axes[1].set_ylabel("Number of Vertices", fontsize=12)
axes[1].set_title("Polygon Complexity (Vertices)", fontsize=14, fontweight="bold")
axes[1].grid(axis="y", alpha=0.3)

# Fill ratio
fill_ratios = [s["fill_ratio"] for s in polygon_stats]
axes[2].bar(range(len(fill_ratios)), fill_ratios, alpha=0.7, color="green")
axes[2].set_xlabel("Apple ID", fontsize=12)
axes[2].set_ylabel("Fill Ratio", fontsize=12)
axes[2].set_title("Bounding Box Fill Ratio", fontsize=14, fontweight="bold")
axes[2].axhline(
    y=np.mean(fill_ratios),
    color="r",
    linestyle="--",
    label=f"Mean: {np.mean(fill_ratios):.3f}",
)
axes[2].legend()
axes[2].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(
    OUTPUT_PATH / f"{Path(image_name).stem}_complexity.png",
    dpi=150,
    bbox_inches="tight",
)
plt.show()

print(f"Saved to: {OUTPUT_PATH / f'{Path(image_name).stem}_complexity.png'}")

# %%
print("\n✓ All visualizations complete!")
print(f"Results saved to: {OUTPUT_PATH}")
