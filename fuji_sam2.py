# %%
"""
Fuji-SfM Dataset Processing with SAM2 Segmentation
Based on SAM2 official image_predictor_example.ipynb
"""
import os
import open3d as o3d
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


# If using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["XDG_SESSION_TYPE"] = "x11"  # For WSL2 display


# Select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Enable optimizations for CUDA if available
if device.type == "cuda":
    # Use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # Turn on tfloat32 for Ampere GPUs
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS."
    )

# %%
# ============================================================================
# VISUALIZATION HELPER FUNCTIONS (from SAM2 official notebook)
# ============================================================================

np.random.seed(3)


def show_mask(mask, ax, random_color=False, borders=True):
    """Display a single mask with optional borders."""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [
            cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours
        ]
        mask_image = cv2.drawContours(
            mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2
        )
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    """Display prompt points (green=foreground, red=background)."""
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    """Display a bounding box."""
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def show_masks(
    image,
    masks,
    scores,
    point_coords=None,
    box_coords=None,
    input_labels=None,
    borders=True,
):
    """Display all masks with prompts."""
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis("off")
        plt.show()


def show_anns(anns, borders=True):
    """
    Display automatic mask generation results.

    Args:
        anns: List of mask annotations from SAM2AutomaticMaskGenerator
        borders: Whether to draw borders around masks
    """
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            contours, _ = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            # Try to smooth contours
            contours = [
                cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
                for contour in contours
            ]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    ax.imshow(img)


def load_fuji_sfm_data(dataset_path: Path):
    """
    Load Fuji-SfM point cloud data from txt file.
    Format: X Y Z R G B (tab-separated)
    """

    file_path = dataset_path / "3-3D_data" / "Fuji_apple_trees_point_cloud.txt"
    print(f"Loading point cloud from {file_path}...")

    # Load the data using numpy
    data = np.loadtxt(str(file_path), delimiter="\t")
    print(f"Loaded {data.shape[0]} points")

    # Extract XYZ coordinates and RGB colors
    points = data[:, :3]  # First 3 columns: X, Y, Z
    colors = data[:, 3:6]  # Last 3 columns: R, G, B
    colors = colors / 255.0  # Normalize colors from 0-255 to 0-1 range

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print(f"Point cloud bounds:")
    print(f"  X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"  Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"  Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")

    return pcd


def save_point_cloud(pcd, base_path: Path):
    """
    Save point cloud in common formats used in 3D and robotics.

    Args:
        pcd: Open3D point cloud object
        base_path: Path object for output files (without extension)
    """
    # Save as PLY (Polygon File Format - widely used in 3D)
    ply_path = base_path.with_suffix(".ply")
    if not ply_path.exists():
        o3d.io.write_point_cloud(str(ply_path), pcd)
    print(f"Saved PLY format: {ply_path}")

    # Save as PCD (Point Cloud Data - standard in robotics/ROS/PCL)
    pcd_path = base_path.with_suffix(".pcd")
    if not pcd_path.exists():
        o3d.io.write_point_cloud(
            str(pcd_path), pcd, write_ascii=False
        )  # Binary PCD is more compact
    print(f"Saved PCD format (binary): {pcd_path}")

    # Optionally save as ASCII PCD for human readability
    pcd_ascii_path = Path(str(base_path) + "_ascii.pcd")
    if not pcd_ascii_path.exists():
        o3d.io.write_point_cloud(str(pcd_ascii_path), pcd, write_ascii=True)
    print(f"Saved PCD format (ASCII): {pcd_ascii_path}")


# %%
HOME_PATH = Path.home()
SAM2_PATH = HOME_PATH / "sam2"
DATASET_PATH = Path("fuji_sfm_data/Fuji-SfM_dataset/")
IMG_PATH = DATASET_PATH / "1-Mask-set" / "raw_images"
OUTPUT_PATH = DATASET_PATH / "sam2_results"
OUTPUT_PATH.mkdir(exist_ok=True)

print(f"Dataset path: {DATASET_PATH}")
print(f"Images path: {IMG_PATH}")
print(f"Output path: {OUTPUT_PATH}")
sam2_checkpoint = SAM2_PATH / "checkpoints" / "sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

print(f"Loading SAM2 model...")
print(f"  Checkpoint: {sam2_checkpoint}")
print(f"  Config: {model_cfg}")

sam2_model = build_sam2(model_cfg, str(sam2_checkpoint), device=device)
predictor = SAM2ImagePredictor(sam2_model)

print("✓ SAM2 model loaded successfully")

# Get first image from dataset
image_files = sorted([f for f in IMG_PATH.iterdir() if f.is_file()])
print(f"Found {len(image_files)} images in dataset")

# Load first image
image_path = image_files[0]
print(f"Loading image: {image_path.name}")

image = cv2.imread(str(image_path))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display image
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.title(f"Image: {image_path.name}")
plt.axis("on")
plt.show()

print(f"Image shape: {image.shape}")

# %%
# ============================================================================
# EXAMPLE 1: Single Point Prompt
# ============================================================================

print("\n=== Example 1: Single Point Prompt ===")

# Set image for prediction
predictor.set_image(image)

# Select a point in the image (center point as example)
input_point = np.array([[image.shape[1] // 2, image.shape[0] // 2]])
input_label = np.array([1])  # 1 = foreground point

# Display the selected point
plt.figure(figsize=(10, 10))
plt.imshow(image)
show_points(input_point, input_label, plt.gca())
plt.title("Selected Point Prompt")
plt.axis("on")
plt.show()

# Predict masks
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

# Sort by score (best first)
sorted_ind = np.argsort(scores)[::-1]
masks = masks[sorted_ind]
scores = scores[sorted_ind]
logits = logits[sorted_ind]

print(f"Generated {len(masks)} masks")
print(f"Scores: {scores}")

# Display all masks
show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label)

# %%
# ============================================================================
# EXAMPLE 2: Multiple Points for Specific Object
# ============================================================================

print("\n=== Example 2: Multiple Points ===")

# Define multiple points (adjust coordinates as needed)
h, w = image.shape[:2]
input_point = np.array([[w // 3, h // 3], [2 * w // 3, 2 * h // 3]])
input_label = np.array([1, 1])  # Both foreground points

# Use the best mask from previous prediction as additional input
mask_input = logits[0:1, :, :]  # Use best mask logits

# Display the selected points
plt.figure(figsize=(10, 10))
plt.imshow(image)
show_points(input_point, input_label, plt.gca())
plt.title("Multiple Point Prompts")
plt.axis("on")
plt.show()

# Predict with multiple points and mask input
masks, scores, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    mask_input=mask_input,
    multimask_output=False,  # Single mask output
)

print(f"Generated {len(masks)} mask")
print(f"Score: {scores[0]:.3f}")

# Display result
show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label)

# %%
# ============================================================================
# EXAMPLE 3: Bounding Box Prompt
# ============================================================================

print("\n=== Example 3: Bounding Box Prompt ===")

# Define a bounding box [x1, y1, x2, y2]
h, w = image.shape[:2]
input_box = np.array([w // 4, h // 4, 3 * w // 4, 3 * h // 4])

# Display the box
plt.figure(figsize=(10, 10))
plt.imshow(image)
show_box(input_box, plt.gca())
plt.title("Bounding Box Prompt")
plt.axis("on")
plt.show()

# Predict with box
masks, scores, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_box[None, :],
    multimask_output=False,
)

print(f"Generated {len(masks)} mask")
print(f"Score: {scores[0]:.3f}")

# Display result
show_masks(image, masks, scores, box_coords=input_box)

# %%
# ============================================================================
# EXAMPLE 4: Combining Points and Box
# ============================================================================

print("\n=== Example 4: Combining Points and Box ===")

# Define both point and box
input_point = np.array([[w // 2, h // 2]])
input_label = np.array([1])
input_box = np.array([w // 4, h // 4, 3 * w // 4, 3 * h // 4])

# Display prompts
plt.figure(figsize=(10, 10))
plt.imshow(image)
show_points(input_point, input_label, plt.gca())
show_box(input_box, plt.gca())
plt.title("Combined Point and Box Prompts")
plt.axis("on")
plt.show()

# Predict with both prompts
masks, scores, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    box=input_box[None, :],
    multimask_output=False,
)

print(f"Generated {len(masks)} mask")
print(f"Score: {scores[0]:.3f}")

# Display result
show_masks(
    image,
    masks,
    scores,
    point_coords=input_point,
    input_labels=input_label,
    box_coords=input_box,
)

# %%
# ============================================================================
# BATCH PROCESSING: Segment All Images
# ============================================================================

print("\n=== Batch Processing All Images ===")

num_images_to_process = min(5, len(image_files))  # Process first 5 images
print(f"Processing {num_images_to_process} images...")

for i, img_path in enumerate(image_files[:num_images_to_process]):
    print(f"\n[{i+1}/{num_images_to_process}] Processing: {img_path.name}")

    # Load and prepare image
    image = cv2.imread(str(img_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    # Use center point as automatic prompt
    h, w = image.shape[:2]
    input_point = np.array([[w // 2, h // 2]])
    input_label = np.array([1])

    # Predict
    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    # Get best mask
    best_idx = np.argmax(scores)
    best_mask = masks[best_idx]
    best_score = scores[best_idx]

    print(f"  Best mask score: {best_score:.3f}")

    # Save visualization
    fig = plt.figure(figsize=(15, 5))

    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original")
    plt.axis("off")

    # Mask overlay
    plt.subplot(1, 3, 2)
    plt.imshow(image)
    show_mask(best_mask, plt.gca(), borders=True)
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Best Mask (score: {best_score:.3f})")
    plt.axis("off")

    # Mask only
    plt.subplot(1, 3, 3)
    plt.imshow(best_mask, cmap="gray")
    plt.title("Mask Only")
    plt.axis("off")

    plt.tight_layout()
    save_path = OUTPUT_PATH / f"{img_path.stem}_segmentation.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {save_path}")

print(f"\n✓ Batch processing complete! Results saved to: {OUTPUT_PATH}")

# %%
# ============================================================================
# AUTOMATIC MASK GENERATION
# ============================================================================

print("\n=== Automatic Mask Generation ===")
print("Generating masks for entire image without prompts...")

# Load the model for automatic mask generation (with apply_postprocessing=False)
sam2_auto = build_sam2(
    model_cfg, str(sam2_checkpoint), device=device, apply_postprocessing=False
)

# Create automatic mask generator with default settings
mask_generator = SAM2AutomaticMaskGenerator(sam2_auto)

# Load an example image
image_path = image_files[0]
print(f"Processing: {image_path.name}")
image = cv2.imread(str(image_path))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Generate masks automatically
print("Generating masks... (this may take a moment)")
masks = mask_generator.generate(image)

print(f"Generated {len(masks)} masks")
print(f"Each mask contains keys: {masks[0].keys()}")

# Show mask statistics
if len(masks) > 0:
    areas = [m["area"] for m in masks]
    ious = [m["predicted_iou"] for m in masks]
    stability_scores = [m["stability_score"] for m in masks]

    print(f"\nMask Statistics:")
    print(
        f"  Areas: min={min(areas):.0f}, max={max(areas):.0f}, mean={np.mean(areas):.0f}"
    )
    print(
        f"  Predicted IoU: min={min(ious):.3f}, max={max(ious):.3f}, mean={np.mean(ious):.3f}"
    )
    print(
        f"  Stability scores: min={min(stability_scores):.3f}, max={max(stability_scores):.3f}, mean={np.mean(stability_scores):.3f}"
    )

# Display results
plt.figure(figsize=(20, 20))
plt.imshow(image)
show_anns(masks, borders=True)
plt.title(f"Automatic Mask Generation - {len(masks)} masks", fontsize=20)
plt.axis("off")
plt.savefig(
    OUTPUT_PATH / f"{image_path.stem}_automatic_masks.png", dpi=150, bbox_inches="tight"
)
plt.show()

print(f"Saved to: {OUTPUT_PATH / f'{image_path.stem}_automatic_masks.png'}")

# %%
# ============================================================================
# AUTOMATIC MASK GENERATION - ADVANCED SETTINGS
# ============================================================================

print("\n=== Automatic Mask Generation - Advanced Settings ===")
print("Using higher quality settings for more detailed segmentation...")

# Create mask generator with advanced settings
mask_generator_advanced = SAM2AutomaticMaskGenerator(
    model=sam2_auto,
    points_per_side=64,  # More points = more masks (default: 32)
    points_per_batch=128,  # Batch size for point processing
    pred_iou_thresh=0.7,  # IoU threshold for keeping masks (default: 0.88)
    stability_score_thresh=0.92,  # Stability threshold (default: 0.95)
    stability_score_offset=0.7,  # Offset for stability calculation
    crop_n_layers=1,  # Number of crop layers (0 = no crops)
    box_nms_thresh=0.7,  # NMS threshold for removing duplicate masks
    crop_n_points_downscale_factor=2,  # Downscaling for crops
    min_mask_region_area=25.0,  # Minimum mask area in pixels
    use_m2m=True,  # Use mask-to-mask refinement
)

print("Generating masks with advanced settings... (this will take longer)")
masks_advanced = mask_generator_advanced.generate(image)

print(f"Generated {len(masks_advanced)} masks (vs {len(masks)} with default settings)")

# Show comparison statistics
if len(masks_advanced) > 0:
    areas = [m["area"] for m in masks_advanced]
    ious = [m["predicted_iou"] for m in masks_advanced]
    stability_scores = [m["stability_score"] for m in masks_advanced]

    print(f"\nAdvanced Mask Statistics:")
    print(
        f"  Areas: min={min(areas):.0f}, max={max(areas):.0f}, mean={np.mean(areas):.0f}"
    )
    print(
        f"  Predicted IoU: min={min(ious):.3f}, max={max(ious):.3f}, mean={np.mean(ious):.3f}"
    )
    print(
        f"  Stability scores: min={min(stability_scores):.3f}, max={max(stability_scores):.3f}, mean={np.mean(stability_scores):.3f}"
    )

# Display results
plt.figure(figsize=(20, 20))
plt.imshow(image)
show_anns(masks_advanced, borders=True)
plt.title(
    f"Advanced Automatic Mask Generation - {len(masks_advanced)} masks", fontsize=20
)
plt.axis("off")
plt.savefig(
    OUTPUT_PATH / f"{image_path.stem}_automatic_masks_advanced.png",
    dpi=150,
    bbox_inches="tight",
)
plt.show()

print(f"Saved to: {OUTPUT_PATH / f'{image_path.stem}_automatic_masks_advanced.png'}")

# %%
# ============================================================================
# COMPARISON: Default vs Advanced Settings
# ============================================================================

print("\n=== Comparison: Default vs Advanced Settings ===")

fig, axes = plt.subplots(1, 3, figsize=(30, 10))

# Original image
axes[0].imshow(image)
axes[0].set_title(f"Original Image\n{image_path.name}", fontsize=16)
axes[0].axis("off")

# Default settings
axes[1].imshow(image)
ax_backup = plt.gca()
plt.sca(axes[1])
show_anns(masks, borders=True)
axes[1].set_title(f"Default Settings\n{len(masks)} masks", fontsize=16)
axes[1].axis("off")

# Advanced settings
axes[2].imshow(image)
plt.sca(axes[2])
show_anns(masks_advanced, borders=True)
axes[2].set_title(f"Advanced Settings\n{len(masks_advanced)} masks", fontsize=16)
axes[2].axis("off")

plt.sca(ax_backup)
plt.tight_layout()
plt.savefig(
    OUTPUT_PATH / f"{image_path.stem}_comparison.png", dpi=150, bbox_inches="tight"
)
plt.show()

print(f"Saved comparison to: {OUTPUT_PATH / f'{image_path.stem}_comparison.png'}")

# %%
# ============================================================================
# BATCH AUTOMATIC MASK GENERATION
# ============================================================================

print("\n=== Batch Automatic Mask Generation ===")

num_images_to_process = min(3, len(image_files))
print(f"Processing {num_images_to_process} images with automatic mask generation...")

# Use default settings for batch processing (faster)
for i, img_path in enumerate(image_files[:num_images_to_process]):
    print(f"\n[{i+1}/{num_images_to_process}] Processing: {img_path.name}")

    # Load image
    image = cv2.imread(str(img_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Generate masks
    masks = mask_generator.generate(image)
    print(f"  Generated {len(masks)} masks")

    # Create visualization
    fig = plt.figure(figsize=(20, 10))

    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image", fontsize=16)
    plt.axis("off")

    # Masks overlay
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    show_anns(masks, borders=True)
    plt.title(f"Automatic Masks ({len(masks)} objects)", fontsize=16)
    plt.axis("off")

    plt.tight_layout()
    save_path = OUTPUT_PATH / f"{img_path.stem}_auto_masks.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {save_path}")

print(f"\n✓ Batch automatic mask generation complete!")

# %%
# ============================================================================
# FILTERING AND ANALYZING MASKS
# ============================================================================

print("\n=== Filtering and Analyzing Masks ===")

# Load an image and generate masks
image_path = image_files[0]
image = cv2.imread(str(image_path))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
masks = mask_generator.generate(image)

print(f"Total masks generated: {len(masks)}")

# Filter masks by different criteria
high_quality_masks = [m for m in masks if m["predicted_iou"] > 0.9]
large_masks = [m for m in masks if m["area"] > 1000]
small_masks = [m for m in masks if m["area"] < 500]
stable_masks = [m for m in masks if m["stability_score"] > 0.95]

print(f"\nFiltered mask counts:")
print(f"  High quality (IoU > 0.9): {len(high_quality_masks)}")
print(f"  Large objects (area > 1000px): {len(large_masks)}")
print(f"  Small objects (area < 500px): {len(small_masks)}")
print(f"  Stable masks (stability > 0.95): {len(stable_masks)}")

# Visualize filtered masks
fig, axes = plt.subplots(2, 2, figsize=(20, 20))

# All masks
axes[0, 0].imshow(image)
plt.sca(axes[0, 0])
show_anns(masks, borders=True)
axes[0, 0].set_title(f"All Masks ({len(masks)})", fontsize=14)
axes[0, 0].axis("off")

# High quality masks
axes[0, 1].imshow(image)
plt.sca(axes[0, 1])
show_anns(high_quality_masks, borders=True)
axes[0, 1].set_title(f"High Quality Masks ({len(high_quality_masks)})", fontsize=14)
axes[0, 1].axis("off")

# Large masks
axes[1, 0].imshow(image)
plt.sca(axes[1, 0])
show_anns(large_masks, borders=True)
axes[1, 0].set_title(f"Large Objects ({len(large_masks)})", fontsize=14)
axes[1, 0].axis("off")

# Stable masks
axes[1, 1].imshow(image)
plt.sca(axes[1, 1])
show_anns(stable_masks, borders=True)
axes[1, 1].set_title(f"Stable Masks ({len(stable_masks)})", fontsize=14)
axes[1, 1].axis("off")

plt.tight_layout()
plt.savefig(
    OUTPUT_PATH / f"{image_path.stem}_filtered_masks.png", dpi=150, bbox_inches="tight"
)
plt.show()

print(
    f"Saved filtered masks to: {OUTPUT_PATH / f'{image_path.stem}_filtered_masks.png'}"
)

# %%
