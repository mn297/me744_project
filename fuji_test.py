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
sam2_model = build_sam2(model_cfg, str(sam2_checkpoint), device=device)
predictor = SAM2ImagePredictor(sam2_model)


print(f"Loading SAM2 model...")
print(f"  Checkpoint: {sam2_checkpoint}")
print(f"  Config: {model_cfg}")
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
