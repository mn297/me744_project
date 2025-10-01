# %%
"""
Simple CNN Training for Apple Segmentation
Trains a U-Net style network on Fuji-SfM dataset
"""
import os
import json
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# For WSL2
os.environ["XDG_SESSION_TYPE"] = "x11"

# %%
# ============================================================================
# DATASET CLASS
# ============================================================================


class AppleDataset(Dataset):
    """Load Fuji-SfM apple images and masks."""

    def __init__(self, data_dir, split="train", image_size=256):
        """
        Args:
            data_dir: Path to Fuji-SfM_dataset
            split: 'train' or 'val'
            image_size: Resize images to this size
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.split = split

        # Set paths based on split
        if split == "train":
            self.img_dir = (
                self.data_dir / "1-Mask-set" / "training_images_and_annotations"
            )
        else:
            self.img_dir = (
                self.data_dir / "1-Mask-set" / "validation_images_and_annotations"
            )

        # Find all mask files
        self.mask_files = sorted(list(self.img_dir.glob("mask_*.csv")))

        # Filter out files with no annotations
        self.valid_samples = []
        for mask_file in self.mask_files:
            image_name = mask_file.name.replace("mask_", "").replace(".csv", ".jpg")
            image_path = self.img_dir / image_name

            if image_path.exists():
                # Check if has annotations
                polygons = self._load_polygons(mask_file)
                if len(polygons) > 0:
                    self.valid_samples.append((image_path, mask_file))

        print(f"{split}: Found {len(self.valid_samples)} images with annotations")

    def _load_polygons(self, csv_path):
        """Load polygon coordinates from CSV."""
        polygons = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                shape_attrs = json.loads(row["region_shape_attributes"])
                x_coords = shape_attrs["all_points_x"]
                y_coords = shape_attrs["all_points_y"]
                polygon = np.array(list(zip(x_coords, y_coords)))
                polygons.append(polygon)
        return polygons

    def _create_mask(self, polygons, image_shape):
        """Create binary mask from polygons."""
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        for polygon in polygons:
            polygon = polygon.astype(np.int32)
            cv2.fillPoly(mask, [polygon], 1)
        return mask

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        image_path, mask_path = self.valid_samples[idx]

        # Load image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load and create mask
        polygons = self._load_polygons(mask_path)
        mask = self._create_mask(polygons, image.shape)

        # Resize
        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size))

        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        return image, mask, str(image_path.name)


# ============================================================================
# SIMPLE U-NET MODEL
# ============================================================================


class ConvBlock(nn.Module):
    """Convolution block: Conv -> BatchNorm -> ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class SimpleUNet(nn.Module):
    """Simple U-Net for binary segmentation."""

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = ConvBlock(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ConvBlock(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        # Output
        self.out = nn.Conv2d(64, out_channels, 1)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Return logits (no sigmoid - will use BCEWithLogitsLoss)
        return self.out(d1)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================


def dice_loss(pred_logits, target, smooth=1e-5):
    """Dice loss for binary segmentation."""
    pred = torch.sigmoid(pred_logits)
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return 1 - dice


def combined_loss(pred_logits, target):
    """Combined BCE + Dice loss. Pred should be logits (no sigmoid)."""
    bce = F.binary_cross_entropy_with_logits(pred_logits, target)
    dice = dice_loss(pred_logits, target)
    return 0.5 * bce + 0.5 * dice


def calculate_iou(pred_logits, target, threshold=0.5):
    """Calculate IoU metric. Pred should be logits (no sigmoid)."""
    pred = torch.sigmoid(pred_logits)
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + 1e-5) / (union + 1e-5)
    return iou.item()


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_iou = 0

    pbar = tqdm(dataloader, desc="Training")
    for images, masks, _ in pbar:
        images, masks = images.to(device), masks.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)

        # Calculate loss
        loss = combined_loss(outputs, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        total_iou += calculate_iou(outputs, masks)

        pbar.set_postfix({"loss": loss.item(), "iou": calculate_iou(outputs, masks)})

    avg_loss = total_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)

    return avg_loss, avg_iou


def validate(model, dataloader, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_iou = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, masks, _ in pbar:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = combined_loss(outputs, masks)

            total_loss += loss.item()
            total_iou += calculate_iou(outputs, masks)

            pbar.set_postfix(
                {"loss": loss.item(), "iou": calculate_iou(outputs, masks)}
            )

    avg_loss = total_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)

    return avg_loss, avg_iou


def visualize_predictions(model, dataset, device, num_samples=6):
    """Visualize model predictions."""
    model.eval()

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

    indices = np.random.choice(len(dataset), num_samples, replace=False)

    for i, idx in enumerate(indices):
        image, mask, name = dataset[idx]

        # Get prediction
        with torch.no_grad():
            pred_logits = model(image.unsqueeze(0).to(device))
            pred = torch.sigmoid(pred_logits).cpu().squeeze()

        # Convert to numpy
        image_np = image.permute(1, 2, 0).numpy()
        mask_np = mask.squeeze().numpy()
        pred_np = pred.numpy()
        pred_binary = (pred_np > 0.5).astype(np.float32)

        # Plot
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title("Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask_np, cmap="gray")
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred_np, cmap="jet", vmin=0, vmax=1)
        axes[i, 2].set_title("Prediction (prob)")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(pred_binary, cmap="gray")
        axes[i, 3].set_title("Prediction (binary)")
        axes[i, 3].axis("off")

    plt.tight_layout()
    return fig


def visualize_comparison(model, dataset, device, idx=0):
    """Detailed comparison for single image."""
    model.eval()

    image, mask, name = dataset[idx]

    # Get prediction
    with torch.no_grad():
        pred_logits = model(image.unsqueeze(0).to(device))
        pred = torch.sigmoid(pred_logits).cpu().squeeze()

    # Convert to numpy
    image_np = image.permute(1, 2, 0).numpy()
    mask_np = mask.squeeze().numpy()
    pred_np = pred.numpy()
    pred_binary = (pred_np > 0.5).astype(np.float32)

    # Calculate metrics
    iou = calculate_iou(pred_logits.cpu(), mask.unsqueeze(0))

    # Create overlay
    overlay = image_np.copy()
    overlay[:, :, 0] = np.maximum(overlay[:, :, 0], mask_np * 0.5)
    overlay[:, :, 1] = np.maximum(overlay[:, :, 1], pred_binary * 0.5)

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(mask_np, cmap="gray")
    axes[0, 1].set_title("Ground Truth Mask")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(pred_binary, cmap="gray")
    axes[0, 2].set_title(f"Predicted Mask\nIoU: {iou:.3f}")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(pred_np, cmap="jet", vmin=0, vmax=1)
    axes[1, 0].set_title("Prediction Confidence")
    axes[1, 0].axis("off")
    plt.colorbar(axes[1, 0].imshow(pred_np, cmap="jet", vmin=0, vmax=1), ax=axes[1, 0])

    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title("Overlay (Red=GT, Green=Pred)")
    axes[1, 1].axis("off")

    # Difference map
    diff = np.abs(mask_np - pred_binary)
    axes[1, 2].imshow(diff, cmap="hot")
    axes[1, 2].set_title("Difference Map")
    axes[1, 2].axis("off")

    plt.suptitle(f"Image: {name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    return fig


# %%
# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

# Configuration
DATASET_PATH = Path("fuji_sfm_data/Fuji-SfM_dataset")
OUTPUT_DIR = Path("training_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = 256
BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Mixed precision training
use_amp = device.type == "cuda"
if use_amp:
    print("Using automatic mixed precision (AMP) for faster training")
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# Create datasets
print("\nLoading datasets...")
train_dataset = AppleDataset(DATASET_PATH, split="train", image_size=IMAGE_SIZE)
val_dataset = AppleDataset(DATASET_PATH, split="val", image_size=IMAGE_SIZE)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# Create model
print("\nInitializing model...")
model = SimpleUNet(in_channels=3, out_channels=1).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

# %%
# Training loop
print("\nStarting training...")
history = {"train_loss": [], "train_iou": [], "val_loss": [], "val_iou": []}

best_val_iou = 0.0

# if model exist
if (OUTPUT_DIR / "best_model.pth").exists():
    model.load_state_dict(torch.load(OUTPUT_DIR / "best_model.pth"))
    print("Loaded best model")
else:
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

        # Train
        train_loss, train_iou = train_epoch(model, train_loader, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")

        # Validate
        val_loss, val_iou = validate(model, val_loader, device)
        print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")

        # Update scheduler
        scheduler.step(val_loss)

        # Save history
        history["train_loss"].append(train_loss)
        history["train_iou"].append(train_iou)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)

        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pth")
            print(f"Saved best model (IoU: {best_val_iou:.4f})")

# %%
# Plot training history
print("\nPlotting training history...")
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(history["train_loss"], label="Train Loss")
axes[0].plot(history["val_loss"], label="Val Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Training and Validation Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history["train_iou"], label="Train IoU")
axes[1].plot(history["val_iou"], label="Val IoU")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("IoU")
axes[1].set_title("Training and Validation IoU")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_history.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Load best model and visualize
print("\nLoading best model for visualization...")
model.load_state_dict(torch.load(OUTPUT_DIR / "best_model.pth"))

# %%
# Visualize validation predictions
print("\nGenerating prediction visualizations...")
fig = visualize_predictions(model, val_dataset, device, num_samples=6)
plt.savefig(OUTPUT_DIR / "predictions_grid.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Detailed comparison for first few images
print("\nGenerating detailed comparisons...")
for i in range(min(3, len(val_dataset))):
    fig = visualize_comparison(model, val_dataset, device, idx=i)
    plt.savefig(OUTPUT_DIR / f"comparison_{i}.png", dpi=150, bbox_inches="tight")
    plt.show()

# %%
print(f"\nTraining complete!")
print(f"Best validation IoU: {best_val_iou:.4f}")
print(f"Outputs saved to: {OUTPUT_DIR}")
