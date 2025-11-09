# %%
"""
Jupyter-style main script for Mask R-CNN training + quick dataset visualization.
- Uses # %% cell delimiters for VS Code / Spyder.
- Safe model builder that handles COCO weights vs custom num_classes.
- Prints first few samples with masks and boxes overlaid.
"""

# %% Imports
import os
from pathlib import Path
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from utils import (
    CocoSegmentationDataset,
    detection_collate,
    fit,
    build_schedulers,
    seed_everything,
)

# %% Config
DATA_TRAIN_IMAGES = "Car-Parts-Segmentation/trainingset/JPEGImages"
DATA_TRAIN_ANNO = "Car-Parts-Segmentation/trainingset/annotations.json"
DATA_VAL_IMAGES = "Car-Parts-Segmentation/testset/JPEGImages"
DATA_VAL_ANNO = "Car-Parts-Segmentation/testset/annotations.json"

EPOCHS = 20
BATCH_SIZE = 2
WORKERS = max(2, os.cpu_count() // 2)
LR = 5e-3
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
OUT_DIR = Path("checkpoints")
SEED = 42
AMP = True
BF16 = True  # if GPU supports bfloat16
COMPILE = True  # torch.compile if available
WARMUP_EPOCHS = 1
COSINE_EPOCHS = 0  # 0 => epochs - warmup
PATIENCE = 10

seed_everything(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

train_ds = CocoSegmentationDataset(DATA_TRAIN_IMAGES, DATA_TRAIN_ANNO, is_train=True)
val_ds = CocoSegmentationDataset(
    DATA_VAL_IMAGES,
    DATA_VAL_ANNO,
    is_train=False,
    catid2contig=train_ds.catid2contig,
    contig2catid=train_ds.contig2catid,
)
print(
    f"train images: {len(train_ds)} | val images: {len(val_ds)} | classes (incl bg): {train_ds.num_classes}"
)

# show category mapping
id2name = {cid: train_ds.coco.cats[cid]["name"] for cid in train_ds.coco.getCatIds()}
print("categories:", {k: id2name[v] for k, v in train_ds.contig2catid.items()})

# %% Visualization helpers


def _to_numpy_img(img_t: torch.Tensor) -> np.ndarray:
    # img_t: [C,H,W] in [0,1]; return HxWxC float
    if img_t.ndim != 3:
        raise ValueError("expected CHW tensor")
    return img_t.permute(1, 2, 0).cpu().numpy()


def show_sample(
    img_t, tgt: dict, ds: CocoSegmentationDataset, ax=None, max_instances: int = 20
):
    """Overlay masks and boxes for a single sample."""
    img = _to_numpy_img(img_t)
    H, W = img.shape[:2]
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img)

    n = min(int(tgt["masks"].shape[0]), max_instances)
    boxes = tgt["boxes"].cpu().numpy() if n > 0 else np.zeros((0, 4))
    labels = tgt["labels"].cpu().numpy() if n > 0 else np.zeros((0,), dtype=int)
    masks = (
        tgt["masks"].cpu().numpy().astype(bool)
        if n > 0
        else np.zeros((0, H, W), dtype=bool)
    )

    rng = np.random.default_rng(123)
    colors = rng.random((n, 3))

    for i in range(n):
        m = masks[i]
        color = colors[i]
        overlay = np.zeros_like(img)
        overlay[m] = color
        ax.imshow(overlay, alpha=0.35)  # transparency overlay

        x1, y1, x2, y2 = boxes[i]
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2)
        rect.set_edgecolor(color)
        ax.add_patch(rect)

        # label text
        contig = int(labels[i])
        catid = ds.contig2catid.get(contig, None)
        name = ds.coco.cats[catid]["name"] if catid in ds.coco.cats else str(contig)
        ax.text(
            x1,
            y1 - 2,
            name,
            fontsize=10,
            color="white",
            bbox=dict(facecolor=color, alpha=0.8, edgecolor="none", pad=1.5),
        )

    ax.set_axis_off()
    return ax


# %% Preview a few training samples
K = min(3, len(train_ds))
for idx in range(K):
    img, tgt = train_ds[idx]
    show_sample(img, tgt, train_ds)
    plt.show()

# %% DataLoaders
from torch.utils.data import DataLoader

pin = device.type == "cuda"
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=WORKERS,
    pin_memory=pin,
    persistent_workers=WORKERS > 0,
    collate_fn=detection_collate,
)
val_loader = DataLoader(
    val_ds,
    batch_size=1,
    shuffle=False,
    num_workers=WORKERS,
    pin_memory=pin,
    persistent_workers=WORKERS > 0,
    collate_fn=detection_collate,
)

# %% Safe model builder
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models import ResNet50_Weights


def build_model_safe(num_classes: int) -> torch.nn.Module:
    """Builds a Mask R-CNN compatible with custom num_classes.
    Strategy:
      1) Try COCO weights + head swap (best fine-tune starting point).
      2) Fallback to ImageNet backbone weights with constructor num_classes.
    """
    try:
        model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
        # swap predictors to match num_classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, 256, num_classes
        )
        return model
    except Exception as e:
        print("COCO-weighted model build failed, falling back:", e)
        return maskrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=ResNet50_Weights.DEFAULT,
            num_classes=num_classes,
        )


# %% Instantiate model
num_classes = train_ds.num_classes  # includes background
model = build_model_safe(num_classes).to(device)

if COMPILE and hasattr(torch, "compile"):
    try:
        model = torch.compile(model)
    except Exception as e:
        print("torch.compile disabled:", e)

# %% Optimizer and schedulers
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
cosine_epochs = COSINE_EPOCHS if COSINE_EPOCHS > 0 else max(EPOCHS - WARMUP_EPOCHS, 1)
sched_warmup, sched_cosine = build_schedulers(optimizer, WARMUP_EPOCHS, cosine_epochs)

# %% Quick smoke-test train (reduce epochs here if needed)
fit(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    dataset=val_ds,
    device=device,
    optimizer=optimizer,
    schedulers=(sched_warmup, sched_cosine),
    epochs=EPOCHS,
    out_dir=str(OUT_DIR),
    writer=None,  # set to SummaryWriter if you want TB in notebooks
    use_amp=AMP,
    use_bf16=BF16,
    early_stop_patience=PATIENCE,
)

print("done")
