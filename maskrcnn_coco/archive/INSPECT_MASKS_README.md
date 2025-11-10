# Mask Inspection Script

Quick script to check if your dataset masks are valid.

## Usage

### Basic Inspection (5 samples)
```bash
cd /home/john/me744_project
python maskrcnn_coco/inspect_masks.py
```

**Output:**
- Mask shapes
- Non-zero pixel counts
- Bounding box info
- Warnings if masks are all zeros

### With Visualization
```bash
python maskrcnn_coco/inspect_masks.py --visualize
```

Saves visualizations to `mask_inspections/` directory.

### Full Dataset Statistics
```bash
python maskrcnn_coco/inspect_masks.py --stats
```

Computes statistics across all images:
- Total masks
- Zero mask count
- Mean/median mask sizes

### Check Training Set
```bash
python maskrcnn_coco/inspect_masks.py \
    --images-dir Fuji-Apple-Segmentation/trainingset/JPEGImages \
    --anno-file Fuji-Apple-Segmentation/trainingset/annotations.json \
    --num-samples 10 \
    --visualize \
    --stats
```

### Check Car Parts Dataset
```bash
python maskrcnn_coco/inspect_masks.py \
    --images-dir Car-Parts-Segmentation/trainingset/JPEGImages \
    --anno-file Car-Parts-Segmentation/trainingset/annotations.json \
    --visualize
```

## What to Look For

### ✅ Good Masks
```
Object 1:
  Label: 1
  Mask shape: (480, 640)
  Non-zero pixels: 12,345 / 307,200 (4.02%)
  ✅ Mask has non-zero pixels
```

### ❌ Bad Masks (All Zeros)
```
Object 1:
  Label: 1
  Mask shape: (480, 640)
  Non-zero pixels: 0 / 307,200 (0.00%)
  ❌ WARNING: Mask is all zeros!
```

If you see zero masks, your annotation conversion has a bug!

## Output Files

When using `--visualize`, creates:
```
mask_inspections/
├── sample_1_image_123.png
├── sample_2_image_456.png
└── ...
```

Each PNG shows:
- Original image
- Each object's mask overlaid
- Bounding boxes

## Quick Checks

**1. Are masks loaded correctly?**
```bash
python maskrcnn_coco/inspect_masks.py --num-samples 3
```
Look for "✅ Mask has non-zero pixels"

**2. Are masks the right size?**
Check that mask shape matches image shape

**3. Do masks cover reasonable area?**
Typical: 1-20% of image (depends on object size)

**4. Any zero masks?**
```bash
python maskrcnn_coco/inspect_masks.py --stats
```
Should show "Zero masks: 0"

