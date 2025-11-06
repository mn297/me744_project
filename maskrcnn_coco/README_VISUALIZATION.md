# Visualization Guide

## 1. Checkpoints

Checkpoints **ARE being saved** automatically during training:
- **Per-epoch checkpoints**: `checkpoints/epoch_001.pth`, `epoch_002.pth`, etc.
- **Best model**: `checkpoints/best_bbox_ap.pth` (saved when bbox mAP improves)
- **Training summary**: `checkpoints/training_summary.json`

### To resume training from a checkpoint:
```bash
uv run main.py --resume checkpoints/epoch_005.pth
```

### To use a checkpoint for inference:
```bash
uv run visualize_predictions.py --checkpoint checkpoints/best_bbox_ap.pth
```

---

## 2. TensorBoard Visualization

TensorBoard logs are automatically saved to `runs/maskrcnn/` during training.

### Start TensorBoard:
```bash
# From the maskrcnn_coco directory
tensorboard --logdir runs/maskrcnn

# Or if tensorboard is not installed globally:
uv run tensorboard --logdir runs/maskrcnn
```

### Access TensorBoard:
1. Open your browser
2. Go to: `http://localhost:6006`
3. You'll see:
   - **Scalars tab**: Training/validation loss, mAP metrics over epochs
   - **Graphs tab**: Model architecture visualization
   - **Images tab**: (if you add image logging)

### What metrics are logged:
- `loss/train` - Training loss
- `loss/val` - Validation loss  
- `mAP/bbox` - Bounding box mean Average Precision
- `mAP/segm` - Segmentation mean Average Precision

### View specific run:
If you have multiple runs, you can view a specific one:
```bash
tensorboard --logdir runs/maskrcnn/run_2024_11_06_01_19_00
```

---

## 3. Visual Predictions Overlay

Visualize model predictions with bounding boxes and masks on images.

### Basic usage:
```bash
uv run visualize_predictions.py \
    --checkpoint checkpoints/best_bbox_ap.pth \
    --num-images 6 \
    --score-threshold 0.5
```

### Options:
- `--checkpoint`: Path to model checkpoint (required)
- `--images`: Path to images directory (default: testset)
- `--annotations`: Path to annotations JSON (default: testset)
- `--num-images`: Number of images to visualize (default: 6)
- `--score-threshold`: Minimum confidence score to display (default: 0.5)
- `--output-dir`: Where to save visualizations (default: `visualizations/`)

### Example with custom settings:
```bash
uv run visualize_predictions.py \
    --checkpoint checkpoints/epoch_010.pth \
    --images Car-Parts-Segmentation/testset/JPEGImages \
    --annotations Car-Parts-Segmentation/testset/annotations.json \
    --num-images 10 \
    --score-threshold 0.3 \
    --output-dir my_visualizations
```

### Output:
- Creates side-by-side comparison:
  - **Left**: Ground truth (green boxes/masks)
  - **Right**: Predictions (red boxes/masks with confidence scores)
- Saves PNG files to `visualizations/` directory
- Each image shows:
  - Bounding boxes
  - Segmentation masks (colored overlay)
  - Class labels
  - Confidence scores (for predictions)

### What you'll see:
- **Green boxes/masks**: Ground truth annotations
- **Red boxes/masks**: Model predictions
- **Colored overlays**: Different colors for different object instances
- **Labels**: Class names with confidence scores

---

## Quick Start Examples

### 1. View training progress in TensorBoard:
```bash
# Terminal 1: Start training
uv run main.py --epochs 20

# Terminal 2: Start TensorBoard
tensorboard --logdir runs/maskrcnn
# Then open http://localhost:6006 in browser
```

### 2. Visualize predictions after training:
```bash
# After training completes
uv run visualize_predictions.py \
    --checkpoint checkpoints/best_bbox_ap.pth \
    --num-images 10
```

### 3. Resume training and monitor:
```bash
# Resume from epoch 5
uv run main.py --resume checkpoints/epoch_005.pth --epochs 20

# In another terminal, watch TensorBoard
tensorboard --logdir runs/maskrcnn
```

---

## Troubleshooting

### TensorBoard not found:
```bash
# Install tensorboard
uv add tensorboard

# Or use uv run
uv run tensorboard --logdir runs/maskrcnn
```

### Checkpoints not found:
- Check that training completed at least 1 epoch
- Verify `checkpoints/` directory exists
- Check file permissions

### Visualization errors:
- Make sure checkpoint path is correct
- Verify images and annotations paths exist
- Check that model was trained with same number of classes

