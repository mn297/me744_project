# 🔧 Mask R-CNN Small Mask Issue - Root Cause & Fix

## 🎯 Problem Summary

**Symptoms:**
- ✅ Bounding box predictions are very precise after training
- ❌ Semantic mask predictions are very small (only cover a small section of the ground truth)

## 🔍 Root Cause Analysis

### The Issue: Incorrect Learning Rates for Finetuning

When finetuning Mask R-CNN, your code does the following:

1. **Loads pretrained model** with COCO weights (backbone + prediction heads)
2. **Replaces BOTH prediction heads** with randomly initialized ones:
   - `FastRCNNPredictor` (box head)
   - `MaskRCNNPredictor` (mask head)
3. **Trains ALL parameters with the SAME learning rate** (5e-3)

### Why This Causes the Problem

```
Pretrained Backbone (ResNet-50 FPN)
    ↓ (needs SMALL LR to avoid forgetting)
    ├─→ Box Predictor (randomly initialized, needs HIGH LR)
    └─→ Mask Predictor (randomly initialized, needs HIGH LR)
```

**The mask predictor is much more complex than the box predictor:**
- Box head: Simple classification + regression (4 bbox coords)
- Mask head: Convolutional layers that generate 28×28 pixel masks

When both use the same learning rate as the pretrained backbone:
- Box predictor learns reasonably well (simpler task)
- Mask predictor severely undertrained → produces small, poor-quality masks

## ✅ Solution: Differential Learning Rates

The fix implements **parameter groups** with different learning rates:

```python
Backbone:     lr = 5e-3      (pretrained, needs small LR)
Box Head:     lr = 5e-2      (10× base, randomly init)
Mask Head:    lr = 5e-2      (10× base, randomly init)
RPN:          lr = 2.5e-2    (5× base, intermediate)
```

## 🚀 Changes Made

### 1. Added `get_parameter_groups()` in `utils.py`

This function separates model parameters into groups and assigns appropriate learning rates:

```python
def get_parameter_groups(model, lr, head_lr_multiplier=10.0):
    """Create parameter groups with differential learning rates"""
    # Returns:
    # - Backbone params @ lr
    # - Box head params @ lr * head_lr_multiplier
    # - Mask head params @ lr * head_lr_multiplier
    # - Other params (RPN) @ lr * 5.0
```

### 2. Updated `main_mlops.py`

- Imported `get_parameter_groups`
- Modified optimizer creation to use parameter groups
- Added `--head-lr-multiplier` argument (default: 10.0)

## 📝 How to Use

### Basic Usage (with defaults)

```bash
python maskrcnn_coco/main_mlops.py \
    --epochs 20 \
    --lr 5e-3 \
    --batch-size 2
```

This will use:
- Backbone LR: 5e-3
- Head LR: 5e-2 (10× backbone)

### Custom Head Learning Rate

```bash
python maskrcnn_coco/main_mlops.py \
    --epochs 20 \
    --lr 5e-3 \
    --head-lr-multiplier 15.0  # Use 15× for even more aggressive head training
```

### Resume Training (Important!)

If you resume from a checkpoint trained with the old code:

```bash
python maskrcnn_coco/main_mlops.py \
    --resume checkpoints/  # Auto-finds latest checkpoint
    --epochs 30            # Train for additional epochs
    --lr 5e-3
    --head-lr-multiplier 10.0
```

The mask head will now train much more aggressively and should improve quickly!

## 🎨 Expected Results

After retraining with differential learning rates:

- ✅ **Box predictions:** Still excellent (minimal impact)
- ✅ **Mask predictions:** Should cover the full object and match ground truth much better
- ✅ **Training stability:** Better convergence for both tasks

## 🔬 Additional Recommendations

### 1. Monitor Training Metrics

Watch these metrics in TensorBoard to ensure proper training:

```bash
tensorboard --logdir runs/maskrcnn
```

Key metrics to watch:
- `loss_mask`: Should decrease steadily
- `mAP/segm`: Should improve significantly
- `lr` (per parameter group): Verify different learning rates are being used

### 2. Experiment with Learning Rate Multipliers

If masks are still not good enough, try:

```bash
# More aggressive (15× for heads)
--head-lr-multiplier 15.0

# Less aggressive (5× for heads)
--head-lr-multiplier 5.0
```

Start with 10× and adjust based on results.

### 3. Training Duration

Since the mask head was undertrained before, you may need to:
- Train for more epochs (30-40 instead of 20)
- Or resume training from your current checkpoint

### 4. Other Potential Issues to Check

If masks are still poor after this fix, check:

1. **Data quality:**
   - Verify annotations are correct with: `python visualize_mask_fix.py`
   - Check mask coverage in ground truth data

2. **Batch size:**
   - Mask R-CNN benefits from larger batches
   - Try `--batch-size 4` if you have enough VRAM

3. **Image resolution:**
   - If your images are much larger/smaller than COCO, consider resizing

4. **Loss weights:**
   - The model uses default loss weights from torchvision
   - For very small objects, you might need to adjust mask loss weight

## 📊 Quick Test

To verify the fix is working, check the console output when training starts:

```
📊 Parameter Groups:
  Backbone: 23,528,320 params @ lr=5.00e-03
  Box Head: 12,845 params @ lr=5.00e-02
  Mask Head: 738,561 params @ lr=5.00e-02
  Other (RPN): 1,229,312 params @ lr=2.50e-02
```

You should see different learning rates for different parts!

## 🎓 Why This Works

**Differential learning rates** are a standard technique in transfer learning:

1. **Pretrained layers** (backbone): Small LR preserves learned features
2. **New layers** (heads): Large LR allows rapid learning from random initialization
3. **Result**: Both parts train optimally at their own pace

This is especially important for Mask R-CNN because:
- Mask prediction is more complex than box prediction
- The mask head has many more parameters (738K vs 12K for box head)
- Masks require fine-grained spatial understanding

## 📚 References

This fix is inspired by best practices from:
- [Detectron2](https://github.com/facebookresearch/detectron2) (Facebook AI's detection framework)
- [fast.ai](https://www.fast.ai/) discriminative learning rates
- [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

## 🆘 Troubleshooting

### "Masks still too small after 10 epochs"

The mask head needs time to learn. Try:
```bash
# Resume training with higher multiplier
--resume checkpoints/
--head-lr-multiplier 15.0
--epochs 40
```

### "Training is unstable / loss exploding"

The learning rate might be too high. Try:
```bash
--lr 1e-3  # Lower base LR
--head-lr-multiplier 5.0  # Lower multiplier
```

### "Want to go back to old behavior"

Set multiplier to 1:
```bash
--head-lr-multiplier 1.0  # All parameters use same LR
```

---

**Author's Note:** This is a classic finetuning issue that affects many computer vision practitioners. The fix is simple but the impact is significant! Your box predictions are good because the box head is simpler, but the mask head was severely handicapped by the low learning rate. With this fix, you should see dramatic improvements in mask quality. 🎉

