# 🔍 Granular Loss Logging - Implementation Summary

## What Changed

I've updated your training pipeline to log **individual loss components** instead of just the total loss. This gives you full visibility into what's happening during training.

## Files Modified

### 1. `maskrcnn_coco/utils.py`

#### ✅ `train_one_epoch()` - Already Returns Dict
Your function was already updated to return a dictionary of losses instead of a single float:

```python
def train_one_epoch(...) -> Dict[str, float]:
    # ... training code ...
    
    # Returns dictionary like:
    # {
    #     'loss_classifier': 0.123,
    #     'loss_box_reg': 0.234,
    #     'loss_mask': 0.345,
    #     'loss_objectness': 0.056,
    #     'loss_rpn_box_reg': 0.078
    # }
    return mean_losses
```

#### ✅ `fit()` - Enhanced Console Output
Added detailed console logging so you can see individual losses during training:

```python
# Print individual training losses for debugging
if isinstance(train_loss, dict):
    print(f"📊 Training Losses:")
    total = sum(train_loss.values())
    print(f"   Total: {total:.4f}")
    for loss_name, loss_value in sorted(train_loss.items()):
        print(f"   {loss_name}: {loss_value:.4f}")
```

Also added metric summaries:
```python
print(f"📈 Validation Loss: {val_loss:.4f}")
print(f"📦 Box mAP: {metrics['bbox']['AP']:.4f} (AP50: {metrics['bbox']['AP50']:.4f})")
print(f"🎭 Mask mAP: {metrics['segm']['AP']:.4f} (AP50: {metrics['segm']['AP50']:.4f})")
```

### 2. `maskrcnn_coco/mlops_modernization.py`

#### ✅ `log_training_metrics()` - Now Accepts Dict
Updated the function signature and implementation:

```python
def log_training_metrics(
    tracker: ExperimentTracker,
    epoch: int,
    train_loss: Dict[str, float] | float,  # ← Can be dict or float now
    val_loss: float,
    metrics: Dict[str, Dict[str, float]],
    lr: float,
):
    """Log all training metrics in organized structure."""
    
    # Handle train_loss - can be dict or float
    if isinstance(train_loss, dict):
        # Log individual loss components
        total_train_loss = sum(train_loss.values())
        tracker.log_scalar("loss/train_total", total_train_loss, step=epoch)
        
        # Log each individual loss component
        for loss_name, loss_value in train_loss.items():
            tracker.log_scalar(f"loss/train_{loss_name}", loss_value, step=epoch)
    else:
        # Backward compatibility: single float
        tracker.log_scalar("loss/train", train_loss, step=epoch)
```

**Key features:**
- ✅ **Modular**: Accepts both dict and float (backward compatible)
- ✅ **Granular**: Logs each loss component separately
- ✅ **Total Loss**: Also logs the sum of all losses
- ✅ **TensorBoard Paths**: Uses clear naming like `loss/train_loss_mask`

## What You'll See Now

### Console Output (Every Epoch)

```
Epoch 5/20
📊 Training Losses:
   Total: 0.8234
   loss_box_reg: 0.1234
   loss_classifier: 0.1456
   loss_mask: 0.3890        ← Monitor this closely!
   loss_objectness: 0.0567
   loss_rpn_box_reg: 0.1087
📈 Validation Loss: 0.9123
📦 Box mAP: 0.6543 (AP50: 0.8901)
🎭 Mask mAP: 0.2134 (AP50: 0.4567)
💾 Saved checkpoint: checkpoints/epoch_005.pth
```

### TensorBoard (Navigate to SCALARS tab)

You'll see these metrics organized by prefix:

**loss/** (Loss components)
- `loss/train_total` - Sum of all training losses
- `loss/train_loss_objectness` - RPN objectness
- `loss/train_loss_rpn_box_reg` - RPN box regression
- `loss/train_loss_classifier` - Classification
- `loss/train_loss_box_reg` - Box regression
- `loss/train_loss_mask` - **Mask prediction** ⭐
- `loss/val` - Validation loss

**mAP/** (Evaluation metrics)
- `mAP/bbox_AP`, `mAP/bbox_AP50`, `mAP/bbox_AP75`
- `mAP/segm_AP`, `mAP/segm_AP50`, `mAP/segm_AP75`

**lr** (Learning rate)
- `lr` - Current learning rate

## How to Use

### 1. Start Training

```bash
python maskrcnn_coco/main_mlops.py \
    --epochs 20 \
    --lr 5e-3 \
    --batch-size 2
```

### 2. Watch Console for Real-Time Loss Breakdown

You'll immediately see which losses are high/low. If `loss_mask` isn't decreasing while others are, you've found the problem!

### 3. Open TensorBoard for Detailed Analysis

```bash
tensorboard --logdir runs/maskrcnn
```

Navigate to **SCALARS** tab and:
- Select multiple losses to compare them
- Apply smoothing (0.6 recommended)
- Look for patterns

### 4. Debug Using Loss Patterns

See `LOSS_BREAKDOWN.md` for detailed debugging guide.

## Expected Loss Components

Mask R-CNN combines these losses during training:

| Loss | What It Trains | Typical Value (Final) |
|------|----------------|----------------------|
| `loss_objectness` | RPN: Is there an object? | 0.02 - 0.08 |
| `loss_rpn_box_reg` | RPN: Where is it (roughly)? | 0.03 - 0.10 |
| `loss_classifier` | Which class? | 0.05 - 0.15 |
| `loss_box_reg` | Precise box coordinates | 0.04 - 0.12 |
| `loss_mask` | Pixel-level segmentation | 0.08 - 0.20 |

**Total**: Usually converges to 0.2 - 0.5

## Debugging Mask Issues

With granular logging, you can now diagnose mask problems:

### ✅ Good Training Pattern
```
Epoch 1:  loss_mask=0.450
Epoch 5:  loss_mask=0.250
Epoch 10: loss_mask=0.150
Epoch 20: loss_mask=0.120
```
→ Mask loss decreasing steadily ✅

### ⚠️ Problem Pattern
```
Epoch 1:  loss_mask=0.450
Epoch 5:  loss_mask=0.420
Epoch 10: loss_mask=0.410
Epoch 20: loss_mask=0.405
```
→ Mask loss barely changing ❌

**This is the pattern you were experiencing!**

### Solution

The mask head needs a higher learning rate. You have two options:

#### Option 1: Use Differential Learning Rates (Recommended)

I've already implemented `get_parameter_groups()` in `utils.py`. To use it:

1. Update `main_mlops.py` to import it:
```python
from utils import (
    ...,
    get_parameter_groups,  # Add this
    ...
)
```

2. Replace the optimizer creation:
```python
# Old (all same LR):
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=args.lr, ...)

# New (differential LRs):
param_groups = get_parameter_groups(model, lr=args.lr, head_lr_multiplier=10.0)
optimizer = torch.optim.SGD(param_groups, ...)
```

This gives:
- Backbone: lr = 5e-3
- Box/Mask heads: lr = 5e-2 (10× higher)

#### Option 2: Increase Overall Learning Rate

```bash
python maskrcnn_coco/main_mlops.py --lr 1e-2  # Double the LR
```

But this might destabilize the pretrained backbone.

## Verification

To verify the logging is working:

1. Start training and check console output shows individual losses
2. Open TensorBoard and verify you see all `loss/train_*` metrics
3. Check that `loss_mask` is being logged separately
4. Verify the sum of individual losses ≈ `loss/train_total`

## Next Steps

1. ✅ **Start training** with the new logging
2. ✅ **Monitor `loss_mask`** specifically in TensorBoard
3. ✅ **Compare** with other loss components
4. ⚠️ **If `loss_mask` plateaus early**: Apply differential learning rates
5. ⚠️ **If `loss_mask` is much higher than others**: Increase mask head LR even more

## Benefits

**Before:**
- Only saw total loss: `loss/train: 0.823`
- No visibility into individual components
- Hard to debug which part is failing

**After:**
- See all components separately
- Can identify mask training issues immediately
- Can tune learning rates for specific components
- Better experiment tracking in TensorBoard

## Compatibility

✅ **Backward compatible**: Still works if `train_loss` is a float
✅ **Modular**: Easy to extend with more metrics
✅ **Framework agnostic**: Works with W&B, MLflow, or plain TensorBoard

---

**Bottom Line**: You now have full visibility into what's happening during training. Watch `loss_mask` closely - if it's not decreasing as fast as other losses, you've confirmed the issue and can apply the differential learning rate fix!

