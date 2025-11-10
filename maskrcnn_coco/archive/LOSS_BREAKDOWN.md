# 📊 Mask R-CNN Loss Breakdown

## Overview

Mask R-CNN uses multiple loss components that are all trained jointly. Understanding each component helps you debug training issues.

## Loss Components

When training, you'll now see these individual losses in **both console output and TensorBoard**:

### 1. **loss_objectness**
- **What it measures**: RPN's ability to distinguish objects from background
- **Range**: Typically 0.01 - 0.5
- **Issue if high**: RPN struggles to find potential object regions
- **TensorBoard path**: `loss/train_loss_objectness`

### 2. **loss_rpn_box_reg**
- **What it measures**: RPN's box coordinate predictions
- **Range**: Typically 0.01 - 0.2
- **Issue if high**: RPN can't accurately localize objects
- **TensorBoard path**: `loss/train_loss_rpn_box_reg`

### 3. **loss_classifier**
- **What it measures**: Classification accuracy (which class is the object?)
- **Range**: Typically 0.05 - 0.5
- **Issue if high**: Model can't distinguish between your classes
- **TensorBoard path**: `loss/train_loss_classifier`

### 4. **loss_box_reg**
- **What it measures**: Final box coordinate refinement
- **Range**: Typically 0.05 - 0.3
- **Issue if high**: Bounding boxes aren't precise
- **TensorBoard path**: `loss/train_loss_box_reg`

### 5. **loss_mask** ⭐
- **What it measures**: Mask prediction accuracy (pixel-level segmentation)
- **Range**: Typically 0.1 - 0.5
- **Issue if high**: **THIS IS YOUR ISSUE** - masks aren't learning properly
- **TensorBoard path**: `loss/train_loss_mask`

## What to Look For

### 🎯 Healthy Training Pattern

```
Epoch 1:
   loss_objectness:   0.450
   loss_rpn_box_reg:  0.180
   loss_classifier:   0.350
   loss_box_reg:      0.250
   loss_mask:         0.400  ← Watch this one!
   Total:             1.630

Epoch 10:
   loss_objectness:   0.080
   loss_rpn_box_reg:  0.050
   loss_classifier:   0.100
   loss_box_reg:      0.080
   loss_mask:         0.150  ← Should be decreasing
   Total:             0.460
```

### ⚠️ Problem Patterns

#### Problem 1: Mask Loss Not Decreasing
```
Epoch 1:  loss_mask: 0.400
Epoch 10: loss_mask: 0.380  ← Barely changed!
Epoch 20: loss_mask: 0.370  ← Still stuck!
```

**Diagnosis**: Mask head undertrained (learning rate too low)
**Solution**: Use differential learning rates (the fix I provided earlier)

#### Problem 2: Mask Loss Exploding
```
Epoch 1:  loss_mask: 0.400
Epoch 5:  loss_mask: 2.100  ← Exploding!
Epoch 10: loss_mask: NaN     ← Crashed!
```

**Diagnosis**: Learning rate too high for mask head
**Solution**: Reduce learning rate or gradient clipping

#### Problem 3: Box Losses Good, Mask Loss Bad
```
Epoch 10:
   loss_objectness:   0.050  ✅
   loss_rpn_box_reg:  0.030  ✅
   loss_classifier:   0.080  ✅
   loss_box_reg:      0.060  ✅
   loss_mask:         0.400  ❌ Still high!
```

**Diagnosis**: Imbalanced learning rates
**Solution**: Differential learning rates (already implemented)

## TensorBoard Visualization

### View All Losses Together

1. Open TensorBoard:
```bash
tensorboard --logdir runs/maskrcnn
```

2. Navigate to the **SCALARS** tab

3. You'll see these groups:
   - `loss/train_loss_objectness`
   - `loss/train_loss_rpn_box_reg`
   - `loss/train_loss_classifier`
   - `loss/train_loss_box_reg`
   - `loss/train_loss_mask` ⭐
   - `loss/train_total`
   - `loss/val`

4. **Select multiple losses** to compare them on the same plot

### Recommended Views

#### View 1: Compare All Training Losses
- Select all `loss/train_*` metrics
- Smoothing: 0.6
- Look for: All losses decreasing together

#### View 2: Focus on Mask Loss
- Select only `loss/train_loss_mask`
- Smoothing: 0.3
- Look for: Steady decrease (not plateauing)

#### View 3: Training vs Validation
- Select `loss/train_total` and `loss/val`
- Look for: Both decreasing, not diverging (overfitting)

## Typical Loss Ratios

In well-trained Mask R-CNN:

```
loss_mask         ≈ 30-40% of total  ← Largest component
loss_classifier   ≈ 20-30% of total
loss_box_reg      ≈ 15-25% of total
loss_rpn_box_reg  ≈ 10-15% of total
loss_objectness   ≈ 5-10% of total
```

If your `loss_mask` is **< 20%** of total, it's undertrained!
If your `loss_mask` is **> 50%** of total, it's struggling!

## Debugging Workflow

### Step 1: Check Console Output
```
Epoch 5/20
📊 Training Losses:
   Total: 0.8234
   loss_box_reg: 0.1234
   loss_classifier: 0.1456
   loss_mask: 0.3890        ← Focus here
   loss_objectness: 0.0567
   loss_rpn_box_reg: 0.1087
📈 Validation Loss: 0.9123
📦 Box mAP: 0.6543 (AP50: 0.8901)
🎭 Mask mAP: 0.2134 (AP50: 0.4567)  ← Low mask mAP!
```

### Step 2: Identify the Problem
- High `loss_mask` + Low Mask mAP = Mask head undertrained
- Low `loss_mask` + Low Mask mAP = Wrong loss or data issue
- High all losses = Overall training issue

### Step 3: Open TensorBoard
```bash
tensorboard --logdir runs/maskrcnn
```

### Step 4: Verify Loss Progression
- Is `loss_mask` decreasing?
- Is it decreasing as fast as other losses?
- Does it plateau early?

### Step 5: Take Action
Based on the pattern, apply the appropriate fix (see above)

## Example Training Log

Here's what you should see with the new logging:

```
Epoch 1/20
📊 Training Losses:
   Total: 1.6234
   loss_box_reg: 0.2456
   loss_classifier: 0.3567
   loss_mask: 0.4890
   loss_objectness: 0.2987
   loss_rpn_box_reg: 0.2334
📈 Validation Loss: 1.7543
📦 Box mAP: 0.1234 (AP50: 0.2567)
🎭 Mask mAP: 0.0456 (AP50: 0.0987)

Epoch 5/20
📊 Training Losses:
   Total: 0.8234
   loss_box_reg: 0.1234
   loss_classifier: 0.1456
   loss_mask: 0.3890        ← Should be decreasing
   loss_objectness: 0.0567
   loss_rpn_box_reg: 0.1087
📈 Validation Loss: 0.9123
📦 Box mAP: 0.4567 (AP50: 0.6234)
🎭 Mask mAP: 0.2345 (AP50: 0.3901)

Epoch 20/20
📊 Training Losses:
   Total: 0.3234
   loss_box_reg: 0.0456
   loss_classifier: 0.0567
   loss_mask: 0.1234        ← Should be much lower now
   loss_objectness: 0.0234
   loss_rpn_box_reg: 0.0743
📈 Validation Loss: 0.3987
📦 Box mAP: 0.8234 (AP50: 0.9456)
🎭 Mask mAP: 0.7567 (AP50: 0.8901)  ← Good!
```

## Quick Reference

| Loss Component | What It Does | Typical Final Value | If Too High |
|----------------|--------------|---------------------|-------------|
| `loss_objectness` | Is there an object? | 0.02 - 0.08 | RPN issues |
| `loss_rpn_box_reg` | Where is the object? (rough) | 0.03 - 0.10 | RPN localization |
| `loss_classifier` | What class is it? | 0.05 - 0.15 | Classification issues |
| `loss_box_reg` | Where exactly? (refined) | 0.04 - 0.12 | Box regression |
| `loss_mask` | Pixel-level segmentation | 0.08 - 0.20 | **Mask training issue** |

---

**Pro Tip**: Always compare `loss_mask` progression with `mAP/segm_AP`. If loss is decreasing but mAP isn't improving, you might have a data quality issue!

