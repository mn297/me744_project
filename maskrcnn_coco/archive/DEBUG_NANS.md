# 🔥 Debugging NaN Losses

## Why You're Getting NaNs

With the new differential learning rates:
- **Backbone**: lr = 5e-3
- **Box/Mask Heads**: lr = **5e-2** (10×)

The head learning rate (0.05) is **very aggressive** and can cause gradient explosion → NaN losses!

## ✅ Solutions (Try in Order)

### Solution 1: Gradient Clipping (DONE ✓)

I just added gradient clipping to your training loop:

```python
# Clips gradients to max_norm=10.0 to prevent explosion
torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm=10.0)
```

This prevents any gradient from getting too large.

### Solution 2: Reduce Learning Rate Multiplier

**Try these from safest to most aggressive:**

```bash
# Conservative (3× for heads)
python maskrcnn_coco/main_mlops.py \
    --lr 5e-3 \
    --head-lr-multiplier 3.0

# Moderate (5× for heads)
python maskrcnn_coco/main_mlops.py \
    --lr 5e-3 \
    --head-lr-multiplier 5.0

# Aggressive (7× for heads) 
python maskrcnn_coco/main_mlops.py \
    --lr 5e-3 \
    --head-lr-multiplier 7.0

# Very aggressive (10× for heads) - this is what caused NaNs
python maskrcnn_coco/main_mlops.py \
    --lr 5e-3 \
    --head-lr-multiplier 10.0
```

### Solution 3: Lower Base Learning Rate

```bash
# Smaller base LR with moderate multiplier
python maskrcnn_coco/main_mlops.py \
    --lr 1e-3 \
    --head-lr-multiplier 10.0
```

This gives:
- Backbone: 1e-3
- Heads: 1e-2 (much safer than 5e-2!)

### Solution 4: Start Training Fresh

If you resumed from a checkpoint that already had NaNs:

```bash
# Delete bad checkpoints and start fresh
rm -rf checkpoints/epoch_*.pth

# Train from scratch with safe settings
python maskrcnn_coco/main_mlops.py \
    --lr 5e-3 \
    --head-lr-multiplier 5.0 \
    --epochs 20
```

## 🔍 How to Monitor

### 1. Check Console Output

Look for NaNs in the loss printout:

```bash
Epoch 1/20
📊 Training Losses:
   Total: nan              ← BAD! 
   loss_box_reg: 0.234
   loss_classifier: nan    ← Problem here!
   loss_mask: nan          ← Problem here!
   ...
```

If you see NaN in **epoch 1**, learning rate is too high!

### 2. Check TensorBoard

```bash
tensorboard --logdir runs/maskrcnn
```

Look at the loss curves:
- **Good**: Smooth decrease
- **Bad**: Sudden spike to infinity → NaN

### 3. Check First Batch

Add this debug code temporarily to see if first batch is NaN:

```python
# In train_one_epoch(), after loss calculation:
if i == 1:  # First batch
    print(f"🔍 First batch losses:")
    for k, v in loss_dict.items():
        print(f"   {k}: {v.item():.6f}")
    if torch.isnan(loss):
        print("❌ NaN detected in first batch! LR too high!")
```

## 📊 Recommended Settings

Based on typical Mask R-CNN finetuning:

### For Small Datasets (< 500 images)
```bash
python maskrcnn_coco/main_mlops.py \
    --lr 1e-3 \
    --head-lr-multiplier 5.0 \
    --epochs 30
```

### For Medium Datasets (500-2000 images)
```bash
python maskrcnn_coco/main_mlops.py \
    --lr 5e-3 \
    --head-lr-multiplier 5.0 \
    --epochs 20
```

### For Large Datasets (> 2000 images)
```bash
python maskrcnn_coco/main_mlops.py \
    --lr 5e-3 \
    --head-lr-multiplier 7.0 \
    --epochs 15
```

## 🎯 Gradient Clipping Values

You can also tune the gradient clipping threshold:

The code now clips to `max_grad_norm=10.0` by default.

To change it, edit `utils.py` line 518:

```python
train_loss = train_one_epoch(
    model, train_loader, optimizer, device, scaler, 
    max_grad_norm=5.0  # ← Stricter clipping (more stable)
)
```

Typical values:
- **1.0**: Very conservative (might slow learning)
- **5.0**: Conservative (safe for high LR)
- **10.0**: Moderate (default, good balance)
- **20.0**: Loose (allows more gradient flow)

## 🔬 Root Cause Analysis

### Why NaNs Happen

1. **High learning rate** (5e-2) → Large parameter updates
2. **Large updates** → Extreme activations  
3. **Extreme activations** → Numerical overflow (inf)
4. **inf in forward pass** → NaN in loss
5. **NaN in loss** → NaN gradients → Everything becomes NaN

### Gradient Clipping Prevents This

```python
# Before clipping: gradients can be huge
grad_norm = 1000.0  ← Will cause NaN!

# After clipping to max_norm=10.0:
grad_norm = 10.0    ← Safe!
```

The gradients are scaled down proportionally:
```
scaled_grad = grad * (max_norm / grad_norm)
```

So instead of updating parameters by a huge amount, we cap the update size.

## 🚦 Quick Decision Tree

```
Is loss NaN in epoch 1?
├─ YES → Learning rate too high
│  └─ Try: --head-lr-multiplier 3.0
│
└─ NO → Loss NaN after several epochs?
   ├─ YES → Gradient accumulation causing instability
   │  └─ Try: Lower --lr to 1e-3
   │
   └─ NO → Loss NaN only sometimes?
      └─ Data issue (check for inf/nan in inputs)
```

## ✅ Test That It's Fixed

After making changes, verify:

1. **First epoch completes** without NaN
2. **Losses decrease** over epochs  
3. **TensorBoard shows** smooth curves (no spikes)
4. **Console shows** all loss values are finite numbers

Example of healthy output:

```
Epoch 1/20
📊 Training Losses:
   Total: 1.4523
   loss_box_reg: 0.2341
   loss_classifier: 0.3456
   loss_mask: 0.4567      ← All finite!
   loss_objectness: 0.1234
   loss_rpn_box_reg: 0.2925
```

## 🎓 Learning Rate Guidelines

| Component | Status | Safe LR Range | Aggressive LR Range |
|-----------|--------|---------------|---------------------|
| Backbone | Pretrained | 1e-4 to 5e-3 | 1e-3 to 1e-2 |
| Box Head | Random Init | 1e-3 to 1e-2 | 1e-2 to 5e-2 |
| Mask Head | Random Init | 1e-3 to 1e-2 | 1e-2 to 5e-2 |
| RPN | Partial Pre | 5e-4 to 5e-3 | 5e-3 to 2e-2 |

**Rule of thumb:** Start conservative, increase if learning is too slow.

---

**TL;DR:** 
1. ✅ Gradient clipping is now enabled (max_norm=10.0)
2. Try: `--head-lr-multiplier 5.0` instead of 10.0
3. If still NaN: Lower to `--lr 1e-3 --head-lr-multiplier 5.0`
4. Monitor first epoch carefully!

