# Epoch Processing: Code Flow

## ✅ YES! Each epoch processes the FULL dataset through batches

---

## 📍 Exact Code Locations

### 1. **Main Epoch Loop** 
**File:** `utils.py`  
**Function:** `fit()`  
**Line:** 395-406

```python
for ep in range(start_epoch + 1, epochs + 1):
    print(f"\nEpoch {ep}/{epochs}")
    
    # ← THIS processes ALL 231 training images (116 batches)
    train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler)
    
    # ← THIS processes ALL 57 validation images (57 batches)
    val_loss = validate_loss(model, val_loader, device)
    metrics = evaluate_coco(model, val_loader, dataset, device)
```

**What happens:**
- Each iteration of this loop = **ONE EPOCH**
- `train_one_epoch()` processes **ALL 231 training images**
- `validate_loss()` processes **ALL 57 validation images**

---

### 2. **Training: Processing All Batches**
**File:** `utils.py`  
**Function:** `train_one_epoch()`  
**Line:** 286-321

```python
def train_one_epoch(model, loader, optimizer, device, scaler=None, log_every: int = 50):
    model.train()
    running, seen = 0.0, 0
    pbar = tqdm(loader, desc="train", leave=False)  # ← Shows "train: 116/116"
    
    # ← THIS LOOP processes ALL batches
    for i, (images, targets) in enumerate(pbar, 1):
        # images = batch of 2 images
        # targets = batch of 2 targets
        
        images, targets = _to_device(images, targets, device)
        loss_dict = model(images, targets)  # Forward pass
        loss = sum(loss_dict.values())
        loss.backward()                    # Backward pass
        optimizer.step()                   # Update weights
        
        # After 116 iterations: ALL 231 images processed!
    
    return running / max(seen, 1)  # Average loss
```

**What happens:**
- `loader` = `train_loader` with 231 images, batch_size=2
- Loop runs **116 times** (one per batch)
- Each iteration processes **2 images** (except last batch: 1 image)
- When loop finishes: **ALL 231 images have been seen!**

---

### 3. **Validation: Processing All Batches**
**File:** `utils.py`  
**Function:** `validate_loss()`  
**Line:** 325-343

```python
@torch.no_grad()
def validate_loss(model, loader, device):
    was_training = model.training
    model.train()  # Need train mode to get losses
    
    total, n = 0.0, 0
    # ← THIS LOOP processes ALL validation batches
    for images, targets in tqdm(loader, desc="val_loss", leave=False):
        # images = batch of 1 image
        # targets = batch of 1 target
        
        images, targets = _to_device(images, targets, device)
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        
        bs = len(images)  # Usually 1
        total += float(loss.item()) * bs
        n += bs
        
        # After 57 iterations: ALL 57 images processed!
    
    return total / max(n, 1)  # Average validation loss
```

**What happens:**
- `loader` = `val_loader` with 57 images, batch_size=1
- Loop runs **57 times** (one per batch)
- Each iteration processes **1 image**
- When loop finishes: **ALL 57 images have been seen!**

---

### 4. **DataLoader: Automatic Batching**
**File:** `main.py`  
**Line:** 106-123

```python
# Training DataLoader
train_loader = DataLoader(
    train_ds,           # 231 images
    batch_size=2,       # Split into batches of 2
    shuffle=True,       # Randomize order each epoch
    ...
)

# Validation DataLoader  
val_loader = DataLoader(
    val_ds,             # 57 images
    batch_size=1,       # Split into batches of 1
    shuffle=False,      # Keep same order
    ...
)
```

**What PyTorch DataLoader does automatically:**
1. Takes your dataset (231 images)
2. Splits into batches (116 batches of size 2)
3. When you iterate: yields one batch at a time
4. When iteration finishes: all images have been processed

---

## 🔄 Complete Flow for ONE EPOCH

```
EPOCH 1 STARTS
│
├─ Step 1: train_one_epoch(train_loader)
│  │
│  ├─ DataLoader yields Batch 1:   [Image 0, Image 1]   → Process → Update model
│  ├─ DataLoader yields Batch 2:   [Image 2, Image 3]   → Process → Update model
│  ├─ DataLoader yields Batch 3:   [Image 4, Image 5]   → Process → Update model
│  ├─ ...
│  ├─ DataLoader yields Batch 115: [Image 228, Image 229] → Process → Update model
│  └─ DataLoader yields Batch 116: [Image 230]         → Process → Update model
│
│  ✅ All 231 training images processed!
│  ✅ Model weights updated 116 times
│
├─ Step 2: validate_loss(val_loader)
│  │
│  ├─ DataLoader yields Batch 1:   [Image 0]   → Process → Calculate loss
│  ├─ DataLoader yields Batch 2:   [Image 1]   → Process → Calculate loss
│  ├─ ...
│  └─ DataLoader yields Batch 57:  [Image 56]  → Process → Calculate loss
│
│  ✅ All 57 validation images processed!
│  ✅ Average validation loss calculated
│
├─ Step 3: evaluate_coco(val_loader)
│  │
│  └─ (Same 57 images again, but for mAP calculation)
│
└─ EPOCH 1 COMPLETE
```

---

## 🎯 Key Points

1. **Each epoch = Full dataset pass**
   - Training: All 231 images (116 batches)
   - Validation: All 57 images (57 batches)

2. **DataLoader handles batching automatically**
   - You don't manually split into batches
   - Just iterate: `for batch in loader:`
   - PyTorch handles the rest!

3. **Shuffle matters**
   - Training: `shuffle=True` → Different order each epoch
   - Validation: `shuffle=False` → Same order every time

4. **Progress bars show batches, not images**
   - `train: 116/116` = 116 batches processed
   - `val_loss: 57/57` = 57 batches processed

---

## 📊 Visual Summary

```
Dataset (231 images)
    ↓
DataLoader (batch_size=2)
    ↓
[Batch 1: 2 images] → [Batch 2: 2 images] → ... → [Batch 116: 1 image]
    ↓                    ↓                           ↓
Process                Process                      Process
    ↓                    ↓                           ↓
Update Model          Update Model                Update Model

After 116 batches: ALL 231 images processed! ✅
```

---

## 🔍 To Verify This Yourself

Add this to `train_one_epoch()` to see it in action:

```python
def train_one_epoch(model, loader, optimizer, device, scaler=None, log_every: int = 50):
    model.train()
    running, seen = 0.0, 0
    pbar = tqdm(loader, desc="train", leave=False)
    
    total_images = 0  # ← Add this
    
    for i, (images, targets) in enumerate(pbar, 1):
        batch_size = len(images)
        total_images += batch_size  # ← Add this
        print(f"Batch {i}: {batch_size} images, Total so far: {total_images}")
        
        # ... rest of code ...
    
    print(f"✅ Processed {total_images} images total!")  # Should be 231
    return running / max(seen, 1)
```

