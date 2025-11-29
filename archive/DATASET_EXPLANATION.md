# Dataset and DataLoader Explanation

## 📊 Your Dataset Breakdown

### Fuji-Apple-Segmentation Dataset

```
Total: 288 images
├── Training set (trainingset/):  231 images  ← Used to TRAIN the model
└── Validation set (testset/):    57 images  ← Used to VALIDATE the model
```

**The "57 images" you see = Validation set (testset/)**

---

## 🔄 What Happens During Training

### Training Phase (231 images)
- **Dataset**: `trainingset/` (231 images)
- **Batch size**: 2 (default)
- **Batches per epoch**: 116
- **Calculation**: 231 ÷ 2 = 115.5 → rounds up to 116 batches
  - First 115 batches: 2 images each = 230 images
  - Last batch: 1 image = 1 image
  - Total: 231 images processed

### Validation Phase (57 images)
- **Dataset**: `testset/` (57 images) ← **This is your "57"!**
- **Batch size**: 1 (hardcoded in code)
- **Batches per epoch**: 57
- **Calculation**: 57 ÷ 1 = 57 batches
  - Each batch: 1 image
  - Total: 57 images processed

---

## 📝 Code Reference

### In `main.py`:

```python
# Line 88-90: Training dataset
train_ds = CocoSegmentationDataset(
    args.train_images,  # "Fuji-Apple-Segmentation/trainingset/JPEGImages"
    args.train_anno,     # "Fuji-Apple-Segmentation/trainingset/annotations.json"
    is_train=True
)

# Line 91-97: Validation dataset
val_ds = CocoSegmentationDataset(
    args.val_images,    # "Fuji-Apple-Segmentation/testset/JPEGImages"
    args.val_anno,       # "Fuji-Apple-Segmentation/testset/annotations.json"
    is_train=False
)
```

### In `main.py` DataLoaders:

```python
# Line 106-114: Training DataLoader
train_loader = DataLoader(
    train_ds,
    batch_size=args.batch_size,  # Default: 2
    shuffle=True,                 # Randomize order
    ...
)
# Result: 231 images ÷ 2 = 116 batches

# Line 115-123: Validation DataLoader
val_loader = DataLoader(
    val_ds,
    batch_size=1,        # ← HARDCODED to 1!
    shuffle=False,       # Don't shuffle validation
    ...
)
# Result: 57 images ÷ 1 = 57 batches
```

---

## 🤔 Why These Numbers?

### Why batch_size=2 for training?
- **Faster**: Process 2 images in parallel
- **Memory efficient**: Small enough for most GPUs
- **231 images ÷ 2 = 116 batches** (last batch has 1 image)

### Why batch_size=1 for validation?
- **Simpler**: One image at a time
- **Less memory**: Good for evaluation
- **Easier debugging**: Can inspect each image individually
- **57 images ÷ 1 = 57 batches**

---

## 📈 What You See During Training

```
Training dataset: 231 images        ← trainingset/
Validation dataset: 57 images      ← testset/ (this is your "57"!)

Training batches: 116 (batch_size=2)
Validation batches: 57 (batch_size=1)
```

**Translation:**
- **116 training batches** = Processing 231 training images (2 at a time)
- **57 validation batches** = Processing 57 validation images (1 at a time)

---

## 🎯 Summary

| Set | Location | Images | Batch Size | Batches | Purpose |
|-----|----------|--------|-----------|---------|---------|
| **Training** | `trainingset/` | 231 | 2 | 116 | Train the model |
| **Validation** | `testset/` | **57** | 1 | **57** | Check model performance |

**The "57" = Validation set size (testset/)**

You're using:
- ✅ **231 images for training** (trainingset/)
- ✅ **57 images for validation** (testset/)
- ✅ **Total: 288 images**

---

## 💡 To Change Batch Sizes

### Training batch size:
```bash
uv run python main.py --batch-size 4
# Now: 231 ÷ 4 = 58 batches (last batch has 3 images)
```

### Validation batch size:
Edit `main.py` line 117:
```python
val_loader = DataLoader(
    val_ds,
    batch_size=2,  # Change from 1 to 2
    ...
)
# Now: 57 ÷ 2 = 29 batches (last batch has 1 image)
```

---

## 🔍 Quick Check

Run this to see actual numbers:
```bash
cd maskrcnn_coco
uv run python dataset_explanation.py
```

