"""
Explanation of how each epoch processes the full dataset through batches.

This shows exactly where in the code the epoch loop happens.
"""

print("=" * 80)
print("HOW EACH EPOCH PROCESSES THE FULL DATASET")
print("=" * 80)

print("""
STEP-BY-STEP PROCESS:

1. EPOCH LOOP (in utils.py, fit() function, line ~394)
   ┌─────────────────────────────────────────────────┐
   │ for ep in range(start_epoch + 1, epochs + 1):  │
   │     # Process ALL training images                │
   │     train_loss = train_one_epoch(...)           │
   │     # Process ALL validation images             │
   │     val_loss = validate_loss(...)               │
   │     metrics = evaluate_coco(...)                 │
   └─────────────────────────────────────────────────┘

2. TRAINING ONE EPOCH (in utils.py, train_one_epoch(), line ~286)
   ┌─────────────────────────────────────────────────┐
   │ def train_one_epoch(model, loader, ...):         │
   │     for i, (images, targets) in enumerate(loader):│
   │         # Process ONE batch                     │
   │         loss = model(images, targets)            │
   │         loss.backward()                          │
   │         optimizer.step()                         │
   │     # After loop: ALL batches processed!        │
   └─────────────────────────────────────────────────┘

3. DATA LOADER AUTOMATICALLY BATCHES (PyTorch DataLoader)
   ┌─────────────────────────────────────────────────┐
   │ DataLoader automatically:                       │
   │   - Takes dataset (231 images)                   │
   │   - Splits into batches (116 batches of size 2) │
   │   - Yields one batch at a time                   │
   │   - When loop finishes: all images processed!    │
   └─────────────────────────────────────────────────┘
""")

print("\n" + "=" * 80)
print("CODE LOCATIONS")
print("=" * 80)

print("""
1. MAIN TRAINING LOOP
   File: utils.py
   Function: fit()
   Line: ~394
   
   Code:
   ┌────────────────────────────────────────────────────┐
   │ for ep in range(start_epoch + 1, epochs + 1):     │
   │     print(f"\\nEpoch {ep}/{epochs}")                │
   │                                                     │
   │     # ← THIS processes ALL 231 training images     │
   │     train_loss = train_one_epoch(                 │
   │         model, train_loader, optimizer, device, scaler│
   │     )                                              │
   │                                                     │
   │     # ← THIS processes ALL 57 validation images    │
   │     val_loss = validate_loss(model, val_loader, device)│
   │     metrics = evaluate_coco(model, val_loader, dataset, device)│
   └────────────────────────────────────────────────────┘

2. PROCESSING ALL TRAINING BATCHES
   File: utils.py
   Function: train_one_epoch()
   Line: ~286
   
   Code:
   ┌────────────────────────────────────────────────────┐
   │ def train_one_epoch(model, loader, ...):          │
   │     pbar = tqdm(loader, desc="train", leave=False)│
   │                                                     │
   │     for i, (images, targets) in enumerate(pbar, 1):│
   │         # images = batch of images (e.g., 2 images)│
   │         # targets = batch of targets              │
   │         loss_dict = model(images, targets)        │
   │         loss = sum(loss_dict.values())            │
   │         loss.backward()                           │
   │         optimizer.step()                          │
   │                                                     │
   │     # When this loop finishes:                    │
   │     # ALL batches have been processed!            │
   │     # All 231 images have been seen!             │
   └────────────────────────────────────────────────────┘

3. PROCESSING ALL VALIDATION BATCHES
   File: utils.py
   Function: validate_loss()
   Line: ~314
   
   Code:
   ┌────────────────────────────────────────────────────┐
   │ def validate_loss(model, loader, device):         │
   │     for images, targets in tqdm(loader, ...):      │
   │         # Process one batch                       │
   │         loss_dict = model(images, targets)         │
   │         loss = sum(loss_dict.values())             │
   │         total += loss.item() * bs                 │
   │                                                     │
   │     # When this loop finishes:                    │
   │     # ALL 57 validation images processed!         │
   └────────────────────────────────────────────────────┘

4. DATA LOADER MAGIC (PyTorch handles this automatically)
   File: main.py
   Line: ~106-114 (training), ~115-123 (validation)
   
   Code:
   ┌────────────────────────────────────────────────────┐
   │ train_loader = DataLoader(                        │
   │     train_ds,      # 231 images                    │
   │     batch_size=2,  # Split into batches of 2       │
   │     shuffle=True, # Randomize order each epoch    │
   │ )                                                  │
   │                                                     │
   │ # When you iterate:                                │
   │ for batch in train_loader:                        │
   │     # batch[0] = 2 images                          │
   │     # batch[1] = 2 targets                         │
   │     # After 116 iterations: all 231 images done!   │
   └────────────────────────────────────────────────────┘
""")

print("\n" + "=" * 80)
print("VISUAL EXAMPLE: EPOCH 1")
print("=" * 80)

print("""
EPOCH 1 STARTS
├─ Training Phase (231 images, batch_size=2)
│  ├─ Batch 1:   Images [0, 1]     (2 images)
│  ├─ Batch 2:   Images [2, 3]     (2 images)
│  ├─ Batch 3:   Images [4, 5]     (2 images)
│  ├─ ...
│  ├─ Batch 115: Images [228, 229] (2 images)
│  └─ Batch 116: Images [230]      (1 image)  ← Last batch
│
│  ✅ All 231 training images processed!
│
├─ Validation Phase (57 images, batch_size=1)
│  ├─ Batch 1:   Image [0]        (1 image)
│  ├─ Batch 2:   Image [1]        (1 image)
│  ├─ Batch 3:   Image [2]        (1 image)
│  ├─ ...
│  └─ Batch 57:  Image [56]       (1 image)
│
│  ✅ All 57 validation images processed!
│
└─ EPOCH 1 COMPLETE

EPOCH 2 STARTS (same process, but images may be in different order due to shuffle=True)
...
""")

print("\n" + "=" * 80)
print("KEY INSIGHT")
print("=" * 80)

print("""
✅ YES! Each epoch processes the FULL dataset through batches.

The DataLoader automatically:
1. Takes your dataset (231 images for training, 57 for validation)
2. Splits it into batches (116 batches of 2, or 57 batches of 1)
3. Yields one batch at a time when you iterate
4. When the loop finishes: ALL images have been processed!

This happens in:
- train_one_epoch(): Loops through ALL training batches
- validate_loss(): Loops through ALL validation batches
- evaluate_coco(): Loops through ALL validation batches again (for mAP)

Each epoch = One complete pass through the entire dataset!
""")

