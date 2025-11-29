"""
Clear explanation of dataset sizes, batch sizes, and data loader behavior.

This explains the "random numbers" you're seeing in training.
"""
from pycocotools.coco import COCO

# Load datasets
fuji_train = COCO("Fuji-Apple-Segmentation/trainingset/annotations.json")
fuji_val = COCO("Fuji-Apple-Segmentation/testset/annotations.json")

print("=" * 80)
print("DATASET BREAKDOWN")
print("=" * 80)

print(f"\n📊 FUJI-APPLE-SEGMENTATION DATASET:")
print(f"   Training set (trainingset/):")
print(f"     • Images: {len(fuji_train.imgs)}")
print(f"     • Annotations: {len(fuji_train.anns)}")
print(f"     • Used for: TRAINING the model")

print(f"\n   Validation set (testset/):")
print(f"     • Images: {len(fuji_val.imgs)}")
print(f"     • Annotations: {len(fuji_val.anns)}")
print(f"     • Used for: VALIDATION (checking model performance)")

print(f"\n   TOTAL: {len(fuji_train.imgs) + len(fuji_val.imgs)} images")

print("\n" + "=" * 80)
print("DATA LOADER EXPLANATION")
print("=" * 80)

# Simulate what happens with different batch sizes
batch_sizes = [1, 2, 4, 8]

print(f"\nWith {len(fuji_train.imgs)} training images:")

for bs in batch_sizes:
    num_batches = (len(fuji_train.imgs) + bs - 1) // bs  # Ceiling division
    print(f"  Batch size {bs:2d}: {num_batches:3d} batches")
    print(f"    → {num_batches} batches × {bs} images/batch = {num_batches * bs} images processed")
    if num_batches * bs > len(fuji_train.imgs):
        print(f"    ⚠️  Last batch has only {len(fuji_train.imgs) - (num_batches - 1) * bs} images")

print(f"\nWith {len(fuji_val.imgs)} validation images:")

for bs in batch_sizes:
    num_batches = (len(fuji_val.imgs) + bs - 1) // bs
    print(f"  Batch size {bs:2d}: {num_batches:3d} batches")
    if num_batches * bs > len(fuji_val.imgs):
        print(f"    ⚠️  Last batch has only {len(fuji_val.imgs) - (num_batches - 1) * bs} images")

print("\n" + "=" * 80)
print("YOUR CURRENT SETUP (from main.py defaults)")
print("=" * 80)

train_bs = 2
val_bs = 1

train_batches = (len(fuji_train.imgs) + train_bs - 1) // train_bs
val_batches = (len(fuji_val.imgs) + val_bs - 1) // val_bs

print(f"""
Training:
  • Dataset: trainingset/ ({len(fuji_train.imgs)} images)
  • Batch size: {train_bs}
  • Number of batches per epoch: {train_batches}
  • Calculation: {len(fuji_train.imgs)} images ÷ {train_bs} batch_size = {train_batches} batches

Validation:
  • Dataset: testset/ ({len(fuji_val.imgs)} images)  ← This is the "57 images"!
  • Batch size: {val_bs}
  • Number of batches per epoch: {val_batches}
  • Calculation: {len(fuji_val.imgs)} images ÷ {val_bs} batch_size = {val_batches} batches

So when you see:
  "Validation batches: 57"
  
It means:
  • You have 57 images in the VALIDATION set (testset/)
  • Batch size is 1 (one image per batch)
  • So: 57 images ÷ 1 = 57 batches
""")

print("=" * 80)
print("WHY DIFFERENT BATCH SIZES?")
print("=" * 80)
print("""
Training batch size = 2:
  • Process 2 images at once
  • Faster training (parallel processing)
  • More memory usage
  • {train_batches} batches per epoch

Validation batch size = 1:
  • Process 1 image at a time
  • Less memory usage (good for evaluation)
  • Easier to handle variable-sized images
  • {val_batches} batches per epoch
""")

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"""
✅ Training: {len(fuji_train.imgs)} images in trainingset/
✅ Validation: {len(fuji_val.imgs)} images in testset/ (this is your "57")
✅ Total: {len(fuji_train.imgs) + len(fuji_val.imgs)} images

Each epoch:
  • Training: {train_batches} batches (processing {len(fuji_train.imgs)} images)
  • Validation: {val_batches} batches (processing {len(fuji_val.imgs)} images)

The "57" you see = number of VALIDATION images (testset/)
""")

