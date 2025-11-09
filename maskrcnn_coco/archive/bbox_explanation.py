"""
Explanation of how bounding boxes are processed from COCO annotations.

Using the user's actual example annotation.
"""

import torch

# Your example annotation
annotation_example = {
    "id": 111,
    "image_id": 1,
    "category_id": 19,
    "segmentation": [
        [132.1, 345.3, 141.5, 352.4, ...],  # Polygon 1
        [391.5, 346.5, 400.4, 354.7, ...],  # Polygon 2
    ],
    "area": 6479,
    "bbox": [99.0, 344.0, 311.0, 80.0],  # ← This is what we use!
    "iscrowd": False,
}

print("=" * 80)
print("UNDERSTANDING COCO BBOX FORMAT")
print("=" * 80)

print("\n1. COCO bbox format: [x, y, width, height]")
x, y, width, height = annotation_example["bbox"]
print(f"   bbox = {annotation_example['bbox']}")
print(f"   x (left) = {x}")
print(f"   y (top) = {y}")
print(f"   width = {width}")
print(f"   height = {height}")

print("\n2. Convert to [x1, y1, x2, y2] format (corner coordinates):")
x1 = x  # Left edge
y1 = y  # Top edge
x2 = x + width  # Right edge
y2 = y + height  # Bottom edge
print(f"   x1 (left) = {x1}")
print(f"   y1 (top) = {y1}")
print(f"   x2 (right) = {x1 + width}")
print(f"   y2 (bottom) = {y1 + height}")
print(f"   Result: [{x1}, {y1}, {x2}, {y2}]")

print("\n" + "=" * 80)
print("COLLECTING BOXES FOR AN IMAGE")
print("=" * 80)

print("\n3. For EACH image, we collect ALL boxes from ALL annotations:")
print("   - Image ID 1 might have multiple annotations")
print("   - Each annotation has ONE bbox")
print("   - We collect all bboxes into a list")

# Simulate multiple annotations for the same image
annotations_for_image_1 = [
    {"bbox": [99.0, 344.0, 311.0, 80.0], "category_id": 19},  # Your example
    {"bbox": [200.0, 100.0, 150.0, 120.0], "category_id": 5},  # Another object
    {"bbox": [50.0, 50.0, 100.0, 80.0], "category_id": 12},  # Another object
]

print(f"\n   Example: Image ID 1 has {len(annotations_for_image_1)} annotations")
print("   Processing each annotation:")

boxes = []
for i, ann in enumerate(annotations_for_image_1):
    x, y, w, h = ann["bbox"]
    x1, y1, x2, y2 = x, y, x + w, y + h
    box = [x1, y1, x2, y2]
    boxes.append(box)
    print(f"   Annotation {i+1}: bbox {ann['bbox']} -> box {box}")

print(f"\n4. Final boxes list for image:")
print(f"   boxes = {boxes}")
print(f"   Number of boxes: {len(boxes)}")

print("\n5. Convert to tensor:")
boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
print(f"   Shape: {boxes_tensor.shape}")
print(f"   Content:")
print(boxes_tensor)

print("\n" + "=" * 80)
print("KEY POINTS")
print("=" * 80)
print(
    """
1. Each annotation has ONE bbox in COCO format: [x, y, width, height]

2. We convert EACH bbox to corner format: [x1, y1, x2, y2]
   - x1, y1 = top-left corner (from x, y)
   - x2, y2 = bottom-right corner (from x+width, y+height)

3. We collect ALL boxes from ALL annotations for the same image
   - If image has 3 objects → 3 boxes in the list
   - If image has 0 objects → empty list (need special handling!)

4. Final tensor shape: [N, 4] where N = number of objects in image
   - Row 0: box for object 1
   - Row 1: box for object 2
   - Row 2: box for object 3
   - etc.

5. The segmentation polygons are SEPARATE - they're used for masks, not boxes!
   - bbox = bounding box (rectangle)
   - segmentation = detailed polygon mask (more precise)
"""
)

print("\n" + "=" * 80)
print("VISUAL EXAMPLE")
print("=" * 80)

print(
    f"""
Your annotation example:
  bbox: [99.0, 344.0, 311.0, 80.0]
  
  This means:
  - Top-left corner: (99, 344)
  - Width: 311 pixels
  - Height: 80 pixels
  - Bottom-right corner: (99+311, 344+80) = (410, 424)
  
  Converted box: [99.0, 344.0, 410.0, 424.0]
  
  If this image has other objects, we'd have:
  boxes = [
    [99.0, 344.0, 410.0, 424.0],  # Object 1 (your example)
    [200.0, 100.0, 350.0, 220.0],  # Object 2 (if exists)
    [50.0, 50.0, 150.0, 130.0],    # Object 3 (if exists)
  ]
  
  Final tensor shape: [3, 4] (if 3 objects)
"""
)
