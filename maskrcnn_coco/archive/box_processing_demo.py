"""
Standalone demonstration of how bounding boxes are processed
from COCO format to PyTorch tensor with shape [N, 4].

This shows the complete pipeline from first principles.
"""

import torch
import numpy as np
from typing import List


def process_boxes_from_coco_annotations(annotations: List[dict]) -> torch.Tensor:
    """
    Process COCO annotations to create boxes tensor with shape [N, 4].

    Args:
        annotations: List of COCO annotation dicts, each with 'bbox' key
                   Format: [x, y, width, height] (COCO format)

    Returns:
        torch.Tensor: Boxes tensor with shape [N, 4] where each row is [x1, y1, x2, y2]
    """
    print("=" * 80)
    print("BOX PROCESSING PIPELINE")
    print("=" * 80)

    # Step 1: Start with empty list
    boxes = []
    print(f"\n1. Initialize empty list: boxes = {boxes}")

    # Step 2: Process each annotation
    for i, ann in enumerate(annotations):
        # COCO format: bbox = [x, y, width, height]
        x, y, w, h = ann["bbox"]
        print(f"\n2.{i+1} Annotation {i+1}:")
        print(f"   COCO bbox format: [x={x}, y={y}, width={w}, height={h}]")

        # Convert to [x1, y1, x2, y2] format (top-left and bottom-right corners)
        x1, y1 = x, y  # Top-left corner
        x2, y2 = x + w, y + h  # Bottom-right corner
        box = [x1, y1, x2, y2]
        boxes.append(box)
        print(f"   Converted to [x1, y1, x2, y2]: {box}")

    print(f"\n3. After processing all annotations:")
    print(f"   boxes list: {boxes}")
    print(f"   Number of boxes: {len(boxes)}")

    # Step 3: Handle empty case
    if boxes:
        # Non-empty: Convert list to tensor directly
        print(f"\n4. Non-empty case: Converting list to tensor")
        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        print(f"   Result shape: {boxes_tensor.shape}")
        print(f"   Expected: [N, 4] where N={len(boxes)}")
    else:
        # Empty case: Create tensor with correct shape [0, 4]
        print(f"\n4. Empty case: Creating tensor with shape [0, 4]")
        boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
        print(f"   Result shape: {boxes_tensor.shape}")
        print(f"   This is critical! Empty list -> tensor([0]) is WRONG")
        print(f"   We need tensor with shape [0, 4] for Mask R-CNN")

    print(f"\n5. Final tensor:")
    print(f"   Shape: {boxes_tensor.shape}")
    print(f"   Dtype: {boxes_tensor.dtype}")
    print(f"   Content:\n{boxes_tensor}")
    print("=" * 80)

    return boxes_tensor


# ============================================================================
# DEMONSTRATION EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Image with annotations")
    print("=" * 80)

    # Example 1: Image with multiple annotations
    annotations_with_boxes = [
        {"bbox": [10, 20, 50, 60]},  # x=10, y=20, w=50, h=60
        {"bbox": [100, 150, 80, 90]},  # x=100, y=150, w=80, h=90
        {"bbox": [200, 300, 40, 50]},  # x=200, y=300, w=40, h=50
    ]

    result1 = process_boxes_from_coco_annotations(annotations_with_boxes)
    assert result1.shape == (3, 4), f"Expected (3, 4), got {result1.shape}"
    print("\n✅ Example 1 passed: Shape is [3, 4]")

    print("\n" + "=" * 80)
    print("EXAMPLE 2: Image with NO annotations (empty case)")
    print("=" * 80)

    # Example 2: Image with no annotations (the problematic case)
    annotations_empty = []

    result2 = process_boxes_from_coco_annotations(annotations_empty)
    assert result2.shape == (0, 4), f"Expected (0, 4), got {result2.shape}"
    print("\n✅ Example 2 passed: Shape is [0, 4] (not [0]!)")

    print("\n" + "=" * 80)
    print("EXAMPLE 3: What happens if we DON'T handle empty case?")
    print("=" * 80)

    # This is what causes the error!
    empty_list = []
    wrong_tensor = torch.as_tensor(empty_list, dtype=torch.float32)
    print(f"\n❌ WRONG way (causes error):")
    print(f"   torch.as_tensor([]) -> shape: {wrong_tensor.shape}")
    print(f"   Mask R-CNN expects [N, 4], but got {wrong_tensor.shape}")
    print(
        f"   This will cause: AssertionError: Expected target boxes to be a tensor of shape [N, 4], got torch.Size([0])"
    )

    correct_tensor = torch.zeros((0, 4), dtype=torch.float32)
    print(f"\n✅ CORRECT way:")
    print(f"   torch.zeros((0, 4)) -> shape: {correct_tensor.shape}")
    print(f"   Mask R-CNN is happy! Shape is [0, 4]")

    print("\n" + "=" * 80)
    print("EXAMPLE 4: Visual representation")
    print("=" * 80)

    # Visual example
    example_boxes = [
        {"bbox": [50, 50, 100, 80]},  # Box 1: x=50, y=50, w=100, h=80
        {"bbox": [200, 100, 150, 120]},  # Box 2: x=200, y=100, w=150, h=120
    ]

    boxes_list = []
    for ann in example_boxes:
        x, y, w, h = ann["bbox"]
        boxes_list.append([x, y, x + w, y + h])

    boxes_tensor = torch.as_tensor(boxes_list, dtype=torch.float32)

    print("\nInput COCO annotations:")
    for i, ann in enumerate(example_boxes):
        x, y, w, h = ann["bbox"]
        print(f"  Box {i+1}: [x={x}, y={y}, width={w}, height={h}]")
        print(f"            -> Top-left: ({x}, {y}), Bottom-right: ({x+w}, {y+h})")

    print(f"\nOutput tensor:")
    print(f"  Shape: {boxes_tensor.shape}")
    print(f"  Content:")
    print(boxes_tensor)
    print(f"\n  Interpretation:")
    print(
        f"    Row 0: [x1={boxes_tensor[0,0]:.0f}, y1={boxes_tensor[0,1]:.0f}, x2={boxes_tensor[0,2]:.0f}, y2={boxes_tensor[0,3]:.0f}]"
    )
    print(
        f"    Row 1: [x1={boxes_tensor[1,0]:.0f}, y1={boxes_tensor[1,1]:.0f}, x2={boxes_tensor[1,2]:.0f}, y2={boxes_tensor[1,3]:.0f}]"
    )

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(
        """
Pipeline:
1. COCO annotation: bbox = [x, y, width, height]
2. Convert to: [x1, y1, x2, y2] = [x, y, x+width, y+height]
3. Append to list: boxes = [[x1, y1, x2, y2], ...]
4. Convert to tensor:
   - If non-empty: torch.as_tensor(boxes) -> [N, 4] ✅
   - If empty: torch.zeros((0, 4)) -> [0, 4] ✅
                  NOT torch.as_tensor([]) -> [0] ❌

Key insight: Empty tensor must have shape [0, 4], not [0]!
    """
    )
