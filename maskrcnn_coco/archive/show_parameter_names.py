"""
Quick script to show how parameter names work in Mask R-CNN
"""

import torch
from utils import build_model

# Build a model
model = build_model(num_classes=2)

print("=" * 80)
print("MASK R-CNN PARAMETER NAMES (showing first few from each component)")
print("=" * 80)

backbone_count = 0
box_count = 0
mask_count = 0
rpn_count = 0
other_count = 0

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    
    # Show examples of parameter names
    if "backbone" in name and backbone_count < 3:
        print(f"\n🏗️  BACKBONE: {name}")
        print(f"   Shape: {param.shape}, Params: {param.numel():,}")
        backbone_count += 1
    
    elif "box_predictor" in name and box_count < 5:
        print(f"\n📦 BOX HEAD: {name}")
        print(f"   Shape: {param.shape}, Params: {param.numel():,}")
        box_count += 1
    
    elif "mask_predictor" in name and mask_count < 5:
        print(f"\n🎭 MASK HEAD: {name}")
        print(f"   Shape: {param.shape}, Params: {param.numel():,}")
        mask_count += 1
    
    elif "rpn" in name and rpn_count < 3:
        print(f"\n🎯 RPN: {name}")
        print(f"   Shape: {param.shape}, Params: {param.numel():,}")
        rpn_count += 1

print("\n" + "=" * 80)
print("HOW IT WORKS:")
print("=" * 80)
print("""
The function `model.named_parameters()` returns parameter names that include
the full path in the model hierarchy:

Examples:
  - "backbone.body.layer1.0.conv1.weight"          → goes to backbone group
  - "roi_heads.box_predictor.cls_score.weight"    → goes to box_head group  
  - "roi_heads.mask_predictor.conv5_mask.weight"  → goes to mask_head group
  - "rpn.head.conv.weight"                         → goes to other group

The code checks if certain strings are IN the parameter name:
  - if "backbone" in name       → backbone group (lr = base_lr)
  - if "box_predictor" in name  → box_head group (lr = base_lr × 10)
  - if "mask_predictor" in name → mask_head group (lr = base_lr × 10)
  - else                        → other group (lr = base_lr × 5)
""")

print("\n" + "=" * 80)
print("VERIFYING THE GROUPING:")
print("=" * 80)

# Count parameters in each group
backbone_params = []
box_head_params = []
mask_head_params = []
other_params = []

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
        
    if "backbone" in name:
        backbone_params.append(param)
    elif "box_predictor" in name:
        box_head_params.append(param)
    elif "mask_predictor" in name:
        mask_head_params.append(param)
    else:
        other_params.append(param)

print(f"\n✅ Backbone:  {len(backbone_params):3d} tensors, {sum(p.numel() for p in backbone_params):12,} params")
print(f"✅ Box Head:  {len(box_head_params):3d} tensors, {sum(p.numel() for p in box_head_params):12,} params")
print(f"✅ Mask Head: {len(mask_head_params):3d} tensors, {sum(p.numel() for p in mask_head_params):12,} params")
print(f"✅ Other:     {len(other_params):3d} tensors, {sum(p.numel() for p in other_params):12,} params")

total = sum([
    sum(p.numel() for p in backbone_params),
    sum(p.numel() for p in box_head_params),
    sum(p.numel() for p in mask_head_params),
    sum(p.numel() for p in other_params)
])
print(f"\n🔢 TOTAL:     {total:,} trainable parameters")

print("\n" + "=" * 80)

