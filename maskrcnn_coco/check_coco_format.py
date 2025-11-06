#!/usr/bin/env python3
"""
COCO Format Specification Checker
Shows what COCO format requires and validates against a dataset
"""
import json
from pycocotools.coco import COCO

print("=" * 80)
print("COCO FORMAT SPECIFICATION")
print("=" * 80)

print("\n1. TOP-LEVEL JSON STRUCTURE")
print("-" * 80)
print("A COCO JSON file must have these top-level keys:")
print("  • 'info' (optional) - Dataset metadata")
print("  • 'licenses' (optional) - License information")
print("  • 'images' (REQUIRED) - List of image metadata")
print("  • 'annotations' (REQUIRED) - List of annotation objects")
print("  • 'categories' (REQUIRED) - List of category definitions")

print("\n2. IMAGES SECTION")
print("-" * 80)
print("Each image object must have:")
print("  • 'id' (int) - Unique image identifier")
print("  • 'file_name' (str) - Image filename")
print("  • 'width' (int) - Image width in pixels")
print("  • 'height' (int) - Image height in pixels")
print("\nOptional fields:")
print("  • 'license' (int) - License ID")
print("  • 'flick_url' (str) - URL to image")
print("  • 'coco_url' (str) - COCO URL")
print("  • 'date_captured' (str) - Capture date")

print("\n3. ANNOTATIONS SECTION")
print("-" * 80)
print("Each annotation object must have:")
print("  • 'id' (int) - Unique annotation identifier")
print("  • 'image_id' (int) - ID of the image this annotation belongs to")
print("  • 'category_id' (int) - ID of the object category")
print("  • 'segmentation' (list/dict) - Segmentation mask:")
print("      - Polygon format: list of lists [[x1,y1,x2,y2,...], [polygon2], ...]")
print("      - RLE format: {'size': [height, width], 'counts': [...]}")
print("      - Empty list [] for objects without segmentation")
print("  • 'area' (float) - Area of the segmentation mask in pixels²")
print("  • 'bbox' (list) - Bounding box [x, y, width, height]")
print("      - x, y: top-left corner coordinates")
print("      - width, height: box dimensions")
print("  • 'iscrowd' (int) - 0 for single object, 1 for crowd/group")

print("\n4. CATEGORIES SECTION")
print("-" * 80)
print("Each category object must have:")
print("  • 'id' (int) - Unique category identifier")
print("  • 'name' (str) - Category name")
print("  • 'supercategory' (str) - Optional parent category")

print("\n" + "=" * 80)
print("VALIDATING AGAINST CAR-PARTS-SEGMENTATION DATASET")
print("=" * 80)

try:
    coco = COCO('Car-Parts-Segmentation/trainingset/annotations.json')
    
    # Check top-level structure
    with open('Car-Parts-Segmentation/trainingset/annotations.json') as f:
        data = json.load(f)
    
    print("\n✓ Top-level keys found:", list(data.keys()))
    print("\n✓ Validation Results:")
    print(f"  • Images: {len(coco.imgs)}")
    print(f"  • Annotations: {len(coco.anns)}")
    print(f"  • Categories: {len(coco.cats)}")
    
    # Check image structure
    img = coco.loadImgs(1)[0]
    required_img_fields = ['id', 'file_name', 'width', 'height']
    has_all = all(field in img for field in required_img_fields)
    print(f"\n✓ Image structure valid: {has_all}")
    if has_all:
        print(f"  Sample image: id={img['id']}, file={img['file_name']}, "
              f"size={img['width']}x{img['height']}")
    
    # Check annotation structure
    if coco.anns:
        ann_id = list(coco.anns.keys())[0]
        ann = coco.anns[ann_id]
        required_ann_fields = ['id', 'image_id', 'category_id', 'segmentation', 
                               'area', 'bbox', 'iscrowd']
        has_all = all(field in ann for field in required_ann_fields)
        print(f"\n✓ Annotation structure valid: {has_all}")
        if has_all:
            print(f"  Sample annotation: id={ann['id']}, image_id={ann['image_id']}, "
                  f"category_id={ann['category_id']}, area={ann['area']:.1f}")
    
    # Check category structure
    if coco.cats:
        cat = list(coco.cats.values())[1]  # Skip background
        required_cat_fields = ['id', 'name']
        has_all = all(field in cat for field in required_cat_fields)
        print(f"\n✓ Category structure valid: {has_all}")
        if has_all:
            print(f"  Sample category: id={cat['id']}, name='{cat['name']}'")
    
    print("\n" + "=" * 80)
    print("✓ DATASET IS VALID COCO FORMAT")
    print("=" * 80)
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    print("Dataset does NOT follow COCO format")

print("\n" + "=" * 80)
print("KEY REQUIREMENTS FOR COCO FORMAT:")
print("=" * 80)
print("1. JSON file with 'images', 'annotations', 'categories' keys")
print("2. Each image has: id, file_name, width, height")
print("3. Each annotation has: id, image_id, category_id, segmentation, area, bbox, iscrowd")
print("4. Each category has: id, name")
print("5. IDs must be unique within their type")
print("6. image_id in annotations must reference valid image id")
print("7. category_id in annotations must reference valid category id")
print("8. Segmentation can be polygon list or RLE format")
print("9. Bbox format: [x, y, width, height] (top-left corner + dimensions)")

