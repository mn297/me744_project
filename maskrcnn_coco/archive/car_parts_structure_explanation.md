# Car-Parts-Segmentation Dataset Structure Explained

## Overview
- **Total Images**: 400
- **Total Annotations**: 3,073
- **Total Categories**: 19
- **Average annotations per image**: 7.68
- **Image size**: 512×512 pixels

## JSON Structure

```json
{
  "images": [...],      // 400 image objects
  "annotations": [...], // 3,073 annotation objects
  "categories": [...]   // 19 category objects
}
```

## 1. IMAGES Array (400 items)

Each image object contains:

### Required Fields (COCO):
- `id` (int) - **Unique image identifier** - Used by annotations to reference this image
- `file_name` (str) - Image filename (e.g., "train1.jpg")
- `width` (int) - Image width in pixels (512)
- `height` (int) - Image height in pixels (512)

### Additional Fields (dataset-specific):
- `dataset_id` (int) - Internal dataset identifier
- `path` (str) - Relative path to image file
- `category_ids` (list) - List of category IDs present in this image
- `annotated` (bool) - Whether image has annotations
- `num_annotations` (int) - Number of annotations for this image
- `metadata` (dict) - Additional metadata
- `deleted`, `milliseconds`, `events`, `regenerate_thumbnail` - Optional fields

### Example:
```json
{
  "id": 1,
  "file_name": "train1.jpg",
  "width": 512,
  "height": 512,
  "path": "JPEGImages/train1.jpg",
  "num_annotations": 3
}
```

## 2. ANNOTATIONS Array (3,073 items)

Each annotation object contains:

### Required Fields (COCO):
- `id` (int) - **Unique annotation identifier**
- `image_id` (int) - **Links to `images[id]`** - Which image this annotation belongs to
- `category_id` (int) - **Links to `categories[id]`** - What object class this is
- `segmentation` (list) - Polygon coordinates defining object mask
  - Format: `[[x1,y1,x2,y2,x3,y3,...], [polygon2], ...]`
  - Multiple polygons for objects with holes or disconnected parts
- `area` (float) - Area of segmented object in pixels²
- `bbox` (list) - Bounding box `[x, y, width, height]`
  - `x, y`: top-left corner coordinates
  - `width, height`: box dimensions
- `iscrowd` (int) - 0 = single object, 1 = crowd/group

### Additional Fields:
- `isbbox` (bool) - Whether annotation is just a bounding box
- `color` (str) - Color code for visualization (e.g., "#e45779")
- `metadata` (dict) - Additional metadata

### Example:
```json
{
  "id": 111,
  "image_id": 1,
  "category_id": 19,
  "segmentation": [[132.1, 345.3, 141.5, 352.4, ...], [391.5, 346.5, ...]],
  "area": 6479.0,
  "bbox": [99.0, 344.0, 311.0, 80.0],
  "iscrowd": 0
}
```

## 3. CATEGORIES Array (19 items)

Each category object contains:

### Required Fields (COCO):
- `id` (int) - **Unique category identifier** - Used by annotations
- `name` (str) - Human-readable category name

### Additional Fields:
- `supercategory` (str) - Parent category (often empty "")
- `color` (str) - Color code for visualization
- `metadata` (dict) - Additional metadata
- `keypoint_colors` (list) - Colors for keypoints if applicable

### All Categories:
1. `_background_` (id: 1) - 0 annotations
2. `back_bumper` (id: 2) - 101 annotations
3. `back_glass` (id: 3) - 112 annotations
4. `back_left_door` (id: 4) - 122 annotations
5. `back_left_light` (id: 5) - 149 annotations
6. `back_right_door` (id: 6) - 103 annotations
7. `back_right_light` (id: 7) - 119 annotations
8. `front_bumper` (id: 8) - 239 annotations (most common)
9. `front_glass` (id: 9) - 218 annotations
10. `front_left_door` (id: 10) - 132 annotations
11. `front_left_light` (id: 11) - 236 annotations
12. `front_right_door` (id: 12) - 105 annotations
13. `front_right_light` (id: 13) - 211 annotations
14. `hood` (id: 14) - 228 annotations
15. `left_mirror` (id: 15) - 240 annotations (most common)
16. `right_mirror` (id: 16) - 233 annotations
17. `tailgate` (id: 17) - 50 annotations (least common)
18. `trunk` (id: 18) - 82 annotations
19. `wheel` (id: 19) - 393 annotations (most common)

## How Everything Relates

The structure forms a **relational graph** using IDs:

```
┌─────────────────┐
│   IMAGES        │
│   id: 1         │ ←──┐
│   file: train1  │    │
│   size: 512x512 │    │
└─────────────────┘    │
                       │ image_id reference
┌─────────────────┐    │
│  ANNOTATIONS    │    │
│   id: 111       │ ───┘
│   image_id: 1   │ ←──┐
│   category_id: 19│   │
│   bbox: [...]   │   │
│   segmentation  │   │
└─────────────────┘   │
                      │ category_id reference
┌─────────────────┐   │
│   CATEGORIES    │   │
│   id: 19        │ ──┘
│   name: wheel   │
└─────────────────┘
```

### Relationship Chain Example:

**Image ID 1** (`train1.jpg`)
  ↓ (referenced by `image_id` in annotations)
**Annotation ID 111** (wheel object)
  ↓ (references `category_id`)
**Category ID 19** (`wheel`)

### Key Relationships:

1. **One Image → Many Annotations**
   - Each image can have multiple annotations (average 7.68 per image)
   - Range: 3 to 13 annotations per image

2. **One Annotation → One Image**
   - Each annotation belongs to exactly one image
   - `annotation["image_id"]` must match `image["id"]`

3. **One Annotation → One Category**
   - Each annotation represents one object class
   - `annotation["category_id"]` must match `category["id"]`

4. **One Category → Many Annotations**
   - Each category can appear in many annotations
   - Example: `wheel` appears in 393 annotations across different images

## Data Flow Example

For image `train1.jpg` (id=1):

1. **Find image**: Look in `images[]` for `id == 1`
   - Found: `train1.jpg`, 512×512 pixels

2. **Find annotations**: Look in `annotations[]` for `image_id == 1`
   - Found: 3 annotations (ids: 111, 112, 113)

3. **For each annotation, find category**:
   - Annotation 111: `category_id = 19` → Category 19 = `wheel`
   - Annotation 112: `category_id = 10` → Category 10 = `front_left_door`
   - Annotation 113: `category_id = 11` → Category 11 = `front_left_light`

4. **Result**: Image shows 3 car parts: 1 wheel, 1 front left door, 1 front left light

## Usage in PyTorch

When loading data:
1. Load image file using `image["file_name"]`
2. Get all annotations where `annotation["image_id"] == image["id"]`
3. For each annotation:
   - Get category name from `categories[]` where `id == annotation["category_id"]`
   - Convert `segmentation` polygons to mask
   - Use `bbox` for bounding box coordinates

This structure allows efficient querying:
- "Which images contain wheels?" → Find annotations with `category_id == 19`
- "What objects are in image 1?" → Find annotations with `image_id == 1`
- "How many wheels are in the dataset?" → Count annotations with `category_id == 19`

