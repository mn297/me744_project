# ME744 Project Map

What lives where:
- `maskrcnn/` — COCO-format pipeline, converters, Mask R-CNN + YOLO flows.
- `unet/Pytorch-UNet/` — standalone U-Net trainer/inferencer (see its README for flags).
- `datasets/` — raw data and produced predictions.
- References: `maskrcnn/torchvision_references/`, `maskrcnn/Mask_RCNN/`, `maskrcnn/JSON2YOLO/` (vendor/upstream helpers).

Data prep (pick what you need):
- Conversions: `maskrcnn/convert_envy_to_coco.py`, `convert_fuji_to_coco.py`, `convert_coco_to_yolo.py`, `convert_coco_to_unet.py`, `convert_envy_to_unet.py`.
- Relabel/merge: `maskrcnn/mix_datasets.py`, `from4lg_to2lb.py`.
- Augment: `maskrcnn/augment_fuji.py`.

Training entrypoints:
- Mask R-CNN: `maskrcnn/train_maskrcnn.py` (YAML configs: `maskrcnn/experiment_*.yaml`; outputs to `maskrcnn/runs` and `checkpoints`).
- YOLO: `maskrcnn/train_yolo.py` (expects YOLO data; weights in `maskrcnn/yolo*.pt`).
- U-Net: `unet/Pytorch-UNet/train.py` (expects U-Net masks).

Inference / visualization:
- Mask R-CNN: `maskrcnn/predict_maskrcnn.py`.
- YOLO: `maskrcnn/predict_yolo.py`, `visualize_yolo.py`, `visualize_yolo_quick.py`.
- U-Net: `unet/Pytorch-UNet/predict.py`, `maskrcnn/predict_unet.py`.
- Fusion/ensembling: `maskrcnn/predict_fusion.py`, `maskrcnn/merge_and_eval.py`.
- Quick helpers: `maskrcnn/visualize_fusion.py`, `maskrcnn/color_picker.py`.

How it connects:
- Convert raw data into the target format (COCO, YOLO, or U-Net masks) with the converters above.
- Train with the matching entrypoint.
- Predict/visualize with the paired script.
- Optionally fuse or aggregate outputs via `predict_fusion.py` or `merge_and_eval.py`.
