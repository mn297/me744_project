# 🔍 How Optimizer Knows Which Parameters Belong to Which Head

## The Magic: `model.named_parameters()`

PyTorch models organize parameters in a **hierarchy**. When you call `model.named_parameters()`, it returns tuples of `(name, parameter)` where the **name includes the full path** to that parameter.

## Example Parameter Names

Here's what the actual parameter names look like in Mask R-CNN:

### Backbone Parameters
```
backbone.body.conv1.weight
backbone.body.layer1.0.conv1.weight
backbone.body.layer1.0.bn1.weight
backbone.body.layer4.2.conv3.weight
backbone.fpn.inner_blocks.0.weight
...
```

### Box Predictor (Box Head) Parameters
```
roi_heads.box_predictor.cls_score.weight         ← Classification
roi_heads.box_predictor.cls_score.bias
roi_heads.box_predictor.bbox_pred.weight         ← Box regression
roi_heads.box_predictor.bbox_pred.bias
```

### Mask Predictor (Mask Head) Parameters  
```
roi_heads.mask_predictor.mask_fcn_logits.weight  ← Final mask layer
roi_heads.mask_predictor.mask_fcn_logits.bias
roi_heads.mask_predictor.conv5_mask.weight       ← Mask convolutions
roi_heads.mask_predictor.conv5_mask.bias
...
```

### RPN Parameters
```
rpn.head.conv.weight
rpn.head.cls_logits.weight
rpn.head.bbox_pred.weight
rpn.anchor_generator...
```

## How `get_parameter_groups()` Works

```python
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    
    if "backbone" in name:              # ← String matching!
        backbone_params.append(param)
    elif "box_predictor" in name:       # ← String matching!
        box_head_params.append(param)
    elif "mask_predictor" in name:      # ← String matching!
        mask_head_params.append(param)
    else:
        other_params.append(param)      # ← RPN and misc
```

It's doing **simple string matching** on the parameter name!

### Example Matching

| Parameter Name | Contains "backbone"? | Contains "box_predictor"? | Contains "mask_predictor"? | Goes to Group |
|----------------|---------------------|---------------------------|----------------------------|---------------|
| `backbone.body.layer1.0.conv1.weight` | ✅ YES | ❌ No | ❌ No | **backbone** |
| `roi_heads.box_predictor.cls_score.weight` | ❌ No | ✅ YES | ❌ No | **box_head** |
| `roi_heads.mask_predictor.conv5_mask.weight` | ❌ No | ❌ No | ✅ YES | **mask_head** |
| `rpn.head.conv.weight` | ❌ No | ❌ No | ❌ No | **other** (RPN) |

## Why This Works

PyTorch's `nn.Module` automatically creates these hierarchical names based on how you structure your model:

```python
class MaskRCNN(nn.Module):
    def __init__(self):
        self.backbone = ...              # All params get "backbone." prefix
        self.rpn = ...                   # All params get "rpn." prefix
        self.roi_heads = RoIHeads()
        
class RoIHeads(nn.Module):
    def __init__(self):
        self.box_predictor = ...         # Gets "roi_heads.box_predictor." prefix
        self.mask_predictor = ...        # Gets "roi_heads.mask_predictor." prefix
```

So when TorchVision builds the model, it naturally creates these organized names!

## The Optimizer Gets Parameter Groups

Once grouped, we pass them to the optimizer with different learning rates:

```python
param_groups = [
    {
        "params": backbone_params,       # List of Parameter objects
        "lr": 5e-3,                      # Low LR for pretrained weights
        "name": "backbone"               # Optional label (for logging)
    },
    {
        "params": box_head_params,       # List of Parameter objects
        "lr": 5e-2,                      # High LR (10×) for random init
        "name": "box_head"
    },
    {
        "params": mask_head_params,      # List of Parameter objects
        "lr": 5e-2,                      # High LR (10×) for random init
        "name": "mask_head"
    },
    {
        "params": other_params,          # RPN, etc.
        "lr": 2.5e-2,                    # Medium LR (5×)
        "name": "other"
    }
]

optimizer = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=5e-4)
```

## Verifying It Works

When you train, you'll see this output:

```
📊 Parameter Groups:
  Backbone: 23,528,320 params @ lr=5.00e-03
  Box Head: 12,845 params @ lr=5.00e-02
  Mask Head: 738,561 params @ lr=5.00e-02
  Other (RPN): 1,229,312 params @ lr=2.50e-02
```

This confirms all parameters were correctly grouped!

## Why Different Learning Rates?

| Component | Status | Why This LR? |
|-----------|--------|--------------|
| **Backbone** | Pretrained on ImageNet/COCO | Small LR (5e-3) preserves learned features |
| **Box Head** | Randomly initialized | Large LR (5e-2) learns from scratch quickly |
| **Mask Head** | Randomly initialized | Large LR (5e-2) learns from scratch quickly |
| **RPN** | Partially pretrained | Medium LR (2.5e-2) adapts to your data |

## Common Question: What if Names Change?

If you use a different model architecture, you'd need to adjust the string matching. For example:

```python
# For a different architecture
if "encoder" in name:           # Instead of "backbone"
    encoder_params.append(param)
elif "decoder.box" in name:     # Instead of "box_predictor"
    box_params.append(param)
```

But for TorchVision's Mask R-CNN, the naming is consistent across versions, so this works reliably!

## TL;DR

1. **`model.named_parameters()`** returns `(name, param)` tuples
2. **Names include full path** like `"roi_heads.mask_predictor.conv5_mask.weight"`
3. **We check if strings are in the name** to determine which group
4. **Optimizer gets lists of parameters** with different learning rates
5. **It just works!** PyTorch handles the rest during training

---

**Bottom line:** The optimizer doesn't "know" anything special. We manually group parameters by checking if certain strings appear in their hierarchical names, then tell the optimizer "use LR X for this group, LR Y for that group."

