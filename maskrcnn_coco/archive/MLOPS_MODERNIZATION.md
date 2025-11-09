# MLOps Modernization Guide

## Current State ✅

Your pipeline already has:
- ✅ **TensorBoard** logging (present in `main.py` line 169)
- ✅ Checkpointing (per-epoch and best model)
- ✅ Early stopping
- ✅ Training summary JSON
- ✅ Progress bars (tqdm)
- ✅ AMP/BF16 support

## What's Missing for Modern MLOps

### 1. Experiment Tracking
- Currently: Only TensorBoard
- Add: Weights & Biases (W&B) or MLflow for better experiment management

### 2. Configuration Management
- Currently: argparse with hardcoded defaults
- Add: Hydra or OmegaConf for structured configs

### 3. Model Versioning
- Currently: Simple file-based checkpoints
- Add: Model registry with metadata

### 4. Enhanced Logging
- Currently: Basic scalar logging
- Add: Images, histograms, model graphs

### 5. Hyperparameter Tracking
- Currently: Not systematically tracked
- Add: HParams logging for TensorBoard

## Quick Start: Add W&B Tracking

### Install:
```bash
uv add wandb
```

### Modify `main.py`:

```python
# Add at top
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# In main(), replace:
writer = SummaryWriter(log_dir=args.logdir)

# With:
writer = SummaryWriter(log_dir=args.logdir)

if WANDB_AVAILABLE and os.getenv("USE_WANDB", "false").lower() == "true":
    wandb.init(
        project="maskrcnn",
        name=f"fuji_apples_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "dataset": "Fuji-Apple-Segmentation",
        },
    )
```

### Modify `utils.py` in `fit()` function:

```python
# After line 407, add:
if WANDB_AVAILABLE and wandb.run is not None:
    wandb.log({
        "epoch": ep,
        "loss/train": train_loss,
        "loss/val": val_loss,
        "mAP/bbox": metrics["bbox"]["AP"],
        "mAP/segm": metrics["segm"]["AP"],
        "lr": optimizer.param_groups[0]["lr"],
    })
```

### Run with W&B:
```bash
USE_WANDB=true uv run python main.py
```

## Full Modernization: Use the MLOps Module

I've created `mlops_modernization.py` with a unified tracking interface.

### 1. Install dependencies:
```bash
uv add wandb mlflow omegaconf
```

### 2. Update `main.py`:

```python
from mlops_modernization import ExperimentTracker, create_experiment_config

# In main():
config = create_experiment_config(args)
tracker = ExperimentTracker(
    experiment_name="maskrcnn_fuji_apples",
    project_name="maskrcnn",
    use_wandb=True,  # Enable W&B
    use_mlflow=False,  # Or enable MLflow
    log_dir=args.logdir,
    config=config,
    tags=["fuji", "apples", "segmentation"],
)

# Pass tracker instead of writer to fit()
fit(
    ...,
    tracker=tracker,  # Instead of writer=writer
    ...
)

tracker.finish()
```

### 3. Update `utils.py` `fit()` function:

Replace `writer` parameter with `tracker` and use:
```python
tracker.log_scalars({
    "loss/train": train_loss,
    "loss/val": val_loss,
    "mAP/bbox": metrics["bbox"]["AP"],
    "mAP/segm": metrics["segm"]["AP"],
}, step=ep)
```

## Benefits

### Weights & Biases:
- 📊 Beautiful dashboards
- 🔍 Experiment comparison
- 📈 Hyperparameter sweeps
- 🎯 Model versioning
- 👥 Team collaboration

### MLflow:
- 📦 Model registry
- 🔄 Model serving
- 📝 Experiment tracking
- 🏷️ Model versioning

### TensorBoard:
- ✅ Already integrated
- 📈 Real-time monitoring
- 🖼️ Image visualization
- 📊 Histogram tracking

## Recommended Workflow

1. **Development**: Use TensorBoard (already set up)
2. **Experiments**: Add W&B for better tracking
3. **Production**: Use MLflow for model registry

## Next Steps

1. ✅ TensorBoard - Already implemented
2. 🔄 Add W&B tracking (optional but recommended)
3. 🔄 Add configuration management (Hydra)
4. 🔄 Add model registry
5. 🔄 Add automated hyperparameter tuning

