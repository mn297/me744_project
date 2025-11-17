"""
MLOps Modernization for Mask R-CNN Training Pipeline

This module adds modern MLOps capabilities:
- Experiment tracking (Weights & Biases / MLflow)
- Configuration management (Hydra)
- Model versioning
- Enhanced logging
- Hyperparameter tracking
- Experiment organization
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import torch
from torch.utils.tensorboard import SummaryWriter

# Optional MLOps libraries
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    from omegaconf import OmegaConf, DictConfig

    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False


class ExperimentTracker:
    """
    Unified experiment tracking interface supporting:
    - TensorBoard (always available)
    - Weights & Biases (optional)
    - MLflow (optional)
    """

    def __init__(
        self,
        experiment_name: str,
        project_name: str = "maskrcnn",
        use_wandb: bool = False,
        use_mlflow: bool = False,
        log_dir: str = "runs",
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None,
    ):
        self.experiment_name = experiment_name
        self.project_name = project_name
        self.config = config or {}
        self.tags = tags or []

        # Create unique run ID
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{experiment_name}_{self.run_id}"

        # TensorBoard (always available)
        log_path = Path(log_dir) / self.run_name
        self.tb_writer = SummaryWriter(log_dir=str(log_path))
        print(f"📊 TensorBoard logs: {log_path}")
        print(f"   View with: tensorboard --logdir {log_dir}")

        # Weights & Biases
        self.wandb_run = None
        if use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=project_name,
                name=self.run_name,
                config=self.config,
                tags=self.tags,
            )
            self.wandb_run = wandb.run
            print(f"🔮 Weights & Biases: {wandb.run.url}")
        elif use_wandb and not WANDB_AVAILABLE:
            print("⚠️  W&B requested but not installed. Install with: pip install wandb")

        # MLflow
        self.mlflow_run = None
        if use_mlflow and MLFLOW_AVAILABLE:
            mlflow.set_experiment(experiment_name)
            self.mlflow_run = mlflow.start_run(run_name=self.run_name)
            if self.config:
                mlflow.log_params(self.config)
            print(f"📦 MLflow run: {self.mlflow_run.info.run_id}")
        elif use_mlflow and not MLFLOW_AVAILABLE:
            print(
                "⚠️  MLflow requested but not installed. Install with: pip install mlflow"
            )

    def log_scalar(self, key: str, value: float, step: int):
        """Log scalar metric to all active trackers."""
        self.tb_writer.add_scalar(key, value, step)

        if self.wandb_run:
            wandb.log({key: value}, step=step)

        if self.mlflow_run:
            mlflow.log_metric(key, value, step=step)

    def log_scalars(self, metrics: Dict[str, float], step: int):
        """Log multiple scalars at once."""
        for key, value in metrics.items():
            self.log_scalar(key, value, step)

    def log_image(self, key: str, image, step: int):
        """Log image to trackers."""
        self.tb_writer.add_image(key, image, step)

        if self.wandb_run:
            wandb.log({key: wandb.Image(image)}, step=step)

        if self.mlflow_run:
            mlflow.log_image(image, f"{key}_step_{step}.png")

    def log_model(self, model_path: str, metadata: Optional[Dict] = None):
        """Log model artifact."""
        if self.wandb_run:
            artifact = wandb.Artifact("model", type="model")
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)

        if self.mlflow_run:
            mlflow.log_artifact(model_path, "models")
            if metadata:
                mlflow.log_dict(metadata, "model_metadata.json")

    def log_config(self, config: Dict[str, Any]):
        """Log configuration/hyperparameters."""
        self.config.update(config)

        # Log to TensorBoard as text
        config_str = json.dumps(config, indent=2)
        self.tb_writer.add_text("config", config_str, 0)

        if self.wandb_run:
            wandb.config.update(config)

        if self.mlflow_run:
            mlflow.log_params(config)

    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Dict[str, float]):
        """Log hyperparameters and final metrics (for hyperparameter tuning)."""
        self.tb_writer.add_hparams(hparams, metrics)

        if self.wandb_run:
            wandb.config.update(hparams)
            wandb.log(metrics)

    def finish(self):
        """Close all tracking sessions."""
        self.tb_writer.close()

        if self.wandb_run:
            wandb.finish()

        if self.mlflow_run:
            mlflow.end_run()


def create_experiment_config(args) -> Dict[str, Any]:
    """Create comprehensive experiment configuration from args."""
    return {
        # Dataset
        "dataset": {
            "train_images": args.train_images,
            "train_anno": args.train_anno,
            "val_images": args.val_images,
            "val_anno": args.val_anno,
        },
        # Training
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "momentum": args.momentum,
        },
        # Optimization
        "optimization": {
            "amp": args.amp,
            "bf16": args.bf16,
            "compile": args.compile,
            "warmup_epochs": args.warmup_epochs,
            "cosine_epochs": args.cosine_epochs,
        },
        # System
        "system": {
            "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
            "seed": args.seed,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": (
                torch.cuda.device_count() if torch.cuda.is_available() else 0
            ),
        },
        # Model
        "model": {
            "architecture": "MaskRCNN_ResNet50_FPN",
            "backbone": "ResNet50",
        },
    }


def log_training_metrics(tracker: ExperimentTracker, metrics: Dict[str, float]):
    """
    Log all training metrics in organized structure.

    Args:
        tracker: ExperimentTracker instance
        metrics: Flat dictionary with all metrics (no nesting):
            - epoch: Current epoch number
            - val_loss: Validation loss
            - train_loss_*: Individual training loss components
            - bbox_*: Bbox AP metrics
            - segm_*: Segmentation AP metrics
            - lr_*: Learning rates per parameter group
    """
    epoch = int(metrics["epoch"])

    # Separate metrics by prefix for organized logging
    train_losses = {}
    bbox_metrics = {}
    segm_metrics = {}
    learning_rates = {}

    for key, value in metrics.items():
        if key.startswith("train_"):
            loss_name = key.replace("train_", "")
            train_losses[loss_name] = value
        elif key.startswith("bbox_"):
            metric_name = key.replace("bbox_", "")
            bbox_metrics[metric_name] = value
        elif key.startswith("segm_"):
            metric_name = key.replace("segm_", "")
            segm_metrics[metric_name] = value
        elif key.startswith("lr_"):
            lr_name = key.replace("lr_", "")
            learning_rates[lr_name] = value

    # Log training losses
    if train_losses:
        total_train_loss = sum(train_losses.values())
        tracker.log_scalar("loss/train_total", total_train_loss, step=epoch)
        for loss_name, loss_value in train_losses.items():
            tracker.log_scalar(f"loss/train_{loss_name}", loss_value, step=epoch)

    # Log validation loss
    tracker.log_scalar("loss/val", metrics["val_loss"], step=epoch)

    # Log bbox metrics
    if bbox_metrics:
        for metric_name, metric_value in bbox_metrics.items():
            tracker.log_scalar(f"mAP/bbox_{metric_name}", metric_value, step=epoch)

    # Log segmentation metrics
    if segm_metrics:
        for metric_name, metric_value in segm_metrics.items():
            tracker.log_scalar(f"mAP/segm_{metric_name}", metric_value, step=epoch)

    # Log learning rates
    for lr_name, lr_value in learning_rates.items():
        tracker.log_scalar(f"lr/{lr_name}", lr_value, step=epoch)


def save_experiment_summary(
    output_dir: Path,
    config: Dict[str, Any],
    best_metrics: Dict[str, float],
    total_epochs: int,
):
    """Save comprehensive experiment summary."""
    summary = {
        "experiment_info": {
            "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "total_epochs": total_epochs,
            "completed_at": datetime.now().isoformat(),
        },
        "config": config,
        "best_metrics": best_metrics,
    }

    summary_path = output_dir / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"📝 Experiment summary saved: {summary_path}")


# Example usage integration
if __name__ == "__main__":
    # Example: How to use in main.py
    print(
        """
To integrate into main.py, replace:

    writer = SummaryWriter(log_dir=args.logdir)
    
With:

    from mlops_modernization import ExperimentTracker, create_experiment_config
    
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
    
    # Then in fit() function, replace writer.add_scalar() with:
    tracker.log_scalars({
        "loss/train": train_loss,
        "loss/val": val_loss,
        "mAP/bbox": metrics["bbox"]["AP"],
    }, step=epoch)
    
    # At the end:
    tracker.finish()
    """
    )
