"""PyTorch Lightning LightningModule for BioMassters AGB estimation."""

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl

from biomassters.losses.losses import build_loss
from biomassters.metrics.metrics import MetricsCollection

logger = logging.getLogger(__name__)


class BioMasstersLitModule(pl.LightningModule):
    """LightningModule wrapping model, loss, metrics, and optimiser.

    Handles training, validation, and test steps with automatic metric
    logging to W&B (or any Logger attached by the Trainer).

    Args:
        model:          Instantiated ``nn.Module`` with ``forward(x) → (B,1,H,W)``.
        loss_name:      Name of the loss function (see ``build_loss``).
        optimizer_cfg:  Dict with keys: ``name``, ``lr``, ``weight_decay``, ``betas``.
        scheduler_cfg:  Dict with keys: ``name``, and scheduler-specific params.
        log_images:     Whether to log prediction visualisations to W&B.
        log_images_n:   Number of samples to visualise per validation epoch.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_name: str = "masked_rmse",
        optimizer_cfg: Optional[Dict[str, Any]] = None,
        scheduler_cfg: Optional[Dict[str, Any]] = None,
        log_images: bool = True,
        log_images_n: int = 4,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = build_loss(loss_name)
        self.optimizer_cfg = optimizer_cfg or {
            "name": "adamw",
            "lr": 1e-4,
            "weight_decay": 1e-4,
        }
        self.scheduler_cfg = scheduler_cfg or {
            "name": "cosine",
            "T_max": 50,
            "eta_min": 1e-6,
        }
        self.log_images = log_images
        self.log_images_n = log_images_n

        # Metrics per split
        self.train_metrics = MetricsCollection(prefix="train/")
        self.val_metrics = MetricsCollection(prefix="val/")
        self.test_metrics = MetricsCollection(prefix="test/")

        # Storage for validation visualisation
        self._val_vis_preds: List[torch.Tensor] = []
        self._val_vis_targets: List[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run inference.

        Args:
            x: Input of shape ``(B, T, C, H, W)``.

        Returns:
            AGB prediction of shape ``(B, 1, H, W)``.
        """
        return self.model(x)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Single training step.

        Args:
            batch:     Dict with ``'image'``, ``'target'``, ``'mask'``.
            batch_idx: Index of the current batch.

        Returns:
            Scalar loss tensor.
        """
        image = batch["image"]  # (B, T, C, H, W)
        target = batch["target"]  # (B, 1, H, W)

        pred = self(image)  # (B, 1, H, W)
        loss = self.loss_fn(pred, target)

        # Update and log training metrics
        with torch.no_grad():
            self.train_metrics.update(pred.detach(), target.detach())

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        """Log training metrics at epoch end."""
        metrics = self.train_metrics.compute()
        self.log_dict(metrics, on_epoch=True, prog_bar=False)
        self.train_metrics.reset()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Single validation step.

        Args:
            batch:     Dict with ``'image'``, ``'target'``, ``'mask'``.
            batch_idx: Index of the current batch.
        """
        image = batch["image"]
        target = batch["target"]

        with torch.no_grad():
            pred = self(image)
            loss = self.loss_fn(pred, target)

        self.val_metrics.update(pred, target)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Store a few samples for visualisation
        if batch_idx == 0 and self.log_images:
            n = min(self.log_images_n, image.shape[0])
            self._val_vis_preds = [pred[:n].cpu()]
            self._val_vis_targets = [target[:n].cpu()]

    def on_validation_epoch_end(self) -> None:
        """Log validation metrics and (optionally) prediction visualisations."""
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, on_epoch=True, prog_bar=True)
        self.val_metrics.reset()

        # Log prediction grid to W&B
        if self.log_images and self._val_vis_preds:
            self._log_prediction_images()
            self._val_vis_preds = []
            self._val_vis_targets = []

    def _log_prediction_images(self) -> None:
        """Log a side-by-side prediction vs. target grid to W&B."""
        try:
            import wandb
            import matplotlib.pyplot as plt
            import matplotlib

            matplotlib.use("Agg")

            preds = torch.cat(self._val_vis_preds, dim=0)  # (N, 1, H, W)
            targets = torch.cat(self._val_vis_targets, dim=0)

            n = preds.shape[0]
            fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
            if n == 1:
                axes = axes[:, None]

            for i in range(n):
                axes[0, i].imshow(
                    targets[i, 0].numpy(), cmap="YlOrBr", vmin=0, vmax=400
                )
                axes[0, i].set_title(f"Target #{i}")
                axes[0, i].axis("off")

                axes[1, i].imshow(preds[i, 0].numpy(), cmap="YlOrBr", vmin=0, vmax=400)
                axes[1, i].set_title(f"Pred #{i}")
                axes[1, i].axis("off")

            plt.tight_layout()
            if self.logger is not None:
                self.logger.experiment.log(  # type: ignore[attr-defined]
                    {"val/prediction_grid": wandb.Image(fig)},
                    step=self.global_step,
                )
            plt.close(fig)
        except Exception as exc:
            logger.debug("Could not log prediction images to W&B: %s", exc)

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Single test step.

        Args:
            batch:     Dict with ``'image'``, ``'target'``, ``'mask'``.
            batch_idx: Index of the current batch.
        """
        image = batch["image"]
        target = batch["target"]

        with torch.no_grad():
            pred = self(image)
            loss = self.loss_fn(pred, target)

        self.test_metrics.update(pred, target)
        self.log("test/loss", loss, on_step=False, on_epoch=True)

    def on_test_epoch_end(self) -> None:
        """Log test metrics."""
        metrics = self.test_metrics.compute()
        self.log_dict(metrics, on_epoch=True)
        self.test_metrics.reset()

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> Any:
        """Configure optimiser and learning rate scheduler.

        Returns:
            Dict compatible with Lightning's ``configure_optimizers`` API.
        """
        cfg = self.optimizer_cfg
        opt_name = cfg.get("name", "adamw").lower()

        optimizer: torch.optim.Optimizer
        if opt_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=float(cfg.get("lr", 1e-4)),
                weight_decay=float(cfg.get("weight_decay", 1e-4)),
                betas=tuple(cfg.get("betas", [0.9, 0.999])),
            )
        elif opt_name == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=float(cfg.get("lr", 1e-4)),
                weight_decay=float(cfg.get("weight_decay", 0.0)),
            )
        elif opt_name == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=float(cfg.get("lr", 1e-3)),
                momentum=float(cfg.get("momentum", 0.9)),
                weight_decay=float(cfg.get("weight_decay", 1e-4)),
            )
        else:
            raise ValueError(f"Unknown optimiser: {opt_name}")

        # Scheduler
        sch_cfg = self.scheduler_cfg
        sch_name = sch_cfg.get("name", "cosine").lower()

        scheduler: torch.optim.lr_scheduler.LRScheduler
        if sch_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(sch_cfg.get("T_max", 50)),
                eta_min=float(sch_cfg.get("eta_min", 1e-6)),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "monitor": "val/rmse",
                },
            }
        elif sch_name == "onecycle":
            total_steps = int(
                self.trainer.estimated_stepping_batches
                if self.trainer is not None
                else 1000
            )
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=float(cfg.get("lr", 1e-4)),
                total_steps=total_steps,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }
        elif sch_name == "none":
            return {"optimizer": optimizer}
        else:
            raise ValueError(f"Unknown scheduler: {sch_name}")
