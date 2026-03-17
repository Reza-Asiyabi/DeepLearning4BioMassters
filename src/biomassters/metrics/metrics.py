"""Evaluation metrics for pixel-wise AGB regression.

All metrics:
    - Extend ``torchmetrics.Metric`` for stateful accumulation across batches.
    - Mask background pixels (target ≈ 0) before computing statistics.
    - Can be used standalone or via ``MetricsCollection``.

Primary competition metric: RMSE (lower is better).
"""

import logging
from typing import Dict, Optional

import torch
from torchmetrics import Metric

logger = logging.getLogger(__name__)

_IGNORE_THRESHOLD = 0.5  # pixels with |target| < threshold are masked


def _apply_mask(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = _IGNORE_THRESHOLD,
) -> tuple:
    """Flatten and mask background pixels.

    Args:
        pred:      ``(B, 1, H, W)`` or ``(N,)`` predictions.
        target:    ``(B, 1, H, W)`` or ``(N,)`` ground truth.
        threshold: Pixels with ``|target| < threshold`` are excluded.

    Returns:
        Tuple of ``(pred_valid, target_valid)`` 1-D tensors.
    """
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    mask = target_flat.abs() >= threshold
    return pred_flat[mask], target_flat[mask]


# ──────────────────────────────────────────────────────────────────────────────
# Individual metrics
# ──────────────────────────────────────────────────────────────────────────────


class MaskedRMSE(Metric):
    """Root Mean Squared Error on valid (non-background) pixels.

    This matches the official BioMassters competition evaluation metric.

    Args:
        ignore_threshold: Pixels with ``|target|`` below this are excluded.
    """

    higher_is_better = False
    full_state_update = False

    # State tensors declared here so mypy knows their type
    sum_sq_err: torch.Tensor
    count: torch.Tensor

    def __init__(self, ignore_threshold: float = _IGNORE_THRESHOLD) -> None:
        super().__init__()
        self.ignore_threshold = ignore_threshold
        self.add_state("sum_sq_err", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate squared errors.

        Args:
            pred:   Predictions ``(B, 1, H, W)``.
            target: Ground truth ``(B, 1, H, W)``.
        """
        p, t = _apply_mask(pred, target, self.ignore_threshold)
        if p.numel() == 0:
            return
        self.sum_sq_err += ((p - t) ** 2).sum()
        self.count += p.numel()

    def compute(self) -> torch.Tensor:
        """Compute final RMSE."""
        if self.count == 0:
            return torch.tensor(float("nan"))
        return torch.sqrt(self.sum_sq_err / self.count)


class MaskedMAE(Metric):
    """Mean Absolute Error on valid pixels.

    Args:
        ignore_threshold: Background mask threshold.
    """

    higher_is_better = False
    full_state_update = False

    # State tensors
    sum_abs_err: torch.Tensor
    count: torch.Tensor

    def __init__(self, ignore_threshold: float = _IGNORE_THRESHOLD) -> None:
        super().__init__()
        self.ignore_threshold = ignore_threshold
        self.add_state("sum_abs_err", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        p, t = _apply_mask(pred, target, self.ignore_threshold)
        if p.numel() == 0:
            return
        self.sum_abs_err += (p - t).abs().sum()
        self.count += p.numel()

    def compute(self) -> torch.Tensor:
        if self.count == 0:
            return torch.tensor(float("nan"))
        return self.sum_abs_err / self.count


class MaskedR2(Metric):
    """Coefficient of determination (R²) on valid pixels.

    R² = 1 - SS_res / SS_tot.

    Args:
        ignore_threshold: Background mask threshold.
    """

    higher_is_better = True
    full_state_update = False

    # State tensors
    ss_res: torch.Tensor
    sum_t: torch.Tensor
    sum_sq_t: torch.Tensor
    count: torch.Tensor

    def __init__(self, ignore_threshold: float = _IGNORE_THRESHOLD) -> None:
        super().__init__()
        self.ignore_threshold = ignore_threshold
        self.add_state("ss_res", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_t", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_sq_t", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        p, t = _apply_mask(pred, target, self.ignore_threshold)
        if p.numel() == 0:
            return
        self.ss_res += ((p - t) ** 2).sum()
        self.sum_t += t.sum()
        self.sum_sq_t += (t**2).sum()
        self.count += t.numel()

    def compute(self) -> torch.Tensor:
        if self.count == 0:
            return torch.tensor(float("nan"))
        mean_t = self.sum_t / self.count
        ss_tot = self.sum_sq_t - self.count * mean_t**2
        if ss_tot == 0:
            return torch.tensor(float("nan"))
        return 1.0 - self.ss_res / ss_tot


class MaskedBias(Metric):
    """Mean prediction bias = mean(pred) - mean(target) on valid pixels.

    A positive bias means the model systematically overestimates AGB.

    Args:
        ignore_threshold: Background mask threshold.
    """

    higher_is_better: Optional[bool] = None  # closer to 0 is better
    full_state_update = False

    # State tensors
    sum_diff: torch.Tensor
    count: torch.Tensor

    def __init__(self, ignore_threshold: float = _IGNORE_THRESHOLD) -> None:
        super().__init__()
        self.ignore_threshold = ignore_threshold
        self.add_state("sum_diff", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        p, t = _apply_mask(pred, target, self.ignore_threshold)
        if p.numel() == 0:
            return
        self.sum_diff += (p - t).sum()
        self.count += p.numel()

    def compute(self) -> torch.Tensor:
        if self.count == 0:
            return torch.tensor(float("nan"))
        return self.sum_diff / self.count


class MaskedRelRMSE(Metric):
    """Relative RMSE = RMSE / mean(target) × 100 (%) on valid pixels.

    Normalises RMSE by the mean target AGB, providing a scale-independent
    error metric useful for comparing models across datasets.

    Args:
        ignore_threshold: Background mask threshold.
    """

    higher_is_better = False
    full_state_update = False

    # State tensors
    sum_sq_err: torch.Tensor
    sum_t: torch.Tensor
    count: torch.Tensor

    def __init__(self, ignore_threshold: float = _IGNORE_THRESHOLD) -> None:
        super().__init__()
        self.ignore_threshold = ignore_threshold
        self.add_state("sum_sq_err", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_t", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        p, t = _apply_mask(pred, target, self.ignore_threshold)
        if p.numel() == 0:
            return
        self.sum_sq_err += ((p - t) ** 2).sum()
        self.sum_t += t.sum()
        self.count += p.numel()

    def compute(self) -> torch.Tensor:
        if self.count == 0 or self.sum_t == 0:
            return torch.tensor(float("nan"))
        rmse = torch.sqrt(self.sum_sq_err / self.count)
        mean_t = self.sum_t / self.count
        return (rmse / mean_t) * 100.0


# ──────────────────────────────────────────────────────────────────────────────
# Metrics collection
# ──────────────────────────────────────────────────────────────────────────────


class MetricsCollection:
    """Container for all AGB evaluation metrics.

    Creates and manages instances of all metrics.  Call ``update()`` on each
    batch and ``compute()`` at epoch end.

    Args:
        ignore_threshold: Background mask threshold shared across all metrics.
        prefix: Optional string prepended to metric names in the output dict.
    """

    def __init__(
        self,
        ignore_threshold: float = _IGNORE_THRESHOLD,
        prefix: str = "",
    ) -> None:
        self.prefix = prefix
        self.metrics = {
            "rmse": MaskedRMSE(ignore_threshold),
            "mae": MaskedMAE(ignore_threshold),
            "r2": MaskedR2(ignore_threshold),
            "bias": MaskedBias(ignore_threshold),
            "rel_rmse": MaskedRelRMSE(ignore_threshold),
        }

    def to(self, device: torch.device) -> "MetricsCollection":
        """Move all metric states to device."""
        for m in self.metrics.values():
            m.to(device)
        return self

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Update all metrics with a batch.

        Args:
            pred:   ``(B, 1, H, W)``
            target: ``(B, 1, H, W)``
        """
        for m in self.metrics.values():
            m.update(pred, target)

    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute and return all metrics as a dict.

        Returns:
            Dict mapping metric name → scalar tensor.
        """
        results = {}
        for name, m in self.metrics.items():
            key = f"{self.prefix}{name}" if self.prefix else name
            results[key] = m.compute()
        return results

    def reset(self) -> None:
        """Reset all accumulated states."""
        for m in self.metrics.values():
            m.reset()
