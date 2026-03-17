"""Loss functions for pixel-wise AGB regression.

All losses:
    - Accept ``(pred, target)`` tensors of shape ``(B, 1, H, W)``.
    - Return a scalar tensor (mean over valid pixels).
    - Are differentiable end-to-end.

Default training loss: ``MaskedLoss(RMSELoss())``.
"""

import logging
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class RMSELoss(nn.Module):
    """Root Mean Squared Error loss.

    Differentiable RMSE computed as ``sqrt(MSE + eps)`` to avoid zero gradients
    at convergence.  Matches the BioMassters competition evaluation metric.

    Args:
        eps: Small constant for numerical stability under the square root.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute RMSE.

        Args:
            pred:   Predicted AGB  ``(B, 1, H, W)``.
            target: Ground-truth AGB ``(B, 1, H, W)``.

        Returns:
            Scalar RMSE.
        """
        mse = F.mse_loss(pred, target, reduction="mean")
        return torch.sqrt(mse + self.eps)


class HuberLoss(nn.Module):
    """Huber (smooth L1) loss, robust to AGB outliers.

    Uses ``delta`` as the threshold between L2 and L1 behaviour.
    Downweights very large errors (e.g., forest patches with extreme AGB).

    Args:
        delta: Huber threshold (Mg/ha units).
    """

    def __init__(self, delta: float = 50.0) -> None:
        super().__init__()
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Huber loss.

        Args:
            pred:   ``(B, 1, H, W)``
            target: ``(B, 1, H, W)``

        Returns:
            Scalar Huber loss.
        """
        return F.huber_loss(pred, target, delta=self.delta, reduction="mean")


class LogCoshLoss(nn.Module):
    """Log-Cosh loss — smooth approximation to MAE.

    Defined as ``mean(log(cosh(pred - target)))``.  Behaves like L2 for small
    errors and L1 for large errors, with smooth gradients everywhere.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Log-Cosh loss.

        Args:
            pred:   ``(B, 1, H, W)``
            target: ``(B, 1, H, W)``

        Returns:
            Scalar Log-Cosh loss.
        """
        diff = pred - target
        # Numerically stable: log(cosh(x)) = x + log(1 + exp(-2x)) - log(2)
        return torch.mean(diff + F.softplus(-2.0 * diff) - math.log(2.0))


class MaskedLoss(nn.Module):
    """Wrapper that masks invalid (zero-biomass background) pixels.

    Background pixels in the AGB map are zero by convention.  Including them
    in the loss would bias the network toward predicting low values.  This
    wrapper excludes pixels where ``|target| < ignore_threshold``.

    Args:
        base_loss: Underlying loss module (e.g., ``RMSELoss()``).
        ignore_threshold: Pixel values below this are masked out.
    """

    def __init__(self, base_loss: nn.Module, ignore_threshold: float = 0.5) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.ignore_threshold = ignore_threshold

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss on valid pixels only.

        Args:
            pred:   ``(B, 1, H, W)``
            target: ``(B, 1, H, W)``

        Returns:
            Scalar masked loss.  Returns 0 if no valid pixels exist.
        """
        mask = target.abs() >= self.ignore_threshold  # (B, 1, H, W), bool

        if mask.sum() == 0:
            logger.warning("MaskedLoss: no valid pixels in batch — returning 0.")
            return pred.sum() * 0.0  # differentiable zero

        pred_masked = pred[mask]
        target_masked = target[mask]

        # Reshape to (N, 1) for loss compatibility
        pred_masked = pred_masked.unsqueeze(-1)
        target_masked = target_masked.unsqueeze(-1)

        return self.base_loss(pred_masked, target_masked)


class CombinedLoss(nn.Module):
    """Weighted sum of multiple loss functions.

    Args:
        losses:  List of loss modules.
        weights: List of scalar weights (must match length of ``losses``).

    Example::

        loss = CombinedLoss(
            [MaskedLoss(RMSELoss()), MaskedLoss(HuberLoss())],
            weights=[0.7, 0.3],
        )
    """

    def __init__(
        self,
        losses: List[nn.Module],
        weights: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        assert len(losses) > 0, "CombinedLoss requires at least one loss."
        self.losses = nn.ModuleList(losses)
        self.weights = weights if weights is not None else [1.0] * len(losses)
        assert len(self.weights) == len(
            losses
        ), "Weights and losses must have the same length."

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute weighted sum of losses.

        Args:
            pred:   ``(B, 1, H, W)``
            target: ``(B, 1, H, W)``

        Returns:
            Scalar combined loss.
        """
        total = torch.tensor(0.0, device=pred.device, requires_grad=True)
        for loss_fn, w in zip(self.losses, self.weights):
            total = total + w * loss_fn(pred, target)
        return total


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

_LOSS_REGISTRY = {
    "rmse": lambda: RMSELoss(),
    "masked_rmse": lambda: MaskedLoss(RMSELoss()),
    "huber": lambda: HuberLoss(),
    "masked_huber": lambda: MaskedLoss(HuberLoss()),
    "logcosh": lambda: LogCoshLoss(),
    "combined": lambda: CombinedLoss(
        [MaskedLoss(RMSELoss()), MaskedLoss(HuberLoss())], weights=[0.7, 0.3]
    ),
}


def build_loss(name: str) -> nn.Module:
    """Build a loss function by name.

    Args:
        name: Loss identifier (see ``_LOSS_REGISTRY``).

    Returns:
        Instantiated loss module.

    Raises:
        ValueError: If ``name`` is not registered.
    """
    if name not in _LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss '{name}'. Available: {list(_LOSS_REGISTRY.keys())}"
        )
    return _LOSS_REGISTRY[name]()
