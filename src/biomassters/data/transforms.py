"""Transforms and augmentations for BioMassters satellite data.

All transforms operate on a sample dict with keys:
    ``'image'``  — ``(T, C, H, W)`` float32 tensor
    ``'target'`` — ``(1, H, W)`` float32 tensor
    ``'mask'``   — ``(1, H, W)`` float32 tensor
    ``'chip_id'``— str
"""

from typing import Callable, Dict, List, Optional

import torch

# ---------------------------------------------------------------------------
# Normalization constants
# ---------------------------------------------------------------------------

# S1 SAR backscatter is stored in dB (typical range −30 to +5 dB).
# Dividing by 25 maps this roughly to [−1.2, +0.2].
_S1_SCALE = 25.0

# S2 surface reflectance is stored as integer × 10 000
# (i.e. real reflectance 0–1 → stored 0–10 000).
# Dividing by 10 000 maps to [0, 1].
_S2_SCALE = 10_000.0

S1_CHANNELS = 4
S2_CHANNELS = 11


def _build_scale_tensor(modalities: List[str]) -> torch.Tensor:
    """Return a (C,) tensor of per-channel scale factors."""
    scales: List[float] = []
    if "s1" in modalities:
        scales.extend([_S1_SCALE] * S1_CHANNELS)
    if "s2" in modalities:
        scales.extend([_S2_SCALE] * S2_CHANNELS)
    return torch.tensor(scales, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Transform classes
# ---------------------------------------------------------------------------


class Normalize:
    """Divide image channels by per-modality scale factors.

    Args:
        modalities: Ordered list of modalities present in the image tensor.
                    Must match the order used when the dataset was created.
    """

    def __init__(self, modalities: List[str]) -> None:
        scale = _build_scale_tensor(modalities)
        # Reshape to (1, C, 1, 1) for broadcasting with (T, C, H, W)
        self._scale = scale.view(1, -1, 1, 1)

    def __call__(self, sample: Dict) -> Dict:
        sample["image"] = sample["image"] / self._scale
        return sample


class RandomHorizontalFlip:
    """Randomly flip image, target, and mask horizontally."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, sample: Dict) -> Dict:
        if torch.rand(1).item() < self.p:
            sample["image"] = sample["image"].flip(-1)
            sample["target"] = sample["target"].flip(-1)
            sample["mask"] = sample["mask"].flip(-1)
        return sample


class RandomVerticalFlip:
    """Randomly flip image, target, and mask vertically."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, sample: Dict) -> Dict:
        if torch.rand(1).item() < self.p:
            sample["image"] = sample["image"].flip(-2)
            sample["target"] = sample["target"].flip(-2)
            sample["mask"] = sample["mask"].flip(-2)
        return sample


class Compose:
    """Apply a sequence of transforms in order."""

    def __init__(self, transforms: List[Callable]) -> None:
        self.transforms = transforms

    def __call__(self, sample: Dict) -> Dict:
        for t in self.transforms:
            sample = t(sample)
        return sample


# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------


def get_train_transforms(
    modalities: Optional[List[str]] = None,
) -> Compose:
    """Return the transform pipeline for training.

    Applies per-modality scale normalisation followed by random
    horizontal and vertical flips.

    Args:
        modalities: Modalities present in the image (e.g. ``['s1', 's2']``).
    """
    mods = [m.lower() for m in (modalities or ["s1", "s2"])]
    return Compose(
        [
            Normalize(mods),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
        ]
    )


def get_val_transforms(
    modalities: Optional[List[str]] = None,
) -> Compose:
    """Return the transform pipeline for validation and inference.

    Applies per-modality scale normalisation only (no augmentation).

    Args:
        modalities: Modalities present in the image (e.g. ``['s1', 's2']``).
    """
    mods = [m.lower() for m in (modalities or ["s1", "s2"])]
    return Compose([Normalize(mods)])
