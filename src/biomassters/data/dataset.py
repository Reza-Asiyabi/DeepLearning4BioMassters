"""BioMasstersDataset — loads multi-modal satellite time series patches.

Directory structure expected::

    root_dir/
      train_features/
        {chip_id}_S1_{month:02d}.npy   # shape (4, H, W)  — SAR backscatter dB
        {chip_id}_S2_{month:02d}.npy   # shape (11, H, W) — surface reflectance
      train_agbm/
        {chip_id}.npy                  # shape (H, W)      — AGB in Mg/ha
      test_features/
        {chip_id}_S1_{month:02d}.npy
        {chip_id}_S2_{month:02d}.npy
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

S1_CHANNELS = 4   # ASC VV, ASC VH, DSC VV, DSC VH
S2_CHANNELS = 11  # B2–B8A, B11, B12, CLP
N_MONTHS = 12


class BioMasstersDataset(Dataset):
    """PyTorch Dataset for BioMassters AGB estimation.

    Loads Sentinel-1 and/or Sentinel-2 monthly time series chips from NPY
    files and returns multi-temporal, multi-modal tensors.

    Args:
        root_dir:   Path to the dataset root directory.
        split:      ``'train'`` or ``'test'``.
        modalities: Modalities to load; any subset of ``['s1', 's2']``.
                    Defaults to both.
        months:     Month indices (0–11) to include. ``None`` = all 12.
        transform:  Callable applied to the sample dict after loading.
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        modalities: Optional[List[str]] = None,
        months: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.modalities = [m.lower() for m in (modalities or ["s1", "s2"])]
        self.months = months if months is not None else list(range(N_MONTHS))
        self.transform = transform

        self._feature_dir = self.root_dir / f"{split}_features"
        self._label_dir = self.root_dir / "train_agbm"
        self._spatial_size: Optional[Tuple[int, int]] = None

        self._chip_ids: List[str] = self._discover_chips()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _discover_chips(self) -> List[str]:
        """Scan the feature directory and return sorted list of chip IDs."""
        if not self._feature_dir.exists():
            raise FileNotFoundError(
                f"Feature directory not found: {self._feature_dir}"
            )

        # Use whichever modality is requested to find chips
        modality_key = "S1" if "s1" in self.modalities else "S2"
        chips: set = set()
        for f in self._feature_dir.glob(f"*_{modality_key}_*.npy"):
            # Filename pattern: {chip_id}_{MODALITY}_{month:02d}.npy
            # chip_id may contain underscores, so split from the right.
            parts = f.stem.rsplit("_", 2)
            if len(parts) == 3:
                chips.add(parts[0])
        return sorted(chips)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._chip_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        chip_id = self._chip_ids[idx]
        image = self._load_image(chip_id)          # (T, C, H, W)
        target, mask = self._load_target(chip_id)  # (1, H, W), (1, H, W)

        sample: Dict[str, Any] = {
            "image": image,
            "target": target,
            "mask": mask,
            "chip_id": chip_id,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    # ------------------------------------------------------------------
    # Internal loaders
    # ------------------------------------------------------------------

    def _get_spatial_size(self) -> Tuple[int, int]:
        """Lazily infer (H, W) from the first available chip file."""
        if self._spatial_size is None:
            for f in self._feature_dir.glob("*.npy"):
                arr = np.load(f, mmap_mode="r")
                if arr.ndim == 3:
                    self._spatial_size = (arr.shape[1], arr.shape[2])
                    break
            else:
                self._spatial_size = (256, 256)  # safe default
        return self._spatial_size

    def _load_npy(self, path: Path, n_channels: int) -> np.ndarray:
        """Load a chip NPY file, returning zeros if the file is missing."""
        if path.exists():
            return np.load(path).astype(np.float32)
        H, W = self._get_spatial_size()
        return np.zeros((n_channels, H, W), dtype=np.float32)

    def _load_image(self, chip_id: str) -> torch.Tensor:
        """Assemble the (T, C, H, W) multi-temporal image tensor."""
        frames = []
        for month in self.months:
            chunks = []
            if "s1" in self.modalities:
                chunks.append(
                    self._load_npy(
                        self._feature_dir / f"{chip_id}_S1_{month:02d}.npy",
                        S1_CHANNELS,
                    )
                )
            if "s2" in self.modalities:
                chunks.append(
                    self._load_npy(
                        self._feature_dir / f"{chip_id}_S2_{month:02d}.npy",
                        S2_CHANNELS,
                    )
                )
            frames.append(np.concatenate(chunks, axis=0))  # (C, H, W)

        image = np.stack(frames, axis=0)  # (T, C, H, W)
        return torch.from_numpy(image)

    def _load_target(
        self, chip_id: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load the AGB target and derive the valid-pixel mask.

        Returns:
            target: ``(1, H, W)`` float32 tensor in Mg/ha.
                    Zero-filled for test split.
            mask:   ``(1, H, W)`` binary float32 (1 = valid forest pixel).
                    Background is defined as ``|AGB| < 0.5``.
        """
        if self.split == "train":
            label_path = self._label_dir / f"{chip_id}.npy"
            if label_path.exists():
                agb = np.load(label_path).astype(np.float32)
            else:
                H, W = self._get_spatial_size()
                agb = np.zeros((H, W), dtype=np.float32)
        else:
            H, W = self._get_spatial_size()
            agb = np.zeros((H, W), dtype=np.float32)

        target = torch.from_numpy(agb).unsqueeze(0)  # (1, H, W)
        mask = (target.abs() >= 0.5).float()          # (1, H, W)
        return target, mask

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_channels(self) -> int:
        """Total feature channels per time step."""
        count = 0
        if "s1" in self.modalities:
            count += S1_CHANNELS
        if "s2" in self.modalities:
            count += S2_CHANNELS
        return count

    @property
    def n_timesteps(self) -> int:
        """Number of time steps (months) per sample."""
        return len(self.months)
