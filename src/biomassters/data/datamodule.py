"""PyTorch Lightning DataModule for the BioMassters dataset."""

from typing import List, Optional

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from biomassters.data.dataset import BioMasstersDataset
from biomassters.data.transforms import get_train_transforms, get_val_transforms


class _TransformDataset(Dataset):
    """Thin wrapper that applies a transform to a ``Subset``."""

    def __init__(self, subset: Subset, transform) -> None:
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        sample = self.subset[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class BioMasstersDataModule(pl.LightningDataModule):
    """LightningDataModule for BioMassters AGB estimation.

    Splits the training data into train/val subsets, applies appropriate
    transforms to each, and creates DataLoaders for every stage.

    Args:
        root_dir:    Path to the dataset root (must contain
                     ``train_features/`` and ``train_agbm/``).
        modalities:  Satellite modalities to load; subset of
                     ``['s1', 's2']``. Defaults to both.
        months:      Month indices (0–11) to include. ``None`` = all 12.
        batch_size:  Samples per mini-batch.
        num_workers: Number of DataLoader worker processes.
        val_split:   Fraction of training chips reserved for validation.
        pin_memory:  Whether to pin DataLoader memory to CUDA pages.
        seed:        RNG seed for the train/val split.
    """

    def __init__(
        self,
        root_dir: str,
        modalities: Optional[List[str]] = None,
        months: Optional[List[int]] = None,
        batch_size: int = 8,
        num_workers: int = 4,
        val_split: float = 0.1,
        pin_memory: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.modalities = modalities or ["s1", "s2"]
        self.months = months
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.pin_memory = pin_memory
        self.seed = seed

        self._train_ds: Optional[Dataset] = None
        self._val_ds: Optional[Dataset] = None
        self._test_ds: Optional[Dataset] = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, stage: Optional[str] = None) -> None:
        """Instantiate datasets for the requested stage.

        Args:
            stage: One of ``'fit'``, ``'validate'``, ``'test'``,
                   ``'predict'``, or ``None`` (all stages).
        """
        if stage in ("fit", "validate", None):
            base_ds = BioMasstersDataset(
                root_dir=self.root_dir,
                split="train",
                modalities=self.modalities,
                months=self.months,
            )
            n_total = len(base_ds)
            n_val = max(1, int(n_total * self.val_split))
            n_train = n_total - n_val

            train_subset, val_subset = random_split(
                base_ds,
                [n_train, n_val],
                generator=torch.Generator().manual_seed(self.seed),
            )

            self._train_ds = _TransformDataset(
                train_subset, get_train_transforms(self.modalities)
            )
            self._val_ds = _TransformDataset(
                val_subset, get_val_transforms(self.modalities)
            )

        if stage in ("test", "predict", None):
            self._test_ds = BioMasstersDataset(
                root_dir=self.root_dir,
                split="test",
                modalities=self.modalities,
                months=self.months,
                transform=get_val_transforms(self.modalities),
            )

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
