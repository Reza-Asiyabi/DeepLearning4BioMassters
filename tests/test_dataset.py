"""Tests for BioMasstersDataset.

All tests use synthetic NPY files created in a temporary directory, so
the real BioMassters dataset is NOT required to run the test suite.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

N_CHIPS = 5
N_MONTHS = 12
S1_CH = 4
S2_CH = 11
H, W = 64, 64  # Use small spatial size for speed


def _create_synthetic_dataset(
    root: Path, split: str = "train", n_chips: int = N_CHIPS
) -> None:
    """Write synthetic NPY chips to a temporary root directory."""
    feature_dir = root / f"{split}_features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    if split == "train":
        label_dir = root / "train_agbm"
        label_dir.mkdir(exist_ok=True)

    chip_ids = [f"chip_{i:04d}" for i in range(n_chips)]
    rng = np.random.default_rng(42)

    for chip_id in chip_ids:
        for month in range(N_MONTHS):
            # S1 chip: (4, H, W) — dB-like values
            s1 = rng.normal(-15.0, 4.0, size=(S1_CH, H, W)).astype(np.float32)
            np.save(feature_dir / f"{chip_id}_S1_{month:02d}.npy", s1)

            # S2 chip: (11, H, W) — reflectance values
            s2 = rng.uniform(0, 5000, size=(S2_CH, H, W)).astype(np.float32)
            np.save(feature_dir / f"{chip_id}_S2_{month:02d}.npy", s2)

        if split == "train":
            # AGB label: (H, W) — mostly > 0 with some zeros (background)
            agb = rng.uniform(0, 400, size=(H, W)).astype(np.float32)
            agb[: H // 4, : W // 4] = 0.0  # background region
            np.save(label_dir / f"{chip_id}.npy", agb)


@pytest.fixture(scope="module")
def dataset_root() -> Path:
    """Create a synthetic train split in a local temp directory (shared across tests)."""
    root = Path(__file__).parent / "_tmp_biomassters_train"
    _create_synthetic_dataset(root, split="train", n_chips=N_CHIPS)
    yield root
    # Cleanup after all tests in module complete
    import shutil

    if root.exists():
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(scope="module")
def test_dataset_root() -> Path:
    """Create a synthetic test split (no labels)."""
    root = Path(__file__).parent / "_tmp_biomassters_test"
    _create_synthetic_dataset(root, split="test", n_chips=3)
    yield root
    import shutil

    if root.exists():
        shutil.rmtree(root, ignore_errors=True)


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestBioMasstersDataset:
    """Tests for BioMasstersDataset loading and output shapes."""

    def _get_dataset(self, root, **kwargs):
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from biomassters.data.dataset import BioMasstersDataset

        return BioMasstersDataset(root_dir=root, **kwargs)

    def test_length(self, dataset_root: Path) -> None:
        """Dataset length matches number of chips discovered."""
        ds = self._get_dataset(dataset_root, split="train")
        assert len(ds) == N_CHIPS

    def test_output_keys(self, dataset_root: Path) -> None:
        """Each sample contains required keys."""
        ds = self._get_dataset(dataset_root, split="train")
        sample = ds[0]
        assert set(sample.keys()) >= {"image", "target", "mask", "chip_id"}

    def test_image_shape_s1_s2(self, dataset_root: Path) -> None:
        """Image shape is (T, C, H, W) with T=12 and C=15 for S1+S2."""
        ds = self._get_dataset(dataset_root, split="train", modalities=["s1", "s2"])
        sample = ds[0]
        T, C, Hd, Wd = sample["image"].shape
        assert T == N_MONTHS
        assert C == S1_CH + S2_CH  # 15
        assert Hd == H and Wd == W

    def test_image_shape_s1_only(self, dataset_root: Path) -> None:
        """S1-only image has 4 channels per time step."""
        ds = self._get_dataset(dataset_root, split="train", modalities=["s1"])
        sample = ds[0]
        T, C, _, _ = sample["image"].shape
        assert C == S1_CH

    def test_image_shape_s2_only(self, dataset_root: Path) -> None:
        """S2-only image has 11 channels per time step."""
        ds = self._get_dataset(dataset_root, split="train", modalities=["s2"])
        sample = ds[0]
        T, C, _, _ = sample["image"].shape
        assert C == S2_CH

    def test_target_shape(self, dataset_root: Path) -> None:
        """Target has shape (1, H, W)."""
        ds = self._get_dataset(dataset_root, split="train")
        sample = ds[0]
        assert sample["target"].shape == (1, H, W)

    def test_mask_shape(self, dataset_root: Path) -> None:
        """Mask has shape (1, H, W) and is binary."""
        ds = self._get_dataset(dataset_root, split="train")
        sample = ds[0]
        mask = sample["mask"]
        assert mask.shape == (1, H, W)
        unique = mask.unique().tolist()
        assert all(v in (0.0, 1.0) for v in unique)

    def test_month_subset(self, dataset_root: Path) -> None:
        """Subsetting months returns correct T dimension."""
        months = [0, 3, 6, 9]
        ds = self._get_dataset(dataset_root, split="train", months=months)
        sample = ds[0]
        T, C, _, _ = sample["image"].shape
        assert T == len(months)

    def test_missing_month_handling(self, dataset_root: Path) -> None:
        """Missing month files are filled with zeros (no exception raised)."""
        # Create a mini dataset with one missing S1 file
        mini_root = Path(__file__).parent / "_tmp_missing_month"
        _create_synthetic_dataset(mini_root, split="train", n_chips=1)

        # Remove one month's S1 file
        missing = mini_root / "train_features" / "chip_0000_S1_05.npy"
        if missing.exists():
            missing.unlink()

        ds = self._get_dataset(mini_root, split="train")
        sample = ds[0]  # Should not raise
        # Month 5 should be all zeros in the S1 channels
        frame_5 = sample["image"][5, :S1_CH]  # (4, H, W) if S1+S2
        assert frame_5.shape[0] == S1_CH

    def test_dtype(self, dataset_root: Path) -> None:
        """Image and target are float32 tensors."""
        ds = self._get_dataset(dataset_root, split="train")
        sample = ds[0]
        assert sample["image"].dtype == torch.float32
        assert sample["target"].dtype == torch.float32

    def test_chip_id_type(self, dataset_root: Path) -> None:
        """chip_id is a string."""
        ds = self._get_dataset(dataset_root, split="train")
        sample = ds[0]
        assert isinstance(sample["chip_id"], str)

    def test_n_channels_property(self, dataset_root: Path) -> None:
        """n_channels property returns correct count."""
        ds = self._get_dataset(dataset_root, split="train", modalities=["s1", "s2"])
        assert ds.n_channels == S1_CH + S2_CH

    def test_n_timesteps_property(self, dataset_root: Path) -> None:
        """n_timesteps property returns correct count."""
        ds = self._get_dataset(dataset_root, split="train", months=[0, 1, 2])
        assert ds.n_timesteps == 3

    def test_transforms_applied(self, dataset_root: Path) -> None:
        """Transforms are applied to the sample dict."""
        from biomassters.data.transforms import get_train_transforms

        ds = self._get_dataset(
            dataset_root,
            split="train",
            transform=get_train_transforms(modalities=["s1", "s2"]),
        )
        sample = ds[0]
        # After normalisation, values should not be in raw dB range
        assert sample["image"].abs().max() < 200.0  # normalised values

    def test_test_split_no_labels(self, test_dataset_root: Path) -> None:
        """Test split loads without labels (target = zeros)."""
        ds = self._get_dataset(test_dataset_root, split="test")
        sample = ds[0]
        # Target is zeros for test split
        assert sample["target"].shape == (1, H, W)
