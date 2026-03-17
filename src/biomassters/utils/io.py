"""I/O utilities: dataset download, chip loading, and prediction saving."""

import logging
from pathlib import Path
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)


def download_biomassters(
    output_dir: Union[str, Path],
    split: str = "train",
    repo_id: str = "nascetti-a/BioMassters",
) -> Path:
    """Download BioMassters dataset from HuggingFace Hub.

    Downloads all files for the specified split into ``output_dir``.
    Already-downloaded files are skipped automatically.

    Args:
        output_dir: Local directory to save the dataset.
        split:      ``'train'`` or ``'test'``.
        repo_id:    HuggingFace dataset repository ID.

    Returns:
        Path to the downloaded split directory.

    Raises:
        ImportError: If ``huggingface_hub`` is not installed.
        ValueError:  If ``split`` is not ``'train'`` or ``'test'``.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required for downloading. "
            "Install with: pip install huggingface_hub"
        ) from e

    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got '{split}'")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading BioMassters '%s' split from HuggingFace…", split)

    # Download only the relevant folders
    patterns = (
        [f"{split}_features/*", f"{split}_agbm/*"]
        if split == "train"
        else [f"{split}_features/*"]
    )

    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(output_dir),
        allow_patterns=patterns,
        ignore_patterns=["*.md", "*.json"],
    )

    logger.info("Download complete → %s", local_dir)
    return Path(local_dir)


def load_chip(chip_path: Union[str, Path]) -> np.ndarray:
    """Load a single NPY chip file.

    Args:
        chip_path: Path to a ``.npy`` satellite chip file.

    Returns:
        Float32 numpy array. Shape depends on the chip type:
            - Sentinel-1: ``(4, 256, 256)``
            - Sentinel-2: ``(11, 256, 256)``
            - AGB target: ``(256, 256)``

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    chip_path = Path(chip_path)
    if not chip_path.exists():
        raise FileNotFoundError(f"Chip file not found: {chip_path}")
    return np.load(chip_path).astype(np.float32)


def save_prediction(
    pred: np.ndarray,
    chip_id: str,
    output_dir: Union[str, Path],
    crs: str = "EPSG:4326",
    transform=None,
) -> Path:
    """Save a model prediction as a GeoTIFF.

    Args:
        pred:       Predicted AGB map of shape ``(H, W)`` or ``(1, H, W)`` (float32).
        chip_id:    Unique chip identifier (used as filename).
        output_dir: Directory to write the output TIF.
        crs:        Coordinate reference system string (default EPSG:4326).
        transform:  Optional rasterio Affine transform. If None, uses a
                    placeholder identity transform.

    Returns:
        Path to the saved GeoTIFF file.
    """
    try:
        import rasterio
        from rasterio.transform import from_bounds

        _RASTERIO_AVAILABLE = True
    except ImportError:
        _RASTERIO_AVAILABLE = False

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{chip_id}.tif"

    if pred.ndim == 3:
        pred = pred[0]  # (H, W)

    if _RASTERIO_AVAILABLE:
        import rasterio
        from rasterio.crs import CRS

        if transform is None:
            from rasterio.transform import from_bounds

            transform = from_bounds(0, 0, 1, 1, pred.shape[1], pred.shape[0])

        with rasterio.open(
            out_path,
            mode="w",
            driver="GTiff",
            height=pred.shape[0],
            width=pred.shape[1],
            count=1,
            dtype="float32",
            crs=CRS.from_string(crs),
            transform=transform,
        ) as dst:
            dst.write(pred.astype("float32"), 1)

        logger.info("Saved prediction GeoTIFF → %s", out_path)
    else:
        # Fallback: save as NPY if rasterio unavailable
        npy_path = output_dir / f"{chip_id}.npy"
        np.save(npy_path, pred)
        logger.warning("rasterio not available — saved as NPY instead: %s", npy_path)
        return npy_path

    return out_path
