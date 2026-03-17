#!/usr/bin/env python
"""Download the BioMassters dataset from HuggingFace Hub.

Usage::

    python scripts/download_data.py --output-dir data/ --split train
    python scripts/download_data.py --output-dir data/ --split test
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download BioMassters dataset from HuggingFace."
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="data/biomassters",
        help="Directory to save the downloaded dataset. (default: data/biomassters)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split to download. (default: train)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="nascetti-a/BioMassters",
        help="HuggingFace dataset repository ID. (default: nascetti-a/BioMassters)",
    )
    return parser.parse_args()


def verify_download(output_dir: Path, split: str) -> bool:
    """Verify that key files exist after download.

    Args:
        output_dir: Root directory of downloaded data.
        split:      Dataset split.

    Returns:
        True if verification passes.
    """
    feature_dir = output_dir / f"{split}_features"
    if not feature_dir.exists():
        logger.error("Feature directory not found: %s", feature_dir)
        return False

    npy_files = list(feature_dir.glob("*.npy"))
    if not npy_files:
        logger.error("No .npy files found in %s", feature_dir)
        return False

    logger.info("Verified %d .npy files in %s", len(npy_files), feature_dir)

    if split == "train":
        label_dir = output_dir / "train_agbm"
        if not label_dir.exists():
            logger.warning("Label directory not found: %s (AGB targets)", label_dir)
        else:
            n_labels = len(list(label_dir.glob("*.npy")))
            logger.info("Verified %d AGB label files in %s", n_labels, label_dir)

    return True


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    logger.info("=" * 60)
    logger.info("BioMassters Data Downloader")
    logger.info("  split      : %s", args.split)
    logger.info("  output_dir : %s", output_dir.resolve())
    logger.info("  repo       : %s", args.repo_id)
    logger.info("=" * 60)

    try:
        from biomassters.utils.io import download_biomassters
    except ImportError:
        # Allow running as a standalone script without package installed
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from biomassters.utils.io import download_biomassters

    try:
        local_dir = download_biomassters(
            output_dir=output_dir,
            split=args.split,
            repo_id=args.repo_id,
        )
        logger.info("Download finished → %s", local_dir)
    except Exception as exc:
        logger.error("Download failed: %s", exc)
        sys.exit(1)

    # Verify
    ok = verify_download(output_dir, args.split)
    if ok:
        logger.info("All checks passed. Ready to train!")
    else:
        logger.error("Verification failed — check download logs.")
        sys.exit(1)


if __name__ == "__main__":
    main()
