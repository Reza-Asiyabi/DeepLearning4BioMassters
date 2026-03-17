#!/usr/bin/env python
"""Run inference on a dataset split and save predictions as GeoTIFFs.

Usage::

    python scripts/predict.py --checkpoint results/utae/checkpoints/best.ckpt \\
                              --config configs/utae.yaml \\
                              --output-dir predictions/utae/
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate AGB predictions and save as GeoTIFFs."
    )
    parser.add_argument("--checkpoint", "-k", type=str, required=True)
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="predictions",
        help="Directory to save prediction TIF files.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run inference on: 'cpu', 'cuda', or 'auto'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from biomassters.utils.config import load_config, merge_configs, config_to_dict
    from biomassters.models.registry import build_model
    from biomassters.data.dataset import BioMasstersDataset
    from biomassters.data.transforms import get_val_transforms
    from biomassters.training.lit_module import BioMasstersLitModule
    from biomassters.utils.io import save_prediction
    from torch.utils.data import DataLoader

    # ── Config ───────────────────────────────────────────────────────────
    base_cfg = load_config(Path(__file__).parent.parent / "configs" / "base.yaml")
    model_cfg = load_config(args.config)
    cfg = merge_configs(base_cfg, model_cfg)
    model_name = cfg.model.name

    # ── Device ───────────────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Running inference on %s", device)

    # ── Model ────────────────────────────────────────────────────────────
    model = build_model(model_name, cfg)
    lit = BioMasstersLitModule.load_from_checkpoint(
        args.checkpoint,
        model=model,
        loss_name=cfg.loss.name,
        optimizer_cfg=config_to_dict(cfg.optimizer),
        scheduler_cfg=config_to_dict(cfg.scheduler),
        log_images=False,
    )
    lit.eval()
    lit.to(device)
    logger.info("Loaded checkpoint: %s", args.checkpoint)

    # ── Dataset ──────────────────────────────────────────────────────────
    data_cfg = cfg.data
    modalities = list(data_cfg.modalities)
    months = list(data_cfg.months) if data_cfg.months else None

    dataset = BioMasstersDataset(
        root_dir=data_cfg.root_dir,
        split=args.split,
        modalities=modalities,
        months=months,
        transform=get_val_transforms(modalities=modalities),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )

    output_dir = Path(args.output_dir) / model_name / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving predictions → %s", output_dir)

    # ── Inference loop ───────────────────────────────────────────────────
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference", unit="batch"):
            images = batch["image"].to(device)  # (B, T, C, H, W)
            chip_ids = batch["chip_id"]  # list of str

            preds = lit(images)  # (B, 1, H, W)
            preds = preds.clamp(min=0.0)  # AGB cannot be negative
            preds_np = preds.cpu().numpy()  # (B, 1, H, W)

            for i, chip_id in enumerate(chip_ids):
                save_prediction(
                    pred=preds_np[i, 0],  # (H, W)
                    chip_id=chip_id,
                    output_dir=output_dir,
                )

    logger.info("Done — %d predictions saved to %s", len(dataset), output_dir)


if __name__ == "__main__":
    main()
