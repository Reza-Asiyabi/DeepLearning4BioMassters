#!/usr/bin/env python
"""Evaluate a trained model checkpoint on the test/validation set.

Usage::

    python scripts/evaluate.py --checkpoint results/utae/checkpoints/best.ckpt \\
                               --config configs/utae.yaml

    # Evaluate on validation split
    python scripts/evaluate.py --checkpoint results/unet/checkpoints/best.ckpt \\
                               --config configs/unet.yaml --split val
"""

import argparse
import json
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
        description="Evaluate a BioMassters checkpoint on the test set."
    )
    parser.add_argument(
        "--checkpoint",
        "-k",
        type=str,
        required=True,
        help="Path to the Lightning checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the model config YAML.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Which split to evaluate on. (default: test)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write metrics.json. Defaults to results/<model_name>/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from biomassters.utils.config import load_config, merge_configs, config_to_dict
    from biomassters.models.registry import build_model
    from biomassters.data.datamodule import BioMasstersDataModule
    from biomassters.training.lit_module import BioMasstersLitModule

    import pytorch_lightning as pl

    # ── Load config ──────────────────────────────────────────────────────
    base_cfg = load_config(Path(__file__).parent.parent / "configs" / "base.yaml")
    model_cfg = load_config(args.config)
    cfg = merge_configs(base_cfg, model_cfg)

    model_name = cfg.model.name

    # ── Build model and load checkpoint ──────────────────────────────────
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
    logger.info("Loaded checkpoint: %s", args.checkpoint)

    # ── DataModule ───────────────────────────────────────────────────────
    data_cfg = cfg.data
    dm = BioMasstersDataModule(
        root_dir=data_cfg.root_dir,
        modalities=list(data_cfg.modalities),
        months=list(data_cfg.months) if data_cfg.months else None,
        batch_size=int(data_cfg.batch_size),
        num_workers=int(data_cfg.num_workers),
        val_split=float(data_cfg.val_split),
    )

    # ── Trainer (no GPU requirement for eval) ────────────────────────────
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        enable_progress_bar=True,
    )

    # ── Evaluate ─────────────────────────────────────────────────────────
    if args.split == "test":
        results = trainer.test(lit, datamodule=dm)[0]
    else:
        results = trainer.validate(lit, datamodule=dm)[0]

    # ── Print table ──────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"  Evaluation results — {model_name.upper()} ({args.split})")
    print("=" * 50)
    for key, val in sorted(results.items()):
        print(f"  {key:<25} {val:.4f}")
    print("=" * 50 + "\n")

    # ── Save to JSON ─────────────────────────────────────────────────────
    out_dir = Path(args.output_dir) if args.output_dir else Path("results") / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"metrics_{args.split}.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "model": model_name,
                "split": args.split,
                "checkpoint": args.checkpoint,
                **{k: float(v) for k, v in results.items()},
            },
            f,
            indent=2,
        )
    logger.info("Metrics saved → %s", metrics_path)


if __name__ == "__main__":
    main()
