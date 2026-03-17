#!/usr/bin/env python
"""Main training entry point for BioMassters AGB estimation.

Usage::

    # Train U-TAE with defaults
    python scripts/train.py --config configs/utae.yaml

    # Override config params inline
    python scripts/train.py --config configs/unet.yaml training.max_epochs=100 data.batch_size=16

    # Disable W&B logging (offline run)
    python scripts/train.py --config configs/utae.yaml --no-wandb
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a BioMassters AGB estimation model."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to model config YAML (e.g., configs/utae.yaml).",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Key=value overrides, e.g. training.max_epochs=100",
    )
    return parser.parse_args()


def build_override_cfg(overrides: list):
    """Convert list of 'key=value' strings into OmegaConf DictConfig."""
    try:
        from omegaconf import OmegaConf

        dot_list = [f"{kv}" for kv in overrides]
        return OmegaConf.from_dotlist(dot_list)
    except ImportError:
        return {}


def main() -> None:
    args = parse_args()

    # ── Add src to path if not installed ────────────────────────────────
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from biomassters.utils.config import load_config, merge_configs, config_to_dict
    from biomassters.models.registry import build_model
    from biomassters.data.datamodule import BioMasstersDataModule
    from biomassters.training.lit_module import BioMasstersLitModule

    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import (
        ModelCheckpoint,
        LearningRateMonitor,
        EarlyStopping,
    )

    # ── Load and merge configs ───────────────────────────────────────────
    base_cfg = load_config(Path(__file__).parent.parent / "configs" / "base.yaml")
    model_cfg = load_config(args.config)
    cfg = merge_configs(base_cfg, model_cfg)

    if args.overrides:
        override_cfg = build_override_cfg(args.overrides)
        cfg = merge_configs(cfg, override_cfg)

    logger.info("Effective config:\n%s", cfg)

    # ── Reproducibility ──────────────────────────────────────────────────
    seed = int(cfg.training.seed)
    pl.seed_everything(seed, workers=True)
    logger.info("Global seed set to %d", seed)

    # ── Resolve model name ───────────────────────────────────────────────
    model_name = cfg.model.name
    logger.info("Building model: %s", model_name)

    model = build_model(model_name, cfg)
    logger.info("Parameters: %s", f"{model.count_parameters():,}")

    # ── DataModule ───────────────────────────────────────────────────────
    data_cfg = cfg.data
    dm = BioMasstersDataModule(
        root_dir=data_cfg.root_dir,
        modalities=list(data_cfg.modalities),
        months=list(data_cfg.months) if data_cfg.months else None,
        batch_size=int(data_cfg.batch_size),
        num_workers=int(data_cfg.num_workers),
        val_split=float(data_cfg.val_split),
        pin_memory=bool(data_cfg.pin_memory),
        seed=seed,
    )

    # ── LightningModule ──────────────────────────────────────────────────
    lit = BioMasstersLitModule(
        model=model,
        loss_name=cfg.loss.name,
        optimizer_cfg=config_to_dict(cfg.optimizer),
        scheduler_cfg=config_to_dict(cfg.scheduler),
        log_images=bool(cfg.logging.log_images) and not args.no_wandb,
        log_images_n=int(cfg.logging.log_images_n),
    )

    # ── Callbacks ────────────────────────────────────────────────────────
    ckpt_dir = Path("results") / model_name / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        monitor=cfg.checkpoint.monitor,
        mode=cfg.checkpoint.mode,
        save_top_k=int(cfg.checkpoint.save_top_k),
        filename=cfg.checkpoint.filename,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    early_stop = EarlyStopping(
        monitor=cfg.checkpoint.monitor,
        patience=15,
        mode=cfg.checkpoint.mode,
        verbose=True,
    )

    callbacks = [checkpoint_cb, lr_monitor, early_stop]

    # ── Logger ───────────────────────────────────────────────────────────
    loggers = []
    if not args.no_wandb:
        try:
            from pytorch_lightning.loggers import WandbLogger

            wandb_logger = WandbLogger(
                project=cfg.logging.project,
                entity=cfg.logging.entity if cfg.logging.entity != "null" else None,
                name=model_name,
                config=config_to_dict(cfg),
                save_dir="wandb",
            )
            loggers.append(wandb_logger)
            logger.info("W&B logging enabled (project=%s)", cfg.logging.project)
        except Exception as exc:
            logger.warning("W&B logger failed to initialise: %s", exc)

    # ── Trainer ──────────────────────────────────────────────────────────
    tr_cfg = cfg.training
    trainer = pl.Trainer(
        max_epochs=int(tr_cfg.max_epochs),
        precision=str(tr_cfg.precision),
        gradient_clip_val=float(tr_cfg.gradient_clip_val),
        accumulate_grad_batches=int(tr_cfg.accumulate_grad_batches),
        log_every_n_steps=int(cfg.logging.log_every_n_steps),
        callbacks=callbacks,
        logger=loggers if loggers else True,
        deterministic=bool(tr_cfg.get("deterministic", False)),
    )

    # ── Train ────────────────────────────────────────────────────────────
    logger.info(
        "Starting training (model=%s, epochs=%d)", model_name, tr_cfg.max_epochs
    )
    trainer.fit(lit, datamodule=dm, ckpt_path=args.resume)

    # ── Save final metrics ───────────────────────────────────────────────
    best_ckpt = checkpoint_cb.best_model_path
    logger.info("Best checkpoint: %s", best_ckpt)

    results = {
        "model": model_name,
        "best_val_rmse": (
            float(checkpoint_cb.best_model_score)
            if checkpoint_cb.best_model_score is not None
            else None
        ),
        "best_checkpoint": best_ckpt,
        "n_params": model.count_parameters(),
        "seed": seed,
    }
    results_dir = Path("results") / model_name
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Metrics saved → %s", results_dir / "metrics.json")


if __name__ == "__main__":
    main()
