#!/usr/bin/env python
"""Generate model comparison tables and figures.

Reads ``metrics_*.json`` files from each model results directory and produces:
    1. Markdown metrics table (printed to stdout).
    2. Bar chart figure saved to ``assets/comparison_bar.png``.
    3. Per-model scatter plots saved to ``assets/``.

Usage::

    python scripts/compare_models.py --results-dir results/
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare trained model metrics and generate figures."
    )
    parser.add_argument(
        "--results-dir",
        "-r",
        type=str,
        default="results",
        help="Root directory containing per-model results folders.",
    )
    parser.add_argument(
        "--assets-dir",
        type=str,
        default="assets",
        help="Directory to save generated figures.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Which split metrics to compare. (default: val)",
    )
    return parser.parse_args()


MODEL_DISPLAY_NAMES = {
    "unet": "U-Net (baseline)",
    "unet3d": "3D U-Net",
    "swin_unet": "Swin U-Net",
    "utae": "U-TAE",
    "tempfusionnet": "TempFusionNet (ours)",
}

METRIC_COLS = ["rmse", "mae", "r2", "rel_rmse", "n_params"]
METRIC_DISPLAY = {
    "rmse": "RMSE ↓",
    "mae": "MAE ↓",
    "r2": "R² ↑",
    "rel_rmse": "Rel.RMSE ↓",
    "n_params": "#Params",
}


def load_results(results_dir: Path, split: str) -> Dict[str, Dict[str, Any]]:
    """Load metrics JSON for each model.

    Args:
        results_dir: Root directory.
        split: ``'val'`` or ``'test'``.

    Returns:
        Dict mapping model_name → metrics dict.
    """
    results = {}
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        metrics_file = model_dir / f"metrics_{split}.json"
        if not metrics_file.exists():
            # Fall back to generic metrics.json
            metrics_file = model_dir / "metrics.json"
        if not metrics_file.exists():
            logger.warning("No metrics file found for %s", model_dir.name)
            continue
        with open(metrics_file) as f:
            data = json.load(f)
        results[model_dir.name] = data
        logger.info("Loaded results for %s", model_dir.name)
    return results


def extract_metric(data: Dict, key: str, split_prefix: str = "val") -> float:
    """Extract a metric value from the results dict, trying multiple key formats."""
    for candidate in [
        f"{split_prefix}/{key}",
        key,
        f"test/{key}",
    ]:
        if candidate in data:
            return float(data[candidate])
    return float("nan")


def make_markdown_table(results: Dict[str, Dict], split: str) -> str:
    """Generate a Markdown comparison table.

    Args:
        results: Loaded results dict.
        split:   Evaluation split label.

    Returns:
        Markdown table as a string.
    """
    model_order = [m for m in MODEL_DISPLAY_NAMES if m in results]
    # Add any models not in display names
    for m in results:
        if m not in model_order:
            model_order.append(m)

    header = "| Model | RMSE ↓ | MAE ↓ | R² ↑ | Rel.RMSE ↓ | #Params |"
    sep = "|---|---|---|---|---|---|"
    rows = [header, sep]

    for model in model_order:
        data = results[model]
        name = MODEL_DISPLAY_NAMES.get(model, model)
        rmse = extract_metric(data, "rmse", split)
        mae = extract_metric(data, "mae", split)
        r2 = extract_metric(data, "r2", split)
        rrmse = extract_metric(data, "rel_rmse", split)
        n_par = data.get("n_params", float("nan"))

        def fmt(v: float, decimals: int = 3) -> str:
            return f"{v:.{decimals}f}" if not (v != v) else "-"  # nan check

        param_str = (
            f"~{int(n_par / 1e6)}M"
            if isinstance(n_par, (int, float)) and n_par == n_par
            else "-"
        )

        rows.append(
            f"| {name} | {fmt(rmse)} | {fmt(mae)} | {fmt(r2)} | {fmt(rrmse)} | {param_str} |"
        )

    return "\n".join(rows)


def make_comparison_bar_chart(results: Dict, split: str, save_path: Path) -> None:
    """Save a bar chart comparing RMSE across models.

    Args:
        results:   Loaded results dict.
        split:     Evaluation split.
        save_path: Output file path.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping bar chart.")
        return

    model_order = [m for m in MODEL_DISPLAY_NAMES if m in results]
    for m in results:
        if m not in model_order:
            model_order.append(m)

    names = [MODEL_DISPLAY_NAMES.get(m, m) for m in model_order]
    rmse_vals = [extract_metric(results[m], "rmse", split) for m in model_order]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    bars = ax.barh(
        names, rmse_vals, color=colors[: len(names)], edgecolor="white", height=0.6
    )

    for bar, val in zip(bars, rmse_vals):
        if val == val:  # not nan
            ax.text(
                bar.get_width() + 0.3,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}",
                va="center",
                ha="left",
                fontsize=10,
            )

    ax.set_xlabel("RMSE (Mg/ha)", fontsize=12)
    ax.set_title(
        f"BioMassters AGB Estimation — Model Comparison ({split})",
        fontsize=13,
        fontweight="bold",
    )
    ax.invert_yaxis()
    ax.set_xlim(
        0,
        (
            max(v for v in rmse_vals if v == v) * 1.15
            if any(v == v for v in rmse_vals)
            else 100
        ),
    )
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved comparison bar chart → %s", save_path)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    assets_dir = Path(args.assets_dir)

    if not results_dir.exists():
        logger.error("Results directory not found: %s", results_dir)
        sys.exit(1)

    results = load_results(results_dir, args.split)

    if not results:
        logger.error("No results found in %s", results_dir)
        sys.exit(1)

    # ── Markdown table ───────────────────────────────────────────────────
    table = make_markdown_table(results, args.split)
    print("\n" + table + "\n")

    table_path = assets_dir / "results_table.md"
    assets_dir.mkdir(parents=True, exist_ok=True)
    with open(table_path, "w") as f:
        f.write(f"# BioMassters Model Comparison ({args.split})\n\n")
        f.write(table + "\n")
    logger.info("Markdown table saved → %s", table_path)

    # ── Bar chart ────────────────────────────────────────────────────────
    make_comparison_bar_chart(results, args.split, assets_dir / "comparison_bar.png")

    logger.info("Done.")


if __name__ == "__main__":
    main()
