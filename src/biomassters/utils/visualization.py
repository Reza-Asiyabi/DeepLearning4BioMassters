"""Visualisation utilities for BioMassters predictions and data exploration."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def make_rgb_composite(
    s2_tensor: np.ndarray,
    rgb_indices: Tuple[int, int, int] = (2, 1, 0),  # B4, B3, B2 at positions 2,1,0
    clip_pct: float = 2.0,
) -> np.ndarray:
    """Create an 8-bit RGB composite from Sentinel-2 bands.

    Applies per-channel percentile clipping (2%–98%) and scales to [0, 255].

    Args:
        s2_tensor:   S2 chip of shape ``(C, H, W)`` with raw reflectance values.
        rgb_indices: Band indices for R, G, B (default: B4=2, B3=1, B2=0).
        clip_pct:    Percentile for low/high clip (e.g., 2 → 2%–98%).

    Returns:
        RGB image of shape ``(H, W, 3)`` as uint8.
    """
    rgb = np.stack([s2_tensor[i] for i in rgb_indices], axis=-1).astype(np.float32)

    # Per-channel percentile clip
    for c in range(3):
        lo = np.percentile(rgb[..., c], clip_pct)
        hi = np.percentile(rgb[..., c], 100 - clip_pct)
        rgb[..., c] = np.clip(rgb[..., c], lo, hi)
        if hi > lo:
            rgb[..., c] = (rgb[..., c] - lo) / (hi - lo)
        else:
            rgb[..., c] = 0.0

    return (rgb * 255).astype(np.uint8)


def plot_sample(
    image: np.ndarray,
    target: np.ndarray,
    prediction: Optional[np.ndarray] = None,
    month: int = 6,
    s1_indices: Tuple[int, int] = (0, 1),
    title: str = "",
    save_path: Optional[str] = None,
) -> None:
    """Plot a multi-panel figure: RGB composite, SAR, AGB target, and prediction.

    Args:
        image:      Image tensor of shape ``(T, C, H, W)`` (raw, before normalisation).
        target:     AGB map ``(H, W)`` or ``(1, H, W)`` in Mg/ha.
        prediction: Optional predicted AGB map of same shape as ``target``.
        month:      Time step index to display (default: 6 = July).
        s1_indices: Channel indices for the SAR false-colour composite (two channels).
        title:      Figure super-title.
        save_path:  If given, save figure to this path instead of displaying.
    """
    import matplotlib.pyplot as plt

    if target.ndim == 3:
        target = target[0]
    if prediction is not None and prediction.ndim == 3:
        prediction = prediction[0]

    n_panels = 4 if prediction is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

    # Determine modality split (15 ch = S1+S2 → 4 S1, 11 S2)
    T, C, H, W = image.shape
    has_s1 = C >= 4
    has_s2 = C >= 15 or (C >= 11 and not has_s1)

    frame = image[month]  # (C, H, W)

    # Panel 1: RGB composite
    ax = axes[0]
    if has_s2:
        s2_start = 4 if has_s1 else 0
        s2 = frame[s2_start : s2_start + 11]
        rgb = make_rgb_composite(s2)
        ax.imshow(rgb)
        ax.set_title(f"S2 RGB (month {month})")
    else:
        ax.imshow(frame[0], cmap="gray")
        ax.set_title(f"S1 VV (month {month})")
    ax.axis("off")

    # Panel 2: SAR false-colour
    ax = axes[1]
    if has_s1:
        vv = frame[s1_indices[0]]
        vh = frame[s1_indices[1]]
        sar = np.stack(
            [
                _normalise_band(vv),
                _normalise_band(vh),
                _normalise_band(vv - vh),
            ],
            axis=-1,
        )
        ax.imshow((sar * 255).astype(np.uint8))
        ax.set_title(f"SAR false-colour (month {month})")
    else:
        ax.imshow(frame[0], cmap="gray")
        ax.set_title("SAR N/A")
    ax.axis("off")

    # Panel 3: AGB target
    agb_cmap = "YlOrBr"
    vmax = max(float(target.max()), 1.0)
    axes[2].imshow(target, cmap=agb_cmap, vmin=0, vmax=vmax)
    axes[2].set_title("AGB Target (Mg/ha)")
    axes[2].axis("off")

    # Panel 4: Prediction (optional)
    if prediction is not None:
        axes[3].imshow(prediction, cmap=agb_cmap, vmin=0, vmax=vmax)
        axes[3].set_title("AGB Prediction (Mg/ha)")
        axes[3].axis("off")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved figure → %s", save_path)
    else:
        plt.show()
    plt.close(fig)


def plot_temporal_sequence(
    image: np.ndarray,
    agb_map: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> None:
    """Visualise all 12 monthly images for a single chip.

    Args:
        image:    ``(T, C, H, W)`` image tensor (raw values).
        agb_map:  Optional AGB map ``(H, W)`` displayed as the last panel.
        save_path: If given, save figure instead of displaying.
    """
    import matplotlib.pyplot as plt

    T, C, H, W = image.shape
    has_s1 = C >= 4
    has_s2 = C >= 15

    n_cols = T + (1 if agb_map is not None else 0)
    fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3))

    for t in range(T):
        ax = axes[t]
        frame = image[t]  # (C, H, W)
        if has_s2:
            s2_start = 4 if has_s1 else 0
            rgb = make_rgb_composite(frame[s2_start : s2_start + 11])
            ax.imshow(rgb)
        else:
            ax.imshow(frame[0], cmap="gray")
        ax.set_title(f"M{t:02d}", fontsize=8)
        ax.axis("off")

    if agb_map is not None:
        axes[-1].imshow(agb_map, cmap="YlOrBr", vmin=0)
        axes[-1].set_title("AGB", fontsize=8)
        axes[-1].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def plot_metrics_comparison(
    results_dict: Dict[str, Dict[str, float]],
    metrics: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """Bar chart comparing all models across key metrics.

    Args:
        results_dict: ``{model_name: {metric_name: value}}``.
        metrics:      Metrics to plot. Defaults to ``['rmse', 'mae', 'r2']``.
        save_path:    If given, save figure instead of displaying.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = metrics or ["rmse", "mae", "r2"]
    model_names = list(results_dict.keys())
    n_metrics = len(metrics)
    x = np.arange(len(model_names))

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        vals = [results_dict[m].get(metric, float("nan")) for m in model_names]
        bars = ax.bar(x, vals, width=0.6, color="steelblue", edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=9)
        ax.set_title(metric.upper(), fontweight="bold")
        ax.set_ylabel(metric)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    plt.suptitle("Model Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def plot_prediction_scatter(
    target: np.ndarray,
    prediction: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None,
    max_points: int = 50_000,
) -> None:
    """Scatter plot of predicted vs. true AGB with R² annotation.

    Args:
        target:     Ground-truth AGB values ``(N,)`` or ``(H, W)``.
        prediction: Predicted AGB values, same shape as ``target``.
        model_name: Label for the title.
        save_path:  If given, save figure instead of displaying.
        max_points: Sub-sample to this many points for readability.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr

    t = target.ravel().astype(np.float32)
    p = prediction.ravel().astype(np.float32)

    # Mask zeros
    valid = t > 0
    t, p = t[valid], p[valid]  # type: ignore[assignment]

    # Sub-sample for speed
    if len(t) > max_points:
        idx = np.random.choice(len(t), max_points, replace=False)
        t, p = t[idx], p[idx]  # type: ignore[assignment]

    r, _ = pearsonr(t, p)
    r2 = r**2

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(t, p, alpha=0.15, s=2, c="steelblue")
    vmax = max(t.max(), p.max())
    ax.plot([0, vmax], [0, vmax], "r--", lw=1, label="1:1 line")
    ax.set_xlabel("True AGB (Mg/ha)")
    ax.set_ylabel("Predicted AGB (Mg/ha)")
    ax.set_title(f"{model_name}  —  R² = {r2:.3f}", fontweight="bold")
    ax.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _normalise_band(arr: np.ndarray, clip_pct: float = 2.0) -> np.ndarray:
    """Percentile-clip and normalise a 2D band to [0, 1]."""
    lo = np.percentile(arr, clip_pct)
    hi = np.percentile(arr, 100 - clip_pct)
    arr = np.clip(arr, lo, hi)
    return (arr - lo) / (hi - lo + 1e-8)
