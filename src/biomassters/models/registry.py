"""Model registry and factory for BioMassters architectures.

Usage::

    from biomassters.models.registry import build_model

    model = build_model("utae", cfg)
"""

import logging
from typing import Any, Dict, Type

import torch.nn as nn

from biomassters.models.unet import UNet
from biomassters.models.unet3d import UNet3D
from biomassters.models.swin_unet import SwinUNet
from biomassters.models.utae import UTAE
from biomassters.models.tempfusionnet import TempFusionNet

logger = logging.getLogger(__name__)

# Registry mapping model name → class
MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "unet": UNet,
    "unet3d": UNet3D,
    "swin_unet": SwinUNet,
    "utae": UTAE,
    "tempfusionnet": TempFusionNet,
}


def build_model(name: str, config: Any) -> nn.Module:
    """Instantiate a model from the registry using the provided config.

    The config object (typically an OmegaConf DictConfig or plain dict) must
    contain a ``model`` sub-key with architecture-specific parameters.
    Common fields ``in_channels`` and ``n_timesteps`` are derived from the
    ``data`` section if not explicitly provided in ``model``.

    Args:
        name: Model identifier — one of the keys in :data:`MODEL_REGISTRY`.
        config: Full experiment config (OmegaConf DictConfig or dict).

    Returns:
        Instantiated ``nn.Module``.

    Raises:
        ValueError: If ``name`` is not in the registry.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )

    model_cls = MODEL_REGISTRY[name]

    # Extract model-specific kwargs from config
    raw_model = (
        getattr(config, "model", None)
        if not isinstance(config, dict)
        else config.get("model", None)
    )
    if raw_model is None:
        model_cfg = {}
    elif isinstance(raw_model, dict):
        model_cfg = dict(raw_model)
    elif hasattr(raw_model, "__dict__") and not isinstance(raw_model, type):
        # OmegaConf DictConfig or similar
        model_cfg = {k: v for k, v in vars(raw_model).items() if not k.startswith("_")}
    elif isinstance(raw_model, type):
        # Plain class (e.g., in tests) — extract class attributes
        model_cfg = {
            k: v
            for k, v in vars(raw_model).items()
            if not k.startswith("_") and not callable(v)
        }
    else:
        try:
            from omegaconf import OmegaConf

            model_cfg = OmegaConf.to_container(raw_model, resolve=True)  # type: ignore[assignment]
        except Exception:
            model_cfg = dict(raw_model)

    # Remove the 'name' key if present (used for routing, not construction)
    model_cfg.pop("name", None)

    # Resolve channel count from data config
    if "in_channels" not in model_cfg:
        raw_data = (
            getattr(config, "data", None)
            if not isinstance(config, dict)
            else config.get("data", None)
        )
        if raw_data is not None and isinstance(raw_data, type):
            modalities = getattr(raw_data, "modalities", ["s1", "s2"])
        elif raw_data is not None:
            modalities = getattr(raw_data, "modalities", ["s1", "s2"])
        else:
            modalities = ["s1", "s2"]
        n_channels = 0
        if "s1" in modalities:
            n_channels += 4
        if "s2" in modalities:
            n_channels += 11
        model_cfg["in_channels"] = n_channels

    if "n_timesteps" not in model_cfg:
        raw_data = (
            getattr(config, "data", None)
            if not isinstance(config, dict)
            else config.get("data", None)
        )
        months = None
        if raw_data is not None:
            months = getattr(raw_data, "months", None)
        model_cfg["n_timesteps"] = len(months) if months is not None else 12

    logger.info("Building model '%s' with kwargs: %s", name, model_cfg)

    return model_cls(**model_cfg)
