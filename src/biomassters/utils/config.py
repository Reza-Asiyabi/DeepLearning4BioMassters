"""Config loading and merging utilities using OmegaConf."""

import logging
from pathlib import Path
from typing import Any, Dict, Union

logger = logging.getLogger(__name__)

try:
    from omegaconf import OmegaConf

    _OMEGACONF_AVAILABLE = True
except ImportError:
    _OMEGACONF_AVAILABLE = False
    logger.warning("OmegaConf not installed — config loading will use plain dicts.")


def load_config(path: Union[str, Path]) -> Any:
    """Load a YAML config file with OmegaConf.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        ``OmegaConf.DictConfig`` if OmegaConf is available, else a plain dict.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if _OMEGACONF_AVAILABLE:
        cfg = OmegaConf.load(path)
        logger.debug("Loaded config from %s", path)
        return cfg
    else:
        import yaml

        with open(path) as f:
            return yaml.safe_load(f)


def merge_configs(base: Any, override: Any) -> Any:
    """Merge override config on top of base config.

    Values in ``override`` take precedence over ``base``.

    Args:
        base:     Base configuration (e.g., loaded from ``configs/base.yaml``).
        override: Model-specific or command-line overrides.

    Returns:
        Merged configuration object.
    """
    if _OMEGACONF_AVAILABLE:
        return OmegaConf.merge(base, override)
    else:
        if isinstance(base, dict) and isinstance(override, dict):
            merged = base.copy()
            for k, v in override.items():
                if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                    merged[k] = merge_configs(merged[k], v)
                else:
                    merged[k] = v
            return merged
        return override


def config_to_dict(cfg: Any) -> Dict[str, Any]:
    """Flatten config to a plain dict (for W&B logging).

    Args:
        cfg: OmegaConf DictConfig or plain dict.

    Returns:
        Flat Python dict (nested dicts preserved for W&B).
    """
    if _OMEGACONF_AVAILABLE and hasattr(cfg, "_metadata"):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    elif isinstance(cfg, dict):
        return cfg
    else:
        return dict(cfg)
