"""Model architectures for BioMassters AGB estimation."""

from biomassters.models.registry import MODEL_REGISTRY, build_model
from biomassters.models.unet import UNet
from biomassters.models.unet3d import UNet3D
from biomassters.models.swin_unet import SwinUNet
from biomassters.models.utae import UTAE
from biomassters.models.tempfusionnet import TempFusionNet

__all__ = [
    "MODEL_REGISTRY",
    "build_model",
    "UNet",
    "UNet3D",
    "SwinUNet",
    "UTAE",
    "TempFusionNet",
]
