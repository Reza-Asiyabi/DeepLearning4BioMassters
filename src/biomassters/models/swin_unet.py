"""Swin Transformer U-Net for AGB estimation.

Design rationale
----------------
Replaces the convolutional encoder of a standard U-Net with a Swin Transformer
backbone (timm), which captures long-range spatial dependencies via hierarchical
shifted-window self-attention.  Temporal fusion is handled before the backbone
via mean-pooling across months — a pragmatic choice that keeps memory footprint
manageable at the cost of ignoring temporal ordering.

Pros:
    - Strong spatial feature extraction from Swin's hierarchical attention.
    - Can be initialised from ImageNet pre-trained weights via timm.
    - Multi-scale FPN-style skip connections for detail preservation.
Cons:
    - No explicit temporal modelling (time averaged before backbone).
    - Swin input expects fixed resolution; requires padding/resizing for other sizes.
    - Higher GPU memory than CNN baselines at equivalent depth.

Reference: Liu et al. "Swin Transformer: Hierarchical Vision Transformer using
Shifted Windows", ICCV 2021.  https://arxiv.org/abs/2103.14030
"""

import logging
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Decoder building blocks
# ──────────────────────────────────────────────────────────────────────────────


class FPNDecoder(nn.Module):
    """Feature Pyramid Network (FPN) style decoder with lateral connections.

    Takes a list of multi-scale feature maps from the backbone and merges them
    top-down with 3×3 lateral convolutions.

    Args:
        in_channels_list: Channel counts for each scale level (coarse→fine).
        out_channels: Unified channel count for FPN feature maps.
    """

    def __init__(self, in_channels_list: List[int], out_channels: int = 256) -> None:
        super().__init__()
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(c, out_channels, 1) for c in in_channels_list]
        )
        self.output_convs = nn.ModuleList(
            [
                nn.Conv2d(out_channels, out_channels, 3, padding=1)
                for _ in in_channels_list
            ]
        )

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Top-down FPN merging.

        Args:
            features: List of feature maps from coarsest to finest.

        Returns:
            List of merged feature maps at each scale.
        """
        # Apply lateral projections
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down merging
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], size=laterals[i].shape[-2:], mode="nearest"
            )

        return [conv(f) for conv, f in zip(self.output_convs, laterals)]


class SegmentationHead(nn.Module):
    """Final upsampling + prediction head.

    Args:
        in_channels: Input feature channels (finest FPN level).
        target_size: Spatial size of the output (H, W).
    """

    def __init__(self, in_channels: int, target_size: tuple = (256, 256)) -> None:
        super().__init__()
        self.target_size = target_size
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x, size=self.target_size, mode="bilinear", align_corners=False
        )
        return self.head(x)


# ──────────────────────────────────────────────────────────────────────────────
# Channel adapter: project multi-modal input → 3-channel for Swin
# ──────────────────────────────────────────────────────────────────────────────


class ChannelAdapter(nn.Module):
    """Project arbitrary-channel input to 3 channels (ImageNet convention).

    Args:
        in_channels: Number of input channels after temporal pooling.
        out_channels: Usually 3 (RGB convention expected by Swin).
    """

    def __init__(self, in_channels: int, out_channels: int = 3) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ──────────────────────────────────────────────────────────────────────────────
# Swin U-Net
# ──────────────────────────────────────────────────────────────────────────────


class SwinUNet(nn.Module):
    """Swin Transformer U-Net for AGB estimation.

    Temporal fusion strategy: mean-pool across T before the backbone so that
    the Swin backbone operates on a single 2D feature map per sample.  A
    learnable channel adapter projects the multi-band input to 3 channels.

    Args:
        in_channels: Channels per time step (e.g., 15 for S1+S2).
        n_timesteps: Number of time steps (used for mean temporal pooling).
        swin_model_name: ``timm`` model name for the Swin backbone.
        pretrained: Load ImageNet-1K pre-trained weights from timm.
        fpn_out_channels: Feature channels in the FPN decoder.
        output_size: Spatial size of the prediction output.

    Input shape:  ``(B, T, C, H, W)``
    Output shape: ``(B, 1, H, W)``
    """

    def __init__(
        self,
        in_channels: int = 15,
        n_timesteps: int = 12,
        swin_model_name: str = "swin_tiny_patch4_window7_224",
        pretrained: bool = False,
        fpn_out_channels: int = 256,
        output_size: tuple = (256, 256),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_timesteps = n_timesteps
        self.output_size = output_size

        # Channel adapter: (C_total → 3) before Swin
        self.channel_adapter = ChannelAdapter(in_channels, out_channels=3)

        # Swin backbone via timm
        try:
            import timm

            self.backbone = timm.create_model(
                swin_model_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=(0, 1, 2, 3),
                img_size=output_size[0],
            )
            feature_info = self.backbone.feature_info.channels()
        except Exception as e:
            logger.warning(
                "timm not available or model load failed (%s). "
                "Falling back to mock backbone for testing.",
                e,
            )
            self.backbone = _MockSwinBackbone(output_size=output_size)
            feature_info = [96, 192, 384, 768]

        # Store expected per-level channel counts (fine → coarse) so that the
        # forward pass can detect and correct NHWC output from newer timm Swin.
        self._backbone_channels: List[int] = list(feature_info)

        # FPN decoder (coarse features first)
        self.fpn = FPNDecoder(list(reversed(feature_info)), fpn_out_channels)

        # Final prediction head
        self.seg_head = SegmentationHead(fpn_out_channels, target_size=output_size)

        logger.info(
            "SwinUNet | in_ch=%d | T=%d | backbone=%s | pretrained=%s | params=%s",
            in_channels,
            n_timesteps,
            swin_model_name,
            pretrained,
            f"{self.count_parameters():,}",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, T, C, H, W)``.

        Returns:
            AGB prediction of shape ``(B, 1, H, W)``.
        """
        B, T, C, H, W = x.shape

        # Temporal mean pooling: (B, T, C, H, W) → (B, C, H, W)
        x = x.mean(dim=1)

        # Project channels to 3 (Swin convention): (B, C, H, W) → (B, 3, H, W)
        x = self.channel_adapter(x)

        # Backbone: list of feature maps from fine to coarse.
        # Newer timm Swin models return NHWC tensors (B, H, W, C); normalise
        # to NCHW by checking whether dim-1 matches the expected channel count.
        features: List[torch.Tensor] = self.backbone(x)
        features = [
            f.permute(0, 3, 1, 2).contiguous() if f.shape[1] != ch else f
            for f, ch in zip(features, self._backbone_channels)
        ]

        # FPN: coarse-to-fine for top-down merging
        fpn_features = self.fpn(list(reversed(features)))  # coarsest first

        # Use finest FPN level for prediction
        finest = fpn_features[-1]

        return self.seg_head(finest)  # (B, 1, H, W)

    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────────────────────
# Fallback backbone for environments without timm
# ──────────────────────────────────────────────────────────────────────────────


class _MockSwinBackbone(nn.Module):
    """Minimal CNN stand-in used when timm is unavailable (e.g., CI tests).

    Produces 4 feature maps at strides 4, 8, 16, 32 with the same channel
    counts as swin_tiny.
    """

    def __init__(self, output_size: tuple = (256, 256)) -> None:
        super().__init__()
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(3, 96, kernel_size=4, stride=4, padding=0),
                    nn.BatchNorm2d(96),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(96, 192, kernel_size=2, stride=2, padding=0),
                    nn.BatchNorm2d(192),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(192, 384, kernel_size=2, stride=2, padding=0),
                    nn.BatchNorm2d(384),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(384, 768, kernel_size=2, stride=2, padding=0),
                    nn.BatchNorm2d(768),
                    nn.ReLU(inplace=True),
                ),
            ]
        )
        self.feature_info = type(
            "FI", (), {"channels": lambda self: [96, 192, 384, 768]}
        )()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features
