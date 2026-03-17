"""TempFusionNet: 3D Encoder + 2D Decoder Hybrid with Channel Attention.

Design rationale
----------------
TempFusionNet is a custom architecture that combines the temporal modelling
capacity of 3D convolutions (in the encoder) with the memory efficiency of 2D
convolutions (in the decoder), bridged by Squeeze-and-Excitation (SE) channel
attention blocks.

The key insight is that early spatial–temporal feature extraction benefits from
3D convolutions (capturing phenological dynamics), while the decoder only needs
to refine 2D spatial detail — so we save compute by collapsing the time
dimension at the bottleneck via global average pooling over T.

SE blocks in the encoder adaptively recalibrate channel responses, helping the
model focus on informative spectral–temporal features (e.g., SWIR bands during
dormant season, C-band backscatter during snow cover).

Architecture:
    Input (B,T,C,H,W)
        ↓
    3D Conv Encoder (with SE blocks)  [levels 0..depth-1]
        ↓
    GAP over T → Bottleneck (B, C_bot, H', W')
        ↓
    2D Decoder (ConvTranspose + skip connections)
        ↓
    Output (B, 1, H, W)

Pros:
    - Richer temporal features than simple mean-pooling (3D early layers).
    - SE attention focuses channels on informative spectral-temporal combos.
    - More memory-efficient than full 3D decoder.
    - Novel combination not published in satellite biomass literature.
Cons:
    - Hard temporal collapse at bottleneck loses fine-grained temporal context.
    - Hyperparameter sensitivity at 3D/2D boundary.
"""

import logging
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Squeeze-and-Excitation Block
# ──────────────────────────────────────────────────────────────────────────────


class SEBlock(nn.Module):
    """Channel-wise Squeeze-and-Excitation attention.

    Globally pools spatial+temporal dimensions, applies an MLP bottleneck,
    and multiplies channel-wise scaling factors back into the feature map.

    Args:
        channels: Number of input/output channels.
        reduction: Bottleneck reduction ratio.

    Reference: Hu et al. "Squeeze-and-Excitation Networks", CVPR 2018.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # (B, C, 1, 1, 1)
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SE attention.

        Args:
            x: Feature tensor ``(B, C, T, H, W)``.

        Returns:
            Channel-recalibrated tensor of same shape.
        """
        scale = self.fc(x).view(x.shape[0], x.shape[1], 1, 1, 1)
        return x * scale


# ──────────────────────────────────────────────────────────────────────────────
# 3D Encoder block
# ──────────────────────────────────────────────────────────────────────────────


class Enc3DBlock(nn.Module):
    """3D Conv-BN-ReLU block with optional Squeeze-and-Excitation.

    Args:
        in_ch: Input channels.
        out_ch: Output channels.
        use_se: Whether to apply SE channel attention.
        se_reduction: SE reduction ratio.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        use_se: bool = True,
        se_reduction: int = 16,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.se = SEBlock(out_ch, reduction=se_reduction) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T, H, W)
        return self.se(self.conv(x))


# ──────────────────────────────────────────────────────────────────────────────
# 2D Decoder block
# ──────────────────────────────────────────────────────────────────────────────


class Dec2DBlock(nn.Module):
    """2D decoder step: upsample + skip + conv.

    Args:
        in_ch: Channels from previous decoder or bottleneck.
        skip_ch: Channels from the T-averaged 3D encoder skip.
        out_ch: Output channels.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        up_ch = in_ch // 2
        self.up = nn.ConvTranspose2d(in_ch, up_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(up_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        return self.conv(torch.cat([x, skip], dim=1))


# ──────────────────────────────────────────────────────────────────────────────
# TempFusionNet
# ──────────────────────────────────────────────────────────────────────────────


class TempFusionNet(nn.Module):
    """TempFusionNet — 3D Encoder + 2D Decoder Hybrid for AGB Estimation.

    A custom architecture combining 3D spatio-temporal feature extraction
    (encoder) with SE channel attention and an efficient 2D decoder.

    Args:
        in_channels: Channels per time step (e.g., 15 for S1+S2).
        n_timesteps: Number of time steps T.
        base_channels: Base channel width of first encoder block.
        depth: Number of encoder/decoder levels.
        use_se: Enable SE blocks in encoder.
        se_reduction: SE reduction ratio.

    Input shape:  ``(B, T, C, H, W)``
    Output shape: ``(B, 1, H, W)``
    """

    def __init__(
        self,
        in_channels: int = 15,
        n_timesteps: int = 12,
        base_channels: int = 32,
        depth: int = 4,
        use_se: bool = True,
        se_reduction: int = 16,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_timesteps = n_timesteps
        self.depth = depth

        # 3D encoder
        self.enc_blocks: nn.ModuleList = nn.ModuleList()
        self.enc_pools: nn.ModuleList = nn.ModuleList()
        enc_channels: List[int] = []
        ch = in_channels
        for i in range(depth):
            out_ch = base_channels * (2**i)
            self.enc_blocks.append(
                Enc3DBlock(ch, out_ch, use_se=use_se, se_reduction=se_reduction)
            )
            # Spatial-only pooling (preserve T in early layers)
            pool_t = 2 if i >= depth // 2 else 1
            self.enc_pools.append(nn.MaxPool3d(kernel_size=(pool_t, 2, 2)))
            enc_channels.append(out_ch)
            ch = out_ch

        # Bottleneck (3D → after collapse over T)
        bot_ch = ch * 2
        self.bottleneck = nn.Sequential(
            nn.Conv3d(ch, bot_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(bot_ch),
            nn.ReLU(inplace=True),
            SEBlock(bot_ch, reduction=se_reduction) if use_se else nn.Identity(),
        )

        # 2D decoder: uses time-mean skips from the 3D encoder
        self.dec_blocks: nn.ModuleList = nn.ModuleList()
        ch = bot_ch
        for skip_ch in reversed(enc_channels):
            self.dec_blocks.append(Dec2DBlock(ch, skip_ch, skip_ch))
            ch = skip_ch

        self.head = nn.Conv2d(ch, 1, kernel_size=1)

        logger.info(
            "TempFusionNet | in_ch=%d | T=%d | base_ch=%d | depth=%d | "
            "use_se=%s | params=%s",
            in_channels,
            n_timesteps,
            base_channels,
            depth,
            use_se,
            f"{self.count_parameters():,}",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input of shape ``(B, T, C, H, W)``.

        Returns:
            AGB prediction of shape ``(B, 1, H, W)``.
        """
        B, T, C, H, W = x.shape
        # Rearrange to 3D conv layout: (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)

        # Encoder — store T-averaged 2D skips
        skips: List[torch.Tensor] = []
        for block, pool in zip(self.enc_blocks, self.enc_pools):
            x = block(x)
            # Time-average for 2D skip: (B, C, T, H, W) → (B, C, H, W)
            skips.append(x.mean(dim=2))
            x = pool(x)

        # Bottleneck (3D)
        x = self.bottleneck(x)

        # Collapse time: global average pooling over T → (B, C_bot, H', W')
        x = x.mean(dim=2)

        # 2D decoder
        for dec_block, skip in zip(self.dec_blocks, reversed(skips)):
            x = dec_block(x, skip)

        return self.head(x)  # (B, 1, H, W)

    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
