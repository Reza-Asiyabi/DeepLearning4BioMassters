"""3D U-Net for spatio-temporal AGB estimation.

Design rationale
----------------
3D convolutions treat the time dimension as a spatial axis, allowing the
network to capture local spatio-temporal patterns (e.g., seasonal phenology
correlated across nearby pixels).  The encoder progressively pools both
spatial and temporal dimensions, while the decoder restores spatial resolution
using 2D transposed convolutions after the time dimension has been collapsed.

Pros:
    - Explicitly models local temporal correlations.
    - Parameter-efficient compared to attention-based methods for short sequences.
Cons:
    - Memory-intensive: 3D feature maps grow cubically.
    - Temporal receptive field limited by kernel size and depth.
    - Collapses temporal information via pooling (no soft attention).
"""

import logging
from typing import List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────────────


class Conv3dBlock(nn.Module):
    """3D Conv-BN-ReLU block (two conv layers).

    Args:
        in_ch: Input channels.
        out_ch: Output channels.
        dropout: Dropout probability.
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T, H, W)
        return self.block(x)


class Conv2dBlock(nn.Module):
    """2D Conv-BN-ReLU block for the decoder.

    Args:
        in_ch: Input channels.
        out_ch: Output channels.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, H, W)
        return self.block(x)


class Decoder2dBlock(nn.Module):
    """2D decoder step: upsample + cat(skip) + conv.

    Args:
        in_ch: Channels coming in (from previous decoder or bottleneck).
        skip_ch: Channels from the corresponding encoder skip.
        out_ch: Output channels.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = Conv2dBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ──────────────────────────────────────────────────────────────────────────────
# 3D U-Net
# ──────────────────────────────────────────────────────────────────────────────


class UNet3D(nn.Module):
    """3D U-Net treating time as a spatial dimension for AGB estimation.

    Architecture:
        Encoder: 3D convolutions over (T, H, W), with 3D max-pooling.
        Bottleneck: 3D conv block.
        Bridge: Collapse time via mean pooling → switch to 2D.
        Decoder: 2D transposed convolutions with 2D skip connections
                 (derived by mean-pooling the 3D encoder skips over T).

    Args:
        in_channels: Channels per time step (e.g., 15 for S1+S2).
        n_timesteps: Number of time steps T.
        base_channels: Base channel width (first encoder block).
        depth: Number of encoder/decoder levels.
        dropout: Dropout probability.

    Input shape:  ``(B, T, C, H, W)``
    Output shape: ``(B, 1, H, W)``
    """

    def __init__(
        self,
        in_channels: int = 15,
        n_timesteps: int = 12,
        base_channels: int = 32,
        depth: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_timesteps = n_timesteps
        self.depth = depth

        # 3D encoder blocks and pooling layers
        self.enc_blocks: nn.ModuleList = nn.ModuleList()
        self.enc_pools: nn.ModuleList = nn.ModuleList()
        ch = in_channels
        enc_channels: List[int] = []
        for i in range(depth):
            out_ch = base_channels * (2**i)
            self.enc_blocks.append(Conv3dBlock(ch, out_ch, dropout=dropout))
            # Pool spatial dims; keep temporal dim for upper levels
            pool_t = 2 if i < depth - 1 else 1  # don't pool T at last level
            self.enc_pools.append(nn.MaxPool3d(kernel_size=(pool_t, 2, 2)))
            enc_channels.append(out_ch)
            ch = out_ch

        # Bottleneck (3D)
        bot_ch = ch * 2
        self.bottleneck = Conv3dBlock(ch, bot_ch, dropout=dropout)

        # 2D decoder
        self.dec_blocks: nn.ModuleList = nn.ModuleList()
        ch = bot_ch
        for skip_ch in reversed(enc_channels):
            self.dec_blocks.append(Decoder2dBlock(ch, skip_ch, skip_ch))
            ch = skip_ch

        self.head = nn.Conv2d(ch, 1, kernel_size=1)

        logger.info(
            "UNet3D | in_ch=%d | T=%d | base_ch=%d | depth=%d | params=%s",
            in_channels,
            n_timesteps,
            base_channels,
            depth,
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
        # Rearrange to 3D conv layout: (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)

        # Encoder
        skips: List[torch.Tensor] = []
        for block, pool in zip(self.enc_blocks, self.enc_pools):
            x = block(x)
            # Collapse T for skip (mean over time → 2D skip)
            skips.append(x.mean(dim=2))  # (B, C, H, W)
            x = pool(x)

        x = self.bottleneck(x)

        # Collapse time dimension → (B, C_bot, H', W')
        x = x.mean(dim=2)

        # 2D decoder
        for dec_block, skip in zip(self.dec_blocks, reversed(skips)):
            x = dec_block(x, skip)

        return self.head(x)  # (B, 1, H, W)

    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
