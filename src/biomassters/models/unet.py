"""Baseline 2D U-Net with Late Temporal Fusion.

Design rationale
----------------
The simplest possible baseline: concatenate all T monthly images along the
channel dimension before feeding into a standard 2D U-Net.  This "late
fusion" strategy makes no assumptions about temporal ordering and lets the
network learn which months matter implicitly.

Pros:
    - Simple, fast, well-understood architecture.
    - Serves as a strong lower-bound baseline.
Cons:
    - Treats time as just another channel — no explicit temporal modelling.
    - Input size grows linearly with T × C; memory scales accordingly.

Reference: Ronneberger et al. "U-Net: Convolutional Networks for Biomedical
Image Segmentation", MICCAI 2015.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────────────


class ConvBlock(nn.Module):
    """Two consecutive Conv-BN-ReLU layers.

    Args:
        in_ch: Number of input channels.
        out_ch: Number of output channels.
        dropout: Dropout probability (applied between the two conv layers).
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, H, W)
        return self.block(x)


class Encoder(nn.Module):
    """U-Net encoder (contracting path).

    Args:
        in_ch: Number of input channels.
        base_ch: Number of channels in the first encoder block.
        depth: Number of encoder levels.
        dropout: Dropout probability.
    """

    def __init__(
        self, in_ch: int, base_ch: int = 64, depth: int = 4, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_ch
        for i in range(depth):
            out_ch = base_ch * (2**i)
            self.blocks.append(ConvBlock(ch, out_ch, dropout=dropout))
            self.pools.append(nn.MaxPool2d(2))
            ch = out_ch
        self.out_channels = ch

    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass returning bottleneck and skip connections.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.

        Returns:
            Tuple of (bottleneck_features, list_of_skip_tensors).
        """
        skips = []
        for block, pool in zip(self.blocks, self.pools):
            x = block(x)
            skips.append(x)
            x = pool(x)
        return x, skips


class Bottleneck(nn.Module):
    """U-Net bottleneck block.

    Args:
        in_ch: Input channels (== last encoder output channels).
        dropout: Dropout probability.
    """

    def __init__(self, in_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = ConvBlock(in_ch, in_ch * 2, dropout=dropout)
        self.out_channels = in_ch * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """Single decoder step: upsample + skip connection + ConvBlock.

    Args:
        in_ch: Channels from the previous decoder level (upsampled).
        skip_ch: Channels from the corresponding encoder skip.
        out_ch: Output channels.
        dropout: Dropout probability.
    """

    def __init__(
        self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ──────────────────────────────────────────────────────────────────────────────
# U-Net model
# ──────────────────────────────────────────────────────────────────────────────


class UNet(nn.Module):
    """2D U-Net with late temporal fusion for AGB estimation.

    All T monthly images are concatenated along the channel dimension before
    being processed by a standard 4-level U-Net.

    Args:
        in_channels: Number of channels per time step (e.g., 15 for S1+S2).
        n_timesteps: Number of time steps T (e.g., 12 months).
        base_channels: Base channel width of the first encoder block.
        depth: Number of encoder/decoder levels.
        dropout: Spatial dropout probability in conv blocks.

    Input shape:  ``(B, T, C, H, W)``
    Output shape: ``(B, 1, H, W)``
    """

    def __init__(
        self,
        in_channels: int = 15,
        n_timesteps: int = 12,
        base_channels: int = 64,
        depth: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_timesteps = n_timesteps
        self.fused_ch = in_channels * n_timesteps  # late fusion channel count

        self.encoder = Encoder(
            self.fused_ch, base_ch=base_channels, depth=depth, dropout=dropout
        )
        self.bottleneck = Bottleneck(self.encoder.out_channels, dropout=dropout)

        # Decoder mirrors encoder
        enc_ch = [base_channels * (2**i) for i in range(depth)]  # per-level channels
        bot_ch = self.bottleneck.out_channels
        self.decoder = nn.ModuleList()
        ch = bot_ch
        for skip_ch in reversed(enc_ch):
            self.decoder.append(DecoderBlock(ch, skip_ch, skip_ch, dropout=dropout))
            ch = skip_ch

        self.head = nn.Conv2d(ch, 1, kernel_size=1)

        logger.info(
            "UNet | in_ch=%d | T=%d | fused_ch=%d | base_ch=%d | depth=%d | params=%s",
            in_channels,
            n_timesteps,
            self.fused_ch,
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
        # Late fusion: (B, T, C, H, W) → (B, T*C, H, W)
        x = x.reshape(B, T * C, H, W)

        x, skips = self.encoder(x)
        x = self.bottleneck(x)

        for dec_block, skip in zip(self.decoder, reversed(skips)):
            x = dec_block(x, skip)

        return self.head(x)  # (B, 1, H, W)

    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
