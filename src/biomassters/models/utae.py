"""U-TAE: U-Net with Temporal Attention Encoder for AGB estimation.

Design rationale
----------------
U-TAE (Garnot & Landrieu 2021) introduces a Lightweight Temporal Attention
Encoder (L-TAE) that replaces temporal pooling with a learned, query-guided
attention mechanism over the T time steps.  At each spatial location, L-TAE
computes a weighted sum of temporal features using multi-head attention with
a *master query* — a global learned query that summarises "what to look for"
— enabling the model to selectively attend to informative time steps (e.g.,
summer observations for deciduous biomass) rather than averaging them.

Architecture:
    1. Shared spatial encoder (2D CNN) applied independently to each time step
       to produce per-timestep feature maps  [B, T, d, H, W].
    2. L-TAE temporal module: multi-head attention over T dimension →
       attended feature map  [B, d_model, H, W].
    3. U-Net decoder with skip connections from the spatial encoder.

Pros:
    - Learns WHICH months matter for biomass — highly relevant for seasonal forests.
    - Very parameter-efficient (L-TAE < 0.1 M params).
    - End-to-end differentiable temporal weighting.
Cons:
    - L-TAE attention is position-wise (each pixel attends independently), so
      spatial context across the attention module is limited.

Reference: Garnot & Landrieu, "Panoptic Segmentation of Satellite Image Time
Series with Convolutional Temporal Attention Networks", ICCV 2021.
https://arxiv.org/abs/2107.07933
"""

import logging
import math
from typing import List, Optional, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# L-TAE: Lightweight Temporal Attention Encoder
# ──────────────────────────────────────────────────────────────────────────────


class LTAE(nn.Module):
    """Lightweight Temporal Attention Encoder (L-TAE).

    Performs multi-head attention over T temporal feature maps using a shared
    master query (learned parameter) projected per head.  The output is a
    single attended feature map of shape ``(B, d_model, H, W)``.

    Args:
        in_channels: Feature channels of each temporal input (d).
        n_head: Number of attention heads.
        d_k: Key/query dimension per head.
        mlp_channels: MLP hidden dimensions applied after attention.
        dropout: Dropout on attention weights.
        T: Maximum sequence length (used for positional encoding).

    Input:  ``(B, T, d, H, W)``
    Output: ``(B, d_model, H, W)``  where d_model = mlp_channels[-1]
    """

    def __init__(
        self,
        in_channels: int = 128,
        n_head: int = 16,
        d_k: int = 4,
        mlp_channels: Optional[List[int]] = None,
        dropout: float = 0.1,
        T: int = 1000,
    ) -> None:
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.in_channels = in_channels
        mlp_channels = mlp_channels if mlp_channels is not None else [256, 128]
        self.d_model = mlp_channels[-1]

        # Master query: one learned vector per head → (n_head, d_k)
        self.master_query = nn.Parameter(torch.randn(n_head, d_k))
        nn.init.normal_(self.master_query, std=0.02)

        # Projections for key and value
        self.k_proj = nn.Linear(in_channels, n_head * d_k, bias=False)
        self.v_proj = nn.Linear(in_channels, n_head * in_channels, bias=False)

        # Positional encoding (sinusoidal, applied to key)
        self.register_buffer("pos_enc", self._make_pos_enc(T, d_k))

        # MLP applied to concatenated head outputs
        mlp_in = n_head * in_channels
        layers: List[nn.Module] = []
        for out_ch in mlp_channels:
            layers += [
                nn.Linear(mlp_in, out_ch),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            ]
            mlp_in = out_ch
        self.mlp = nn.Sequential(*layers)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_k)

    @staticmethod
    def _make_pos_enc(T: int, d: int) -> torch.Tensor:
        """Sinusoidal positional encoding.

        Returns:
            Tensor of shape ``(T, d)``.
        """
        pos = torch.arange(T, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d)
        )
        enc = torch.zeros(T, d)
        enc[:, 0::2] = torch.sin(pos * div)
        enc[:, 1::2] = torch.cos(pos * div)
        return enc  # (T, d)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """L-TAE forward pass.

        Args:
            x:    Feature sequence of shape ``(B, T, d, H, W)``.
            mask: Optional boolean mask of shape ``(B, T)`` — True = valid time step.

        Returns:
            Tuple of:
                - Attended feature map ``(B, d_model, H, W)``.
                - Attention weights       ``(B, n_head, T, H, W)``.
        """
        B, T, d, H, W = x.shape

        # Flatten spatial dims for attention: (B*H*W, T, d)
        x_flat = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, d)

        # Keys with positional encoding: (B*H*W, T, n_head * d_k)
        assert isinstance(self.pos_enc, torch.Tensor)
        pos = self.pos_enc[:T].unsqueeze(0)  # (1, T, d_k)
        # Project to key space and add positional encoding
        k = self.k_proj(x_flat)  # (B*H*W, T, n_head * d_k)
        k = k.view(B * H * W, T, self.n_head, self.d_k)
        k = k + pos.unsqueeze(2)  # broadcast pos over heads

        # Values: (B*H*W, T, n_head, d)
        v = self.v_proj(x_flat).view(B * H * W, T, self.n_head, d)

        # Queries: master query broadcast → (B*H*W, n_head, d_k, 1)
        q = self.master_query.unsqueeze(0).expand(
            B * H * W, -1, -1
        )  # (B*H*W, n_head, d_k)

        # Attention scores: (B*H*W, n_head, T)
        # Q: (B*H*W, n_head, d_k) ; K: (B*H*W, T, n_head, d_k)
        k_t = k.permute(0, 2, 1, 3)  # (B*H*W, n_head, T, d_k)
        scores = (
            torch.einsum("bnd,bntd->bnt", q, k_t) / self.scale
        )  # (B*H*W, n_head, T)

        # Mask padded time steps
        if mask is not None:
            # mask: (B, T) → expand to (B*H*W, n_head, T)
            m = mask.unsqueeze(1).expand(B, H * W, T).reshape(B * H * W, T)
            m = m.unsqueeze(1).expand(-1, self.n_head, -1)
            scores = scores.masked_fill(~m, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)  # (B*H*W, n_head, T)
        attn_weights = self.dropout(attn_weights)

        # Aggregate values: (B*H*W, n_head, d)
        v_t = v.permute(0, 2, 1, 3)  # (B*H*W, n_head, T, d)
        out = torch.einsum("bnt,bntd->bnd", attn_weights, v_t)  # (B*H*W, n_head, d)

        # Concatenate heads and apply MLP: (B*H*W, n_head*d)
        out = out.reshape(B * H * W, self.n_head * d)
        out = self.mlp(out)  # (B*H*W, d_model)

        # Reshape back to spatial: (B, d_model, H, W)
        out = out.view(B, H, W, self.d_model).permute(0, 3, 1, 2)

        # Attention weights for visualisation: (B, n_head, T, H, W)
        attn_spatial = attn_weights.view(B, H, W, self.n_head, T).permute(0, 3, 4, 1, 2)

        return out, attn_spatial


# ──────────────────────────────────────────────────────────────────────────────
# Spatial encoder (shared across time steps)
# ──────────────────────────────────────────────────────────────────────────────


class SpatialEncoder(nn.Module):
    """Lightweight 2D CNN applied independently to each time step.

    Args:
        in_channels: Input channels per time step.
        out_channels: Output feature channels (fed into L-TAE).
        n_levels: Number of downsampling levels.
    """

    def __init__(
        self,
        in_channels: int = 15,
        out_channels: int = 128,
        n_levels: int = 3,
    ) -> None:
        super().__init__()
        self.n_levels = n_levels

        self.enc_blocks: nn.ModuleList = nn.ModuleList()
        self.pools: nn.ModuleList = nn.ModuleList()
        ch = in_channels
        enc_ch: List[int] = []
        for i in range(n_levels):
            mid_ch = min(64 * (2**i), out_channels)
            self.enc_blocks.append(
                nn.Sequential(
                    nn.Conv2d(ch, mid_ch, 3, padding=1, bias=False),
                    nn.BatchNorm2d(mid_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
                    nn.BatchNorm2d(mid_ch),
                    nn.ReLU(inplace=True),
                )
            )
            self.pools.append(nn.MaxPool2d(2))
            enc_ch.append(mid_ch)
            ch = mid_ch

        # Final projection to out_channels
        self.proj = nn.Conv2d(ch, out_channels, 1)
        self.enc_channels = enc_ch  # for skip connections

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Apply spatial encoder to a single time step.

        Args:
            x: ``(B, C, H, W)``

        Returns:
            Tuple of (bottleneck_features (B, out_ch, H', W'),
                      skip_list [(B, ch, H_i, W_i)])
        """
        skips = []
        for block, pool in zip(self.enc_blocks, self.pools):
            x = block(x)
            skips.append(x)
            x = pool(x)
        return self.proj(x), skips


# ──────────────────────────────────────────────────────────────────────────────
# Decoder
# ──────────────────────────────────────────────────────────────────────────────


class UTAEDecoder(nn.Module):
    """U-Net decoder that uses skip connections from the spatial encoder.

    Skip connections are the mean over all T time step skips (a soft form
    of temporal aggregation at each spatial scale).

    Args:
        ltae_out_ch: Output channels of L-TAE (d_model).
        enc_channels: Per-level channel counts from the spatial encoder.
    """

    def __init__(self, ltae_out_ch: int, enc_channels: List[int]) -> None:
        super().__init__()
        self.dec_blocks: nn.ModuleList = nn.ModuleList()
        ch = ltae_out_ch
        for skip_ch in reversed(enc_channels):
            up_ch = ch // 2 if ch > skip_ch else ch
            self.dec_blocks.append(
                nn.ModuleDict(
                    {
                        "up": nn.ConvTranspose2d(ch, up_ch, 2, stride=2),
                        "conv": nn.Sequential(
                            nn.Conv2d(
                                up_ch + skip_ch, skip_ch, 3, padding=1, bias=False
                            ),
                            nn.BatchNorm2d(skip_ch),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(skip_ch, skip_ch, 3, padding=1, bias=False),
                            nn.BatchNorm2d(skip_ch),
                            nn.ReLU(inplace=True),
                        ),
                    }
                )
            )
            ch = skip_ch
        self.out_channels = ch

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        """Decode with skip connections.

        Args:
            x:     Attended feature map ``(B, d_model, H', W')``.
            skips: Mean skip tensors (fine→coarse order → reversed inside).

        Returns:
            Decoded feature map at full spatial resolution.
        """
        for block, skip in zip(self.dec_blocks, reversed(skips)):
            block_dict = cast(nn.ModuleDict, block)
            x = block_dict["up"](x)
            # Handle potential size mismatch from pooling
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)
            x = block_dict["conv"](x)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Full U-TAE model
# ──────────────────────────────────────────────────────────────────────────────


class UTAE(nn.Module):
    """U-TAE: U-Net with Temporal Attention Encoder.

    Combines a shared lightweight spatial encoder (2D CNN per time step) with
    an L-TAE temporal module that uses multi-head attention to aggregate the
    T feature maps into a single attended representation, which is then decoded
    by a U-Net decoder with mean-aggregated skip connections.

    Args:
        in_channels: Channels per time step (e.g., 15 for S1+S2).
        n_timesteps: Number of time steps T (12 for full year).
        encoder_widths: Feature channels output by each spatial encoder level.
        ltae_n_head: Number of L-TAE attention heads.
        ltae_d_k: L-TAE key/query dimension per head.
        ltae_mlp: L-TAE MLP hidden dims after attention.
        dropout: Dropout in L-TAE.

    Input shape:  ``(B, T, C, H, W)``
    Output shape: ``(B, 1, H, W)``
    """

    def __init__(
        self,
        in_channels: int = 15,
        n_timesteps: int = 12,
        encoder_widths: Optional[List[int]] = None,
        ltae_n_head: int = 16,
        ltae_d_k: int = 4,
        ltae_mlp: Optional[List[int]] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_timesteps = n_timesteps
        encoder_widths = (
            encoder_widths if encoder_widths is not None else [64, 128, 128]
        )
        ltae_mlp = ltae_mlp if ltae_mlp is not None else [256, 128]

        # Spatial encoder (shared across time steps via weight sharing)
        self.spatial_encoder = SpatialEncoder(
            in_channels=in_channels,
            out_channels=encoder_widths[-1],
            n_levels=len(encoder_widths) - 1 if len(encoder_widths) > 1 else 1,
        )

        # L-TAE temporal module
        self.ltae = LTAE(
            in_channels=encoder_widths[-1],
            n_head=ltae_n_head,
            d_k=ltae_d_k,
            mlp_channels=ltae_mlp,
            dropout=dropout,
            T=max(n_timesteps, 1000),
        )

        # Decoder
        self.decoder = UTAEDecoder(
            ltae_out_ch=ltae_mlp[-1],
            enc_channels=self.spatial_encoder.enc_channels,
        )

        self.head = nn.Conv2d(self.decoder.out_channels, 1, kernel_size=1)

        logger.info(
            "UTAE | in_ch=%d | T=%d | encoder_widths=%s | n_head=%d | params=%s",
            in_channels,
            n_timesteps,
            encoder_widths,
            ltae_n_head,
            f"{self.count_parameters():,}",
        )

    def forward(
        self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x:        Input of shape ``(B, T, C, H, W)``.
            pad_mask: Optional boolean mask ``(B, T)`` True = valid time step.

        Returns:
            AGB prediction of shape ``(B, 1, H, W)``.
        """
        B, T, C, H, W = x.shape

        # Apply spatial encoder to each time step independently
        # Merge B and T for batch efficiency: (B*T, C, H, W)
        x_bt = x.reshape(B * T, C, H, W)
        feats_bt, skips_bt = self.spatial_encoder(x_bt)  # (B*T, d, H', W')

        _, d, H2, W2 = feats_bt.shape
        feats = feats_bt.view(B, T, d, H2, W2)  # (B, T, d, H', W')

        # Reshape skip connections: each is (B*T, ch_i, H_i, W_i)
        # Aggregate over T → mean skip: (B, ch_i, H_i, W_i)
        mean_skips = []
        for skip in skips_bt:
            _, ch_s, Hs, Ws = skip.shape
            skip_b = skip.view(B, T, ch_s, Hs, Ws).mean(dim=1)  # (B, ch_s, Hs, Ws)
            mean_skips.append(skip_b)

        # L-TAE temporal attention: (B, T, d, H', W') → (B, d_model, H', W')
        attended, _attn = self.ltae(feats, mask=pad_mask)  # (B, d_model, H', W')

        # Decode: upsample back to (H, W) using mean skips
        out = self.decoder(attended, mean_skips)

        # Upsample to full resolution if encoder downsampled
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)

        return self.head(out)  # (B, 1, H, W)

    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
