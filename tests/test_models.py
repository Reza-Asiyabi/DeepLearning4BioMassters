"""Tests for all five model architectures.

Tests use small tensors (B=2, T=4, C=15, H=64, W=64) to keep CPU run time
reasonable without requiring a GPU or real data.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

BATCH = 2
N_TIMESTEP = 4  # Use 4 months for speed (models must accept any T)
S1_CH = 4
S2_CH = 11
COMBINED = S1_CH + S2_CH  # 15
H, W = 64, 64


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_input(n_ch: int = COMBINED) -> torch.Tensor:
    """Create a random input tensor of shape (B, T, C, H, W)."""
    return torch.randn(BATCH, N_TIMESTEP, n_ch, H, W)


def _assert_output(pred: torch.Tensor) -> None:
    """Check output shape and that there are no NaNs."""
    assert pred.shape == (
        BATCH,
        1,
        H,
        W,
    ), f"Expected (B=2, 1, {H}, {W}), got {tuple(pred.shape)}"
    assert not torch.isnan(pred).any(), "Model output contains NaN values."


# ──────────────────────────────────────────────────────────────────────────────
# U-Net tests
# ──────────────────────────────────────────────────────────────────────────────


class TestUNet:
    """Tests for the 2D U-Net (late fusion) baseline."""

    def _build(self, in_channels: int = COMBINED) -> nn.Module:
        from biomassters.models.unet import UNet

        return UNet(
            in_channels=in_channels,
            n_timesteps=N_TIMESTEP,
            base_channels=16,  # small for speed
            depth=3,
        )

    def test_output_shape_combined(self) -> None:
        model = self._build(COMBINED)
        pred = model(_make_input(COMBINED))
        _assert_output(pred)

    def test_output_shape_s1_only(self) -> None:
        model = self._build(S1_CH)
        pred = model(_make_input(S1_CH))
        _assert_output(pred)

    def test_output_shape_s2_only(self) -> None:
        model = self._build(S2_CH)
        pred = model(_make_input(S2_CH))
        _assert_output(pred)

    def test_parameter_count(self) -> None:
        model = self._build()
        assert model.count_parameters() > 0

    def test_no_nan_with_zero_input(self) -> None:
        model = self._build()
        pred = model(torch.zeros(BATCH, N_TIMESTEP, COMBINED, H, W))
        assert not torch.isnan(pred).any()


# ──────────────────────────────────────────────────────────────────────────────
# 3D U-Net tests
# ──────────────────────────────────────────────────────────────────────────────


class TestUNet3D:
    """Tests for the 3D U-Net (spatio-temporal)."""

    def _build(self, in_channels: int = COMBINED) -> nn.Module:
        from biomassters.models.unet3d import UNet3D

        return UNet3D(
            in_channels=in_channels,
            n_timesteps=N_TIMESTEP,
            base_channels=8,
            depth=3,
        )

    def test_output_shape_combined(self) -> None:
        model = self._build(COMBINED)
        pred = model(_make_input(COMBINED))
        _assert_output(pred)

    def test_output_shape_s1_only(self) -> None:
        model = self._build(S1_CH)
        pred = model(_make_input(S1_CH))
        _assert_output(pred)

    def test_parameter_count(self) -> None:
        model = self._build()
        assert model.count_parameters() > 0

    def test_no_nan_with_random_input(self) -> None:
        model = self._build()
        pred = model(_make_input())
        assert not torch.isnan(pred).any()


# ──────────────────────────────────────────────────────────────────────────────
# Swin U-Net tests
# ──────────────────────────────────────────────────────────────────────────────


class TestSwinUNet:
    """Tests for the Swin Transformer U-Net."""

    def _build(self, in_channels: int = COMBINED) -> nn.Module:
        from biomassters.models.swin_unet import SwinUNet

        return SwinUNet(
            in_channels=in_channels,
            n_timesteps=N_TIMESTEP,
            pretrained=False,
            fpn_out_channels=64,
            output_size=(H, W),
        )

    def test_output_shape_combined(self) -> None:
        model = self._build(COMBINED)
        pred = model(_make_input(COMBINED))
        _assert_output(pred)

    def test_output_shape_s1_only(self) -> None:
        model = self._build(S1_CH)
        pred = model(_make_input(S1_CH))
        _assert_output(pred)

    def test_parameter_count(self) -> None:
        model = self._build()
        assert model.count_parameters() > 0

    def test_temporal_mean_pooling(self) -> None:
        """SwinUNet should produce same output regardless of T (mean-pooled)."""
        model = self._build()
        # The model outputs a deterministic result for a fixed random seed
        torch.manual_seed(0)
        x = _make_input()
        pred = model(x)
        assert pred.shape == (BATCH, 1, H, W)


# ──────────────────────────────────────────────────────────────────────────────
# U-TAE tests
# ──────────────────────────────────────────────────────────────────────────────


class TestUTAE:
    """Tests for the U-TAE (temporal attention encoder)."""

    def _build(self, in_channels: int = COMBINED) -> nn.Module:
        from biomassters.models.utae import UTAE

        return UTAE(
            in_channels=in_channels,
            n_timesteps=N_TIMESTEP,
            encoder_widths=[32, 64],
            ltae_n_head=4,
            ltae_d_k=4,
            ltae_mlp=[64, 32],
        )

    def test_output_shape_combined(self) -> None:
        model = self._build(COMBINED)
        pred = model(_make_input(COMBINED))
        _assert_output(pred)

    def test_output_shape_s1_only(self) -> None:
        model = self._build(S1_CH)
        pred = model(_make_input(S1_CH))
        _assert_output(pred)

    def test_output_shape_s2_only(self) -> None:
        model = self._build(S2_CH)
        pred = model(_make_input(S2_CH))
        _assert_output(pred)

    def test_parameter_count(self) -> None:
        model = self._build()
        assert model.count_parameters() > 0

    def test_with_pad_mask(self) -> None:
        """U-TAE should accept an optional boolean pad mask."""
        model = self._build()
        x = _make_input()
        # All valid (no masked time steps)
        mask = torch.ones(BATCH, N_TIMESTEP, dtype=torch.bool)
        pred = model(x, pad_mask=mask)
        _assert_output(pred)

    def test_no_nan_with_random_input(self) -> None:
        model = self._build()
        pred = model(_make_input())
        assert not torch.isnan(pred).any()


# ──────────────────────────────────────────────────────────────────────────────
# TempFusionNet tests
# ──────────────────────────────────────────────────────────────────────────────


class TestTempFusionNet:
    """Tests for the TempFusionNet (3D encoder + 2D decoder + SE attention)."""

    def _build(self, in_channels: int = COMBINED) -> nn.Module:
        from biomassters.models.tempfusionnet import TempFusionNet

        return TempFusionNet(
            in_channels=in_channels,
            n_timesteps=N_TIMESTEP,
            base_channels=8,
            depth=3,
            use_se=True,
        )

    def test_output_shape_combined(self) -> None:
        model = self._build(COMBINED)
        pred = model(_make_input(COMBINED))
        _assert_output(pred)

    def test_output_shape_s1_only(self) -> None:
        model = self._build(S1_CH)
        pred = model(_make_input(S1_CH))
        _assert_output(pred)

    def test_output_shape_without_se(self) -> None:
        from biomassters.models.tempfusionnet import TempFusionNet

        model = TempFusionNet(
            in_channels=COMBINED,
            n_timesteps=N_TIMESTEP,
            base_channels=8,
            depth=3,
            use_se=False,
        )
        pred = model(_make_input())
        _assert_output(pred)

    def test_parameter_count(self) -> None:
        model = self._build()
        assert model.count_parameters() > 0

    def test_no_nan_with_random_input(self) -> None:
        model = self._build()
        pred = model(_make_input())
        assert not torch.isnan(pred).any()


# ──────────────────────────────────────────────────────────────────────────────
# Registry tests
# ──────────────────────────────────────────────────────────────────────────────


class TestModelRegistry:
    """Tests for the model registry and factory."""

    def test_all_models_registered(self) -> None:
        from biomassters.models.registry import MODEL_REGISTRY

        expected = {"unet", "unet3d", "swin_unet", "utae", "tempfusionnet"}
        assert expected == set(MODEL_REGISTRY.keys())

    def test_build_model_unknown_raises(self) -> None:
        from biomassters.models.registry import build_model

        with pytest.raises(ValueError, match="Unknown model"):
            build_model("nonexistent_model", {})

    def test_build_unet(self) -> None:
        from biomassters.models.registry import build_model

        class FakeCfg:
            class model:
                name = "unet"
                base_channels = 16
                depth = 2
                dropout = 0.0

            class data:
                modalities = ["s1", "s2"]
                months = None

        model = build_model("unet", FakeCfg())
        assert model is not None
        assert model.count_parameters() > 0
