"""Tests for evaluation metrics.

Validates metric correctness against known analytical values, masked
computation, and R² bounds.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestMaskedRMSE:
    """Tests for MaskedRMSE metric."""

    def _get(self):
        from biomassters.metrics.metrics import MaskedRMSE

        return MaskedRMSE(ignore_threshold=0.5)

    def test_perfect_prediction(self) -> None:
        """RMSE is 0 for perfect predictions."""
        metric = self._get()
        t = torch.ones(2, 1, 4, 4) * 100.0
        metric.update(t.clone(), t)
        assert metric.compute().item() == pytest.approx(0.0, abs=1e-4)

    def test_known_value(self) -> None:
        """RMSE matches analytical value for constant offset."""
        metric = self._get()
        # All valid pixels (target = 10), prediction off by 3 → RMSE = 3
        target = torch.full((1, 1, 4, 4), 10.0)
        pred = torch.full((1, 1, 4, 4), 13.0)
        metric.update(pred, target)
        assert metric.compute().item() == pytest.approx(3.0, rel=1e-4)

    def test_masks_background_pixels(self) -> None:
        """Zero-target pixels are excluded from RMSE computation."""
        metric = self._get()
        target = torch.zeros(1, 1, 4, 4)
        target[0, 0, 2:, 2:] = 100.0  # valid pixels in bottom-right quadrant
        pred = target.clone() + 5.0  # offset only on valid pixels
        metric.update(pred, target)
        assert metric.compute().item() == pytest.approx(5.0, rel=1e-3)

    def test_accumulates_across_batches(self) -> None:
        """RMSE correctly accumulates over multiple update() calls."""
        metric = self._get()
        t = torch.ones(1, 1, 2, 2) * 50.0
        p = torch.ones(1, 1, 2, 2) * 54.0  # error = 4 each batch
        metric.update(p, t)
        metric.update(p, t)
        assert metric.compute().item() == pytest.approx(4.0, rel=1e-4)

    def test_reset(self) -> None:
        """After reset, compute() returns NaN (no data)."""
        metric = self._get()
        t = torch.ones(1, 1, 4, 4) * 10.0
        metric.update(t, t)
        metric.reset()
        result = metric.compute()
        assert torch.isnan(result)

    def test_all_background_returns_nan(self) -> None:
        """If all pixels are background (target=0), result is NaN."""
        metric = self._get()
        t = torch.zeros(1, 1, 4, 4)
        p = torch.ones(1, 1, 4, 4) * 50.0
        metric.update(p, t)
        result = metric.compute()
        assert torch.isnan(result)


class TestMaskedMAE:
    """Tests for MaskedMAE metric."""

    def _get(self):
        from biomassters.metrics.metrics import MaskedMAE

        return MaskedMAE(ignore_threshold=0.5)

    def test_perfect_prediction(self) -> None:
        metric = self._get()
        t = torch.ones(1, 1, 4, 4) * 50.0
        metric.update(t.clone(), t)
        assert metric.compute().item() == pytest.approx(0.0, abs=1e-4)

    def test_known_value(self) -> None:
        metric = self._get()
        # Prediction is always +5 above target
        t = torch.full((1, 1, 8, 8), 20.0)
        p = torch.full((1, 1, 8, 8), 25.0)
        metric.update(p, t)
        assert metric.compute().item() == pytest.approx(5.0, rel=1e-4)

    def test_asymmetric_error(self) -> None:
        """MAE is symmetric for + and - errors."""
        metric = self._get()
        t = torch.full((1, 1, 2, 2), 10.0)
        p = torch.tensor([[[[5.0, 15.0], [0.5, 20.0]]]])  # errors: 5, 5, 9.5, 10
        p[0, 0, 1, 0] = 0.5 + 10.0  # = 10.5 → error = 0.5  (avoid zero-target mask)
        metric.update(p, t)
        result = metric.compute().item()
        assert result > 0


class TestMaskedR2:
    """Tests for MaskedR2 (coefficient of determination)."""

    def _get(self):
        from biomassters.metrics.metrics import MaskedR2

        return MaskedR2(ignore_threshold=0.5)

    def test_perfect_prediction_r2_is_one(self) -> None:
        metric = self._get()
        t = torch.arange(16, dtype=torch.float32).view(1, 1, 4, 4) + 1.0
        metric.update(t.clone(), t)
        assert metric.compute().item() == pytest.approx(1.0, abs=1e-4)

    def test_constant_prediction_r2_is_zero_or_less(self) -> None:
        """Predicting the mean gives R² ≈ 0; worse than mean gives R² < 0."""
        metric = self._get()
        t = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
        mean_t = t.mean()
        p = torch.full_like(t, mean_t.item())  # predict mean → R² ≈ 0
        metric.update(p, t)
        r2 = metric.compute().item()
        assert r2 == pytest.approx(0.0, abs=1e-3)

    def test_r2_range(self) -> None:
        """R² should be ≤ 1.0 for any inputs."""
        metric = self._get()
        t = torch.rand(2, 1, 8, 8) * 200.0 + 1.0  # non-zero targets
        p = torch.rand(2, 1, 8, 8) * 200.0
        metric.update(p, t)
        r2 = metric.compute().item()
        assert r2 <= 1.0 + 1e-5


class TestMaskedBias:
    """Tests for MaskedBias (mean systematic error)."""

    def _get(self):
        from biomassters.metrics.metrics import MaskedBias

        return MaskedBias(ignore_threshold=0.5)

    def test_positive_bias(self) -> None:
        metric = self._get()
        t = torch.ones(1, 1, 4, 4) * 50.0
        p = torch.ones(1, 1, 4, 4) * 60.0
        metric.update(p, t)
        assert metric.compute().item() == pytest.approx(10.0, rel=1e-4)

    def test_negative_bias(self) -> None:
        metric = self._get()
        t = torch.ones(1, 1, 4, 4) * 50.0
        p = torch.ones(1, 1, 4, 4) * 40.0
        metric.update(p, t)
        assert metric.compute().item() == pytest.approx(-10.0, rel=1e-4)

    def test_zero_bias(self) -> None:
        metric = self._get()
        t = torch.ones(1, 1, 4, 4) * 50.0
        metric.update(t.clone(), t)
        assert metric.compute().item() == pytest.approx(0.0, abs=1e-4)


class TestMaskedRelRMSE:
    """Tests for MaskedRelRMSE (relative RMSE in %)."""

    def _get(self):
        from biomassters.metrics.metrics import MaskedRelRMSE

        return MaskedRelRMSE(ignore_threshold=0.5)

    def test_known_value(self) -> None:
        """Rel.RMSE = (RMSE / mean_target) * 100."""
        metric = self._get()
        t = torch.full((1, 1, 4, 4), 100.0)
        p = torch.full((1, 1, 4, 4), 110.0)  # RMSE = 10, mean_t = 100 → 10%
        metric.update(p, t)
        assert metric.compute().item() == pytest.approx(10.0, rel=1e-4)

    def test_perfect_prediction_zero(self) -> None:
        metric = self._get()
        t = torch.ones(1, 1, 4, 4) * 50.0
        metric.update(t.clone(), t)
        assert metric.compute().item() == pytest.approx(0.0, abs=1e-4)


class TestMetricsCollection:
    """Tests for the MetricsCollection container."""

    def _get(self, prefix: str = "val/"):
        from biomassters.metrics.metrics import MetricsCollection

        return MetricsCollection(prefix=prefix)

    def test_compute_returns_all_keys(self) -> None:
        col = self._get("val/")
        t = torch.ones(1, 1, 4, 4) * 50.0
        col.update(t.clone(), t)
        results = col.compute()
        expected_keys = {"val/rmse", "val/mae", "val/r2", "val/bias", "val/rel_rmse"}
        assert expected_keys == set(results.keys())

    def test_reset_clears_state(self) -> None:
        col = self._get()
        t = torch.ones(1, 1, 4, 4) * 50.0
        col.update(t.clone(), t)
        col.reset()
        results = col.compute()
        # After reset, all metrics should be NaN
        for v in results.values():
            assert torch.isnan(v)

    def test_no_prefix(self) -> None:
        from biomassters.metrics.metrics import MetricsCollection

        col = MetricsCollection(prefix="")
        t = torch.ones(1, 1, 4, 4) * 50.0
        col.update(t.clone(), t)
        results = col.compute()
        assert "rmse" in results
