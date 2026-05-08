import numpy as np
import pytest

pytest.importorskip("enn")

from analysis.fitting_time import fitting_time as ft
from analysis.fitting_time.fitting_time import fit_enn


def test_fit_enn_posterior_chunking_matches_one_shot(monkeypatch) -> None:
    """Large ``x_test`` should match one-shot posterior; chunking only caps peak memory."""
    rng = np.random.default_rng(42)
    n, d = 120, 3
    x_np = rng.uniform(0.0, 1.0, size=(n, d)).astype(np.float64)
    y_np = (x_np[:, :1] ** 2 + 0.01 * rng.standard_normal((n, 1))).astype(np.float64)
    x_te = rng.uniform(0.0, 1.0, size=(n, d)).astype(np.float64)

    monkeypatch.setattr(ft, "_ENN_POSTERIOR_CHUNK", 10**9)
    _, y_full, v_full = fit_enn(x_np, y_np, x_te)
    monkeypatch.setattr(ft, "_ENN_POSTERIOR_CHUNK", 25)
    _, y_part, v_part = fit_enn(x_np, y_np, x_te)
    np.testing.assert_allclose(y_full, y_part, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(v_full, v_part, rtol=1e-12, atol=1e-12)
