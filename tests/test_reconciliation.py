import numpy as np
import pandas as pd
import pytest

from retailcast.data.hierarchy import build_hierarchy
from retailcast.reconciliation.mint import (
    METHODS,
    Reconciler,
    reconcile,
    reconcile_forecast_frame,
    shrink_covariance,
)


@pytest.fixture
def hier(panel):
    meta = panel.drop_duplicates("series_id")[["series_id", "item_id", "cat_id", "store_id"]]
    return build_hierarchy(meta)


def _coherent(nodes: np.ndarray, S: np.ndarray) -> bool:
    bottom = nodes[:, -S.shape[1] :]
    return np.allclose(nodes, bottom @ S.T, atol=1e-6)


def test_bottom_up_keeps_bottom(hier):
    rng = np.random.default_rng(0)
    base = rng.uniform(0, 10, size=(5, hier.n_nodes))
    out = reconcile(base, hier.S, "bottom_up")
    np.testing.assert_allclose(out[:, -hier.n_bottom :], base[:, -hier.n_bottom :])
    assert _coherent(out, hier.S)


@pytest.mark.parametrize("method", [m for m in METHODS if m != "none"])
def test_all_methods_coherent(hier, method):
    rng = np.random.default_rng(1)
    resid = rng.normal(size=(120, hier.n_nodes)) @ (
        0.5 * np.eye(hier.n_nodes) + 0.5 * np.ones((hier.n_nodes, hier.n_nodes)) / hier.n_nodes
    )
    base = rng.uniform(0, 10, size=(7, hier.n_nodes))
    out = reconcile(base, hier.S, method, residuals=resid)
    assert out.shape == base.shape
    assert _coherent(out, hier.S)


def test_reconciliation_is_projection(hier):
    """Reconciling already-coherent forecasts leaves them unchanged (S G S = S)."""
    rng = np.random.default_rng(2)
    bottom = rng.uniform(0, 10, size=(4, hier.n_bottom))
    coherent = bottom @ hier.S.T
    resid = rng.normal(size=(100, hier.n_nodes))
    for method in ["ols", "wls_struct", "wls_var", "mint_shrink"]:
        np.testing.assert_allclose(reconcile(coherent, hier.S, method, resid), coherent, atol=1e-6)


def test_shrinkage_lambda_in_unit_interval():
    rng = np.random.default_rng(3)
    W, lam = shrink_covariance(rng.normal(size=(50, 8)))
    assert 0.0 <= lam <= 1.0
    assert W.shape == (8, 8)
    assert np.allclose(W, W.T)
    # with independent noise, shrinkage toward the diagonal should be strong
    assert lam > 0.5


def test_mint_improves_noisy_aggregate_forecasts(hier):
    """MinT should reduce error when upper-level base forecasts are informative."""
    rng = np.random.default_rng(4)
    true_bottom = rng.uniform(5, 15, size=(28, hier.n_bottom))
    true_nodes = true_bottom @ hier.S.T
    noise_scale = np.sqrt(
        hier.S.sum(axis=1)
    )  # noise grows with sqrt(#bottom series) -> aggregates relatively precise
    base = true_nodes + rng.normal(size=true_nodes.shape) * noise_scale
    resid = rng.normal(size=(200, hier.n_nodes)) * noise_scale
    err_base = np.mean((base - true_nodes) ** 2)
    err_mint = np.mean((reconcile(base, hier.S, "mint_shrink", resid) - true_nodes) ** 2)
    err_bu = np.mean((reconcile(base, hier.S, "bottom_up") - true_nodes) ** 2)
    assert err_mint < err_base
    assert err_mint < err_bu


def test_reconcile_forecast_frame_quantiles_monotone(hier, panel):
    dates = pd.date_range("2020-01-01", periods=3)
    rows = []
    rng = np.random.default_rng(5)
    for sid in hier.bottom_ids:
        for h, d in enumerate(dates, 1):
            m = rng.uniform(1, 5)
            rows.append(
                {
                    "series_id": sid,
                    "date": d,
                    "h": h,
                    "yhat": m,
                    "q_0.05": m * 0.5,
                    "q_0.50": m,
                    "q_0.95": m * 1.5,
                }
            )
    fc = pd.DataFrame(rows)
    out = reconcile_forecast_frame(fc, hier, "wls_struct")
    assert len(out) == len(fc)
    assert (out["q_0.05"] <= out["q_0.50"] + 1e-9).all() and (
        out["q_0.50"] <= out["q_0.95"] + 1e-9
    ).all()
    rec = Reconciler(hier, "ols")
    wide = fc.pivot(index="date", columns="series_id", values="yhat")
    nodes = rec.reconcile_bottom(wide)
    assert _coherent(nodes.to_numpy(), hier.S)
