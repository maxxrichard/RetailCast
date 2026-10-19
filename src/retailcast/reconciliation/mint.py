"""Hierarchical forecast reconciliation: bottom-up, OLS, WLS and MinT (Wickramasuriya et al. 2019).

Given base forecasts ``yhat`` for all nodes (n_nodes x h) and the summing matrix ``S``
(n_nodes x n_bottom), every linear reconciliation method computes

    y_tilde = S @ G @ yhat,   with   G = (S' W^{-1} S)^{-1} S' W^{-1}

for a method-specific ``W``:
    * ``ols``:         W = I
    * ``wls_struct``:  W = diag(S @ 1)             (number of bottom series per node)
    * ``wls_var``:     W = diag(var(residuals))
    * ``mint_shrink``: W = shrinkage estimate of the residual covariance (Schäfer & Strimmer)
    * ``mint_sample``: W = sample residual covariance (needs n_obs >> n_nodes)
    * ``bottom_up``:   G = [0 | I]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from retailcast.data.hierarchy import Hierarchy

METHODS = ["none", "bottom_up", "ols", "wls_struct", "wls_var", "mint_shrink", "mint_sample"]


def shrink_covariance(resid: np.ndarray) -> tuple[np.ndarray, float]:
    """Schäfer-Strimmer shrinkage of the residual covariance toward its diagonal.

    ``resid`` is (n_obs, n_nodes). Returns (W, lambda).
    """
    resid = np.asarray(resid, dtype=float)
    resid = resid[~np.isnan(resid).any(axis=1)]
    n, p = resid.shape
    if n < 2:
        raise ValueError("need at least two residual observations for shrinkage")
    cov = np.cov(resid, rowvar=False, ddof=1)
    diag = np.diag(np.diag(cov))
    sd = np.sqrt(np.diag(cov))
    sd = np.where(sd > 0, sd, 1.0)
    x = (resid - resid.mean(axis=0)) / sd
    r = (x.T @ x) / (n - 1)
    # variance of the correlation estimates: sum_i (x_i x_i' - r)^2 = (x^2)'(x^2) - (n-2) r^2
    x2 = x**2
    v = (x2.T @ x2 - (n - 2) * r**2) * n / ((n - 1) ** 3)
    off = ~np.eye(p, dtype=bool)
    denom = np.sum(r[off] ** 2)
    lam = float(np.clip(np.sum(v[off]) / denom, 0.0, 1.0)) if denom > 0 else 1.0
    W = lam * diag + (1 - lam) * cov
    return W, lam


def _regularize(W: np.ndarray, rel: float = 1e-3) -> np.ndarray:
    """Floor the diagonal so nodes with (near-)zero residual variance don't blow up W^-1."""
    d = np.diag(W)
    floor = max(rel * float(np.mean(d[d > 0])) if (d > 0).any() else 1.0, 1e-8)
    return W + floor * np.eye(len(W))


def _projection(S: np.ndarray, W: np.ndarray | None, method: str) -> np.ndarray:
    n_nodes, n_bottom = S.shape
    if method == "bottom_up":
        G = np.zeros((n_bottom, n_nodes))
        G[:, n_nodes - n_bottom :] = np.eye(n_bottom)
        return G
    if method == "ols" or W is None:
        W = np.eye(n_nodes)
    elif method == "wls_struct":
        W = np.diag(S.sum(axis=1))
    if method in ("mint_shrink", "mint_sample"):
        W_inv_S = np.linalg.solve(W, S)
    else:
        W_inv_S = S / np.diag(W)[:, None]
    A = S.T @ W_inv_S  # = S' W^-1 S
    return np.linalg.solve(A, W_inv_S.T)


def reconcile(
    yhat: np.ndarray,
    S: np.ndarray,
    method: str = "mint_shrink",
    residuals: np.ndarray | None = None,
) -> np.ndarray:
    """Reconcile base forecasts ``yhat`` (h x n_nodes) -> coherent forecasts (h x n_nodes)."""
    if method == "none":
        return np.asarray(yhat, dtype=float)
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    S = np.asarray(S, dtype=float)
    W = None
    if method == "wls_var":
        if residuals is None:
            raise ValueError("wls_var needs residuals")
        W = np.diag(np.nanvar(residuals, axis=0, ddof=1) + 1e-8)
    elif method == "mint_shrink":
        if residuals is None:
            raise ValueError("mint_shrink needs residuals")
        W, _ = shrink_covariance(residuals)
        W = _regularize(W)
    elif method == "mint_sample":
        if residuals is None:
            raise ValueError("mint_sample needs residuals")
        W = _regularize(np.cov(np.asarray(residuals, dtype=float), rowvar=False, ddof=1))
    G = _projection(S, W, method)
    out = np.asarray(yhat, dtype=float) @ (S @ G).T
    out[np.abs(out) < 1e-9] = 0.0  # remove floating-point dust around zero
    return out


@dataclass
class Reconciler:
    hierarchy: Hierarchy
    method: str = "mint_shrink"
    residuals: pd.DataFrame | None = None  # date x node_id in-sample residuals

    def _resid_matrix(self) -> np.ndarray | None:
        if self.residuals is None:
            return None
        return self.residuals.reindex(columns=self.hierarchy.node_ids).to_numpy(dtype=float)

    def reconcile_wide(self, base_nodes: pd.DataFrame) -> pd.DataFrame:
        """``base_nodes``: date x node_id base forecasts for every node (aggregated base)."""
        yhat = base_nodes.reindex(columns=self.hierarchy.node_ids).to_numpy(dtype=float)
        out = reconcile(yhat, self.hierarchy.S, self.method, self._resid_matrix())
        return pd.DataFrame(
            np.clip(out, 0, None), index=base_nodes.index, columns=self.hierarchy.node_ids
        )

    def reconcile_bottom(self, base_bottom: pd.DataFrame) -> pd.DataFrame:
        """Reconcile from bottom-level base forecasts only (aggregating them first)."""
        return self.reconcile_wide(self.hierarchy.aggregate(base_bottom))


def reconcile_forecast_frame(
    fc: pd.DataFrame, hierarchy: Hierarchy, method: str, residuals: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Apply reconciliation to a long ForecastFrame (point + quantile columns), returning the
    bottom-level long frame. Quantile columns are reconciled with the same G and re-sorted."""
    if method == "none":
        return fc
    rec = Reconciler(hierarchy, method, residuals)
    cols = ["yhat"] + [c for c in fc.columns if c.startswith("q_")]
    out = fc[["series_id", "date", "h"]].copy()
    for c in cols:
        wide = fc.pivot(index="date", columns="series_id", values=c)
        nodes = rec.reconcile_bottom(wide)
        bottom = nodes[hierarchy.bottom_ids]
        long = bottom.stack().rename(c).reset_index()
        long.columns = ["date", "series_id", c]
        out = out.merge(long, on=["date", "series_id"], how="left")
    qcols = sorted([c for c in out.columns if c.startswith("q_")], key=lambda c: float(c[2:]))
    if len(qcols) > 1:
        out[qcols] = np.maximum.accumulate(out[qcols].to_numpy(dtype=float), axis=1)
    return out
