"""Forecast accuracy metrics.

All point metrics operate on wide frames: index = horizon steps/dates, columns = series.
``train`` is the in-sample history (same columns) needed by scale-dependent metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _align(a: pd.DataFrame, b: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    cols = a.columns.intersection(b.columns)
    return a[cols].to_numpy(dtype=float), b[cols].to_numpy(dtype=float)


def wape(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    """Weighted absolute percentage error = sum|e| / sum|y| over all series and steps."""
    yt, yp = _align(y_true, y_pred)
    denom = np.abs(yt).sum()
    return float(np.abs(yt - yp).sum() / denom) if denom > 0 else float("nan")


def rmsse(y_true: pd.DataFrame, y_pred: pd.DataFrame, train: pd.DataFrame) -> pd.Series:
    """Root mean squared scaled error per series (M5 definition, seasonal lag 1)."""
    cols = y_true.columns.intersection(y_pred.columns).intersection(train.columns)
    yt = y_true[cols].to_numpy(dtype=float)
    yp = y_pred[cols].to_numpy(dtype=float)
    tr = train[cols].to_numpy(dtype=float)
    # scale: mean squared one-step naive error on history, after the first non-zero sale
    scale = np.empty(len(cols))
    for j in range(len(cols)):
        s = tr[:, j]
        nz = np.nonzero(s)[0]
        s = s[nz[0] :] if len(nz) else s
        d = np.diff(s)
        scale[j] = np.mean(d**2) if len(d) else np.nan
    mse = np.mean((yt - yp) ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.sqrt(mse / scale)
    return pd.Series(out, index=cols, name="rmsse")


def series_weights(
    train: pd.DataFrame, prices: pd.DataFrame | None = None, last_n: int = 28
) -> pd.Series:
    """Revenue share of each series over the last ``last_n`` training days (M5 weighting).

    If ``prices`` is None, unit volume is used instead of revenue.
    """
    tail = train.tail(last_n)
    if prices is not None:
        rev = (tail * prices.reindex_like(tail).fillna(0.0)).sum()
    else:
        rev = tail.sum()
    total = rev.sum()
    w = rev / total if total > 0 else pd.Series(1.0 / len(rev), index=rev.index)
    return w.astype(float)


def wrmsse(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    train: pd.DataFrame,
    weights: pd.Series | None = None,
    prices: pd.DataFrame | None = None,
) -> float:
    """Weighted RMSSE across series at a single aggregation level.

    For the full M5 metric call this per level and average the level scores
    (see :func:`wrmsse_hierarchical`).
    """
    r = rmsse(y_true, y_pred, train)
    if weights is None:
        weights = series_weights(train[r.index], prices)
    weights = weights.reindex(r.index).fillna(0.0)
    mask = r.notna() & np.isfinite(r)
    if weights[mask].sum() == 0:
        return float("nan")
    return float((r[mask] * weights[mask]).sum() / weights[mask].sum())


def wrmsse_hierarchical(
    y_true_nodes: pd.DataFrame,
    y_pred_nodes: pd.DataFrame,
    train_nodes: pd.DataFrame,
    level_of: dict[str, str],
    prices_nodes: pd.DataFrame | None = None,
) -> dict[str, float]:
    """WRMSSE per hierarchy level plus the equally-weighted average across levels."""
    levels = sorted(set(level_of.values()), key=lambda lv: list(level_of.values()).index(lv))
    out: dict[str, float] = {}
    for lv in levels:
        cols = [n for n in y_true_nodes.columns if level_of.get(n) == lv]
        if not cols:
            continue
        p = prices_nodes[cols] if prices_nodes is not None else None
        out[lv] = wrmsse(y_true_nodes[cols], y_pred_nodes[cols], train_nodes[cols], prices=p)
    out["mean"] = float(np.nanmean([v for k, v in out.items()]))
    return out


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Mean pinball (quantile) loss at level q."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    diff = yt - yp
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def scaled_pinball_loss(
    y_true: pd.DataFrame, quantile_preds: dict[float, pd.DataFrame], train: pd.DataFrame
) -> dict[float, float]:
    """M5-uncertainty style pinball loss scaled by the naive in-sample MAE, mean over series."""
    cols = y_true.columns
    tr = train[cols].to_numpy(dtype=float)
    scale = np.mean(np.abs(np.diff(tr, axis=0)), axis=0)
    scale = np.where(scale > 0, scale, np.nan)
    out: dict[float, float] = {}
    for q, pred in quantile_preds.items():
        yt = y_true[cols].to_numpy(dtype=float)
        yp = pred[cols].to_numpy(dtype=float)
        diff = yt - yp
        pl = np.mean(np.maximum(q * diff, (q - 1) * diff), axis=0)
        out[q] = float(np.nanmean(pl / scale))
    return out


def coverage(y_true: pd.DataFrame, lower: pd.DataFrame, upper: pd.DataFrame) -> float:
    """Empirical coverage of the interval [lower, upper]."""
    yt, lo = _align(y_true, lower)
    _, up = _align(y_true, upper)
    return float(np.mean((yt >= lo) & (yt <= up)))


def interval_width(lower: pd.DataFrame, upper: pd.DataFrame) -> float:
    lo, up = _align(lower, upper)
    return float(np.mean(up - lo))


def summarize_point(
    y_true: pd.DataFrame, y_pred: pd.DataFrame, train: pd.DataFrame
) -> dict[str, float]:
    return {
        "wrmsse": wrmsse(y_true, y_pred, train),
        "wape": wape(y_true, y_pred),
        "rmsse_mean": float(np.nanmean(rmsse(y_true, y_pred, train))),
    }
