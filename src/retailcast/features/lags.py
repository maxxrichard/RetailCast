"""Lag and rolling-window features on the target.

With ``horizon_safe=True`` every lag is shifted by at least ``horizon`` days so that the
feature is observable at forecast origin for a direct multi-horizon model.
"""

from __future__ import annotations

import pandas as pd


def _shift(df: pd.DataFrame, col: str, by: int, id_col: str) -> pd.Series:
    return df.groupby(id_col, sort=False)[col].shift(by)


def add_lag_features(
    df: pd.DataFrame,
    lags: list[int],
    target_col: str = "y",
    id_col: str = "series_id",
    horizon: int = 0,
) -> pd.DataFrame:
    """Add ``lag_k`` columns. ``horizon`` offsets every lag (horizon-safe lags)."""
    for lag in lags:
        shift = lag + horizon
        df[f"lag_{lag}"] = _shift(df, target_col, shift, id_col).astype("float32")
    return df


def add_rolling_features(
    df: pd.DataFrame,
    windows: list[int],
    stats: list[str],
    target_col: str = "y",
    id_col: str = "series_id",
    horizon: int = 0,
) -> pd.DataFrame:
    """Add rolling statistics over the last ``w`` observed days (ending at origin)."""
    base = df.groupby(id_col, sort=False)[target_col].shift(1 + horizon)
    grp = base.groupby(df[id_col], sort=False)
    for w in windows:
        roll = grp.rolling(w, min_periods=max(1, w // 2))
        for stat in stats:
            s = getattr(roll, stat)()
            s.index = s.index.droplevel(0)
            df[f"roll_{stat}_{w}"] = s.astype("float32")
    # intermittency features
    nonzero = (base > 0).astype("float32")
    nz_grp = nonzero.groupby(df[id_col], sort=False)
    nzr = nz_grp.rolling(28, min_periods=7).mean()
    nzr.index = nzr.index.droplevel(0)
    df["nonzero_rate_28"] = nzr.astype("float32")

    # days since last sale, computed on the shifted (observable) series
    def _days_since(s: pd.Series) -> pd.Series:
        idx = s.index.to_series()
        last = idx.where(s > 0).ffill()
        return (idx - last).astype("float32")

    df["days_since_sale"] = (
        base.groupby(df[id_col], sort=False).transform(_days_since).fillna(0).astype("float32")
    )
    return df
