"""Promotion features: current flag, upcoming/previous promo distance, promo intensity."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_promo_features(
    df: pd.DataFrame, promo_col: str = "promo", id_col: str = "series_id"
) -> pd.DataFrame:
    if promo_col not in df.columns:
        df[promo_col] = np.int8(0)
    g = df.groupby(id_col, sort=False)[promo_col]
    df["promo_next_7"] = g.transform(
        lambda s: s[::-1].rolling(7, min_periods=1).sum()[::-1]
    ).astype("float32")
    df["promo_prev_7"] = g.transform(lambda s: s.rolling(7, min_periods=1).sum()).astype("float32")
    df["promo_rate_28"] = g.transform(lambda s: s.rolling(28, min_periods=7).mean()).astype(
        "float32"
    )

    def _days_since_promo(s: pd.Series) -> pd.Series:
        idx = s.index.to_series()
        last = idx.where(s > 0).ffill()
        return (idx - last).astype("float32")

    df["days_since_promo"] = g.transform(_days_since_promo).fillna(0).astype("float32")
    return df
