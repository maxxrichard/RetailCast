"""Price features: relative price vs. item/category history and recent price changes.

Prices for the next 28 days are typically known from the price list, so they are treated
as future-known covariates (no horizon shift).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_price_features(
    df: pd.DataFrame,
    price_col: str = "sell_price",
    id_col: str = "series_id",
) -> pd.DataFrame:
    if price_col not in df.columns or df[price_col].isna().all():
        for c in [
            "price_rel_item",
            "price_rel_cat",
            "price_chg_7",
            "price_chg_28",
            "price_rank_cat",
        ]:
            df[c] = np.float32(0.0)
        return df
    p = df[price_col].astype("float32")
    g = df.groupby(id_col, sort=False)[price_col]
    df["price_rel_item"] = (p / g.transform("mean")).astype("float32")
    cat_mean = df.groupby(["cat_id", "store_id", "date"], sort=False)[price_col].transform("mean")
    df["price_rel_cat"] = (p / cat_mean).astype("float32")
    df["price_chg_7"] = (p / g.shift(7) - 1.0).astype("float32")
    df["price_chg_28"] = (p / g.shift(28) - 1.0).astype("float32")
    df["price_rank_cat"] = (
        df.groupby(["cat_id", "store_id", "date"], sort=False)[price_col]
        .rank(pct=True)
        .astype("float32")
    )
    df["log_price"] = np.log1p(p).astype("float32")
    return df
