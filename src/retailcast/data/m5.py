"""Loader for the M5 Forecasting (Walmart) dataset into the canonical long panel.

Expected files (Kaggle ``m5-forecasting-accuracy``):
    sales_train_evaluation.csv (or sales_train_validation.csv), calendar.csv, sell_prices.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from retailcast.data.schema import validate_long

SNAP_COLS = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}


def load_m5(
    data_dir: str | Path,
    sales_file: str = "sales_train_evaluation.csv",
    stores: list[str] | None = None,
    n_items: int | None = None,
    start_day: int = 1,
) -> pd.DataFrame:
    data_dir = Path(data_dir)
    sales = pd.read_csv(data_dir / sales_file)
    cal = pd.read_csv(data_dir / "calendar.csv", parse_dates=["date"])
    prices = pd.read_csv(data_dir / "sell_prices.csv")

    if stores:
        sales = sales[sales["store_id"].isin(stores)]
    if n_items:
        sales = sales.groupby("store_id", group_keys=False).head(n_items)

    day_cols = [c for c in sales.columns if c.startswith("d_")]
    day_cols = [c for c in day_cols if int(c.split("_")[1]) >= start_day]
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    long = sales.melt(id_vars=id_cols, value_vars=day_cols, var_name="d", value_name="y")
    long = long.merge(
        cal[["d", "date", "wm_yr_wk", "event_name_1", "snap_CA", "snap_TX", "snap_WI"]], on="d"
    )
    long = long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    snap = np.zeros(len(long), dtype=np.int8)
    for state, col in SNAP_COLS.items():
        mask = long["state_id"].values == state
        snap[mask] = long.loc[mask, col].values.astype(np.int8)
    long["promo"] = ((snap == 1) | long["event_name_1"].notna()).astype(np.int8)
    long["series_id"] = long["item_id"] + "_" + long["store_id"]
    out = long[["series_id", "item_id", "cat_id", "store_id", "date", "y", "sell_price", "promo"]]
    # M5 has no price before the product was listed -> no sales; drop those rows.
    out = out[out["sell_price"].notna()]
    return validate_long(out)
