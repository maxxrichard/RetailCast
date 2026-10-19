"""Loader for the Corporación Favorita grocery dataset into the canonical long panel.

Expected files (Kaggle ``favorita-grocery-sales-forecasting``):
    train.csv, items.csv, stores.csv, oil.csv (optional), holidays_events.csv (optional)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from retailcast.data.schema import validate_long


def load_favorita(
    data_dir: str | Path,
    stores: list[int] | None = None,
    start: str | None = "2016-01-01",
    n_items: int | None = None,
) -> pd.DataFrame:
    data_dir = Path(data_dir)
    usecols = ["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"]
    train = pd.read_csv(data_dir / "train.csv", usecols=usecols, parse_dates=["date"])
    if start:
        train = train[train["date"] >= pd.Timestamp(start)]
    if stores:
        train = train[train["store_nbr"].isin(stores)]
    items = pd.read_csv(data_dir / "items.csv")[["item_nbr", "family"]]
    if n_items:
        keep = train["item_nbr"].drop_duplicates().head(n_items)
        train = train[train["item_nbr"].isin(keep)]

    train["unit_sales"] = train["unit_sales"].clip(lower=0)  # returns are negative in raw data
    train["onpromotion"] = train["onpromotion"].fillna(False).astype("int8")
    train = train.merge(items, on="item_nbr", how="left")

    # Fill missing (item, store, date) combinations with zero demand.
    full = (
        train.set_index(["item_nbr", "store_nbr", "date"])
        .unstack("date")
        .stack("date", future_stack=True)
        .reset_index()
    )
    full["unit_sales"] = full["unit_sales"].fillna(0.0)
    full["onpromotion"] = full["onpromotion"].fillna(0).astype("int8")
    full["family"] = full.groupby("item_nbr")["family"].transform("first")

    out = pd.DataFrame(
        {
            "series_id": full["item_nbr"].astype(str) + "_" + full["store_nbr"].astype(str),
            "item_id": full["item_nbr"].astype(str),
            "cat_id": full["family"].astype(str),
            "store_id": full["store_nbr"].astype(str),
            "date": full["date"],
            "y": full["unit_sales"],
            "promo": full["onpromotion"],
        }
    )
    return validate_long(out)
