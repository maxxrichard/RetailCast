"""Canonical long-format panel schema shared by all loaders.

One row per (series_id, date). ``series_id`` is the bottom-level (article x store) key.
"""

from __future__ import annotations

import pandas as pd

LONG_COLUMNS: dict[str, str] = {
    "series_id": "string",
    "item_id": "string",
    "cat_id": "string",
    "store_id": "string",
    "date": "datetime64[ns]",
    "y": "float32",
    "sell_price": "float32",
    "promo": "int8",
}

REQUIRED = ["series_id", "item_id", "cat_id", "store_id", "date", "y"]


def validate_long(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce dtypes and assert the long-format invariants."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"long panel missing columns: {missing}")
    out = df.copy()
    for col, dtype in LONG_COLUMNS.items():
        if col in out.columns:
            out[col] = out[col].astype(dtype)
    if "sell_price" not in out.columns:
        out["sell_price"] = pd.Series(float("nan"), index=out.index, dtype="float32")
    if "promo" not in out.columns:
        out["promo"] = pd.Series(0, index=out.index, dtype="int8")
    dup = out.duplicated(["series_id", "date"]).sum()
    if dup:
        raise ValueError(f"{dup} duplicate (series_id, date) rows")
    if (out["y"] < 0).any():
        raise ValueError("negative demand found")
    return out.sort_values(["series_id", "date"]).reset_index(drop=True)
