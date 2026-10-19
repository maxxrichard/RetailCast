"""Synthetic retail panel with hierarchy, seasonality, price elasticity and promotions.

Used for tests, demos and CI. The generator is deliberately structured like M5:
stores x categories x items, intermittent demand, weekly/annual seasonality, price
changes with a negative elasticity and promotional uplift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from retailcast.data.schema import validate_long


def make_synthetic_panel(
    n_items: int = 60,
    n_stores: int = 3,
    n_cats: int = 3,
    n_days: int = 365,
    start: str = "2015-01-01",
    seed: int = 7,
    elasticity: float = -1.3,
    promo_uplift: float = 0.45,
    promo_rate: float = 0.06,
    zero_inflation: float = 0.35,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days, freq="D")
    t = np.arange(n_days)
    dow = dates.dayofweek.values
    doy = dates.dayofyear.values

    weekly = np.array([0.95, 0.9, 0.92, 0.98, 1.1, 1.3, 1.25])[dow]
    annual = 1.0 + 0.25 * np.sin(2 * np.pi * (doy - 330) / 365.25)
    trend = 1.0 + 0.0004 * t

    rows = []
    cats = [f"CAT_{c}" for c in range(n_cats)]
    for item in range(n_items):
        item_id = f"ITEM_{item:04d}"
        cat_id = cats[item % n_cats]
        base_level = rng.gamma(shape=1.5, scale=2.0)
        base_price = rng.uniform(2.0, 25.0)
        item_zero_p = rng.beta(2, 4) * zero_inflation * 2
        for store in range(n_stores):
            store_id = f"STORE_{store}"
            store_mult = rng.uniform(0.6, 1.5)
            # price path: piecewise constant with occasional changes
            price = np.full(n_days, base_price, dtype=float)
            for change in rng.choice(n_days, size=max(1, n_days // 120), replace=False):
                price[change:] *= rng.uniform(0.85, 1.15)
            promo = (rng.uniform(size=n_days) < promo_rate).astype(np.int8)
            # promo runs 3 days
            promo = np.maximum.reduce([promo, np.roll(promo, 1), np.roll(promo, 2)])
            rel_price = price / base_price
            mu = (
                base_level
                * store_mult
                * weekly
                * annual
                * trend
                * rel_price**elasticity
                * (1.0 + promo_uplift * promo)
            )
            lam = rng.gamma(shape=4.0, scale=mu / 4.0)
            y = rng.poisson(lam).astype(float)
            y[rng.uniform(size=n_days) < item_zero_p] = 0.0
            rows.append(
                pd.DataFrame(
                    {
                        "series_id": f"{item_id}_{store_id}",
                        "item_id": item_id,
                        "cat_id": cat_id,
                        "store_id": store_id,
                        "date": dates,
                        "y": y,
                        "sell_price": price.round(2),
                        "promo": promo,
                    }
                )
            )
    return validate_long(pd.concat(rows, ignore_index=True))
