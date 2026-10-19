"""Programmatic backtest example (equivalent to `retailcast backtest`).

    python scripts/run_backtest.py --n-items 100 --n-days 420 --folds 3
"""

from __future__ import annotations

import argparse

from retailcast.data.synthetic import make_synthetic_panel
from retailcast.evaluation.backtest import RollingOriginBacktest
from retailcast.models.baselines import ETSForecaster, SeasonalNaive
from retailcast.models.lightgbm_model import LightGBMDirect

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-items", type=int, default=100)
    ap.add_argument("--n-days", type=int, default=420)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=400)
    args = ap.parse_args()

    panel = make_synthetic_panel(n_items=args.n_items, n_days=args.n_days)
    q = [0.05, 0.25, 0.5, 0.75, 0.95]
    models = {
        "seasonal_naive": lambda: SeasonalNaive(quantiles=q),
        "ets": lambda: ETSForecaster(quantiles=q),
        "lightgbm": lambda: LightGBMDirect(quantiles=q, num_boost_round=args.rounds),
    }
    bt = RollingOriginBacktest(n_folds=args.folds, reconciliations=["none", "bottom_up", "mint_shrink"], quantiles=q)
    res = bt.run(panel, models)
    print(res.leaderboard())
    print(res.vs_baseline())
