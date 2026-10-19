"""Command line interface: ``retailcast --help``."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rich import print as rprint
from rich.table import Table

from retailcast.config import load_config

app = typer.Typer(
    help="RetailCast: hierarchical demand forecasting for e-commerce assortments.",
    no_args_is_help=True,
)


def _load_panel(cfg) -> pd.DataFrame:
    src = cfg.data.source
    if src == "synthetic":
        path = Path(cfg.data.path)
        if path.exists():
            return pd.read_parquet(path)
        from retailcast.data.synthetic import make_synthetic_panel

        return make_synthetic_panel()
    if src == "m5":
        from retailcast.data.m5 import load_m5

        return load_m5(cfg.data.m5_dir)
    if src == "favorita":
        from retailcast.data.favorita import load_favorita

        return load_favorita(cfg.data.favorita_dir)
    raise typer.BadParameter(f"unknown data source {src}")


def _print_df(df: pd.DataFrame, title: str) -> None:
    t = Table(title=title)
    t.add_column(df.index.name or "")
    for c in df.columns:
        t.add_column(str(c), justify="right")
    for idx, row in df.iterrows():
        t.add_row(str(idx), *[f"{v:.4f}" if isinstance(v, float) else str(v) for v in row])
    rprint(t)


@app.command()
def synth(
    n_items: int = typer.Option(200, help="items per category block"),
    n_stores: int = 3,
    n_days: int = 400,
    out: Path = Path("data/processed/synthetic.parquet"),
    seed: int = 7,
) -> None:
    """Generate a synthetic hierarchical retail panel."""
    from retailcast.data.synthetic import make_synthetic_panel

    panel = make_synthetic_panel(n_items=n_items, n_stores=n_stores, n_days=n_days, seed=seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out, index=False)
    rprint(
        f"[green]wrote[/green] {len(panel):,} rows / "
        f"{panel['series_id'].nunique():,} series -> {out}"
    )


@app.command()
def features(
    config: Path = Path("configs/default.yaml"),
    out: Path = Path("data/processed/features.parquet"),
    horizon: int = 0,
) -> None:
    """Build the pandas feature table from the configured panel."""
    from retailcast.features.pipeline import build_features

    cfg = load_config(config)
    panel = _load_panel(cfg)
    table, fp = build_features(panel, cfg.features.to_dict(), horizon=horizon)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    rprint(
        f"[green]wrote[/green] {table.shape[0]:,} rows x {len(fp.feature_names_)} features -> {out}"
    )


@app.command()
def backtest(
    config: Path = Path("configs/backtest.yaml"),
    models: str | None = typer.Option(
        None, help="comma-separated override, e.g. seasonal_naive,lightgbm"
    ),
    n_folds: int | None = None,
    num_boost_round: int | None = None,
    out: Path | None = None,
) -> None:
    """Run the rolling-origin backtest with hierarchical evaluation and reconciliation."""
    from retailcast.evaluation.backtest import RollingOriginBacktest
    from retailcast.models.registry import make_model
    from retailcast.tracking import log_metrics, log_params, tracked_run

    cfg = load_config(config)
    panel = _load_panel(cfg)
    bt_cfg = cfg.backtest
    names = models.split(",") if models else list(bt_cfg.models)
    lgb_cfg = load_config("lightgbm.yaml").model.to_dict()
    model_kwargs = {
        "lightgbm": {
            "horizon_groups": lgb_cfg["horizon_groups"],
            "params": {
                **lgb_cfg["params"],
                "objective": lgb_cfg["objective"],
                "tweedie_variance_power": lgb_cfg["tweedie_variance_power"],
            },
            "num_boost_round": num_boost_round or lgb_cfg["num_boost_round"],
            "early_stopping_rounds": lgb_cfg["early_stopping_rounds"],
            "quantile_heads": lgb_cfg["quantile_heads"],
            "feature_cfg": cfg.features.to_dict(),
            "sample_weight": cfg.training.sample_weight,
            "min_history": cfg.training.min_history,
        }
    }
    factories = {
        n: (lambda n=n: make_model(n, cfg.horizon, list(cfg.quantiles), **model_kwargs.get(n, {})))
        for n in names
    }
    bt = RollingOriginBacktest(
        horizon=bt_cfg.horizon,
        n_folds=n_folds or bt_cfg.n_folds,
        step=bt_cfg.step,
        reconciliations=list(bt_cfg.reconcile),
        quantiles=list(cfg.quantiles),
    )
    ml = cfg.get("mlflow")
    with tracked_run(
        bool(ml and ml.enabled),
        ml.experiment if ml else None,
        ml.tracking_uri if ml else None,
        run_name="backtest",
    ) as run:
        log_params(run, cfg.to_dict())
        res = bt.run(panel, factories)
        lb = res.leaderboard()
        _print_df(lb.reset_index().set_index("model"), "Backtest leaderboard (mean over folds)")
        _print_df(
            res.vs_baseline(), "Relative to seasonal naive (WRMSSE, no reconciliation)"
        ) if "seasonal_naive" in names else None
        best = lb.iloc[0]
        log_metrics(run, {f"best_{k}": v for k, v in best.items()})
    out_dir = out or Path(bt_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res.metrics.to_csv(out_dir / "metrics.csv", index=False)
    res.forecasts.to_parquet(out_dir / "forecasts.parquet", index=False)
    res.timings.to_csv(out_dir / "timings.csv", index=False)
    rprint(f"[green]saved[/green] metrics/forecasts/timings -> {out_dir}")


@app.command()
def ablation(
    config: Path = Path("configs/backtest.yaml"), n_folds: int = 3, num_boost_round: int = 300
) -> None:
    """Incremental value of price and promotion data (LightGBM, cumulative feature groups)."""
    from retailcast.evaluation.ablation import AblationStudy
    from retailcast.evaluation.backtest import RollingOriginBacktest

    cfg = load_config(config)
    panel = _load_panel(cfg)
    bt = RollingOriginBacktest(
        horizon=cfg.horizon,
        n_folds=n_folds,
        step=cfg.backtest.step,
        reconciliations=["none"],
        quantiles=list(cfg.quantiles),
        verbose=False,
    )
    study = AblationStudy(
        bt,
        model_kwargs={
            "horizon": cfg.horizon,
            "quantiles": list(cfg.quantiles),
            "num_boost_round": num_boost_round,
        },
    )
    _print_df(study.run(panel), "Incremental value of data sources")


@app.command()
def train(
    config: Path = Path("configs/lightgbm.yaml"),
    out: Path = Path("artifacts/forecast.parquet"),
    num_boost_round: int | None = None,
) -> None:
    """Fit LightGBM on the full panel and write the next-horizon forecast (point + quantiles)."""
    from retailcast.models.lightgbm_model import LightGBMDirect

    cfg = load_config(config)
    panel = _load_panel(cfg)
    m = cfg.model.to_dict()
    model = LightGBMDirect(
        horizon=cfg.horizon,
        horizon_groups=m["horizon_groups"],
        params={
            **m["params"],
            "objective": m["objective"],
            "tweedie_variance_power": m["tweedie_variance_power"],
        },
        num_boost_round=num_boost_round or m["num_boost_round"],
        early_stopping_rounds=m["early_stopping_rounds"],
        quantiles=list(cfg.quantiles),
        quantile_heads=m["quantile_heads"],
        feature_cfg=cfg.features.to_dict(),
        sample_weight=cfg.training.sample_weight,
        min_history=cfg.training.min_history,
    ).fit(panel)
    origin = panel["date"].max()
    last = panel.sort_values("date").groupby("series_id").tail(1)
    future = last.loc[last.index.repeat(cfg.horizon)].copy()
    future["date"] = list(pd.date_range(origin + pd.Timedelta(days=1), periods=cfg.horizon)) * len(
        last
    )
    fc = model.predict(future.drop(columns=["y"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    fc.to_parquet(out, index=False)
    _print_df(model.feature_importance().head(15).set_index("feature"), "Top features (gain)")
    rprint(f"[green]wrote[/green] {len(fc):,} forecast rows -> {out}")


if __name__ == "__main__":
    app()
