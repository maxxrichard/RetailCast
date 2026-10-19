from pathlib import Path

from typer.testing import CliRunner

from retailcast.cli import app

runner = CliRunner()


def test_synth_and_train(tmp_path: Path, monkeypatch):
    out = tmp_path / "synthetic.parquet"
    r = runner.invoke(
        app, ["synth", "--n-items", "6", "--n-stores", "2", "--n-days", "150", "--out", str(out)]
    )
    assert r.exit_code == 0, r.output
    assert out.exists()

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "extends: lightgbm.yaml\n"
        f"data:\n  source: synthetic\n  path: {out}\n"
        "training:\n  min_history: 40\n"
        "features:\n  lags: [1, 7, 28]\n  rolling_windows: [7]\n"
        "model:\n  horizon_groups: [[1, 28]]\n  early_stopping_rounds: null\n"
        "  quantile_heads: false\n"
        "  params:\n    num_leaves: 7\n    min_data_in_leaf: 5\n"
    )
    import shutil

    from retailcast.config import CONFIG_DIR

    shutil.copy(CONFIG_DIR / "lightgbm.yaml", tmp_path / "lightgbm.yaml")
    shutil.copy(CONFIG_DIR / "default.yaml", tmp_path / "default.yaml")
    fc = tmp_path / "fc.parquet"
    r = runner.invoke(
        app, ["train", "--config", str(cfg), "--out", str(fc), "--num-boost-round", "5"]
    )
    assert r.exit_code == 0, r.output
    assert fc.exists()


def test_backtest_cli(tmp_path: Path):
    out = tmp_path / "synthetic.parquet"
    runner.invoke(
        app, ["synth", "--n-items", "6", "--n-stores", "2", "--n-days", "150", "--out", str(out)]
    )
    import shutil

    from retailcast.config import CONFIG_DIR

    for f in ["default.yaml", "lightgbm.yaml", "backtest.yaml"]:
        shutil.copy(CONFIG_DIR / f, tmp_path / f)
    cfg = tmp_path / "bt.yaml"
    cfg.write_text(
        "extends: backtest.yaml\n"
        f"data:\n  source: synthetic\n  path: {out}\n"
        "horizon: 14\n"
        "backtest:\n  horizon: 14\n  step: 7\n  reconcile: [none, bottom_up]\n"
    )
    r = runner.invoke(
        app,
        [
            "backtest",
            "--config",
            str(cfg),
            "--models",
            "seasonal_naive",
            "--n-folds",
            "2",
            "--out",
            str(tmp_path / "bt"),
        ],
    )
    assert r.exit_code == 0, r.output
    assert (tmp_path / "bt" / "metrics.csv").exists()
