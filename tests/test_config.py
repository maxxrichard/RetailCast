from retailcast.config import load_config


def test_extends_and_override():
    cfg = load_config("backtest.yaml")
    assert cfg.horizon == 28  # inherited from default.yaml
    assert cfg.backtest.n_folds == 5
    cfg2 = cfg.override(horizon=14, backtest__n_folds=2)
    assert cfg2.horizon == 14 and cfg2.backtest.n_folds == 2
    assert cfg.backtest.n_folds == 5  # original untouched


def test_lightgbm_config_loads():
    cfg = load_config("lightgbm.yaml")
    assert cfg.model.name == "lightgbm"
    assert cfg.model.params.num_leaves == 127
    assert cfg.features.lags[0] == 1
