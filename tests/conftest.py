import pandas as pd
import pytest

from retailcast.data.synthetic import make_synthetic_panel

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]


@pytest.fixture(scope="session")
def panel() -> pd.DataFrame:
    """Small synthetic panel: 12 items x 2 stores x 3 cats, 200 days."""
    return make_synthetic_panel(n_items=12, n_stores=2, n_cats=3, n_days=200, seed=3)


@pytest.fixture(scope="session")
def split(panel):
    origin = panel["date"].max() - pd.Timedelta(days=28)
    train = panel[panel["date"] <= origin]
    test = panel[panel["date"] > origin]
    return train, test, origin
