import pytest

pyspark = pytest.importorskip("pyspark")


@pytest.fixture(scope="module")
def spark():
    from pyspark.sql import SparkSession

    s = (
        SparkSession.builder.master("local[2]")
        .appName("retailcast-test")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    yield s
    s.stop()


def test_spark_features_match_pandas(spark, panel):
    import numpy as np
    import pandas as pd

    from retailcast.features.pipeline import FeaturePipeline
    from retailcast.spark.features import build_feature_table

    sdf = spark.createDataFrame(panel.assign(date=panel["date"].dt.date))
    out = build_feature_table(sdf, lags=[1, 7], windows=[7], horizon_offset=0).toPandas()
    out["date"] = pd.to_datetime(out["date"])
    ref = FeaturePipeline(lags=[1, 7], rolling_windows=[7], rolling_stats=["mean"]).transform(panel)
    m = out.merge(ref, on=["series_id", "date"], suffixes=("_s", "_p"))
    np.testing.assert_allclose(m["lag_7_s"].to_numpy(), m["lag_7_p"].to_numpy(), equal_nan=True)
    np.testing.assert_allclose(m["dow_s"].to_numpy(), m["dow_p"].to_numpy())
    np.testing.assert_allclose(m["promo_next_7_s"].to_numpy(), m["promo_next_7_p"].to_numpy())
    ok = m["roll_mean_7_s"].notna() & m["roll_mean_7_p"].notna()
    np.testing.assert_allclose(m.loc[ok, "roll_mean_7_s"], m.loc[ok, "roll_mean_7_p"], rtol=1e-4)
