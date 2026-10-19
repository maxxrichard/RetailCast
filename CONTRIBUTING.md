# Contributing

```bash
pip install -e ".[dev]"
pre-commit install
make lint && make test
```

* Every model implements `BaseForecaster.fit/predict` and returns a long `ForecastFrame`.
* New features go into `retailcast.features` **and** `retailcast.spark.features`; the Spark test
  (`tests/test_spark.py`, requires Java + PySpark) asserts both implementations agree.
* Keep lags horizon-safe: anything derived from `y` must respect the `horizon` offset.
* Add a test for every metric or reconciliation change; coherence (`S G S = S`) is a hard invariant.
