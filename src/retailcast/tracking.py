"""Thin MLflow helper that degrades gracefully when MLflow is not installed/enabled."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

try:
    import mlflow

    HAS_MLFLOW = True
except Exception:  # pragma: no cover
    HAS_MLFLOW = False


@contextlib.contextmanager
def tracked_run(
    enabled: bool,
    experiment: str | None = None,
    tracking_uri: str | None = None,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
) -> Iterator[Any]:
    if not (enabled and HAS_MLFLOW):
        yield None
        return
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    if experiment:
        mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        yield run


def log_params(run: Any, params: dict[str, Any], prefix: str = "") -> None:
    if run is None:
        return
    flat = {}

    def _walk(d: dict[str, Any], p: str) -> None:
        for k, v in d.items():
            key = f"{p}{k}"
            if isinstance(v, dict):
                _walk(v, key + ".")
            else:
                flat[key[:250]] = str(v)[:250]

    _walk(params, prefix)
    mlflow.log_params(flat)


def log_metrics(run: Any, metrics: dict[str, float], step: int | None = None) -> None:
    if run is None:
        return
    mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v == v}, step=step)


def log_table(run: Any, df: Any, name: str) -> None:
    if run is None:
        return
    mlflow.log_table(df, artifact_file=name)
