.PHONY: install install-all lint fmt test test-cov synth backtest clean

PY ?= python

install:
	$(PY) -m pip install -e ".[dev]"

install-all:
	$(PY) -m pip install -e ".[all]"

lint:
	ruff check src tests
	ruff format --check src tests

fmt:
	ruff check --fix src tests
	ruff format src tests

test:
	pytest

test-cov:
	pytest --cov=retailcast --cov-report=term-missing

synth:
	retailcast synth --n-items 200 --n-days 400 --out data/processed/synthetic.parquet

backtest:
	retailcast backtest --config configs/backtest.yaml

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
