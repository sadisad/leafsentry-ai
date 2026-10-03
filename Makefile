SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

.PHONY: help sync format lint type test verify build model-smoke benchmark docker-build docker-up

help:
	@printf '%s\n' 'sync         Install locked dev dependencies' 'verify       Run formatting, lint, typing, tests, and build' 'model-smoke   Exercise the pinned model on three checksum-verified samples' 'benchmark     Evaluate the cached pinned test split' 'docker-build  Build the non-root runtime image'

sync:
	uv sync --frozen --extra dev --no-install-project

format:
	uv run --no-sync ruff format .

lint:
	uv run --no-sync ruff check .

type:
	uv run --no-sync mypy src

test:
	PYTHONPATH=src uv run --no-sync pytest -m "not model" --cov=leafsentry --cov-report=term-missing

verify:
	uv run --no-sync ruff format --check .
	uv run --no-sync ruff check .
	uv run --no-sync mypy src
	PYTHONPATH=src uv run --no-sync pytest -m "not model" --cov=leafsentry --cov-report=term-missing
	uv build --no-sources

build:
	uv build --no-sources

model-smoke:
	PYTHONPATH=src uv run --no-sync python scripts/model_smoke.py --output docs/evaluation/model-smoke.json

benchmark:
	PYTHONPATH=src uv run --no-sync python scripts/benchmark_test_split.py .cache/dataset-eval/test.zip --output docs/evaluation/test-split-report.json

docker-build:
	docker build --tag leafsentry-ai:local .

docker-up:
	docker compose up --build
