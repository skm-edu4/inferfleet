.PHONY: install test lint run demo up down logs seed bench clean

VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

install:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .

test:
	$(PY) -m pytest tests -q

lint:
	$(VENV)/bin/ruff check src tests
	$(VENV)/bin/ruff format --check src tests || true

run:
	$(PY) -m inferfleet

demo:
	bash scripts/demo.sh

# A/B: static vs continuous batching on an identical closed-loop workload.
# Other sweeps: see BENCHMARKS.md (load sweep + cache absorption runs).
bench:
	bash scripts/bench.sh ab-static static --requests 400 --concurrency 32 --pool 20000 --max-tokens 64
	bash scripts/bench.sh ab-continuous continuous --requests 400 --concurrency 32 --pool 20000 --max-tokens 64

seed:
	$(PY) scripts/seed.py

up:
	docker compose up --build -d --scale worker=3

down:
	docker compose down -v

logs:
	docker compose logs -f gateway worker history

clean:
	rm -rf .pytest_cache .ruff_cache data *.egg-info src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
