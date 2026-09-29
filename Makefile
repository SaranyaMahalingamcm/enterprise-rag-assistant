.PHONY: help install lint format test eval run docker-build docker-run clean

PYTHON ?= python
PORT ?= 8000

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

install:  ## Install the package with dev dependencies
	$(PYTHON) -m pip install -e ".[dev]"

lint:  ## Run ruff lint and format check
	ruff check .
	ruff format --check .

format:  ## Auto-format and fix lint issues
	ruff format .
	ruff check . --fix

test:  ## Run the test suite (offline)
	$(PYTHON) -m pytest

eval:  ## Run the evaluation harness and write reports/eval_report.md
	$(PYTHON) -m rag_assistant.eval

run:  ## Start the API locally with auto-reload
	$(PYTHON) -m uvicorn rag_assistant.api.app:app --reload --port $(PORT)

docker-build:  ## Build the container image
	docker build -t enterprise-rag-assistant:local .

docker-run:  ## Run the container image
	docker run --rm -p $(PORT):8000 --env-file .env.example enterprise-rag-assistant:local

clean:  ## Remove caches and build artefacts
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache -o -name "*.egg-info" \) -prune -exec rm -rf {} +
	rm -rf build dist
