PY=python

.PHONY: bootstrap test test_all ci

bootstrap:
	@echo "[bootstrap] Creating venv and installing dependencies."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry install; \
	else \
		$(PY) -m pip install -U pip; \
		$(PY) -m pip install -r requirements.txt; \
	fi

test:
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run pytest -q; \
	else \
		pytest -q; \
	fi

test_all: test

ci: test

