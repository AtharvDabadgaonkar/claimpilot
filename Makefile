# ClaimPilot — developer entry points.
# Python 3.11 is required (matches Streamlit Community Cloud).

PYTHON ?= python3.11
VENV   := .venv
BIN    := $(VENV)/bin

.PHONY: help setup data simulate models score run test metrics lint check-deploy clean all

help:
	@echo "ClaimPilot targets:"
	@echo "  make setup         create .venv and install dev dependencies"
	@echo "  make data          download + process the source dataset"
	@echo "  make simulate      generate simulated narratives/policies/costs"
	@echo "  make models        train and persist the models"
	@echo "  make score         score the whole portfolio (triage + routing)"
	@echo "  make all           data + simulate + metrics"
	@echo "  make run           launch the Streamlit app locally"
	@echo "  make test          run the pytest suite"
	@echo "  make metrics       recompute reports/metrics.json"
	@echo "  make lint          ruff check"
	@echo "  make check-deploy  fresh venv, runtime deps only, boot the app"
	@echo "  make clean         remove venv, caches and raw data"

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip

setup: $(BIN)/python
	$(BIN)/python -m pip install -r requirements-dev.txt
	@echo "Setup complete. Next: make all"

data: setup
	$(BIN)/python scripts/ingest.py

simulate: data
	$(BIN)/python scripts/simulate.py

models: simulate
	$(BIN)/python scripts/train_models.py

score: models
	$(BIN)/python scripts/score.py

metrics: setup
	$(BIN)/python scripts/metrics.py

all: score metrics
	@echo "Pipeline complete: data -> simulate -> models -> score -> metrics."

run: setup
	$(BIN)/streamlit run streamlit_app.py

test: setup
	$(BIN)/python -m pytest -q

lint: setup
	$(BIN)/ruff check claimpilot scripts tests streamlit_app.py

# Mirrors Streamlit Community Cloud: a clean venv with ONLY requirements.txt,
# then boot the app headless and confirm it starts without error.
check-deploy:
	@echo "== check-deploy: fresh venv with runtime deps only =="
	rm -rf .venv-deploy
	$(PYTHON) -m venv .venv-deploy
	.venv-deploy/bin/python -m pip install --upgrade pip --quiet
	.venv-deploy/bin/python -m pip install -r requirements.txt --quiet
	@echo "== booting app for 20s =="
	@.venv-deploy/bin/streamlit run streamlit_app.py \
		--server.headless true --server.port 8599 > /tmp/claimpilot-deploy.log 2>&1 & \
	APP_PID=$$!; \
	sleep 20; \
	if kill -0 $$APP_PID 2>/dev/null; then \
		kill $$APP_PID; \
		if grep -qiE "traceback|exception|error" /tmp/claimpilot-deploy.log; then \
			echo "FAIL: errors in startup log:"; cat /tmp/claimpilot-deploy.log; exit 1; \
		fi; \
		echo "PASS: app booted cleanly on a runtime-only environment."; \
	else \
		echo "FAIL: app exited early:"; cat /tmp/claimpilot-deploy.log; exit 1; \
	fi
	rm -rf .venv-deploy

clean:
	rm -rf $(VENV) .venv-deploy .pytest_cache .ruff_cache data/raw
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
