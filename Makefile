SHELL := /bin/bash
.DEFAULT_GOAL := help

# Paths
VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

DATA ?= data/all_telemetry_track_data.csv
CONF ?= config/default.yaml
ETLC ?= config/etl.yaml
MODEL_DIR ?= models
MODEL ?= $(MODEL_DIR)/model.joblib
ARTIFACTS ?= artifacts

# environments
venv:
	python3 -m venv $(VENV)
	$(PY) -m pip install -U pip setuptools wheel
	$(PIP) install -r requirements.txt

# workflows
train: | venv
	$(PY) -m src.f1lap.train --data $(DATA) --config $(CONF)

predict: | venv
	mkdir -p $(ARTIFACTS)
	$(PY) -m src.f1lap.predict_cli --data $(DATA) --config $(CONF) --model $(MODEL) --out $(ARTIFACTS)/preds.csv

etl: | venv
	$(PY) -m src.f1lap.build_dataset_cli --config $(ETLC)

etl-train: | venv
	$(PY) -m src.f1lap.build_dataset_cli --config $(ETLC)
	$(PY) -m src.f1lap.train --data $(DATA) --config $(CONF)

# quality
format: | venv
	$(PY) -m black .
	$(PY) -m isort .

lint: | venv
	$(PY) -m flake8 .

# clean (keeps curated artifacts you commit)
clean:
	@mkdir -p $(ARTIFACTS)
	@find $(ARTIFACTS) -type f \
	  ! -name 'metrics_from_preds.json' \
	  ! -name 'slice_by_race.csv' \
	  ! -name 'slice_by_compound.csv' \
	  ! -name 'slice_by_stint.csv' \
	  ! -name 'residual_hist.png' -delete || true
	@rm -f $(MODEL_DIR)/*.joblib

# wipe everything generated
clean-all:
	rm -rf $(ARTIFACTS) $(MODEL_DIR)

distclean: clean-all
	rm -rf $(VENV) **/__pycache__

# help
help:
	@echo "Targets:"
	@echo "  venv         Create and populate virtual environment"
	@echo "  train        Train model on $(DATA)"
	@echo "  predict      Write predictions to $(ARTIFACTS)/preds.csv"
	@echo "  etl          Run dataset build pipeline (uses $(ETLC))"
	@echo "  etl-train    ETL then train"
	@echo "  format       Run black + isort"
	@echo "  lint         Run flake8"
	@echo "  clean        Remove ephemeral artifacts and model binaries, keep curated files"
	@echo "  clean-all    Remove ALL artifacts and models"
	@echo "  distclean    clean-all + remove venv and __pycache__"

.PHONY: venv train predict etl etl-train format lint clean clean-all distclean help
