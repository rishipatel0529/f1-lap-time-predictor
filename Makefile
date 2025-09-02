SHELL := /bin/bash

VENV := .venv
PY := $(VENV)/bin/python

venv:
	python3 -m venv $(VENV); source $(VENV)/bin/activate; pip install -r requirements.txt

train:
	$(PY) -m src.f1lap.train --data data/all_telemetry_track_data.csv --config config/default.yaml

clean:
	rm -rf artifacts models/*.joblib

.PHONY: venv train clean

predict:
	$(PY) -m src.f1lap.predict_cli --data data/all_telemetry_track_data.csv --config config/default.yaml --model models/model.joblib --out artifacts/preds.csv

etl:
	$(PY) -m src.f1lap.build_dataset_cli --config config/etl.yaml

etl-train:
	$(PY) -m src.f1lap.build_dataset_cli --config config/etl.yaml && \
	$(PY) -m src.f1lap.train --data data/all_telemetry_track_data.csv --config config/default.yaml
