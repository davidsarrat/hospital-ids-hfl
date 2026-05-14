SHELL := /bin/bash
.SHELLFLAGS := -euo pipefail -c

.PHONY: install data partition compose flower-config up down train eval predict demo demo-docker render-runtime-notebook vignettes pages baselines flat local centralized clean

SEED ?= 123
GLOBAL_ROUNDS ?= 3
REGIONAL_ROUNDS ?= 2
EPOCHS ?= 3
BATCH_SIZE ?= 1024
DEMO_GLOBAL_ROUNDS ?= 1
DEMO_REGIONAL_ROUNDS ?= 1
DEMO_BATCH_SIZE ?= 8192
COMPOSE_PARALLEL_LIMIT ?= 1
SUPERLINK_PORTS ?= 19093,29093,39093,49093

install:
	python -m pip install -e ".[docs]"

data:
	python scripts/download_kaggle.py
	python scripts/prepare_cicids.py

partition:
	python scripts/make_partitions.py --seed $(SEED)

compose:
	python scripts/generate_compose.py --output docker-compose.yml

flower-config:
	python scripts/configure_flower_profiles.py

up:
	COMPOSE_PARALLEL_LIMIT=$(COMPOSE_PARALLEL_LIMIT) docker compose --ansi never --progress plain up --build -d

down:
	docker compose down --remove-orphans

train:
	python scripts/run_hierarchical_rounds.py \
		--global-rounds $(GLOBAL_ROUNDS) \
		--regional-rounds $(REGIONAL_ROUNDS)

eval:
	python scripts/evaluate_global_model.py \
		--checkpoint shared/checkpoints/global/round_$(GLOBAL_ROUNDS).pt

predict:
	python scripts/predict_with_checkpoint.py \
		--checkpoint shared/checkpoints/global/round_$(GLOBAL_ROUNDS).pt \
		--hospital-id hospital_eu_01 \
		--output reports/predictions_hospital_eu_01.csv

demo: flower-config
	python scripts/run_demo.py \
		--global-rounds $(DEMO_GLOBAL_ROUNDS) \
		--regional-rounds $(DEMO_REGIONAL_ROUNDS) \
		--batch-size $(DEMO_BATCH_SIZE)

demo-docker: flower-config
	python scripts/generate_compose.py --output reports/docker-compose-hierarchical-demo.yml --hierarchical-only
	docker compose --ansi never -f reports/docker-compose-hierarchical-demo.yml down --remove-orphans
	COMPOSE_PARALLEL_LIMIT=$(COMPOSE_PARALLEL_LIMIT) docker compose --ansi never --progress plain -f reports/docker-compose-hierarchical-demo.yml up --build -d
	python scripts/wait_for_superlinks.py --ports 19093,29093,39093 --timeout 120 --settle-seconds 20
	python scripts/run_demo.py \
		--runtime existing \
		--runtime-shared-dir /shared \
		--global-rounds $(DEMO_GLOBAL_ROUNDS) \
		--regional-rounds $(DEMO_REGIONAL_ROUNDS) \
		--batch-size $(DEMO_BATCH_SIZE)

render-runtime-notebook:
	python scripts/render_runtime_notebook.py

vignettes:
	python scripts/render_vignettes.py
	python scripts/build_pages.py

pages:
	python scripts/build_pages.py

centralized:
	python scripts/centralized_mlp_baseline.py --epochs $(EPOCHS) --batch-size $(BATCH_SIZE)

local:
	python scripts/local_only_baseline.py --epochs $(EPOCHS) --batch-size $(BATCH_SIZE)

flat:
	python scripts/flat_fl_baseline.py --rounds $(GLOBAL_ROUNDS) --batch-size $(BATCH_SIZE)

baselines: centralized local flat

clean:
	rm -rf shared/checkpoints/* shared/metrics/* reports/*.csv reports/*.metadata.json reports/demo_transcript.txt
