.PHONY: install data partition compose flower-config up down train eval render-runtime-notebook baselines flat local centralized clean

SEED ?= 123
GLOBAL_ROUNDS ?= 3
REGIONAL_ROUNDS ?= 2
EPOCHS ?= 3
BATCH_SIZE ?= 1024

install:
	python -m pip install -e .

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
	docker compose up --build -d

down:
	docker compose down --remove-orphans

train:
	python scripts/run_hierarchical_rounds.py \
		--global-rounds $(GLOBAL_ROUNDS) \
		--regional-rounds $(REGIONAL_ROUNDS)

eval:
	python scripts/evaluate_global_model.py \
		--checkpoint shared/checkpoints/global/round_$(GLOBAL_ROUNDS).pt

render-runtime-notebook:
	python scripts/render_runtime_notebook.py

centralized:
	python scripts/centralized_mlp_baseline.py --epochs $(EPOCHS) --batch-size $(BATCH_SIZE)

local:
	python scripts/local_only_baseline.py --epochs $(EPOCHS) --batch-size $(BATCH_SIZE)

flat:
	python scripts/flat_fl_baseline.py --rounds $(GLOBAL_ROUNDS) --batch-size $(BATCH_SIZE)

baselines: centralized local flat

clean:
	rm -rf shared/checkpoints/* shared/metrics/* reports/*.csv reports/*.metadata.json
