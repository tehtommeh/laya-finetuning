# Thin wrappers over the commands you would otherwise retype constantly.
.PHONY: help download check verify up down logs test test-batch test-coalescing load-test shell stats clean \
        train-build example-data validate train evaluate calibrate publish learning-curve summarize experiments

help:
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | expand -t22

download:  ## Fetch model weights into ./models
	python3 scripts/download.py $$(grep '^MODEL_REPO=' .env | cut -d= -f2)

check:     ## Are there upstream model updates?
	python3 scripts/download.py --check

verify:    ## Do local weights still match the lock file?
	python3 scripts/download.py --verify

up:        ## Build and start the stack
	docker compose up -d --build

down:      ## Stop the stack
	docker compose down

logs:      ## Follow logs
	docker compose logs -f

test:      ## Run the endpoint smoke tests
	python3 scripts/smoke_test.py --wait 600

test-batch:  ## Check /v1/decide/batch matches per-state /v1/decide, and time both
	python3 scripts/test_batch_equivalence.py

test-coalescing:  ## Scheduler + CUDA-graph tests, then a live concurrent leak test against the SDK (~5 min)
	docker compose exec -T api python - < scripts/test_scheduler.py
	docker compose exec -T api python - < scripts/test_cuda_graphs.py
	docker compose exec -T api python - < scripts/test_coalescing.py

load-test:  ## Throughput and latency under 1-128 concurrent clients
	python3 scripts/load_test.py --clients 1 4 16 64 128

shell:     ## Shell into the API container
	docker compose exec api /bin/bash

stats:     ## GPU and container resource usage
	@nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
	@docker stats --no-stream

clean:     ## Remove containers and images, keep the weights
	docker compose down --rmi local --volumes

# ---------------------------------------------------------------- fine-tuning
# See docs/FINETUNING.md. Variables:
#   DATA=/data/<dir>  directory with train.jsonl [calib.jsonl] [test.jsonl] [questions.json]
#   RUN=<name>        run name (output in ./runs/<name>)
#   BASE=english|multilingual|typed-decisions      ARGS="--epochs 6 ..." extra train.py flags
TRAIN = docker compose --profile train run --rm train
DATA ?= /data/typed-decisions
BASE ?= english
RUN  ?= my-run
data_flags = --train $(DATA)/train.jsonl \
	$(if $(wildcard .$(DATA)/calib.jsonl),--calib $(DATA)/calib.jsonl) \
	$(if $(wildcard .$(DATA)/test.jsonl),--test $(DATA)/test.jsonl) \
	$(if $(wildcard .$(DATA)/questions.json),--questions $(DATA)/questions.json)

train-build:  ## Build the training image
	docker compose --profile train build train

# Revision used for docs/EXPERIMENTS.md; override with REV= (empty = latest).
REV ?= c76749ec58bd8c3d2ea706b31c333a9059c38f90
example-data:  ## Fetch the typed-decisions example dataset into ./data/typed-decisions
	$(TRAIN) prepare_example.py $(if $(REV),--revision $(REV))

validate:  ## Check a dataset:            make validate DATA=/data/mine
	$(TRAIN) validate.py $(data_flags) --base $(BASE)

train: validate  ## Fine-tune:             make train DATA=/data/mine RUN=mine-v1 [BASE=..] [ARGS=..]
	$(TRAIN) train.py --name $(RUN) --base $(BASE) $(data_flags) $(ARGS)

evaluate:  ## Score a run vs its base:  make evaluate RUN=mine-v1 [ARGS="--compare english typed-decisions"]
	$(TRAIN) evaluate.py --run /runs/$(RUN) $(ARGS)

calibrate:  ## Refit a run's temperatures: make calibrate RUN=mine-v1 [ARGS="--target soft | --calib /data/x.jsonl"]
	$(TRAIN) calibrate.py --run /runs/$(RUN) $(ARGS)

publish:  ## Serve a run in the API:   make publish RUN=mine-v1   (restarts the API)
	$(TRAIN) publish.py /runs/$(RUN) $(ARGS)
	docker compose restart api

SIZES ?= 75 150 300 600
learning-curve:  ## Train+evaluate at several data sizes: make learning-curve DATA=.. RUN=mine SIZES="100 300"
	@for n in $(SIZES); do \
	  $(TRAIN) train.py --name $(RUN)-n$$n --base $(BASE) $(data_flags) --limit $$n $(ARGS) && \
	  $(TRAIN) evaluate.py --run /runs/$(RUN)-n$$n || exit 1; \
	done
	$(TRAIN) summarize.py --prefix $(RUN)- --out /runs/$(RUN)-summary.md

summarize:  ## Table of all evaluated runs:  make summarize [RUN=prefix]
	$(TRAIN) summarize.py --prefix "$(RUN)" --out /runs/summary.md

experiments:  ## Re-run every experiment in docs/EXPERIMENTS.md (~90 min on an RTX 3090)
	bash scripts/reproduce_experiments.sh
