#!/usr/bin/env bash
# Run a benchmark from scripts/bench inside the running api container (it needs the GPU, torch and laya).
#   bash scripts/bench/run.sh pass_anatomy.py
#   bash scripts/bench/run.sh replicas.py process 2 3 8
# Stop other GPU work first (training, load tests): these measure the GPU directly.
set -euo pipefail
cd "$(dirname "$0")/../.."
docker compose exec -T api mkdir -p /tmp/bench
for f in scripts/bench/*.py; do docker compose cp "$f" api:/tmp/bench/ >/dev/null 2>&1; done
[ -f data/typed-decisions/test.jsonl ] && docker compose cp data/typed-decisions/test.jsonl api:/tmp/bench/test.jsonl >/dev/null 2>&1
script="$1"; shift
docker compose exec -T -w /tmp/bench api python "$script" "$@"
