#!/usr/bin/env bash
# Reproduce docs/EXPERIMENTS.md: fetch the pinned example dataset, then train and
# evaluate every run in the study. Runs that already have an eval.json are skipped,
# so the script can be re-run after an interruption. ~90 min on an RTX 3090.
#
#   bash scripts/reproduce_experiments.sh            # everything
#   ONLY="td-n30 td-n75" bash scripts/reproduce_experiments.sh
set -euo pipefail
cd "$(dirname "$0")/.."

T=(docker compose --profile train run --rm train)
# The recorded study (sections 1-5) was trained before hard-label calibration became the
# default, so reproduce it with the calibration it actually used; section 6 then compares.
D=(--train /data/typed-decisions/train.jsonl --test /data/typed-decisions/test.jsonl --calibrate-on soft)
REV=c76749ec58bd8c3d2ea706b31c333a9059c38f90   # LocalLLaMA/typed-decisions revision used in the write-up
mkdir -p runs/logs

[ -f data/typed-decisions/train.jsonl ] || "${T[@]}" prepare_example.py --revision "$REV"

# name | train.py flags | evaluate.py flags
RUNS=(
  "td-full||--compare english typed-decisions"
  "td-n30|--limit 30|"
  "td-n75|--limit 75|"
  "td-n150|--limit 150|"
  "td-n300|--limit 300|"
  "td-n600|--limit 600|"
  "td-n150-s1|--limit 150 --seed 1|"
  "td-n600-s1|--limit 600 --seed 1|"
  "td-n150-e12|--limit 150 --epochs 12|"
  "td-notebook|--calib-frac 0 --no-shuffle-options|"
  "td-ml-n300|--base multilingual --limit 300|"
)
for spec in "${RUNS[@]}"; do
  IFS='|' read -r name tflags eflags <<<"$spec"
  if [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $name "* ]]; then continue; fi
  if [ -f "runs/$name/eval.json" ]; then echo "== $name: done, skipping"; continue; fi
  echo "== $name: train.py $tflags"
  # shellcheck disable=SC2086
  "${T[@]}" train.py --name "$name" "${D[@]}" $tflags 2>&1 | tee "runs/logs/$name.log"
  # shellcheck disable=SC2086
  "${T[@]}" evaluate.py --run "/runs/$name" $eflags 2>&1 | tee -a "runs/logs/$name.log"
done
# Section 6: the same td-full weights with three calibrations (no retraining).
for target in hard none; do
  name=td-full-cal-$target
  if [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $name "* ]]; then continue; fi
  if [ -f "runs/$name/eval.json" ]; then echo "== $name: done, skipping"; continue; fi
  rm -rf "runs/$name"; mkdir -p "runs/$name"; cp -r runs/td-full/model "runs/$name/model"; rm -f "runs/$name/model/eval.json"
  sed -i "s/\"name\": \"td-full\"/\"name\": \"$name\"/" "runs/$name/model/training_summary.json"
  "${T[@]}" calibrate.py --run "/runs/$name" --target "$target" 2>&1 | tee "runs/logs/$name.log"
  "${T[@]}" evaluate.py --run "/runs/$name" --compare 2>&1 | tee -a "runs/logs/$name.log"
done
"${T[@]}" summarize.py --prefix td- --out /runs/td-summary.md
