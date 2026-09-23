#!/usr/bin/env bash
# Reproduce docs/BAKEOFF.md: every run of the bake-off and the specialist comparison, then the tables.
# ~4 hours on an RTX 3090 (mostly model downloads and the NLI fine-tune). Needs the typed-decisions
# example data (make example-data) and the td-full / td-n150 runs (scripts/reproduce_experiments.sh).
#
# Disk: models are cached in runs/.hf-cache (~18 GB at peak, incl. the 11.6 GB Granite Guardian 5B)
# and deleted at the end; Laya fine-tunes' weights are deleted after scoring. Stops if < 100 GB free.
set -euo pipefail
cd "$(dirname "$0")/.."
T=(docker compose --profile train run --rm -e HF_HOME=/runs/.hf-cache train)
mkdir -p runs/logs
disk_ok() { local f; f=$(df --output=avail -BG . | tail -1 | tr -dc 0-9); [ "$f" -ge 100 ] || { echo "STOP: only ${f} GB free"; exit 1; }; }

# the bake-off's calibrated Laya checkpoint (hard-label temperatures), as in docs/EXPERIMENTS.md section 6
if [ ! -f runs/td-full-cal-hard/model/rl_agent_config.json ]; then
  ONLY="td-full-cal-hard" bash scripts/reproduce_experiments.sh
fi

# GLiClass needs its own package; installed apart from the image so it cannot touch pinned deps
docker compose --profile train run --rm --entrypoint sh train -c 'pip install -q --no-deps --target /runs/.pylib gliclass' || true

B() { disk_ok; echo "=== bakeoff $*"; "${T[@]}" bakeoff.py "$@"; }
B laya --name laya-zeroshot --model /models/convaiinnovations__laya
B laya --name laya-ft-150 --model /runs/td-n150/model
B laya --name laya-ft-1075 --model /runs/td-full-cal-hard/model
B nli-zeroshot --name deberta-v3-large-zs --hf MoritzLaurer/deberta-v3-large-zeroshot-v2.0
B nli-zeroshot --name modernbert-large-zs --hf MoritzLaurer/ModernBERT-large-zeroshot-v2.0
B gliclass --name gliclass-large-v3-zs --hf knowledgator/gliclass-large-v3.0
B heads --name heads-ft-150 --limit 150
B heads --name heads-ft-1075
B nli-finetune --name nli-modernbert-ft-150 --hf MoritzLaurer/ModernBERT-large-zeroshot-v2.0 --limit 150
B nli-finetune --name nli-modernbert-ft-1075 --hf MoritzLaurer/ModernBERT-large-zeroshot-v2.0

"${T[@]}" prepare_specialists.py
S() { disk_ok; echo "=== specialists $*"; "${T[@]}" specialists.py run "$@"; }
for task in prompt-injection toxicity sentiment; do
  S --task $task --method specialist
  S --task $task --method nli --hf MoritzLaurer/deberta-v3-large-zeroshot-v2.0 --name deberta-v3-large-zs
  S --task $task --method laya --model /models/convaiinnovations__laya --name laya-zeroshot
  D=/data/specialists/$task
  disk_ok; "${T[@]}" train.py --name spec-$task --train $D/train.jsonl --test $D/test.jsonl --questions $D/questions.json
  S --task $task --method laya --model /runs/spec-$task/model --name laya-ft-450
  rm -f runs/spec-$task/model/model.safetensors
done
G=ibm-granite/granite-guardian-3.2-5b     # risk names are the ones its chat template defines
S --task prompt-injection --method guardian --hf $G --risk jailbreak --name granite-guardian-5b
S --task toxicity --method guardian --hf $G --risk harm --name granite-guardian-5b
S --task toxicity --method guardian --hf $G --risk profanity --name granite-guardian-5b-profanity

"${T[@]}" bakeoff.py table
"${T[@]}" specialists.py table
# the Hub stores weights in a shared blobs/ folder, so delete the whole cache rather than per-model folders
rm -rf runs/.hf-cache runs/.pylib
echo "done; results in runs/bakeoff/"
