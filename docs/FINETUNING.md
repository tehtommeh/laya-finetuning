# Fine-tuning Laya

This guide covers turning your own labelled examples into a Laya checkpoint, measuring whether it got better, and
serving it next to the shipped checkpoints. Everything runs in Docker on the GPU you already use for serving.

```
data/<you>/train.jsonl ──make validate──► make train ──► runs/<name>/model ──make evaluate──► eval.json
                                                                      └──make publish──► finetuned/<name> ──► API + UI
```

- [When fine-tuning is worth it](#when-fine-tuning-is-worth-it)
- [What to collect](#what-to-collect)
- [Data format](#data-format)
- [How much data](#how-much-data)
- [Hardware and timings](#hardware-and-timings)
- [Running it](#running-it)
- [What comes out](#what-comes-out)
- [Evaluating](#evaluating)
- [Serving the fine-tuned model](#serving-the-fine-tuned-model)
- [Measured results](#measured-results-typed-decisions-rtx-3090)
- [Tuning and troubleshooting](#tuning-and-troubleshooting)

## When fine-tuning is worth it

The shipped checkpoints are a **base to specialise**, not a zero-shot decision engine: on the typed-decisions
benchmark the base English checkpoint scores **below the majority-label baseline** (measured below). Fine-tune when:

- you ask the same questions repeatedly (a routing schema, a triage form, a review rubric), and
- you can produce a few hundred labelled examples of real inputs.

Fine-tuning teaches the model **your schema**: the question wording, the option labels and their descriptions are
part of the input. Serve the fine-tuned model with the same questions you trained on. Rewording or renaming an
option moves you away from what it learned. `make publish` records the trained schemas, and the UI offers them as
presets.

## What to collect

For each example (a "case"):

1. **The state**: the input exactly as you will send it in production (text, an email object, a ticket, an agent
   trace as JSON). Strip boilerplate you would strip in production (signatures, quoted threads) and nothing else.
2. **The questions**: your fixed schema. Usually one shared `questions.json`, so each case only carries a state
   and answers.
3. **The gold answers**: for each question, the right answer. Unanswered questions are fine; they are skipped.

Guidance:

- **Sample from real traffic**, including hard, ambiguous and rare cases. A model trained only on clean examples
  is confidently wrong on messy ones.
- **Cover every option.** An option that never appears as the answer will not be learned. `validate` warns when
  one label dominates (>90%) or a question has a single label.
- **Soft labels help.** If two annotators disagree, or you sample an LLM teacher several times, record the split
  (`{"billing": 0.6, "technical": 0.4}`). Laya is trained with a proper scoring rule against the whole distribution,
  which is what makes its probabilities meaningful. Hard labels work too.
- **Keep a held-out test set** that no one tuned on. `train.py` also carves out a calibration set (10% by default)
  that is used only to fit temperatures.
- **Language:** fine-tune `BASE=multilingual` if your traffic is not English. The English base collapses on
  non-Latin scripts.

## Data format

JSON Lines, one case per line. Lines starting with `//` are ignored. A worked sample is in
[`data/sample/`](../data/sample). To build it from exports, annotator votes or an LLM teacher, with deduplication
and leakage-free splits, see [DATA_PREP.md](DATA_PREP.md).

```json
{"id": "t-001",
 "state": {"subject": "Duplicate charge", "body": "Billed twice for March. Refund today or we cancel."},
 "gold": {"department": "billing", "urgency": 2, "refund_requested": true, "churn_risk": 0.8}}
```

| field | required | notes |
|---|---|---|
| `id` | recommended | stable id; splits are made by hashing it, so adding data never reshuffles old cases |
| `state` | yes | string, object or list, exactly what you will send to `/v1/decide` |
| `questions` | if no `questions.json` | same shape as the API: `{id: {type, instructions, criteria}}`; per-line entries override the shared file |
| `gold` | yes | `{question_id: answer}`; `null` or absent = unlabelled |
| `workflow` | optional | groups cases for per-workflow metrics and UI presets |

Gold answers, short or full form:

| type | shorthand | full form |
|---|---|---|
| `choice` | `"billing"` | `{"probabilities": {"billing": 0.7, "technical": 0.3}}` (missing options = 0) |
| `score` | `2` (level index, 0-based) | `{"probabilities": {"0": 0.1, "1": 0.3, "2": 0.6}}` |
| `noul` | `true` / `false` / `0.8` (P(true)) | `{"probabilities": {"false": 0.2, "true": 0.8}}` |

Probabilities are normalised, so vote counts work too (`{"billing": 3, "technical": 1}`). The
LocalLLaMA/typed-decisions format (`label`, `confidence`, `probabilities`, `score`, `noul` keys) is read as is.

Question schema (`questions.json`), identical to the API:

```json
{
  "department": {"type": "choice", "instructions": "Which team should handle this ticket?",
                 "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages", "other": "anything else"}},
  "urgency":    {"type": "score", "instructions": "How urgent is this ticket?",
                 "criteria": ["can wait", "this week", "today", "blocking right now"]},
  "refund_requested": {"type": "noul", "instructions": "Does the customer ask for money back?"}
}
```

Put the files in one directory under `./data/`:

```
data/mine/train.jsonl       required
data/mine/questions.json    optional shared schema
data/mine/test.jsonl        optional; otherwise 10% of train is held out by id hash
data/mine/calib.jsonl       optional; otherwise 10% of train is held out by id hash
```

Token budget: each question is encoded as `[instructions + options (head_max_len=256)] + [state]`, up to
`max_len=1024` tokens. `validate` reports how many inputs are truncated and any question whose options do not
fit. For more than ~20 options, raise `--head-max-len` or split the question into coarse and fine steps.

## How much data

Measured on the typed-decisions example (4 workflows × 5 questions, English base, same 400 test cases; details
and controls in [EXPERIMENTS.md](EXPERIMENTS.md)):

| training cases | per workflow | accuracy | Brier ↓ | train time (3090) |
|---|---|---|---|---|
| 0 (base, zero-shot) | – | 0.360 | 0.316 | – |
| 30 | ~8 | 0.514 | 0.171 | 0.4 min |
| 75 | ~19 | 0.529 | 0.171 | 1.0 min |
| 150 | ~38 | 0.591 (0.623 with 12 epochs) | 0.137 | 2.0 min |
| 300 | ~75 | 0.657 | 0.115 | 3.9 min |
| 600 | ~150 | **0.735** | **0.071** | 8.2 min |
| 1,075 | ~270 | 0.733 | 0.078 | 14.0 min |

(Majority-label baseline: 0.452 accuracy.) What that suggests for planning your own dataset:

- **Pilot with ~20–40 cases per question set** (75–150 in total here). That already clearly beats the base model
  and shows whether the questions are learnable before you invest in more labelling.
- **Aim for ~150 cases (≈750 answers) per distinct question set.** On this benchmark that is where the curve
  flattened (600 in total), and two independent 600-case samples agreed within 0.2 points.
- **From ~20 to ~150 per question set, each doubling was worth ~6–8 accuracy points.** Run-to-run noise was under
  1 point, so these steps are real.
- **Below ~300 cases, train longer:** `ARGS="--epochs 12"` added +3.2 points at 150 cases.
- Label **the hard questions** more: per-question results (`eval.json`) show where extra data pays. Here, the
  8-option `disposition` and 4-level `urgency` questions lagged.

These are one benchmark's numbers. A schema with many more options per question, or much less consistent labels,
will need more data. Measure your own curve with `make learning-curve DATA=/data/mine RUN=mine SIZES="50 100 200 400"`
once you have a few hundred cases.

## Hardware and timings

Measured on an RTX 3090 (24 GB, Ampere, bf16), with the serving API running on the same GPU:

| | English base (ModernBERT-large, 421M) | multilingual base (mmBERT-base, 322M) |
|---|---|---|
| peak VRAM, training (defaults: batch 8, grad checkpointing) | **8.6 GB** (8.8 reserved) | **6.4 GB** (6.8 reserved) |
| training time, 4 epochs | **~1.3 min per 100 cases** (500 answers) | **~0.7 min per 100 cases** |
| throughput | ~7,000–7,500 tokens/s | ~15,300 tokens/s |
| per run overhead | ~20 s container start + model build | same |
| evaluation, 400 test cases | ~20 s load + ~25 s scoring per model | ~20 s load + ~10 s |
| checkpoint on disk | 804 MB | 615 MB |
| inference latency afterwards | unchanged (~48 ms / 5-question case) | unchanged (~25 ms) |

What that means for other hardware:

- **GPU:** NVIDIA with CUDA. Peak memory does not depend on dataset size, only on batch and sequence length (these
  inputs were ≤ 634 tokens; states near `max_len=1024` need more). A **12 GB** card should fit the defaults with
  the API stopped, and a 16 GB T4 is what the official notebook trained on. Neither was tested here. If memory is
  short: `ARGS="--batch 4"` (same results; the effective batch is kept at 64 by accumulation).
- **Pre-Ampere GPUs** (T4, V100, RTX 20xx) train in fp16 with gradient scaling automatically. That code path is
  the notebook's, but was not exercised on this machine.
- **CPU:** not supported; `train.py` exits. At these throughputs even a 100-case run would take hours.
- **Serving alongside:** API (3 checkpoints, ~6.4 GB) plus training (~8.8 GB) ≈ 15 GB, which fits on a 24 GB GPU.
  Stop the API on smaller cards.
- **Disk:** ~0.8 GB per run. The 11-run study in EXPERIMENTS.md used 8.6 GB under `runs/`; delete run directories
  you no longer need.
- **Timing rule of thumb:** `minutes ≈ 1.3 × cases/100 × epochs/4` on a 3090 for the English base. Scale by your
  GPU's relative bf16/fp16 throughput.

## Running it

```bash
make train-build                                  # once: build the training image

# optional: prove the pipeline on the example dataset first
make example-data                                 # LocalLLaMA/typed-decisions -> data/typed-decisions
make train RUN=td-full                            # DATA defaults to /data/typed-decisions
make evaluate RUN=td-full ARGS="--compare english typed-decisions"

# your data
make validate DATA=/data/mine                     # fix every ERROR, read every WARN
make train    DATA=/data/mine RUN=mine-v1         # BASE=multilingual for non-English traffic
make evaluate RUN=mine-v1
make calibrate RUN=mine-v1 ARGS="--calib ..."     # optional: refit temperatures on other held-out data
make publish  RUN=mine-v1                         # copies to finetuned/, restarts the API (ARGS=--force to replace)
```

`DATA` paths are as seen inside the container: `./data/mine` on the host is `/data/mine`. `make train` runs
`validate` first and stops on errors. Extra `train.py` flags go in `ARGS`, e.g.
`ARGS="--epochs 6 --limit 300"`. Every flag is listed in `docker compose --profile train run --rm train train.py --help`.

The serving stack can stay up while you train: on a 24 GB GPU both fit (measured below). On a smaller GPU, run
`docker compose stop api` first.

Multi-GPU: `docker compose --profile train run --rm --entrypoint torchrun train --standalone --nproc_per_node=2
train.py --name ... --train ...`. The effective batch stays 64 sequences, so results are comparable.

## What comes out

`runs/<name>/`:

| file | what |
|---|---|
| `model/model.safetensors` | fine-tuned weights, fp16 (804 MB for the ModernBERT-large bases, 615 MB multilingual) |
| `model/rl_agent_config.json` | the SDK config: fitted temperatures (+ `calibration` before/after), `max_len`/`head_max_len`, provenance |
| `model/encoder/`, `model/tokenizer/` | copied from the base unchanged; fine-tuning never changes shapes or vocabulary |
| `model/training_summary.json` | every argument, data counts per question, timings, peak VRAM, calibration before/after |
| `model/README.md` | a model card with the trained questions |
| `model/eval.json`, `eval.md` | after `make evaluate`: held-out metrics vs baselines |
| `train_log.jsonl` | loss / cross-entropy / reward every 10 optimiser steps |

`model/` is a normal Laya checkpoint, loadable anywhere with `laya.Agent("runs/<name>/model")`, and the same size
and speed as its base. Inference latency does not change.

The model is calibrated at the end of training. One temperature per `(question type, option count)` bucket is fitted
on the held-out calibration split and written to `rl_agent_config.json`, which the SDK applies at inference. Buckets
with fewer than 30 calibration answers fall back to one temperature per question type. The printed
"Calibration ... ECE a -> b" line shows the effect on the calibration split.

The fit targets the gold **labels** by default: afterwards a stated 80% means right about 80% of the time, which is
what you want for thresholds and escalation. On the example it cut test ECE from 0.123 to 0.034. `--calibrate-on
soft` fits to the gold **distributions** instead, as the official notebook did. That matches annotator/teacher
spread, but it made the model under-confident on held-out data ([EXPERIMENTS.md §6](EXPERIMENTS.md)). With hard
labels only, the two are the same. Temperatures never change which answer wins, so accuracy is unaffected.

Recalibrate any time without retraining, e.g. on a fresh `calib.jsonl` from newer traffic:

```bash
make calibrate RUN=mine-v1 ARGS="--calib /data/mine/calib-2026-10.jsonl"   # or --target soft / none, --dry-run
make evaluate  RUN=mine-v1 && make publish RUN=mine-v1 ARGS=--force
```

## Evaluating

`make evaluate RUN=<name>` loads every model through `laya.Agent`, the exact serving code path, and scores the
held-out test split against:

- the **base checkpoint** you started from (add more with `ARGS="--compare english typed-decisions"`)
- a **majority-label baseline**: always answer the most common training label for each question. A model that
  does not beat it has learned nothing useful.

| metric | meaning | better |
|---|---|---|
| `accuracy` | top answer = gold label (score: most likely level) | higher |
| `soft_acc` | Σ p·gold: agreement with the full gold distribution | higher |
| `brier` | Σ (p − gold)²: accuracy and calibration together | lower |
| `nll` | −log p(gold label): punishes confident mistakes hard | lower |
| `ece` | gap between stated confidence and actual accuracy | lower |
| `score_mae` | error of the expected level on `score` questions | lower |

Results are broken down by question type, by question and by workflow in `eval.json`. The **per-question**
breakdown is where to look for what to label next.

## Serving the fine-tuned model

`make publish RUN=<name>` copies the checkpoint to `finetuned/<name>/`, writes `schemas.json` (the trained question
sets with an example input), and restarts the API. Every directory in `finetuned/` is loaded at startup.

- API: pass `"model": "<name>"` to `/v1/decide`, `/v1/decide/batch` or `/v1/route`. `/v1/compare` includes it
  automatically, and `/v1/models` lists it with its training summary and eval results.
- UI: it appears in every checkpoint menu. The Decide tab offers `<name>: <workflow>` presets that load the
  trained questions, an example input and the model together. The Deployment tab shows its eval table.
- Automatic routing never picks a fine-tune; it is only used when asked for by name.
- Each published model adds ~0.9 GB of VRAM (ModernBERT-large) to the API.
- Unpublish: `rm -r finetuned/<name> && docker compose restart api`.

## Measured results (typed-decisions, RTX 3090)

Fine-tuning works. On typed-decisions' 400 held-out test cases, the English base improved from **0.360 → 0.733
accuracy** (Brier 0.316 → 0.078) with the full training set, in 14 minutes, and a 600-case subset matched it in
8 minutes. Hard-label calibration then cut ECE to 0.034. The full write-up, with the learning curve, seed and
epoch controls, a notebook-faithful comparison, the multilingual base, the calibration study, and the commands to
reproduce every number, is in [EXPERIMENTS.md](EXPERIMENTS.md).

## Tuning and troubleshooting

| symptom | what to do |
|---|---|
| `CUDA out of memory` | `ARGS="--batch 4"` (the effective batch stays 64 via accumulation), or stop the API |
| accuracy no better than the majority baseline | more data, or check labels: `validate` warns on dominant labels; look at per-question results |
| one question stays bad | it probably needs more examples of its minority options, or clearer option descriptions |
| calibration ECE barely moves | the calibration split is small; supply a larger `calib.jsonl` (≥ 30 answers per bucket) |
| still under-confident after calibration | a temperature hit the SDK's floor (0.5), so it cannot sharpen further; check `temperature_by_options` in `rl_agent_config.json` |
| you want probabilities that match annotator spread | `make calibrate RUN=... ARGS="--target soft"` (lower Brier vs soft gold, higher ECE) |
| inputs truncated (`validate` WARN) | shorten states, or `ARGS="--max-len 2048"` (slower, more VRAM) |
| options do not fit | `ARGS="--head-max-len 512"`, or fewer / shorter option descriptions |
| training is slow | `ARGS="--no-grad-checkpointing"` if VRAM allows; multi-GPU via torchrun |

The optimiser defaults are the official notebook's (4 epochs, AdamW lr 2.5e-5 encoder / 1e-4 head, cosine,
64 sequences per step, group size 4, exploration σ 0.4→0.1). This implementation deliberately differs from the
notebook in four ways, described at the top of [`train/train.py`](../train/train.py): bf16 on Ampere+, per-example
choice-option shuffling, held-out per-bucket calibration against labels, and a constant effective batch whatever
the GPU count. A notebook-exact run (all data, no shuffling) scored the same accuracy ([EXPERIMENTS.md
§4](EXPERIMENTS.md)).
