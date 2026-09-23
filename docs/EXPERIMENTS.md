# Fine-tuning experiments: how much data does Laya need?

A record of the experiments run while building the fine-tuning pipeline in this repo (2026-09-23), in the order
they were run, with the reasoning at each step. The question: **does fine-tuning work, what does it cost, and how
much labelled data does it take to make a real difference?**

All runs used [`docs/FINETUNING.md`](FINETUNING.md)'s pipeline. Every number below is on the **same 400 held-out
test cases (2,000 answers)** and comes from `evaluate.py`. Nothing was tuned on the test set.

One change was made *because of* these experiments. Sections 1–5 were run with temperatures fitted to the soft
gold distributions (the notebook's approach, and the default at the time). Section 6 found that this makes the
model under-confident, so the default became hard-label calibration (`--calibrate-on hard`). Accuracy is
unaffected: temperature never changes which answer wins. The reproduction script trains sections 1–5 with
`--calibrate-on soft` so the tables reproduce as recorded.
Reproduction steps are [at the end](#reproducing-this).

## Setup

| | |
|---|---|
| Dataset | [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) (Apache-2.0), revision `c76749ec58bd8c3d2ea706b31c333a9059c38f90` |
| Task | 4 workflows (agent-trace observability, customer service, invoice processing, security incidents); each case asks 5 typed questions (choice / score / noul) about one JSON state; gold answers are soft distributions from repeated teacher samples |
| Splits | official `test` (400 cases) for every evaluation; official `train` (1,200 cases) split by id hash into 1,075 train + 125 calibration (seed 0) |
| Base | `convaiinnovations/laya` English checkpoint (ModernBERT-large, 421M), revision `1c5edc17a7ac` |
| Recipe | `train.py` defaults = the official notebook's: 4 epochs, AdamW 2.5e-5 / 1e-4, cosine, 64 sequences/step, group size 4, σ 0.4→0.1, max_len 1024 / head 256 |
| Hardware | 1× RTX 3090 24 GB, driver 580, `pytorch:2.14.0-cuda12.6` image, bf16 autocast; the serving API (3 checkpoints, ~6 GB) stayed up throughout |
| Subsets | `--limit N` takes a deterministic, **nested** random sample (the 150-case set contains the 75-case set), so learning-curve points differ only in data volume |

Validation (`make validate`) passed with no errors: median sequence 307 tokens, p95 433, none truncated. Two
question ids (`action`, `disposition`) are reused across workflows with different option sets, so the
majority-label baseline in `evaluate.py` keys on id and options together.

## 1. Does fine-tuning work? Full training set

`make train RUN=td-full` then `make evaluate RUN=td-full ARGS="--compare english typed-decisions"`:

| model | accuracy | soft acc | Brier ↓ | NLL ↓ | ECE ↓ | score MAE ↓ |
|---|---|---|---|---|---|---|
| majority label (baseline) | 0.452 | 0.379 | 0.202 | 1.057 | 0.043 | 0.629 |
| `english` base, zero-shot | 0.360 | 0.331 | 0.316 | 1.331 | 0.176 | 0.694 |
| **`td-full`, fine-tuned here (1,075 cases)** | **0.733** | **0.490** | 0.078 | **0.682** | 0.123 | 0.292 |
| `typed-decisions`, official fine-tune (1,200 cases) | 0.767 | 0.471 | 0.061 | 0.707 | 0.214 | 0.242 |

**Yes.** Accuracy doubles relative to the base (0.360 → 0.733), Brier drops 4×, and the model clears the
majority-label baseline by 28 points. The base checkpoint is *below* that baseline: it has no idea what these
workflows want until it is shown.

- Training took **14.0 min**, 336 optimiser steps, ~7,500 tokens/s, peak **8.6 GB** VRAM alongside the running API.
- Latency is unchanged: ~48 ms per 5-question case in evaluation (the same encoder, just new weights).
- Per question, the weakest are `urgency` (0.61, a 4-level ordinal), `disposition` (0.66, 8 options) and
  `outcome` / `action` (0.67–0.69). The best are `category` (0.95) and `duplicate` (0.93).
- Calibration (soft target, as run): temperatures fitted on the 125 held-out calibration cases came out at
  1.00–1.18, and ECE on the calibration set moved 0.143 → 0.154. That looked like noise at the time. Section 6
  shows it was a systematic effect of the soft target.

It is 3.4 accuracy points behind the official checkpoint, while better on soft accuracy, NLL and ECE. Section 4
tests whether that gap comes from this pipeline's deviations from the notebook.

## 2. How much data? The learning curve

`make learning-curve RUN=td SIZES="30 75 150 300 600"`: the same recipe on nested random subsets.

| training cases | answers | optimiser steps | train time | accuracy | soft acc | Brier ↓ | NLL ↓ | score MAE ↓ |
|---|---|---|---|---|---|---|---|---|
| 0 (base) | – | – | – | 0.360 | 0.331 | 0.316 | 1.331 | 0.694 |
| 30 | 150 | 12 | 0.4 min | 0.514 | 0.383 | 0.171 | 1.018 | 0.545 |
| 75 | 375 | 24 | 1.0 min | 0.529 | 0.393 | 0.171 | 0.987 | 0.488 |
| 150 | 750 | 48 | 2.0 min | 0.591 | 0.425 | 0.137 | 0.879 | 0.451 |
| 300 | 1,500 | 96 | 3.9 min | 0.657 | 0.450 | 0.115 | 0.806 | 0.411 |
| 600 | 3,000 | 188 | 8.2 min | **0.735** | **0.499** | **0.071** | **0.671** | **0.271** |
| 1,075 | 5,375 | 336 | 14.0 min | 0.733 | 0.490 | 0.078 | 0.682 | 0.292 |

Reading it:

- **Any data helps.** 30 cases (150 answers) already move accuracy +15 points and nearly halve the Brier score.
  But that lands only 6 points above the majority-label baseline: at this size the model mostly learns what each
  question's answers usually are, not yet how to read the state.
- **Gains are roughly steady per doubling from 75 to 600** (+6–8 points per doubling).
- **The curve flattens at ~600 cases** (~150 per workflow, 3,000 answers). 1,075 cases added nothing measurable.
- Training time is linear in data, ~1.3 min per 100 cases (500 answers) for 4 epochs, and peak VRAM does not depend
  on dataset size (8.4–8.6 GB).

## 3. Controls: is the curve real?

Three follow-up runs, because a single-seed curve could mislead:

| run | question | result |
|---|---|---|
| `td-n150-s1`: 150 cases, seed 1 (different subset, shuffle and calibration split) | how noisy is one run? | **0.590** vs 0.591 (Brier 0.131 vs 0.137) |
| `td-n600-s1`: 600 cases, seed 1 | is the plateau real? | **0.733** vs 0.735, same as 1,075 cases |
| `td-n150-e12`: 150 cases, 12 epochs (144 steps instead of 48) | were the small runs just under-trained? | **0.623** vs 0.591 at 4 epochs; 300 cases at 4 epochs gets 0.657 |

- **Seed variation is under a point**, so the steps in section 2 are real differences, not noise.
- **The plateau is real**: two different 600-case samples both match the full set.
- **Small datasets are partly under-trained** by the 4-epoch default. Tripling the epochs recovers +3.2 points,
  about half of what doubling the data gives, for 3× the training time. With fewer than ~300 cases, use
  `ARGS="--epochs 12"`. More data is still the bigger lever.

## 4. The gap to the official checkpoint

`td-full` differs from the official notebook in two data-side ways: it holds out 125 training cases for
calibration (1,075 vs 1,200 trained on), and it shuffles choice options per example. `td-notebook` removes both:
all 1,200 cases (`--calib-frac 0`) and `--no-shuffle-options`.

| model | train cases | shuffle | accuracy | soft acc | Brier ↓ | NLL ↓ | ECE ↓ | train time |
|---|---|---|---|---|---|---|---|---|
| `td-full` | 1,075 | yes | 0.733 | 0.490 | 0.078 | 0.682 | 0.123 | 14.0 min |
| `td-notebook` | 1,200 | no | 0.732 | 0.496 | 0.074 | 0.677 | 0.115 | 16.7 min* |
| official `typed-decisions` | 1,200 | no | 0.767 | 0.471 | 0.061 | 0.707 | 0.214 | (2×T4) |

\* 12% more data, and ~7% lower throughput on this run (6,980 vs 7,480 tokens/s; run-to-run variation on the 3090 was 6,980–7,560).

**Neither deviation costs accuracy.** The notebook-faithful run lands exactly where the default does. The
remaining 3.4-point gap to the published checkpoint is not explained by anything tested here. Untested
candidates: fp16 autocast with gradient scaling (the notebook, on T4s) vs bf16 here; 2-GPU DDP (each rank
shuffles its own half) vs 1 GPU; library versions; or the published weights being the best of several runs.
The model card's own figure for the checkpoint is 0.766, consistent with our 0.767 re-measurement, so the
evaluation is not the difference. The defaults stay as they are: shuffling and a held-out calibration split are
free, and the calibrated models score better on NLL and ECE than the official one.

## 5. Multilingual base

`td-ml-n300`: 300 cases on the **multilingual** base (mmBERT-base, 322M), compared with `td-n300` (English base,
same 300 cases):

| model | accuracy | Brier ↓ | NLL ↓ | ECE ↓ | train time | peak VRAM | latency |
|---|---|---|---|---|---|---|---|
| multilingual base, zero-shot | 0.352 | 0.463 | 1.862 | 0.316 | – | – | 25.6 ms |
| **`td-ml-n300`**, multilingual base | 0.649 | 0.112 | 0.791 | 0.058 | **2.0 min** | **6.4 GB** | **25.3 ms** |
| `td-n300`, English base | 0.657 | 0.115 | 0.806 | 0.094 | 3.9 min | 8.5 GB | 48.5 ms |

On English data, the smaller multilingual base fine-tunes to within a point of the English base. It trains in half
the time (~15,300 vs ~7,400 tokens/s), with 2 GB less memory, and serves twice as fast. Zero-shot it is the
worst-calibrated model in the study (Brier 0.463), which fine-tuning fixes along with accuracy. For non-English
traffic it is the only sensible base. For English, the English base's small edge may not be worth 2× the latency.

## 6. Calibration: soft or hard target?

A check done while writing this up: `td-full` scored *better* on the test set with its fitted temperatures switched
off. The fit minimised cross-entropy against the teacher's soft distributions, which are deliberately spread out
(e.g. 0.43 / 0.28 / 0.25). That pushes temperatures above 1, flattening a model whose top answer is right more
often than the teacher's spread implies. `calibrate.py` refits an existing checkpoint without retraining. The same
`td-full` weights, three ways:

| `td-full` weights, calibrated | temperatures | accuracy | ECE ↓ | NLL (gold label) ↓ | Brier vs soft gold ↓ | soft acc |
|---|---|---|---|---|---|---|
| soft target (notebook; sections 1–5) | 1.00–1.18 | 0.733 | 0.123 | 0.682 | **0.078** | 0.490 |
| none (T = 1) | 1.0 | 0.733 | 0.111 | 0.672 | 0.080 | 0.496 |
| **hard target (new default)** | 0.50–0.71 | 0.733 | **0.034** | **0.625** | 0.120 | **0.539** |

It is a genuine trade-off, not a free win:

- **Hard target**: stated confidence matches how often the model is right (ECE 0.123 → 0.034, 3.6× better; the
  official checkpoint's is 0.214). This is what you need to set thresholds ("auto-approve above 0.9") or decide
  when to escalate, so it is the default.
- **Soft target**: probabilities match the annotators' / teacher's spread (lower Brier against the soft gold).
  Prefer it when the distribution itself is the product, e.g. to reproduce benchmark numbers that score against
  soft gold (`--calibrate-on soft`).
- With hard labels only, the two targets are identical.
- The choice temperature hit the SDK's lower clamp (0.5), so the unclamped optimum is sharper still.

The served `td-full` model (`finetuned/td-full`) is this hard-calibrated version.

## Conclusions

1. **Fine-tuning works, and cheaply.** On a 4-workflow, 15-question benchmark it took the English base from 0.360
   (below the 0.452 majority baseline) to **0.733** accuracy, in 14 minutes on one RTX 3090, using 8.6 GB of VRAM
   next to the running API. Latency is unchanged.
2. **How much data:** gains start immediately (+15 points from 30 cases) and continue at ~6–8 points per doubling,
   then **flatten at ~150 cases per question set** (600 here). Run-to-run noise is under 1 point.
3. **Small datasets:** train longer (`--epochs 12`, +3 points at 150 cases), but more data is the bigger lever.
4. **The recipe is sound:** the notebook-exact variant scores the same (0.732). A 3-point gap to the published
   checkpoint remains unexplained. It is not caused by this pipeline's deviations.
5. **Calibrate against labels, not teacher distributions,** when the probabilities will be used as confidence
   (ECE 0.123 → 0.034). This changed the default.
6. **The multilingual base is a viable, faster choice** even for English (−0.8 points, 2× faster training and
   serving).

## Reproducing this

The data is not in the repo. It is fetched from the Hugging Face Hub at a pinned revision.

```bash
# 0. the stack's prerequisites: base weights and the training image
python3 scripts/download.py convaiinnovations/laya      # -> models/convaiinnovations__laya (2.3 GB)
make train-build

# 1. fetch the dataset at the revision used here -> data/typed-decisions/{train,test}.jsonl + SOURCE.json
make example-data            # = prepare_example.py --revision c76749ec58bd8c3d2ea706b31c333a9059c38f90

# 2. everything above, skipping runs that already have an eval.json (~90 min on an RTX 3090)
make experiments             # = bash scripts/reproduce_experiments.sh

# or one run at a time, e.g.
ONLY="td-n300" bash scripts/reproduce_experiments.sh

# 3. the tables
make summarize RUN=td-       # -> runs/summary.md
cat runs/td-*/eval.md        # per-run tables; per-question/workflow detail in runs/<run>/eval.json
```

What to expect: with the same revisions, seeds and pinned images, accuracy should reproduce within about a point
(seed variation measured in section 3). GPU kernels are not bit-deterministic, so exact decimals will differ.
Timings scale with GPU: the 3090 sustains ~7,500 training tokens/s. Runs are written to `runs/<name>/` (the
checkpoint, `training_summary.json`, `train_log.jsonl`, `eval.json`, `eval.md`) and logs to `runs/logs/`.

To serve any of them: `make publish RUN=td-full` (the UI then offers `td-full: <workflow>` presets with each
workflow's questions and an example input).
