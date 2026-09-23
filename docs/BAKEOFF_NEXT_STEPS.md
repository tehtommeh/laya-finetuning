# Bake-off next steps: making the comparison fully fair

The [bake-off](BAKEOFF.md) left some conclusions open. Its [fairness review](BAKEOFF.md#fairness-review) found that
Part 1 is tilted slightly toward Laya (Laya's recipe was built for the benchmark; the competitors ran untuned
defaults), and that Part 2's fine-tuned rows compare Laya with in-domain data against models without it. These
tests close those gaps. None have been run yet.

| # | test | settles | GPU time (RTX 3090) |
|---|---|---|---|
| 1 | Equal tuning budget for every fine-tuned method | whether "Laya is best at ~150 cases" survives | ~2 h |
| 2 | Three training seeds per fine-tuned configuration | seed variance, which the bootstrap does not cover | ~1.5 h |
| 3 | Fine-tuned competitors in the specialist round | whether Laya beats other methods given the same in-domain data | ~1 h |
| 4 | Uniform calibration and symmetric prompts in the specialist round | fair calibration and zero-shot numbers | ~20 min |
| 5 | Latency on equal serving paths | whether "heads are 1.5× faster to serve" holds against Laya's served path | ~20 min |
| 6 | A benchmark with human labels, or changing questions | whether the typed-decisions results generalise | depends on data |

Tests 1–4 are the priority: together, about 4–5 hours unattended.

## 1. Equal tuning budget

**Why.** Laya's recipe (4 epochs, lr 2.5e-5 / 1e-4, 64-sequence steps) comes from its authors' typed-decisions
notebook. The heads model and the NLI fine-tune used generic defaults. The NLI fine-tune ran only 2 epochs, and the
heads model got just 76 optimiser steps at 150 cases. Small data is exactly where that matters.

**How.** Give every fine-tuned method, Laya included, the same small grid, and choose on the **calibration
split**, never the test set:

- epochs ∈ {4, 8, 12}
- encoder learning rate ∈ {1e-5, 2.5e-5, 5e-5}
- heads model only: head learning rate ∈ {1e-4, 1e-3}

At 150 and 1,075 cases, that is ~9–18 runs per method and size. The 150-case runs are fast (under 3 min each).
Score the selected configuration once on the test set.

**Changes needed.**

- `bakeoff.py heads` and `nli-finetune` take `--epochs` and `--lr`. They also need a `--select` mode that
  scores on the calibration split and writes no test results.
- `train.py` already takes `--epochs` and `--lr-encoder`.
- A small driver script runs the grid and picks the best configuration by calibration-split NLL.

**Success criterion.** Report each method's best configuration. The claim "Laya is best with little data" holds if
it still leads at 150 cases, with a paired bootstrap interval that excludes 0.

## 2. Seeds

**Why.** `bakeoff.py bootstrap` covers test-set sampling (about ±2.3 points), but each model was trained once.
Laya's earlier seed repeats varied by under 1 point; the other models' variation is unknown.

**How.** Train the selected configuration from test 1 with seeds 0, 1 and 2, for each method at 150 and 1,075
cases. Report the mean and the spread, and bootstrap over test cases *and* seeds together.

**Changes needed.** A `--seed` flag for `bakeoff.py heads` / `nli-finetune` (currently fixed at 0), and bootstrap
support for pooling several runs of one configuration.

## 3. Fine-tuned competitors in the specialist round

**Why.** Only Laya was fine-tuned on the 450 in-domain examples. "Laya fine-tuned beats the specialist" therefore
measures in-domain data, not Laya's method.

**How.** On each task's same 450/50 split, fine-tune:

- the **specialist itself** (continue training `protectai/…`, `unitary/toxic-bert`, `cardiffnlp/…` on the 450);
- the **ModernBERT-large heads** model;
- the **DeBERTa-v3-large NLI model** (as in `nli-finetune`).

Compare them with Laya fine-tuned on the same data, using the test-1 tuning budget.

**Changes needed.** A `finetune` method in `specialists.py` that trains a `AutoModelForSequenceClassification` on
a task's `train.jsonl` (soft labels for toxicity) and scores it with the existing metrics.

## 4. Uniform calibration and symmetric prompts (specialist round)

**Why.** The Laya fine-tunes had a held-out calibration step; every other model was scored raw, so the ECE and
Brier comparisons are not like for like. Laya zero-shot got detailed instructions and definitions, while the NLI
model got one-line hypotheses.

**How.**

- Report every model both raw and after the same temperature fit on each task's 50-row calibration split (as
  `bakeoff.py` already does for Part 1).
- Give the NLI model hypotheses built from the same instructions and criteria Laya sees. Also report Laya with a
  one-line question, so both directions are visible.
- Report accuracy and F1 at a threshold chosen on the calibration split, as well as at 0.5, because several models
  (Granite Guardian especially) rank well but have a badly placed default threshold.
- Give Granite Guardian its risk names up front (`jailbreak`; `profanity` and `harm`), not after seeing results.
- Enlarge the samples. Prompt injection has only 116 test rows; add a second injection dataset (e.g. the
  `jackhhao/jailbreak-classification` test split, which protectai *did* train on, reported separately as its home
  turf), and raise toxicity and sentiment to 3,000 test rows.

## 5. Latency on equal serving paths

**Why.** Every model's latency was measured eager. Laya's actual served path uses CUDA graphs (~7–15 ms per
request), so "heads are 1.5× faster to serve" compares the heads model with an unoptimised Laya.

**How.** Measure both ways: (a) all eager, as now; (b) each model with CUDA graphs over the same bucket scheme
(`GraphRunner` generalises to any fixed-shape forward pass), batch of one case. Also measure throughput at
batch 64.

## 6. A second benchmark

**Why.** typed-decisions is one synthetic benchmark, and "correct" there means agreeing with the teacher LLM that
produced the labels. It also suits fixed per-question heads, because every test question appears in training.

**Options.**

- **Human labels:** a multi-question dataset with human gold, for example `Tobi-Bueck/customer-support-tickets`
  (queue, priority, type and tags per ticket; CC-BY-NC).
- **Changing questions:** hold out one question per workflow from training (or reword the test questions) and
  measure each method on unseen questions. Laya can answer them directly; heads would need new heads. This tests
  Laya's main structural claim.

## Running it

All of these extend `train/bakeoff.py` and `train/specialists.py`, and would be driven by an extended
`scripts/reproduce_bakeoff.sh`. The disk guards stay the same: models cached in `runs/.hf-cache` and deleted at the
end, a stop below 100 GB free, and fine-tune weights deleted after scoring. Results should update the tables and
the [fairness review](BAKEOFF.md#fairness-review) in place, and keep the earlier numbers for comparison.
