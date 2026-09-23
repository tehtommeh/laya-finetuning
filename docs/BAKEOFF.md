# Bake-off: does Laya live up to the hype?

Laya is pitched as a general "System 1" decision model: typed questions in, calibrated probabilities out, much
faster than LLMs, and no hallucinations. This puts it next to the established ways of doing the same jobs, on equal
terms:

1. **Multi-question decisions**, the job Laya is built for, on the typed-decisions benchmark. Laya is compared with
   the most popular zero-shot classifiers, a plain fine-tune of its own backbone, and a fine-tuned zero-shot model,
   each zero-shot and at two training-set sizes.
2. **Specialists on their home turf.** Purpose-built prompt-injection, toxicity and sentiment models, plus an LLM
   guard model, each on its own kind of benchmark, against Laya zero-shot and Laya fine-tuned on 450 examples.

Run 2026-09-23/24 on one RTX 3090. Code: [`train/bakeoff.py`](../train/bakeoff.py),
[`train/specialists.py`](../train/specialists.py), [`train/prepare_specialists.py`](../train/prepare_specialists.py).
Reproduce everything with `bash scripts/reproduce_bakeoff.sh` (about 4 hours, mostly downloads and the NLI
fine-tune).

## Verdict

- **Out of the box, Laya is the weakest multi-question model tested.** Zero-shot on typed-decisions it scores
  0.360, below always answering the most common label (0.452). A popular zero-shot NLI model, DeBERTa-v3-large,
  scores 0.539 with no training at all.
- **Laya learns fast from little data.** With 150 labelled cases it beats every other approach given the same data
  (0.591, against 0.545 and 0.506).
- **With ~1,000 cases, a plain fine-tune of the same encoder wins.** ModernBERT-large with ordinary per-question
  classification heads scores 0.781 against Laya's 0.733. It is also better on every question type, 1.5× faster to
  serve, and 5.6× faster to train. Laya's typed-question machinery helps when data is scarce, and costs accuracy
  when it is not.
- **Laya's calibration is genuinely good.** It is the only model that is well calibrated before any
  post-calibration (raw ECE 0.034). After everyone gets the same temperature fit, the gap closes.
- **Against specialists it is mixed.** Laya matches or beats the toxicity and prompt-injection specialists, and
  loses to the sentiment specialist on its home benchmark. With 450 in-domain examples it becomes excellent at
  prompt injection (0.948 accuracy). Specialists are 2–3× faster.

So the concept is sound and Laya is a good base to fine-tune from a small dataset. The "general decision engine"
framing does not hold up: zero-shot it loses to a free, widely used NLI model, and once you have ~1,000 labels a
conventional fine-tune does better.

## Which should you use?

A Laya fine-tune, or a traditional fine-tune (a ModernBERT-style encoder with per-question classification heads)?

**Laya is the better choice when:**

- **You have little data** (a few hundred labelled cases or fewer). It led by 5–9 points at 150 cases.
- **You want honest probabilities with no calibration step** (raw ECE 0.034).
- **Your questions change.** You can reword questions or add options without re-architecting. Accuracy on new
  questions stays weak until you retrain, but the pipeline doesn't change.
- **You want what this repo already provides:** serving, CUDA graphs, fine-tuning, evaluation, the API and the UI
  are all built around Laya.

**A traditional fine-tune is the better choice when:**

- **You have ~1,000+ labelled cases.** It was 5 points more accurate and won on every question type, especially the
  ordinal 0–4 scales.
- **Throughput and training cost matter.** It is 1.5× faster to serve and 5.6× faster to train, because it encodes
  each input once rather than once per question.
- **You want standard, well-supported tooling:** a classification head and ordinary Hugging Face training, with no
  dependency on a small, new SDK.

**Its costs:**

- It needs a calibration step for honest probabilities (raw ECE 0.154 before it).
- Changing a question means retraining.
- This repo does not serve it: `bakeoff.py heads` trains and scores it for comparison, but does not save a model the
  API can load.

**Recommendation.** Start with a Laya fine-tune while you collect your first few hundred labels: it gets you to
something useful fastest, and it is already wired into this stack. Once you have ~1,000 labels and a stable schema,
train the traditional model on the same data, compare both on the same held-out test set (`bakeoff.py heads`
against `bakeoff.py laya`, or `make evaluate`), and switch if it wins. On this benchmark it did.

## Part 1: typed-decisions (multi-question workflows)

**Task.** [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions): 4 workflows, 5
questions per case (choice, 0–4 score, yes/no), with soft gold labels from repeated teacher samples. Test: the
official 400 cases (2,000 answers). Training: 1,075 cases (and a nested 150-case subset); 125 held out for
calibration. The majority-label baseline (always answer the most common training label) scores **0.452**.

**Methods.**

| method | what it is | trained on |
|---|---|---|
| Laya | `convaiinnovations/laya` English checkpoint, fine-tuned with this repo's `train.py` (defaults) | 0 / 150 / 1,075 cases |
| DeBERTa-v3-large zero-shot | [`MoritzLaurer/deberta-v3-large-zeroshot-v2.0`](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0), the most-downloaded modern zero-shot classifier; each option phrased as a hypothesis | 0 |
| ModernBERT-large zero-shot | [`MoritzLaurer/ModernBERT-large-zeroshot-v2.0`](https://huggingface.co/MoritzLaurer/ModernBERT-large-zeroshot-v2.0), the same idea on Laya's backbone family | 0 |
| GLiClass | [`knowledgator/gliclass-large-v3.0`](https://huggingface.co/knowledgator/gliclass-large-v3.0), the design closest to Laya's (labels and text in one pass) | 0 |
| ModernBERT heads | `answerdotai/ModernBERT-large` (Laya's own base encoder) + one linear head per question, soft cross-entropy, 4 epochs | 150 / 1,075 |
| NLI fine-tune | ModernBERT-large-zeroshot fine-tuned on the task as a cross-encoder (softmax over options' entailment scores), 2 epochs | 150 / 1,075 |

**Shared across all methods.** The same splits and soft labels. The same hard-label temperature calibration on the
125-case calibration set, over a wide range [0.05, 50]. The same metrics code (`evaluate.py`). Latency is model
time per test case (all its questions), one case at a time, bf16. No method's hyper-parameters were tuned, Laya's
included.

**Results** (calibrated; accuracy never depends on calibration):

| method | train cases | accuracy | soft acc | Brier ↓ | NLL ↓ | ECE ↓ | score MAE ↓ | ms/case | train time |
|---|---|---|---|---|---|---|---|---|---|
| majority label | – | 0.452 | – | – | – | – | – | – | – |
| Laya | 0 | 0.360 | 0.323 | 0.233 | 1.197 | 0.025 | 0.687 | 30 | – |
| DeBERTa-v3-large zero-shot | 0 | **0.539** | **0.388** | **0.188** | **1.020** | 0.063 | **0.516** | 86 | – |
| ModernBERT-large zero-shot | 0 | 0.449 | 0.366 | 0.198 | 1.070 | 0.050 | 0.516 | 54 | – |
| GLiClass-large-v3 | 0 | 0.460 | 0.335 | 0.237 | 1.159 | 0.073 | 0.630 | 155 | – |
| **Laya** | 150 | **0.591** | **0.451** | **0.156** | **0.855** | **0.032** | **0.438** | 30 | 2.0 min |
| ModernBERT heads | 150 | 0.506 | 0.403 | 0.189 | 1.003 | 0.068 | 0.562 | 21 | 0.4 min |
| NLI fine-tune | 150 | 0.545 | 0.440 | 0.171 | 0.924 | 0.061 | 0.451 | 55 | 2.4 min |
| Laya | 1,075 | 0.733 | 0.539 | 0.120 | 0.625 | **0.033** | 0.318 | 30 | 14.0 min |
| **ModernBERT heads** | 1,075 | **0.781** | **0.583** | 0.124 | **0.512** | 0.037 | **0.270** | **20** | **2.5 min** |
| NLI fine-tune | 1,075 | 0.655 | 0.497 | 0.151 | 0.773 | 0.034 | 0.371 | 59 | 18.4 min |

For reference, the official `laya-typed-decisions` checkpoint (all 1,200 cases, from its authors) scores 0.767.

**Accuracy by question type:**

| method | train cases | choice | yes/no | score |
|---|---|---|---|---|
| Laya | 0 | 0.285 | 0.488 | 0.321 |
| DeBERTa-v3-large zero-shot | 0 | **0.570** | **0.635** | 0.444 |
| Laya | 150 | **0.597** | **0.680** | **0.520** |
| ModernBERT heads | 150 | 0.468 | 0.643 | 0.432 |
| NLI fine-tune | 150 | 0.535 | 0.613 | 0.501 |
| Laya | 1,075 | 0.718 | 0.822 | 0.676 |
| ModernBERT heads | 1,075 | **0.748** | **0.840** | **0.762** |
| NLI fine-tune | 1,075 | 0.637 | 0.757 | 0.594 |

The ordinal score questions show the biggest gap at full data (0.762 vs 0.676): plain heads handle the 0–4 scales
better than Laya's option-marker scoring.

**Calibration before any post-fit** (raw ECE, lower is better): Laya 0.034 (1,075) and 0.061 (150); ModernBERT
heads 0.154 and 0.056; NLI fine-tune 0.100 and 0.028; zero-shot models 0.13–0.54. Laya's RLCD training really does
produce honest probabilities by default. The others get there only with a calibration step, which this repo's
pipeline applies to Laya too.

**Why the result flips with data size.** Laya reads each question, and each option's description, as part of its
input. So it can transfer what it learned from one question to another, which helps most when every question has
only a few dozen examples. The plain heads treat each question as its own classifier, and need more data per
question, but with enough they fit each one more closely and pay no per-question encoding cost. There is also a
structural trade-off the table cannot show: heads are fixed to the questions they were trained on, while Laya can
take a reworded or new question at request time (though zero-shot it answers them poorly).

## Part 2: specialists on their home tasks

**Setup.** Each task uses a seeded sample from its public dataset (fetched with
[`train/prepare_specialists.py`](../train/prepare_specialists.py), through the Hub rows API, with no bulk download).
Laya fine-tunes use this repo's `train.py` on 500 training rows (450 train / 50 calibration). The specialists are
used off the shelf, as you would deploy them. Probabilities are scored as each model outputs them, with no
recalibration. Positive class for toxicity: at least half of raters said toxic; Brier is scored against the rater
fraction.

**Prompt injection:** [`deepset/prompt-injections`](https://huggingface.co/datasets/deepset/prompt-injections),
the full 116-case test split (60 injections; some prompts are German).

| method | accuracy | F1 (injection) | AUROC | Brier ↓ | ECE ↓ | ms/item |
|---|---|---|---|---|---|---|
| protectai/deberta-v3-base-prompt-injection-v2 (specialist) | 0.672 | 0.537 | 0.899 | 0.321 | 0.324 | 19 |
| Granite Guardian 3.2 5B (risk `jailbreak`) | 0.569 | 0.306 | 0.887 | 0.380 | 0.391 | 53 |
| DeBERTa-v3-large zero-shot | 0.595 | 0.505 | 0.710 | 0.302 | 0.287 | 37 |
| Laya zero-shot | 0.724 | 0.667 | 0.865 | 0.201 | 0.212 | 24 |
| **Laya fine-tuned (450)** | **0.948** | **0.948** | **0.997** | **0.039** | **0.037** | 23 |

**Toxicity:** [`google/civil_comments`](https://huggingface.co/datasets/google/civil_comments), 1,000 sampled test
comments (87 toxic).

| method | accuracy | F1 (toxic) | AUROC | Brier ↓ | ECE ↓ | ms/item |
|---|---|---|---|---|---|---|
| unitary/toxic-bert (specialist) | 0.944 | 0.627 | 0.922 | 0.027 | 0.028 | **8** |
| Granite Guardian 3.2 5B (risk `harm`) | 0.720 | 0.300 | 0.801 | 0.132 | 0.114 | 49 |
| Granite Guardian 3.2 5B (risk `profanity`) | 0.905 | 0.481 | 0.874 | 0.040 | 0.020 | 60 |
| DeBERTa-v3-large zero-shot | 0.595 | 0.281 | 0.860 | 0.297 | 0.313 | 37 |
| Laya zero-shot | 0.951 | **0.703** | **0.962** | 0.033 | 0.168 | 22 |
| **Laya fine-tuned (450)** | **0.952** | 0.684 | 0.955 | **0.020** | **0.019** | 23 |

**Sentiment:** [`cardiffnlp/tweet_eval`](https://huggingface.co/datasets/cardiffnlp/tweet_eval) (`sentiment`),
1,000 sampled test tweets.

| method | accuracy | macro-F1 | Brier ↓ | ECE ↓ | ms/item |
|---|---|---|---|---|---|
| **cardiffnlp/twitter-roberta-base-sentiment-latest (specialist)** | **0.712** | **0.718** | **0.392** | 0.056 | **8** |
| DeBERTa-v3-large zero-shot | 0.664 | 0.661 | 0.610 | 0.286 | 37 |
| Laya zero-shot | 0.608 | 0.605 | 0.528 | 0.107 | 22 |
| Laya fine-tuned (450) | 0.651 | 0.648 | 0.467 | **0.030** | 22 |

**Reading these fairly:**

- **The sentiment specialist is on its home benchmark:** it was fine-tuned on TweetEval's training data. The other
  specialists were not trained on these exact datasets.
- **Laya fine-tuned on in-domain data,** the specialists did not. "Laya + 450 examples beats an off-the-shelf
  specialist on that domain" is what the fine-tuned rows show. Prompt injection, where deepset's mild, partly German
  "injections" differ from what the specialist and Granite were built for, is the clearest case.
- **Laya's zero-shot strength on toxicity and injection** may reflect its training data, which is not published.
  Moderation and guardrails are among its advertised uses, and its SDK ships presets for them. It is weakest where
  it had no such head start (sentiment, typed-decisions).
- **Granite Guardian depends heavily on the risk name.** `profanity` fits civil_comments far better than the broader
  `harm` (0.905 vs 0.720 accuracy). Its AUROC stays high while accuracy suffers, which means its default yes/no
  threshold is off for these datasets. It is also the slowest model here, and generative.
- **Specialists are the fastest:** 8 ms, against Laya's ~22 ms. Laya's latency here is the eager SDK path; the
  API's CUDA graphs bring a lone request to ~7 ms.
- **`meta-llama/Llama-Prompt-Guard-2-86M` was not tested:** it is gated, and this account has not been granted
  access.

## Limitations

- **One run per configuration.** Laya's earlier seed repeats varied by under 1 point; the others were not repeated.
- **Untuned defaults for everyone.** The NLI fine-tune ran 2 epochs (it costs one pass per option), and more might
  help it. Laya's own recipe is its authors'.
- **Test-set size.** Specialist test sets are samples (1,000 rows; 116 for prompt injection), so differences of a
  couple of points there are within noise.
- **One domain.** typed-decisions is one benchmark, with synthetic teacher labels. Results on your own schema can
  differ, and `make evaluate` against a majority baseline is the check that matters.
