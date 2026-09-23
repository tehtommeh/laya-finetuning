# Laya, running locally

[convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) (Apache 2.0) is a
**non-autoregressive decision model**. You give it a *state* (text, an email, a ticket, any JSON) and
typed *questions*. It answers every question in **one encoder forward pass** with probabilities.
It never generates text.

| type | you give | you get |
|---|---|---|
| `choice` | `criteria: {label: description}` (or a list of labels) | `choice`, `probabilities`, `confidence` |
| `score` | `criteria: [level0, level1, ...]` | `score` (expected level), `probabilities` |
| `noul` | a yes/no question (optional `{"true":..., "false":...}`) | `noul` = P(true) |

The repo bundles three checkpoints, and this stack keeps all three resident on the GPU:

| checkpoint | encoder | params | context (options budget) | best at |
|---|---|---|---|---|
| `english` (repo root) | ModernBERT-large | 421M | 512 (192) | English text, guardrails, email triage |
| `multilingual` | mmBERT-base | 322M | 1024 (256) | 100+ languages |
| `typed-decisions` | ModernBERT-large | 421M | 1024 (256) | the 4 typed-decisions workflows it was fine-tuned on |

Requests go through the official `laya` SDK's `Router`. It detects the script and language in under a
millisecond and sends non-English text to `multilingual`. `typed-decisions` is only used when you ask for it.

This repo does two things:

1. **Run Laya locally** with an API and a UI, in Docker on your GPU. See [Start / stop](#start--stop) below.
2. **Fine-tune Laya on your own decisions**, evaluate the result against the base model, and serve it
   alongside the shipped checkpoints. See [Fine-tuning](#fine-tuning) and the full guide in
   [`docs/FINETUNING.md`](docs/FINETUNING.md).

### Requirements

- An NVIDIA GPU with Docker GPU access (`scripts/preflight.py` checks this). Serving all three checkpoints takes
  ~6 GB of VRAM; fine-tuning ModernBERT-large needs ~9 GB more. CPU-only works for serving (slowly), not for training.
- Docker with compose v2, ~5 GB of disk for weights, and ~10 GB for images.
- `python3` on the host for the helper scripts (stdlib only; they bootstrap their own dependencies).

### Layout

| path | what |
|---|---|
| `api/` | FastAPI server (laya SDK `Router` + published fine-tunes) |
| `frontend/` | Gradio UI |
| `train/` | fine-tuning image: `validate.py`, `train.py`, `evaluate.py`, `publish.py`, `prepare_example.py`, `summarize.py` |
| `scripts/` | `download.py` (weights + update check), `smoke_test.py`, `preflight.py` |
| `models/` | downloaded base weights (read-only in containers) |
| `data/` | your training data; `data/sample/` shows the format |
| `runs/` | training outputs, one directory per run |
| `finetuned/` | published fine-tunes; the API serves each subdirectory |
| `docs/FINETUNING.md` | data format, amounts, hardware, timings, measured results |

## Start / stop

First run on a new machine:

```bash
python3 scripts/preflight.py --workdir . --ports api=8001,frontend=7860   # GPU, Docker GPU access, ports, disk
cp .env.example .env                        # then set LOCAL_UID/LOCAL_GID (`id -u`/`id -g`) and ports if needed
python3 scripts/download.py convaiinnovations/laya   # 2.3 GB of weights -> models/, pinned in download.lock.json
```

Then:

```bash
docker compose up -d --build     # first build ~5 min; startup ~55 s (loads + warms 3 checkpoints)
python3 scripts/smoke_test.py    # 19 checks against the live stack (make test waits for startup)
docker compose logs -f api
docker compose down
```

`make help` lists the shortcuts (`make up`, `make test`, `make check`, ...).

| URL | what |
|---|---|
| http://localhost:7860 | Gradio UI |
| http://localhost:8001/docs | interactive API docs |
| http://localhost:8001/info | GPU, per-checkpoint device/dtype/latency, warnings |

The API is on **8001** because 8000 was already taken on this host. Change `API_PORT` in `.env`.

## API

```bash
curl -s localhost:8001/v1/decide -H 'content-type: application/json' -d '{
  "state": {"subject": "Duplicate charge", "body": "Billed twice for March. Refund today or we cancel."},
  "questions": {
    "department": {"type": "choice", "instructions": "Which department should handle this?",
                   "criteria": {"billing": "invoices, refunds", "technical": "bugs, outages", "sales": "pricing", "other": "else"}},
    "urgency":    {"type": "score", "instructions": "How urgent?", "criteria": ["not urgent", "soon", "critical"]},
    "churn_risk": {"type": "noul",  "instructions": "Does the user threaten to cancel?"}
  }}'
```

| endpoint | purpose |
|---|---|
| `POST /v1/decide` (alias `/v1/system_one`) | answer questions; optional `model` (`auto`/`english`/`multilingual`/`typed-decisions`/a published fine-tune), `lang`, `task` |
| `POST /v1/route` | which checkpoint would answer, and why, without running the model |
| `POST /v1/compare` | the same input on every checkpoint |
| `POST /v1/decide/batch` | many states, one question set |
| `GET /v1/presets` | the SDK's question sets (triage, email, guard, moderation, router) plus each fine-tune's trained questions |
| `GET /health`, `/info`, `/v1/models` | introspection |

## Question types

Every question has a `type`, `instructions` and (usually) `criteria`. Laya answers all of a request's questions in
one batched forward pass. It never generates text, so an answer is always one of the options you defined, with
probabilities. Pick the type by the *shape* of the answer you need:

| type | answer shape | you get | typical use |
|---|---|---|---|
| `noul` | yes / no | `noul` = P(true) | flags, guardrails, gates |
| `choice` | exactly one of N labels | `choice`, `probabilities`, `confidence` | routing, categorisation, next action |
| `score` | a level on an ordered scale | `score` (expected level), `probabilities` | severity, urgency, rubric marking |

Measured accuracy on the typed-decisions test set (400 cases; [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)):

| type | always-most-common-label baseline | English base, zero-shot | fine-tuned (`td-full`) |
|---|---|---|---|
| `noul` | 0.642 | 0.488 | **0.822** |
| `choice` | 0.418 | 0.285 | **0.718** |
| `score` (level exactly right) | 0.335 | 0.321 | **0.676** (mean error 0.32 levels) |

Zero-shot, the base checkpoint does worse than the baseline on every type. Fine-tuning is what makes these
reliable for a specific schema (see [Fine-tuning](#fine-tuning)).

### `noul`: yes / no

```json
"refund": {"type": "noul", "instructions": "Does the customer ask for money back?",
           "criteria": {"true": "explicitly asks for a refund or chargeback", "false": "no request for money back"}}
```
→ `{"type": "noul", "noul": 0.90, "confidence": 0.90}`

- **Output:** `noul` is P(true). `confidence` is `max(p, 1 − p)`, i.e. how far from a coin flip.
- **`criteria` is optional.** Without it, the options are "yes, the statement holds" / "no, the statement does not
  hold". Spelling out what counts as true and false helps with borderline cases.
- **Good for:** flags and gates (`is_phishing`, `needs_human`, `contains_pii`), guardrails (`jailbreak`,
  `prompt_injection`), and anything you will threshold. It is the most accurate type after fine-tuning (0.82).
- **Thresholds:** after hard-label calibration (the default for fine-tunes), P(true) = 0.9 means right about 90%
  of the time, so you can pick a cut-off from the precision you need. The shipped checkpoints are
  over-confident, so don't threshold them without calibrating first.
- **Limits:** one statement per question. "Is it urgent *and* billing-related?" is two questions. Zero-shot on
  an unfamiliar schema it was worse than always answering the majority (0.49 vs 0.64).

### `choice`: one of N labels

```json
"team": {"type": "choice", "instructions": "Which team should handle this ticket?",
         "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages, errors",
                      "sales": "pricing, quotes", "other": "none of the above"}}
```
→ `{"type": "choice", "choice": "billing", "probabilities": {"billing": 0.94, "technical": 0.03, ...}, "confidence": 0.78}`

- **Output:** `choice` is the most likely label, and `probabilities` sums to 1 over your labels. `confidence` is
  1 − normalised entropy (Jev-style), **not** the probability of the top label. To threshold "how sure is it
  about this answer", use `probabilities[choice]`.
- **`criteria`:** `{label: description}` or a plain list of labels. The descriptions are part of the input, and
  good ones matter more than label names.
- **Good for:** routing, categorisation, picking a next action or disposition, intent.
- **Single-label only.** The probabilities compete, so "which of these apply?" should be one `noul` per label.
- **Always offer an escape option** (`other`, `none of the above`). Otherwise every input is forced into one of
  your labels, confidently.
- **Option count:** best under ~20. All options share a 256-token budget (192 on the English base), each one cut at
  48 tokens, so beyond ~20 the label texts get squeezed. The model card measures 0.425 on 77-option Banking77.
  For large label sets, raise `head_max_len`, split the question into coarse and fine steps, or pre-filter with
  the SDK's `laya.shortlist_choice` / `predict_shortlist`. The shipped temperature for 11+ options is clamped as
  invalid, so treat confidence there as uncalibrated.
- **Measured:** 0.72 accuracy fine-tuned. The 8-option `disposition` was the weakest choice question (0.66).

### `score`: a level on an ordered scale (rubrics)

```json
"quality": {"type": "score", "instructions": "Mark the agent's reply against the rubric.",
            "criteria": ["1 - wrong or unhelpful; ignores the question",
                         "2 - partly answers; missing key steps",
                         "3 - correct and complete, but unclear or impersonal",
                         "4 - correct, complete, clear, and addresses the customer's tone"]}
```
→ `{"type": "score", "score": 2.4, "probabilities": {"0": 0.02, "1": 0.10, "2": 0.35, "3": 0.53}, "legend": {...}}`

- **`criteria` is the rubric:** a list of level descriptions, lowest first. The model reads them when marking.
- **Output:** `score` is the *expected* level (0-based), a decimal that can fall between levels. Add 1 for a scale
  starting at 1 (2.4 → 3.4 above). `probabilities` shows the spread; the most likely level (here "4") can differ
  from the rounded expectation. Use `score` for averages, rankings and dashboards, and the top level when you
  need one discrete mark.
- **Good for:** severity, urgency, risk, sentiment intensity, and rubric-style marking of answers, replies or
  documents.
- **Multi-criterion rubrics:** one `score` question per criterion (accuracy, clarity, tone…), combined with your own
  weights. All criteria are marked in the same pass.
- **Limits:**
  - It is the weakest type. The model card rates ordinal `score` the weakest primitive (SST-5 0.372), and
    fine-tuned here it had the lowest exact-level accuracy (0.68), with `urgency` at 0.61. But it is usually
    close: mean error 0.32 levels.
  - Keep descriptors short: levels share the same 256-token option budget, each cut at 48 tokens. 3–5 levels
    work best, and 10+ levels with long descriptors get truncated (`make validate` reports it).
  - Descriptors can be structured (`{"desc": ..., "example": ...}`, rendered as compact JSON), but that uses the
    budget faster than plain text.
- **If exact marks matter, fine-tune on marked examples** and keep the rubric wording fixed afterwards.

### Behaviour shared by all types

- **Input budget:** each question is encoded as `[type + instructions + options] + [state]`. The English base
  allows 512 tokens in total with 192 for the question, which leaves roughly 320–500 for the state depending on
  option length. Fine-tunes here use 1024 / 256. Longer states are cut from the end, so put what matters first or
  pre-clean it (the SDK's `laya.email_state` strips quoted threads and signatures).
- **Cost:** each question is its own sequence in the batch. Measured through the API on the 3090 (one
  typed-decisions state): 1 question 24 ms, 5 questions 27 ms, 10 questions 41 ms. That is nearly flat up to ~5
  questions, then ~3 ms each, so asking several questions at once is cheap.
- **Wording is part of the model's input.** After fine-tuning, ask with the exact wording and options you trained
  on. The UI's `<model>: <workflow>` presets load them.
- **`action.act_probability`** (on every answer) comes from a separate "act vs escalate" head. This repo's
  fine-tuning does not train that head, so on fine-tuned models treat it as unreliable, and on the shipped
  checkpoints treat it as uncalibrated. For escalation decisions, prefer an explicit `noul` such as
  `needs_human`.
- **Nothing to parse and nothing to hallucinate,** but also no explanation. When you need a reason, ask an LLM
  afterwards, with Laya's decision as input.

## Fine-tuning

The base checkpoints are a starting point: on a real multi-workflow benchmark the English base scores *below* a
majority-label guess until fine-tuned. The `train` service does the whole loop in Docker on the same GPU:

```bash
make train-build                          # once
make validate DATA=/data/mine             # checks format, label balance, token budget
make train    DATA=/data/mine RUN=mine-v1 # ~1.3 min per 100 cases on an RTX 3090, ~9 GB VRAM
make evaluate RUN=mine-v1                 # held-out metrics vs the base and a majority baseline
make publish  RUN=mine-v1                 # served by the API/UI as model "mine-v1"
```

Data is JSONL: `{"state": ..., "gold": {"question": answer}}` with a shared `questions.json`. See
[`data/sample/`](data/sample) and the full guide, [`docs/FINETUNING.md`](docs/FINETUNING.md): what to collect,
format, how much, hardware, timings, what comes out, and how to serve it.

**Measured here** ([`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)), on LocalLLaMA/typed-decisions (4 workflows,
400-case test set):

| training cases | 0 | 30 | 150 | 300 | 600 | 1,075 |
|---|---|---|---|---|---|---|
| accuracy | 0.360 | 0.514 | 0.591 | 0.657 | **0.735** | 0.733 |
| train time (RTX 3090) | – | 0.4 min | 2.0 min | 3.9 min | 8.2 min | 14.0 min |

Gains flatten at ~150 cases per question set. Hard-label calibration then brings ECE to 0.034. `make experiments`
reproduces every number (the dataset is fetched at a pinned revision, not stored in the repo). The published
example model is `td-full`; check it with `python3 scripts/smoke_test.py --stack stack.finetuned.json`.

## Weights

- `./models/convaiinnovations__laya/` holds 2.3 GB for all three checkpoints plus the model card and assets. It is mounted **read-only** into the API container, and nothing is downloaded at runtime (`HF_HUB_OFFLINE=1`).
- The pinned revision is in `models/download.lock.json` (currently `1c5edc17a7ac`).

```bash
python3 scripts/download.py --check     # has upstream changed?
python3 scripts/download.py --update convaiinnovations/laya && docker compose restart api
python3 scripts/download.py --verify    # do local files match the lock?
```

## Measured on this host (RTX 3090 24 GB, driver 580, CUDA 12.6 image)

- **VRAM:** 4.7 GB allocated by torch (5.8 GB reserved) with all three checkpoints resident, in bf16.
- **Startup:** 53 s total (about 16–21 s per checkpoint, mostly encoder construction).
- **Latency:** 17–24 ms for a single question. The model-card email with 4 questions takes ~24 ms on `english`, ~20 ms on `multilingual`, and ~24 ms/state in a sequential batch.

## Things worth knowing

- **Short Latin-script text defaults to English.** The SDK needs at least two function words, or diacritics,
  to call Latin-script text non-English. "Mein Konto wurde zweimal belastet" (4 words, 1 hit, no umlauts) routes to
  `english`, while the full German sentence routes to `multilingual`. Non-Latin scripts are always detected. If you
  know the language, pass `"lang": "de"` (or `"model": "multilingual"`).
- **Probabilities ship over-confident.** Per the model card, fit a temperature on your own data before trusting
  them (`make calibrate` does this for fine-tunes; the shipped checkpoints keep their own temperatures). The `choice:11+` temperature shipped with `english`/`typed-decisions` (0.10) is clamped to 0.5 by the SDK,
  so treat confidence on 11+-option questions as uncalibrated. `/info` reports this under `warnings`.
- **The base checkpoints are a base to fine-tune, not a zero-shot oracle** (0.36 on typed-decisions against a 0.45
  majority baseline). `typed-decisions` scores 0.77 only on the workflows it was trained on. See [Fine-tuning](#fine-tuning).
- **Many options share a fixed token budget** (192/256 tokens). Beyond about 20 options, labels get truncated and
  accuracy drops. Split the options hierarchically.
- **CPU fallback is surfaced.** The SDK silently falls back to CPU on OOM or an unsupported GPU. `/info` reports
  `all_on_gpu`, and the smoke test fails if it is false.
- **Other hardware:** pre-Ampere GPUs automatically use fp16. On CPU, set `DEVICE=cpu` and drop the
  `deploy:` GPU block (the model card quotes ~200–460 ms per call). To save memory, trim `PRELOAD`.
- The image pins `transformers==5.17.0` because the shipped encoder configs use the v5 format.
