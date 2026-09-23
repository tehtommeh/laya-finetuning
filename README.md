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
