"""FastAPI server for Laya, a non-autoregressive System 1 decision model.

Laya never generates text. It takes a *state* (text, an email, a ticket, any
JSON) plus typed *questions* and answers all of them in one encoder forward
pass with calibrated probabilities:

  choice  pick one of N labels            -> {"choice", "probabilities", "confidence"}
  score   ordinal level 0..K-1            -> {"score" (expected level), "probabilities"}
  noul    yes/no probability              -> {"noul" (P(true))}

Serving uses the official `laya` SDK so behaviour matches the model card:
its Router picks the English (ModernBERT-large) or multilingual (mmBERT-base)
checkpoint from the state's script/language, and the typed-decisions
fine-tune is reachable explicitly. All checkpoints come from the local,
read-only ./models mount; nothing is fetched at runtime.

Design notes worth preserving when you edit:
  * Checkpoints load once at startup and stay resident (Router preload).
    With max_loaded=1 the SDK would rebuild a model on every language switch.
  * Inference runs in a worker thread so /health stays responsive, behind a
    lock because the SDK's GPU->CPU OOM fallback mutates the agent.
  * /v1/decide/batch does not call system_one per state: batching.py runs
    every (state, question) sequence in length-sorted chunks, taking the lock
    per chunk. scripts/test_batch_equivalence.py guards it against drift from
    the SDK's post-processing.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import threading
import time
import warnings
from contextlib import asynccontextmanager
from typing import Any, Literal, Optional, Union  # noqa: F401

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("api")

MODEL_DIR = os.environ.get("MODEL_DIR", "/models/convaiinnovations__laya")
SERVED_MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "laya")
DEVICE = os.environ.get("DEVICE", "cuda")
DEFAULT_CHECKPOINT = os.environ.get("DEFAULT_CHECKPOINT", "english")
PRELOAD = [p.strip() for p in os.environ.get("PRELOAD", "english,multilingual,typed-decisions").split(",") if p.strip()]
SHADOW_ROOT = os.environ.get("SHADOW_ROOT", "/tmp/laya-shadow")
# Every subdirectory holding an rl_agent_config.json is served as an extra, explicitly
# selectable checkpoint named after the directory (see train/publish.py).
FINETUNED_DIR = os.environ.get("FINETUNED_DIR", "/finetuned")
# /v1/decide/batch: padded tokens per forward pass. 8192 was near-best at every batch size on an
# RTX 3090 (4k-64k measured: throughput saturates ~44k tok/s; bigger budgets only add padding and VRAM).
BATCH_TOKEN_BUDGET = int(os.environ.get("BATCH_TOKEN_BUDGET", "8192"))
BATCH_MAX_STATES = int(os.environ.get("BATCH_MAX_STATES", "1024"))

# checkpoint name -> subfolder inside the bundle repo (None = repo root)
SUBFOLDERS = {"english": None, "multilingual": "multilingual", "typed-decisions": "typed-decisions"}

STATE: dict[str, Any] = {"ready": False, "error": None, "checkpoints": {}, "warnings": [], "custom": {}}
LOCK = threading.Lock()


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _tokenizer_needs_fix(ckpt_dir: str) -> bool:
    """Mirror laya.agent._fix_tokenizer_config's trigger conditions.

    The SDK patches tokenizer_config.json *in place* when it is in a format some
    transformers versions reject. Our weights are mounted read-only, so the SDK's
    write would fail silently and the tokenizer would then fail to load.
    """
    path = os.path.join(ckpt_dir, "tokenizer", "tokenizer_config.json")
    try:
        with open(path) as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return False
    return cfg.get("tokenizer_class") in (None, "TokenizersBackend") or isinstance(cfg.get("extra_special_tokens"), list)


def _checkpoint_path(name: str) -> tuple[str, Optional[str]]:
    """(path, subfolder) for Router."""
    sub = SUBFOLDERS[name]
    src = os.path.join(MODEL_DIR, sub) if sub else MODEL_DIR
    if not os.path.isfile(os.path.join(src, "model.safetensors")):
        raise FileNotFoundError("checkpoint %r not found at %s - run scripts/download.py" % (name, src))
    shadow = _shadow_if_needed(name, src)
    return (shadow, None) if shadow else (MODEL_DIR, sub)


def _shadow_if_needed(name: str, src: str) -> Optional[str]:
    """If the tokenizer config needs the SDK's fix-up, build a writable shadow
    directory: symlinks to the read-only weights plus a private copy of the (tiny)
    tokenizer folder the SDK is allowed to rewrite."""
    if not _tokenizer_needs_fix(src):
        return None
    dst = os.path.join(SHADOW_ROOT, name)
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)
    for entry in ("rl_agent_config.json", "model.safetensors", "encoder"):
        os.symlink(os.path.join(src, entry), os.path.join(dst, entry))
    shutil.copytree(os.path.join(src, "tokenizer"), os.path.join(dst, "tokenizer"))
    log.info("checkpoint %s: tokenizer config needs the SDK fix-up; using writable shadow %s", name, dst)
    return dst


WARMUP_QUESTIONS = {"ok": {"type": "noul", "instructions": "Is this a test message?"}}


def load_models():
    from laya import Router

    t_all = time.time()
    names = [n for n in PRELOAD if n in SUBFOLDERS]
    models = {n: _checkpoint_path(n) for n in SUBFOLDERS}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        router = Router(models=models, device=DEVICE, max_loaded=max(1, len(names)), default=DEFAULT_CHECKPOINT)
        for n in names:
            t0 = time.time()
            agent = router.load(n)
            # First call compiles/initialises CUDA kernels; do it now, not on a user's request.
            agent.system_one("warm up", WARMUP_QUESTIONS)
            if agent.device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
            agent.system_one("warm up", WARMUP_QUESTIONS)
            STATE["checkpoints"][n] = _describe(agent, models[n][0] if not models[n][1] else os.path.join(*models[n]),
                                                t0, t1, "shipped")
            log.info("checkpoint %s ready: %s", n, STATE["checkpoints"][n])
        _load_finetuned()
    STATE["warnings"] = sorted({str(w.message) for w in caught if "laya" in str(w.message).lower()}) + STATE["warnings"]
    STATE["router"] = router
    STATE["load_seconds"] = round(time.time() - t_all, 1)
    if DEVICE.startswith("cuda"):
        off_gpu = [n for n, c in STATE["checkpoints"].items() if c["device"] != "cuda"]
        if off_gpu:  # the SDK falls back to CPU on OOM/arch errors with only a print
            STATE["warnings"].append("checkpoints fell back to CPU: %s" % ", ".join(off_gpu))
            log.error("checkpoints fell back to CPU: %s", off_gpu)
    STATE["ready"] = True
    log.info("all checkpoints ready in %ss", STATE["load_seconds"])


def _describe(agent, path, t0, t1, kind) -> dict:
    return {"kind": kind, "path": path, "device": agent.device.type,
            "dtype": str(agent.dtype).replace("torch.", ""),
            "encoder": agent.cfg.get("encoder"),
            "max_len": agent.cfg.get("max_len"),
            "head_max_len": agent.cfg.get("head_max_len"),
            "params_millions": round(sum(p.numel() for p in agent.model.parameters()) / 1e6, 1),
            "load_seconds": round(t1 - t0, 2),
            "warm_latency_ms": round((time.time() - t1) * 1000, 1),
            "temperature_clamped": {k: [round(v, 4), agent.temperature_by_options[k]]
                                    for k, v in agent.temperature_by_options_raw.items()
                                    if agent.temperature_by_options.get(k) != v}}


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _load_finetuned():
    """Load every published fine-tune. A broken one is reported, not fatal."""
    from laya import Agent
    from laya.router import _ALIASES
    if not os.path.isdir(FINETUNED_DIR):
        return
    for name in sorted(os.listdir(FINETUNED_DIR)):
        src = os.path.join(FINETUNED_DIR, name)
        if not os.path.isfile(os.path.join(src, "rl_agent_config.json")):
            continue
        if name in SUBFOLDERS or name in _ALIASES or name in ("auto", SERVED_MODEL_NAME):
            STATE["warnings"].append("fine-tuned %r skipped: name clashes with a shipped checkpoint" % name)
            continue
        try:
            t0 = time.time()
            agent = Agent(_shadow_if_needed(name, src) or src, device=DEVICE)
            agent.system_one("warm up", WARMUP_QUESTIONS)
            t1 = time.time()
            agent.system_one("warm up", WARMUP_QUESTIONS)
        except Exception as e:
            STATE["warnings"].append("fine-tuned %r failed to load: %s: %s" % (name, type(e).__name__, e))
            log.exception("fine-tuned %s failed to load", name)
            continue
        d = _describe(agent, src, t0, t1, "fine-tuned")
        summ = _read_json(os.path.join(src, "training_summary.json")) or {}
        ev = _read_json(os.path.join(src, "eval.json")) or {}
        d["training"] = {k: summ.get(k) for k in ("base", "finished_at", "gpu", "timing", "memory")} | {
            "train_cases": (summ.get("data") or {}).get("train_cases"),
            "questions": sorted(((summ.get("data") or {}).get("per_question") or {}))}
        d["eval"] = {m: r.get("overall") for m, r in (ev.get("results") or {}).items()}
        STATE.setdefault("schemas", {})[name] = _read_json(os.path.join(src, "schemas.json")) or {}
        STATE["checkpoints"][name] = d
        STATE["custom"][name] = agent
        log.info("fine-tuned checkpoint %s ready from %s", name, src)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_models()
    except Exception as e:  # keep the container up so /health can report why
        STATE["error"] = "{}: {}".format(type(e).__name__, e)
        log.exception("Model failed to load")
    yield


app = FastAPI(title="Laya decision API", lifespan=lifespan,
              description="Typed, calibrated decisions from a non-autoregressive encoder. See /docs.")


def _router():
    if STATE["error"]:
        raise HTTPException(503, detail=STATE["error"])
    if not STATE["ready"]:
        raise HTTPException(503, detail="model still loading")
    return STATE["router"]


# --------------------------------------------------------------------------
# Introspection
# --------------------------------------------------------------------------
@app.get("/health")
def health():
    _router()
    return {"status": "ok", "model": SERVED_MODEL_NAME, "checkpoints": list(STATE["checkpoints"])}


@app.get("/info")
def info():
    gpu = None
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "count": torch.cuda.device_count(),
            "vram_total_gb": round(total / 1e9, 2),
            "vram_free_gb": round(free / 1e9, 2),
            "vram_used_gb": round((total - free) / 1e9, 2),
            "torch_allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
            "torch_reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability())),
        }
    cks = STATE["checkpoints"]
    import laya
    import transformers
    return {
        "model": SERVED_MODEL_NAME, "model_dir": MODEL_DIR, "modality": "decision",
        "ready": STATE["ready"], "error": STATE["error"], "load_seconds": STATE.get("load_seconds"),
        "default_checkpoint": DEFAULT_CHECKPOINT, "checkpoints": cks,
        "all_on_gpu": bool(cks) and all(c["device"] == "cuda" for c in cks.values()),
        "warnings": STATE["warnings"],
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "laya": laya.__version__},
        "gpu": gpu,
    }


@app.get("/v1/models")
def list_models():
    data = [{"id": SERVED_MODEL_NAME, "object": "model", "owned_by": "local",
             "description": "auto-routed across the loaded checkpoints"}]
    data += [{"id": n, "object": "model", "owned_by": "local", "kind": c["kind"], "encoder": c["encoder"],
              "max_len": c["max_len"], **({"training": c["training"], "eval": c["eval"]} if c["kind"] == "fine-tuned" else {})}
             for n, c in STATE["checkpoints"].items()]
    return {"object": "list", "data": data}


@app.get("/v1/presets")
def presets():
    """The SDK's ready-made question sets for common workflows."""
    import laya
    out = {}
    for name in ("triage_questions", "email_questions", "guard_questions", "moderation_questions", "router_questions"):
        fn = getattr(laya, name, None)
        if fn is None:
            continue
        try:
            out[name.replace("_questions", "")] = fn()
        except TypeError:
            pass  # presets that need arguments are not offered as defaults
    for name, sch in STATE.get("schemas", {}).items():  # questions each fine-tune was trained on
        for wf, v in sch.items():
            out["%s: %s" % (name, wf)] = v["questions"]
    return out


@app.get("/v1/preset_examples")
def preset_examples():
    """An example state for each fine-tune schema preset (from its held-out data)."""
    return {"%s: %s" % (name, wf): {"state": v.get("example_state"), "model": name}
            for name, sch in STATE.get("schemas", {}).items() for wf, v in sch.items()}


# --------------------------------------------------------------------------
# Decisions
# --------------------------------------------------------------------------
class Question(BaseModel):
    type: Literal["choice", "score", "noul"]
    instructions: Union[str, dict, list]
    # choice: {label: description|null} or [labels]; score: [level descriptions];
    # noul: optional {"true": ..., "false": ...}
    criteria: Optional[Union[dict[str, Optional[str]], list[str]]] = None


# "auto"/None routes; a shipped name or alias forces it; any published fine-tune name selects it.
Checkpoint = Optional[str]


class DecideRequest(BaseModel):
    state: Union[str, dict, list] = Field(..., description="text, JSON object, or conversation turns")
    questions: dict[str, Question] = Field(..., min_length=1)
    model: Checkpoint = Field(None, description="checkpoint to force; omit/auto to route")
    task: Optional[str] = Field(None, description='"typed_decisions" to force the fine-tuned checkpoint')
    lang: Optional[str] = Field(None, description="skip detection: 'en' -> english, else multilingual")


class RouteRequest(BaseModel):
    state: Union[str, dict, list]
    questions: Optional[dict[str, Question]] = None
    model: Checkpoint = None
    task: Optional[str] = None
    lang: Optional[str] = None


class CompareRequest(BaseModel):
    state: Union[str, dict, list]
    questions: dict[str, Question] = Field(..., min_length=1)


class BatchRequest(BaseModel):
    states: list[Union[str, dict, list]] = Field(..., min_length=1)
    questions: dict[str, Question] = Field(..., min_length=1)
    model: Checkpoint = None


def _questions(qs: dict[str, Question]) -> dict:
    out = {}
    for qid, q in qs.items():
        d = q.model_dump(exclude_none=True)
        crit = d.get("criteria")
        if q.type == "choice" and (not crit or len(crit) < 2):
            raise HTTPException(422, "question %r: choice needs at least 2 criteria (labels)" % qid)
        if q.type == "score" and (not isinstance(crit, list) or len(crit) < 2):
            raise HTTPException(422, "question %r: score needs criteria as a list of >= 2 levels" % qid)
        if q.type == "noul" and crit is not None and not isinstance(crit, dict):
            raise HTTPException(422, "question %r: noul criteria must be {\"true\": ..., \"false\": ...}" % qid)
        out[qid] = d
    return out


def _model_arg(m: Optional[str]) -> Optional[str]:
    if m in (None, "", "auto", SERVED_MODEL_NAME):
        return None
    if m in STATE["custom"]:
        return m
    from laya.router import normalise_name
    try:
        return normalise_name(m)
    except ValueError:
        raise HTTPException(400, "unknown model %r; available: auto, %s" % (m, ", ".join(STATE["checkpoints"])))


def _ensure_loaded(name: str):
    if name not in STATE["checkpoints"]:
        raise HTTPException(400, "checkpoint %r is not loaded (PRELOAD=%s)" % (name, ",".join(PRELOAD)))


def _resolve(router, state, questions, **kw):
    """(routing decision, agent) for one state."""
    if kw.get("model") in STATE["custom"]:
        decision = {"model": kw["model"], "repo": STATE["checkpoints"][kw["model"]]["path"],
                    "reason": "explicit fine-tuned checkpoint %r" % kw["model"], "detection": None, "workflow": None}
        return decision, STATE["custom"][kw["model"]]
    decision = router.route(state, questions, **kw)
    _ensure_loaded(decision["model"])
    return decision, router.load(decision["model"])


def _routing_view(decision) -> dict:
    return {k: v for k, v in dict(decision).items() if k != "detection"} | {
        "detection": {k: v for k, v in (decision.get("detection") or {}).items() if k != "script_profile"} or None}


def _predict(router, state, questions, **kw) -> dict:
    decision, agent = _resolve(router, state, questions, **kw)
    t0 = time.perf_counter()
    try:
        with LOCK:
            res = agent.system_one(state, questions)
    except ValueError as e:  # e.g. options do not fit in head_max_len
        raise HTTPException(422, str(e))
    res["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    res["routing"] = _routing_view(decision)
    return res


@app.post("/v1/route")
def route(req: RouteRequest):
    """Which checkpoint would answer this, and why - without running the model."""
    router = _router()
    qs = _questions(req.questions) if req.questions else None
    m = _model_arg(req.model)
    if m in STATE["custom"]:
        return {"model": m, "reason": "explicit fine-tuned checkpoint %r" % m}
    return dict(router.route(req.state, qs, model=m, task=req.task, lang=req.lang))


@app.post("/v1/decide")
async def decide(req: DecideRequest):
    """Answer every question about `state` in one forward pass (Jev-style system_one)."""
    router = _router()
    qs = _questions(req.questions)
    return await asyncio.to_thread(_predict, router, req.state, qs,
                                   model=_model_arg(req.model), task=req.task, lang=req.lang)


# Jev-compatible alias: the request/response shape is the same.
app.add_api_route("/v1/system_one", decide, methods=["POST"], include_in_schema=True)


@app.post("/v1/compare")
async def compare(req: CompareRequest):
    """Same state and questions on every loaded checkpoint, side by side."""
    router = _router()
    qs = _questions(req.questions)
    auto = router.route(req.state, qs)["model"]

    def run():
        return {n: _predict(router, req.state, qs, model=n) for n in STATE["checkpoints"]}
    return {"auto_route": auto, "results": await asyncio.to_thread(run)}


@app.post("/v1/decide/batch")
async def decide_batch(req: BatchRequest):
    """Many states, one question set, truly batched on the GPU (see batching.py).

    Each state is routed independently, then states are grouped by checkpoint and
    every (state, question) sequence runs in length-sorted chunks. Results match
    /v1/decide per state (up to bf16 rounding); `latency_ms` per result is the
    amortised share of its checkpoint group's time.
    """
    from batching import decide_many
    router = _router()
    if len(req.states) > BATCH_MAX_STATES:
        raise HTTPException(422, "at most %d states per batch (BATCH_MAX_STATES)" % BATCH_MAX_STATES)
    qs = _questions(req.questions)
    model = _model_arg(req.model)

    def run():
        t0 = time.perf_counter()
        groups: dict[str, list[int]] = {}
        resolved = []
        for i, s in enumerate(req.states):
            decision, agent = _resolve(router, s, qs, model=model)
            resolved.append((decision, agent))
            groups.setdefault(decision["model"], []).append(i)
        results: list[Optional[dict]] = [None] * len(req.states)
        stats = {"sequences": 0, "forward_passes": 0, "tokens": 0, "groups": {}}
        for name, idx in groups.items():
            agent = resolved[idx[0]][1]
            tg = time.perf_counter()
            try:
                out, st = decide_many(agent, [req.states[i] for i in idx], qs,
                                      token_budget=BATCH_TOKEN_BUDGET, lock=LOCK)
            except ValueError as e:  # options do not fit in head_max_len
                raise HTTPException(422, str(e))
            share = (time.perf_counter() - tg) * 1000 / len(idx)
            for i, r in zip(idx, out):
                r["latency_ms"] = round(share, 2)
                r["routing"] = _routing_view(resolved[i][0])
                results[i] = r
            for k in ("sequences", "forward_passes", "tokens"):
                stats[k] += st[k]
            stats["groups"][name] = {"states": len(idx), **st}
        return results, (time.perf_counter() - t0) * 1000, stats
    results, total, stats = await asyncio.to_thread(run)
    return {"results": results, "total_ms": round(total, 1),
            "states_per_second": round(len(results) / max(total / 1000, 1e-9), 1), "batching": stats}
