"""True cross-state batching for laya Agents.

`laya.Agent.system_one(state, questions)` runs one forward pass per *state*:
all that state's questions are batched together, but N states cost N passes.
Nothing in the model requires that - every (state, question) pair is an
independent sequence. `decide_many` encodes all pairs for many states, sorts
them by length to minimise padding, runs them in chunks under a token budget,
and splits the results back per state.

The encoding (`build_sequence`, `_to_internal`, `collate_items`) is the SDK's
own; `_answer` mirrors `system_one`'s post-processing line for line (same
temperatures, rounding and output shape). scripts/test_batch_equivalence.py
checks the two agree on a live stack. If the laya pin is bumped, re-run it.
"""
from __future__ import annotations

import threading
from typing import Any, Callable, Optional

import numpy as np
import torch

from laya.common import (QTYPES, build_sequence, collate_items, confidence_from_probs, render_options,
                         temp_bucket)


def _answer(agent, iq: dict, logits_row: np.ndarray, act_row: np.ndarray, k: int) -> dict:
    """One question's answer, exactly as laya.Agent.system_one builds it."""
    qt = QTYPES[iq["t"]]
    t_scale = agent.temperature_by_options.get(temp_bucket(qt, k), agent.temperature[qt])
    z = logits_row[:k] / t_scale
    p = np.exp(z - z.max())
    p = p / p.sum()
    conf_score = round(confidence_from_probs(p, k), 4)
    ext = {"act_probability": round(float(act_row[0]), 4)}
    if iq["t"] == "choice":
        keys = list(iq["crit"].keys())
        return {"type": "choice", "choice": keys[int(p.argmax())],
                "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
                "confidence": conf_score, "action": ext}
    if iq["t"] == "score":
        return {"type": "score", "score": round(float((np.arange(k) * p).sum()), 4),
                "legend": {str(i): c for i, c in enumerate(iq["crit"])},
                "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                "confidence": conf_score, "action": ext}
    return {"type": "noul", "noul": round(float(p[1]), 4),
            "confidence": round(max(float(p[1]), 1.0 - float(p[1])), 4), "action": ext}


@torch.no_grad()
def _forward(agent, items: list[dict], lock: Optional[threading.Lock]):
    """Run one chunk. On CUDA OOM, split it in half and retry, instead of the
    SDK's silent move of the whole model to CPU."""
    b = collate_items([items], agent.tok.pad_token_id)
    try:
        if lock is not None:
            lock.acquire()
        try:
            with torch.autocast(device_type=agent.device.type, dtype=agent.dtype,
                                enabled=agent.device.type == "cuda"):
                logits, act = agent.model(b["input_ids"].to(agent.device), b["attention_mask"].to(agent.device),
                                          b["marker_pos"].to(agent.device), b["marker_mask"].to(agent.device),
                                          b["qtype"].to(agent.device))
            return logits.float().cpu().numpy(), torch.softmax(act.float(), -1).cpu().numpy(), 1
        finally:
            if lock is not None:
                lock.release()
    except torch.cuda.OutOfMemoryError:
        if len(items) == 1:
            raise
        torch.cuda.empty_cache()
        half = len(items) // 2
        la, aa, na = _forward(agent, items[:half], lock)
        lb, ab, nb = _forward(agent, items[half:], lock)
        width = max(la.shape[1], lb.shape[1])
        pad = lambda x: np.pad(x, ((0, 0), (0, width - x.shape[1])), constant_values=-1e4)
        return np.concatenate([pad(la), pad(lb)]), np.concatenate([aa, ab]), na + nb


def decide_many(agent, states: list, questions: dict, token_budget: int = 16384, max_seqs: int = 256,
                lock: Optional[threading.Lock] = None) -> tuple[list[dict], dict]:
    """Answer the same questions for many states. Returns (results, stats).

    results[i] has the same shape as agent.system_one(states[i], questions).
    token_budget caps padded tokens per forward pass (longest sequence x count).
    The lock, if given, is held per chunk, so other requests can interleave.
    Raises ValueError, like system_one, if a question's options do not fit.
    """
    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)
    ids = list(questions)
    internal = {qid: agent._to_internal(questions[qid]) for qid in ids}

    items = []
    for si, state in enumerate(states):
        for qid in ids:
            iq = internal[qid]
            seq, markers = build_sequence(agent.tok, state, iq, max_len, head_max_len)
            if len(markers) != len(render_options(iq)):
                raise ValueError("question %r options exceed head_max_len=%d" % (qid, head_max_len))
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[iq["t"]], "_s": si, "_q": qid})

    # length-sorted chunks under the padded-token budget
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    chunks, cur, cur_max = [], [], 0
    for i in order:
        n = len(items[i]["ids"])
        if cur and (max(cur_max, n) * (len(cur) + 1) > token_budget or len(cur) >= max_seqs):
            chunks.append(cur)
            cur, cur_max = [], 0
        cur.append(i)
        cur_max = max(cur_max, n)
    if cur:
        chunks.append(cur)

    answers: list[dict] = [dict() for _ in states]
    tokens = [0] * len(states)
    passes, padded = 0, 0
    for chunk in chunks:
        chunk_items = [items[i] for i in chunk]
        logits, act, n = _forward(agent, chunk_items, lock)
        passes += n
        padded += max(len(it["ids"]) for it in chunk_items) * len(chunk_items)
        for r, it in enumerate(chunk_items):
            k = len(it["markers"])
            answers[it["_s"]][it["_q"]] = _answer(agent, internal[it["_q"]], logits[r], act[r], k)
            tokens[it["_s"]] += len(it["ids"])

    results = [{"model": "laya-rl-agent", "answers": {qid: answers[s][qid] for qid in ids},
                "usage": {"input_tokens": tokens[s], "output_tokens": 0}} for s in range(len(states))]
    real = sum(len(it["ids"]) for it in items)
    return results, {"sequences": len(items), "forward_passes": passes, "tokens": real,
                     "padding_overhead": round(padded / max(1, real) - 1, 3)}
