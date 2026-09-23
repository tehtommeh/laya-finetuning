#!/usr/bin/env python3
"""Check /v1/decide/batch agrees with per-state /v1/decide on a live stack.

    python3 scripts/test_batch_equivalence.py [--api http://localhost:8001] [--n 60] [--tol 0.02]

The batch endpoint re-implements laya's post-processing to run many states in
one forward pass (api/batching.py), so this is the guard against drift. It
compares, per state and question: routing and token counts (exact), every
probability (within --tol; bf16 kernels differ slightly with batch shape and
padding), the expected score (within 3x --tol), and the top answer (must match
unless the top two options are within --tol of each other). Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

QUESTIONS = {
    "team": {"type": "choice", "instructions": "Which team should handle this?",
             "criteria": {"billing": "payments, refunds", "technical": "bugs, outages", "sales": "pricing",
                          "other": "anything else"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["can wait", "this week", "today", "right now"]},
    "needs_human": {"type": "noul", "instructions": "Does this need a human?"},
}
EXTRA_STATES = [  # different lengths and scripts, so auto-routing produces several groups
    "Refund please.",
    "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।",
    "Mein Konto wurde zweimal belastet. Bitte erstatten Sie mir den Betrag, sonst kündige ich.",
    {"subject": "Outage", "body": "API returns 500 on every request since the deploy. " * 20},
    ["user: hi", "agent: hello, how can I help?", "user: cancel my plan"],
]


def post(api, path, body):
    req = urllib.request.Request(api + path, json.dumps(body).encode(), {"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def get(api, path):
    with urllib.request.urlopen(api + path, timeout=30) as r:
        return json.load(r)


def compare(single, batched, tol):
    problems, worst = [], 0.0
    if single["routing"]["model"] != batched["routing"]["model"]:
        problems.append("routing %s vs %s" % (single["routing"]["model"], batched["routing"]["model"]))
    if single["usage"] != batched["usage"]:
        problems.append("usage %s vs %s" % (single["usage"], batched["usage"]))
    for qid, a in single["answers"].items():
        b = batched["answers"][qid]
        pa = a.get("probabilities") or {"true": a["noul"], "false": 1 - a["noul"]}
        pb = b.get("probabilities") or {"true": b["noul"], "false": 1 - b["noul"]}
        diff = max(abs(pa[k] - pb[k]) for k in pa)
        worst = max(worst, diff)
        if diff > tol:
            problems.append("%s: probability differs by %.4f" % (qid, diff))
        if a["type"] == "score" and abs(a["score"] - b["score"]) > 3 * tol:
            problems.append("%s: score %.3f vs %.3f" % (qid, a["score"], b["score"]))
        if a["type"] == "choice" and a["choice"] != b["choice"]:
            top = sorted(pa.values(), reverse=True)
            if top[0] - top[1] > tol:
                problems.append("%s: choice %s vs %s" % (qid, a["choice"], b["choice"]))
    return problems, worst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", default="http://localhost:8001")
    ap.add_argument("--n", type=int, default=60, help="states to test (from the example test set if present)")
    ap.add_argument("--tol", type=float, default=0.02)
    a = ap.parse_args()

    states = list(EXTRA_STATES)
    path = os.path.join(os.path.dirname(__file__), "..", "data", "typed-decisions", "test.jsonl")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            states += [json.loads(line)["state"] for _, line in zip(range(a.n), f)]
    models = ["auto"] + [m["id"] for m in get(a.api, "/v1/models")["data"] if m.get("kind") == "fine-tuned"][:1]

    failures = 0
    for model in models:
        status, batch = post(a.api, "/v1/decide/batch", {"states": states, "questions": QUESTIONS, "model": model})
        if status != 200:
            print("FAIL %s: batch returned HTTP %s %s" % (model, status, batch))
            failures += 1
            continue
        worst, bad = 0.0, 0
        t0 = time.perf_counter()
        for i, s in enumerate(states):
            status, single = post(a.api, "/v1/decide", {"state": s, "questions": QUESTIONS, "model": model})
            problems, w = compare(single, batch["results"][i], a.tol)
            worst = max(worst, w)
            if problems:
                bad += 1
                print("  state %d: %s" % (i, "; ".join(problems)))
        seq_ms = (time.perf_counter() - t0) * 1000
        failures += bad
        print("%s %-8s %d states: %d mismatched, max probability diff %.4f | batch %.0f ms vs sequential %.0f ms "
              "(%.1fx) | %s" % ("PASS" if not bad else "FAIL", model, len(states), bad, worst, batch["total_ms"],
                                seq_ms, seq_ms / max(batch["total_ms"], 1e-9),
                                {k: v for k, v in batch["batching"].items() if k != "groups"}))
        print("         groups: %s" % {g: v["states"] for g, v in batch["batching"]["groups"].items()})

    status, _ = post(a.api, "/v1/decide/batch", {"states": ["x"], "questions": {
        "q": {"type": "choice", "instructions": "?", "criteria": ["only-one"]}}})
    print("%s invalid question -> HTTP %s (want 422)" % ("PASS" if status == 422 else "FAIL", status))
    failures += status != 422
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
