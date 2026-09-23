#!/usr/bin/env python3
"""Live leak test for request coalescing: concurrent API results vs the SDK itself.

    docker compose exec -T api python - < scripts/test_coalescing.py     # or: make test-coalescing

Runs inside the API container. Loads its own reference copies of the checkpoints
through laya.Agent (the SDK's untouched system_one path), then fires a mixed,
concurrent workload at the live API: /v1/decide on every checkpoint, batches,
/v1/compare and invalid requests. Every request has unique question ids, option
labels and state text, so any result that crossed between requests shows up as
a wrong key. Every value is then checked against system_one for the same
(state, questions, checkpoint), and within each batch every state must be
closer to its own reference than to any other state's (catches swaps that
keys cannot, since a batch's states share question ids). Exit status 1 on any
mismatch.

Three separate guarantees are checked:

  1. No leaks (exact, no tolerance): question ids, option labels, routing and
     token counts per result, plus the within-batch swap check above.
  2. Exact computation: 100 requests re-sent one at a time, after the
     concurrent phase, must match system_one to 1e-4. A request alone is
     encoded and batched exactly as the SDK does it, so any post-processing
     or encoding bug shows here.
  3. Values under concurrency within bf16 batch-shape noise: the same row in a
     differently padded batch gives slightly different numbers in bf16. With the
     SDK alone (no scheduler) that reached 0.13 in probability for sensitive
     inputs (probability spread across neighbouring options), so each value must
     be within TOL = 0.15, and the p99 difference within 0.05 so a systematic
     drift still fails.
"""
from __future__ import annotations

import concurrent.futures
import glob
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

import laya

API = os.environ.get("API", "http://localhost:8000")
REQUESTS = int(os.environ.get("REQUESTS", "600"))
CLIENTS = int(os.environ.get("CLIENTS", "64"))
TOL = float(os.environ.get("TOL", "0.15"))            # per value under concurrency; expected score gets 2x
P99_MAX = 0.05                                            # the bulk of the distribution must stay tight
EXACT = 1e-4                                              # solo requests vs system_one
MODEL_DIR = os.environ.get("MODEL_DIR", "/models/convaiinnovations__laya")

TEXTS = [
    "I was charged twice for invoice {n}, please refund one.", "The API has returned 500 errors since deploy {n}.",
    "Could you send a quote for {n} seats with SSO?", "Password reset email {n} never arrived.",
    "Cancel subscription {n} today, the product never worked.", "Dashboard {n} takes a minute to load.",
    "मुझसे ऑर्डर {n} के लिए दो बार शुल्क लिया गया, कृपया पैसे वापस करें।",
    "Mein Konto wurde für Rechnung {n} zweimal belastet. Bitte erstatten Sie den Betrag, sonst kündige ich.",
    "注文{n}の請求が二重になっています。返金をお願いします。",
]
WORDS = "payments refunds bugs outages pricing quotes access login security billing delivery other".split()


def post(path, body):
    req = urllib.request.Request(API + path, json.dumps(body).encode(), {"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def get(path):
    with urllib.request.urlopen(API + path, timeout=30) as r:
        return json.load(r)


def make_state(rnd, rid, j):
    n = "%d-%d" % (rid, j)
    kind = rnd.random()
    if kind < 0.6:
        return rnd.choice(TEXTS).format(n=n)
    if kind < 0.85:
        return {"ticket": n, "subject": rnd.choice(TEXTS).format(n=n),
                "body": " ".join(rnd.choice(TEXTS).format(n=n) for _ in range(rnd.randint(1, 8)))}
    return ["user: " + rnd.choice(TEXTS).format(n=n), "agent: how can I help with %s?" % n,
            "user: " + rnd.choice(TEXTS).format(n=n)]


def make_questions(rnd, rid):
    qs = {}
    for j in range(rnd.randint(1, 5)):
        qid = "q%d_%d" % (rid, j)
        t = rnd.choice(["choice", "score", "noul"])
        if t == "choice":
            k = rnd.randint(2, 6)
            qs[qid] = {"type": "choice", "instructions": "Which area is request %d about?" % rid,
                       "criteria": {"L%d_%d_%d" % (rid, j, i): rnd.choice(WORDS) for i in range(k)}}
        elif t == "score":
            k = rnd.randint(2, 5)
            qs[qid] = {"type": "score", "instructions": "How urgent is request %d?" % rid,
                       "criteria": ["level %d of %d (r%d)" % (i, k, rid) for i in range(k)]}
        else:
            qs[qid] = {"type": "noul", "instructions": "Does request %d ask for money back?" % rid}
    return qs


def build_workload(checkpoints):
    rnd = random.Random(1234)
    work = []
    for rid in range(REQUESTS):
        r = rnd.random()
        qs = make_questions(rnd, rid)
        if r < 0.70:
            work.append(("decide", rid, {"state": make_state(rnd, rid, 0), "questions": qs,
                                         "model": rnd.choice(["auto"] + checkpoints)}))
        elif r < 0.85:
            work.append(("batch", rid, {"states": [make_state(rnd, rid, j) for j in range(rnd.randint(2, 40))],
                                        "questions": qs, "model": rnd.choice(["auto"] + checkpoints)}))
        elif r < 0.95:
            work.append(("compare", rid, {"state": make_state(rnd, rid, 0), "questions": qs}))
        else:
            work.append(("invalid", rid, {"state": "x", "questions": {"bad%d" % rid: {
                "type": "choice", "instructions": "?", "criteria": ["only-one"]}}}))
    return work


def distance(a, b):
    """Largest probability difference between two results with the same questions."""
    d = 0.0
    for qid, x in a["answers"].items():
        y = b["answers"][qid]
        px = x.get("probabilities") or {"t": x["noul"]}
        py = y.get("probabilities") or {"t": y["noul"]}
        d = max(d, max(abs(px[k] - py[k]) for k in px))
    return d


def compare_answers(got, ref, qs):
    """Problems between one API result and the SDK reference for the same input."""
    probs = []
    if set(got.get("answers", {})) != set(qs):
        return ["question ids %s != requested %s" % (sorted(got.get("answers", {}))[:4], sorted(qs)[:4])]
    if got["usage"] != ref["usage"]:
        probs.append("usage %s != %s" % (got["usage"], ref["usage"]))
    for qid, a in got["answers"].items():
        b = ref["answers"][qid]
        if a["type"] != b["type"]:
            probs.append("%s type %s != %s" % (qid, a["type"], b["type"]))
            continue
        if a["type"] == "noul":
            if abs(a["noul"] - b["noul"]) > TOL:
                probs.append("%s noul %.4f != %.4f" % (qid, a["noul"], b["noul"]))
            continue
        if set(a["probabilities"]) != set(b["probabilities"]):
            probs.append("%s option labels %s != %s" % (qid, sorted(a["probabilities"]), sorted(b["probabilities"])))
            continue
        diff = max(abs(a["probabilities"][k] - b["probabilities"][k]) for k in a["probabilities"])
        if diff > TOL:
            probs.append("%s probability diff %.4f" % (qid, diff))
        if a["type"] == "score" and abs(a["score"] - b["score"]) > 2 * TOL:
            probs.append("%s score %.3f != %.3f" % (qid, a["score"], b["score"]))
        if a["type"] == "choice" and a["choice"] != b["choice"]:
            top = sorted(b["probabilities"].values(), reverse=True)
            if top[0] - top[1] > TOL:
                probs.append("%s choice %s != %s" % (qid, a["choice"], b["choice"]))
    return probs


def main():
    fine_tunes = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob("/finetuned/*/rl_agent_config.json"))
    served = [m["id"] for m in get("/v1/models")["data"] if m.get("kind")]
    fine_tunes = [f for f in fine_tunes if f in served][:1]
    checkpoints = ["english", "multilingual"] + fine_tunes
    print("Loading SDK reference agents: %s" % checkpoints)
    ref = {"english": laya.Agent(MODEL_DIR, device="cuda"),
           "multilingual": laya.Agent(MODEL_DIR, device="cuda", subfolder="multilingual")}
    for f in fine_tunes:
        ref[f] = laya.Agent(os.path.join("/finetuned", f), device="cuda")
    router = laya.Router(models={"english": (MODEL_DIR, None), "multilingual": (MODEL_DIR, "multilingual"),
                                 "typed-decisions": (MODEL_DIR, "typed-decisions")})

    def expected_model(state, model):
        return model if model not in (None, "auto") else router.route(state)["model"]

    work = build_workload(checkpoints)
    before = get("/info")["scheduler"]
    print("Firing %d requests from %d concurrent clients..." % (len(work), CLIENTS))
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(CLIENTS) as ex:
        responses = list(ex.map(lambda w: (w, post({"decide": "/v1/decide", "batch": "/v1/decide/batch",
                                                    "compare": "/v1/compare", "invalid": "/v1/decide"}[w[0]], w[2])),
                                work))
    wall = time.perf_counter() - t0
    after = get("/info")["scheduler"]
    passes = after["passes"] - before["passes"]
    tickets = after["tickets"] - before["tickets"]
    print("  done in %.1fs; %d GPU passes carried %d ticket-slots (avg %.1f tickets/pass, max %d)"
          % (wall, passes, tickets, tickets / max(1, passes), after["max_tickets_per_pass"]))

    print("Checking every result against laya.Agent.system_one ...")
    failures, checked, diffs, swap_checks = [], 0, [], 0
    cache: dict = {}

    def reference(model_name, state, questions, rid):
        key = (model_name, rid, json.dumps(state, sort_keys=True, ensure_ascii=False))
        if key not in cache:
            cache[key] = ref[model_name].system_one(state, questions)
        return cache[key]
    for (kind, rid, body), (status, resp) in responses:
        if kind == "invalid":
            if status != 422:
                failures.append("req %d invalid: HTTP %s (want 422)" % (rid, status))
            continue
        if status != 200:
            failures.append("req %d %s: HTTP %s %s" % (rid, kind, status, str(resp)[:150]))
            continue
        qs = body["questions"]
        if kind == "decide":
            pairs = [(body["state"], body.get("model"), resp)]
        elif kind == "batch":
            if len(resp["results"]) != len(body["states"]):
                failures.append("req %d batch: %d results for %d states" % (rid, len(resp["results"]), len(body["states"])))
                continue
            pairs = [(s, body.get("model"), r) for s, r in zip(body["states"], resp["results"])]
        else:
            pairs = [(body["state"], name, r) for name, r in resp["results"].items() if name in ref]
            if set(resp["results"]) != set(served):
                failures.append("req %d compare: checkpoints %s" % (rid, sorted(resp["results"])))
        for state, model, got in pairs:
            checked += 1
            if "error" in got:
                failures.append("req %d %s: state error %s" % (rid, kind, got["error"]))
                continue
            want_model = expected_model(state, model)
            if got["routing"]["model"] != want_model:
                failures.append("req %d %s: routed to %s, want %s" % (rid, kind, got["routing"]["model"], want_model))
                continue
            if want_model not in ref:
                continue  # typed-decisions via auto is never chosen; nothing else to compare
            own = reference(want_model, state, qs, rid)
            problems = compare_answers(got, own, qs)
            if problems:
                failures.append("req %d %s (%s): %s" % (rid, kind, want_model, "; ".join(problems[:3])))
                continue
            diffs.append(distance(got, own))
            if kind == "batch":  # swap check: closer to its own reference than to any clearly different sibling's
                d_own = distance(got, own)
                for other in body["states"]:
                    if other is state or expected_model(other, model) != want_model:
                        continue
                    sib = reference(want_model, other, qs, rid)
                    if distance(own, sib) > 0.15:
                        swap_checks += 1
                        if distance(got, sib) <= d_own:
                            failures.append("req %d batch: a state's result matches a sibling state better than itself"
                                            % rid)
                            break
    kinds = {k: sum(1 for w in work if w[0] == k) for k in ("decide", "batch", "compare", "invalid")}
    print("  %s requests; %d state results checked against the SDK; %d batch swap comparisons"
          % (kinds, checked, swap_checks))
    if diffs:
        diffs.sort()
        print("  difference from system_one: median %.4f  p95 %.4f  p99 %.4f  max %.4f (tolerance %.2f)"
              % (diffs[len(diffs) // 2], diffs[int(len(diffs) * .95)], diffs[int(len(diffs) * .99)], diffs[-1], TOL))
    if diffs and diffs[int(len(diffs) * .99)] > P99_MAX:
        failures.append("p99 difference %.4f exceeds %.2f - systematic drift, not noise"
                        % (diffs[int(len(diffs) * .99)], P99_MAX))

    print("Re-sending 100 requests one at a time: must match system_one exactly (<= %g) ..." % EXACT)
    solo = [(w, r) for w, r in responses if w[0] == "decide" and r[0] == 200][:100]
    worst_solo = 0.0
    for (kind, rid, body), _ in solo:
        status, resp = post("/v1/decide", body)
        want = expected_model(body["state"], body.get("model"))
        if status != 200 or want not in ref:
            continue
        d = distance(resp, reference(want, body["state"], body["questions"], rid))
        worst_solo = max(worst_solo, d)
        if d > EXACT:
            failures.append("req %d solo: differs from system_one by %.5f" % (rid, d))
    print("  %d solo requests, max difference %.5f" % (len(solo), worst_solo))

    for f in failures[:20]:
        print("FAIL", f)
    ok = not failures and tickets > passes
    if tickets <= passes:
        print("FAIL no coalescing happened (tickets/pass <= 1)")
    print("%s: %d mismatches in %d checked results" % ("ALL PASSED" if ok else "FAILED", len(failures), checked))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
