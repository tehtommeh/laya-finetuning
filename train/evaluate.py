"""Score checkpoints on a held-out test split, the way they will be served.

    python evaluate.py --run /runs/my-run                  # test split recorded by train.py, vs its base
    python evaluate.py --model /runs/my-run/model --test /data/mine/test.jsonl --compare english multilingual

Every model is loaded through `laya.Agent` - the exact code path the API uses,
including the shipped/fitted temperatures and each checkpoint's own max_len.
A per-question majority-label baseline (from the training split, when known)
is reported alongside, because a model that does not beat "always answer the
most common label" has not learned anything useful.

Metrics (per answer, then averaged):
  accuracy    argmax matches the gold label (score: most likely level)
  soft_acc    sum_i p_i * gold_i  - agreement with the full gold distribution
  brier       sum_i (p_i - gold_i)^2              (lower is better)
  nll         -log p[gold label]                  (lower is better)
  ece         expected calibration error of max-probability confidence (lower is better)
  score_mae   |expected level - gold expected level| for score questions
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from ckpt import BASES, resolve_base
from data import load_splits, option_keys, subsample


def answer_dist(q, ans):
    keys = option_keys(q)
    if q["type"] == "noul":
        p = float(ans["noul"])
        d = [1 - p, p]
    else:
        probs = ans["probabilities"]
        d = [float(probs.get(k, 0.0)) for k in keys]
    s = sum(d)
    return [v / s for v in d] if s > 0 else [1 / len(d)] * len(d)


def score_rows(rows):
    """rows: list of dicts with p, g, label, type."""
    from laya.common import ece_score
    if not rows:
        return {}
    acc, soft, brier, nll, conf, mae = [], [], [], [], [], []
    for r in rows:
        p, g = np.asarray(r["p"]), np.asarray(r["g"])
        ok = float(int(p.argmax()) == r["label"])
        acc.append(ok)
        conf.append(float(p.max()))
        soft.append(float((p * g).sum()))
        brier.append(float(((p - g) ** 2).sum()))
        nll.append(-float(np.log(max(p[r["label"]], 1e-12))))
        if r["type"] == "score":
            lv = np.arange(len(p))
            mae.append(abs(float((lv * p).sum()) - float((lv * g).sum())))
    out = {"n": len(rows), "accuracy": np.mean(acc), "soft_acc": np.mean(soft), "brier": np.mean(brier),
           "nll": np.mean(nll), "ece": ece_score(np.array(conf), np.array(acc))}
    if mae:
        out["score_mae"] = np.mean(mae)
    return {k: (round(float(v), 4) if k != "n" else v) for k, v in out.items()}


def grouped(rows):
    res = {"overall": score_rows(rows), "by_type": {}, "by_question": {}, "by_workflow": {}}
    for key, field in (("by_type", "type"), ("by_question", "qid"), ("by_workflow", "workflow")):
        vals = sorted({r[field] for r in rows if r.get(field)})
        res[key] = {v: score_rows([r for r in rows if r.get(field) == v]) for v in vals}
    return res


def run_model(path, cases):
    import laya
    import torch
    t0 = time.time()
    agent = laya.Agent(path, device="cuda" if torch.cuda.is_available() else "cpu")
    load_s = time.time() - t0
    rows, lat = [], []
    agent.system_one("warm up", {"x": {"type": "noul", "instructions": "Is this a test?"}})
    for c in cases:
        t = time.perf_counter()
        try:
            res = agent.system_one(c["state"], c["questions"])
        except ValueError as e:  # options do not fit this checkpoint's head_max_len
            print("  skip %s: %s" % (c["id"], e))
            continue
        lat.append((time.perf_counter() - t) * 1000)
        for qid, g in c["gold"].items():
            q = c["questions"][qid]
            rows.append({"p": answer_dist(q, res["answers"][qid]), "g": g["dist"], "label": g["label"],
                         "type": q["type"], "qid": qid, "workflow": c.get("workflow")})
    out = grouped(rows)
    out["latency_ms_p50"] = round(float(np.percentile(lat, 50)), 1) if lat else None
    out["load_seconds"] = round(load_s, 1)
    out["device"] = agent.device.type
    out["max_len"] = agent.cfg.get("max_len")
    del agent
    torch.cuda.empty_cache()
    return out


def majority_baseline(train_cases, test_cases):
    """Always answer the most common training label for that question, with the training
    label frequencies as p. Keyed by question id *and* its options: a schema can reuse an
    id (e.g. `action`) with different options in different workflows."""
    def key(c, qid):
        return qid, tuple(option_keys(c["questions"][qid]))
    freq = {}
    for c in train_cases:
        for qid, g in c["gold"].items():
            freq.setdefault(key(c, qid), np.zeros(len(g["dist"])))[g["label"]] += 1
    rows = []
    for c in test_cases:
        for qid, g in c["gold"].items():
            f = freq.get(key(c, qid))
            if f is None:
                continue
            p = (f + 0.5) / (f + 0.5).sum()  # add-half smoothing so nll stays finite
            rows.append({"p": p.tolist(), "g": g["dist"], "label": g["label"], "type": c["questions"][qid]["type"],
                         "qid": qid, "workflow": c.get("workflow")})
    return grouped(rows)


def table(results):
    cols = ["accuracy", "soft_acc", "brier", "nll", "ece", "score_mae"]
    lines = ["| model | n | " + " | ".join(cols) + " | p50 ms |", "|---" * (len(cols) + 3) + "|"]
    for name, r in results.items():
        o = r["overall"]
        lines.append("| %s | %s | %s | %s |" % (name, o.get("n"), " | ".join(
            "%.3f" % o[c] if c in o else "-" for c in cols), r.get("latency_ms_p50") or "-"))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="a train.py run dir: uses its model, test split and base")
    ap.add_argument("--model", nargs="*", default=[], help="extra checkpoint dirs to score")
    ap.add_argument("--test", help="JSONL test file (overrides the run's recorded split)")
    ap.add_argument("--questions")
    ap.add_argument("--compare", nargs="*", help="baselines to score too: english multilingual typed-decisions "
                                                 "(default: the run's base)")
    ap.add_argument("--limit", type=int, help="score only the first N test cases")
    ap.add_argument("--out", help="where to write eval.json / eval.md (default: the run dir)")
    a = ap.parse_args(argv)

    models, train_cases, test_cases = {}, [], []
    if a.run:
        with open(os.path.join(a.run, "model", "training_summary.json")) as f:
            s = json.load(f)
        ar = s["args"]
        splits, errors = load_splits(ar["train"], ar.get("calib"), a.test or ar.get("test"),
                                     a.questions or ar.get("questions"), ar["calib_frac"], ar["test_frac"], ar["seed"])
        if errors:
            sys.exit("data errors: %s" % errors[:3])
        train_cases = subsample(splits["train"], ar.get("limit"), ar["seed"])
        test_cases = splits["test"]
        models[s["name"] + " (fine-tuned)"] = os.path.join(a.run, "model")
        compare = a.compare if a.compare is not None else [s["base"]]
    else:
        if not a.test:
            sys.exit("give --run, or --model with --test")
        splits, errors = load_splits(a.test, None, None, a.questions, 0.0, 0.0)
        test_cases = splits["train"]
        compare = a.compare or []
    for m in a.model:
        name = os.path.basename(os.path.normpath(m))
        if name == "model":  # <run>/model -> name it after the run
            name = os.path.basename(os.path.dirname(os.path.normpath(m)))
        while name in models:
            name += "'"
        models[name] = m
    for b in compare:
        models["%s (base)" % b if b in BASES else b] = resolve_base(b)
    if a.limit:
        test_cases = test_cases[:a.limit]
    if not test_cases:
        sys.exit("no test cases (train.py was run with --test-frac 0 and no --test?)")
    print("Scoring %d test cases (%d answers)" % (len(test_cases), sum(len(c["gold"]) for c in test_cases)))

    results = {}
    if train_cases:
        results["majority label (baseline)"] = majority_baseline(train_cases, test_cases)
    for name, path in models.items():
        print("  %s  <- %s" % (name, path))
        results[name] = run_model(path, test_cases)
    md = table(results)
    print("\n" + md)
    by_q = next((r for n, r in results.items() if "fine-tuned" in n), None)
    if by_q:
        print("\nFine-tuned, per question:")
        for qid, m in by_q["by_question"].items():
            print("  %-28s acc %.3f  brier %.3f  n=%d" % (qid, m["accuracy"], m["brier"], m["n"]))

    out = a.out or a.run
    if out:
        report = {"test_cases": len(test_cases), "results": results,
                  "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        for d in {out, os.path.join(a.run, "model")} if a.run else {out}:
            with open(os.path.join(d, "eval.json"), "w") as f:
                json.dump(report, f, indent=2)
        with open(os.path.join(out, "eval.md"), "w") as f:
            f.write(md + "\n")
        print("\nWrote %s/eval.json and eval.md" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
