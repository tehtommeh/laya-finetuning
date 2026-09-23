"""Specialist models on their home tasks vs Laya (zero-shot and fine-tuned) and a generalist NLI model.

    python specialists.py run --task toxicity --method specialist
    python specialists.py run --task toxicity --method laya --model /runs/spec-toxicity/model --name laya-ft-450
    python specialists.py run --task toxicity --method nli --hf MoritzLaurer/deberta-v3-large-zeroshot-v2.0
    python specialists.py table

Data from prepare_specialists.py (/data/specialists/<task>). Every method returns a
probability per class for each test item; the probabilities are scored as they come
(nothing is recalibrated here). Binary tasks report accuracy, F1 of the positive class
and AUROC (toxicity: positive = at least half of raters said toxic; Brier is against the
raw rater fraction); sentiment reports accuracy and macro-F1. Latency is the model time per
item, one item at a time, bf16 autocast, median.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA = "/data/specialists"
OUT = "/runs/bakeoff/specialists"
SPECIALISTS = {
    "prompt-injection": "protectai/deberta-v3-base-prompt-injection-v2",
    "toxicity": "unitary/toxic-bert",
    "sentiment": "cardiffnlp/twitter-roberta-base-sentiment-latest",
}
CLASSES = {"prompt-injection": ["false", "true"], "toxicity": ["false", "true"],
           "sentiment": ["negative", "neutral", "positive"]}


def load(task):
    rows = [json.loads(line) for line in open(os.path.join(DATA, task, "test.jsonl"), encoding="utf-8")]
    questions = json.load(open(os.path.join(DATA, task, "questions.json")))
    return rows, questions


def text_of(row):
    return next(iter(row["state"].values()))


def gold(task, row):
    g = next(iter(row["gold"].values()))
    if task == "sentiment":
        return CLASSES[task].index(g), None
    frac = float(g)
    return int(frac >= 0.5), frac


def timed(fn, rows):
    for r in rows[:5]:
        fn(r)
    out, times = [], []
    for r in rows:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.perf_counter()
        out.append(fn(r))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1000)
    return out, round(float(np.median(times)), 1)


# --------------------------------------------------------------------------- methods
def specialist(task):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    name = SPECIALISTS[task]
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).cuda().eval()
    labels = {i: l.lower() for i, l in model.config.id2label.items()}

    @torch.no_grad()
    def predict(row):
        enc = tok(text_of(row), truncation=True, max_length=512, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**enc).logits.float()[0]
        if task == "toxicity":                 # multi-label sigmoid head: use the "toxic" output
            idx = [i for i, l in labels.items() if l == "toxic"][0]
            p = torch.sigmoid(logits[idx]).item()
            return [1 - p, p]
        probs = torch.softmax(logits, -1).cpu().numpy()
        if task == "prompt-injection":         # SAFE / INJECTION
            inj = [i for i, l in labels.items() if "inject" in l][0]
            return [1 - float(probs[inj]), float(probs[inj])]
        return [float(probs[[i for i, l in labels.items() if l == c][0]]) for c in CLASSES[task]]
    return predict, name


def laya_method(task, model_dir):
    import laya
    agent = laya.Agent(model_dir, device="cuda")
    _, qs = load(task)
    qid, q = next(iter(qs.items()))

    def predict(row):
        a = agent.system_one(row["state"], qs)["answers"][qid]
        if q["type"] == "noul":
            return [1 - a["noul"], a["noul"]]
        return [a["probabilities"][c] for c in CLASSES[task]]
    return predict, "laya:" + model_dir


def nli_method(task, hf):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf)
    model = AutoModelForSequenceClassification.from_pretrained(hf).cuda().eval()
    ent = int([i for i, l in model.config.id2label.items() if l.lower().startswith("entail")][0])
    _, qs = load(task)
    q = next(iter(qs.values()))
    if q["type"] == "noul":
        hyps = ["This text is %s." % q["criteria"]["false"], "This text is a %s." % q["criteria"]["true"]
                if task == "prompt-injection" else "This text is %s." % q["criteria"]["true"]]
    else:
        hyps = ["The sentiment of this text is %s." % c for c in CLASSES[task]]

    @torch.no_grad()
    def predict(row):
        enc = tok([text_of(row)] * len(hyps), hyps, truncation="only_first", max_length=512, padding=True,
                  return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(**enc).logits.float()
        return torch.softmax(lg[:, ent] - lg[:, 1 - ent], -1).tolist()
    return predict, "nli-zeroshot:" + hf


def guardian_method(task, hf, risk):
    """IBM Granite Guardian (an LLM judge): P(risky) = softmax of the "Yes" vs "No" logits for
    the first generated token, per its model card (safe token "No", unsafe token "Yes")."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf)
    model = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16).cuda().eval()
    yes = sorted({tok.encode(v, add_special_tokens=False)[0] for v in ("Yes", " Yes")})
    no = sorted({tok.encode(v, add_special_tokens=False)[0] for v in ("No", " No")})

    @torch.no_grad()
    def predict(row):
        ids = tok.apply_chat_template([{"role": "user", "content": text_of(row)}], guardian_config={"risk_name": risk},
                                      add_generation_prompt=True, return_tensors="pt")
        ids = (ids["input_ids"] if hasattr(ids, "keys") else ids).cuda()
        logits = model(ids).logits[0, -1].float()
        z = torch.stack([torch.logsumexp(logits[no], 0), torch.logsumexp(logits[yes], 0)])
        return torch.softmax(z, 0).tolist()
    return predict, "guardian:%s (risk=%s)" % (hf, risk)


# --------------------------------------------------------------------------- metrics
def f1(true_mask, pred_mask) -> float:
    tp = float(np.sum(true_mask & pred_mask))
    fp = float(np.sum(~true_mask & pred_mask))
    fn = float(np.sum(true_mask & ~pred_mask))
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def auroc(scores, labels) -> float:
    """Rank-based AUROC (Mann-Whitney U), ties get average ranks."""
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    s_sorted = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    pos, neg = labels == 1, labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def metrics(task, rows, preds):
    from laya.common import ece_score
    y = np.array([gold(task, r)[0] for r in rows])
    P = np.array(preds, dtype=np.float64)
    P = P / P.sum(1, keepdims=True)
    pred = P.argmax(1)
    out = {"n": len(rows), "accuracy": float((pred == y).mean()),
           "ece": float(ece_score(P.max(1), (pred == y).astype(float)))}
    if task == "sentiment":
        out["macro_f1"] = float(np.mean([f1(y == c, pred == c) for c in range(3)]))
        onehot = np.eye(3)[y]
        out["brier"] = float(((P - onehot) ** 2).sum(1).mean())
    else:
        out["f1_positive"] = f1(y == 1, pred == 1)
        out["auroc"] = auroc(P[:, 1], y)
        target = np.array([gold(task, r)[1] for r in rows])      # raw fraction (toxicity) or 0/1
        out["brier"] = float(((P[:, 1] - target) ** 2).mean())
        out["positives"] = int(y.sum())
    return out


def run(a):
    rows, _ = load(a.task)
    if a.method == "specialist":
        fn, desc = specialist(a.task)
        name = a.name or "specialist"
    elif a.method == "laya":
        fn, desc = laya_method(a.task, a.model)
        name = a.name or "laya"
    elif a.method == "guardian":
        fn, desc = guardian_method(a.task, a.hf, a.risk)
        name = a.name or "granite-guardian"
    else:
        fn, desc = nli_method(a.task, a.hf)
        name = a.name or "nli-zeroshot"
    preds, ms = timed(fn, rows)
    res = {"task": a.task, "name": name, "model": desc, "ms_per_item": ms, **metrics(a.task, rows, preds)}
    os.makedirs(os.path.join(OUT, a.task), exist_ok=True)
    json.dump(res, open(os.path.join(OUT, a.task, name + ".json"), "w"), indent=2)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}))


def table(_a):
    for task in SPECIALISTS:
        files = sorted(glob.glob(os.path.join(OUT, task, "*.json")))
        if not files:
            continue
        print("\n### %s" % task)
        cols = ["accuracy", "macro_f1", "brier", "ece"] if task == "sentiment" else ["accuracy", "f1_positive", "auroc", "brier", "ece"]
        print("| method | model | " + " | ".join(cols) + " | ms/item |")
        print("|---" * (len(cols) + 3) + "|")
        for f in files:
            r = json.load(open(f))
            print("| %s | %s | %s | %s |" % (r["name"], r["model"], " | ".join("%.3f" % r[c] for c in cols), r["ms_per_item"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "table"])
    ap.add_argument("--task", choices=list(SPECIALISTS))
    ap.add_argument("--method", choices=["specialist", "laya", "nli", "guardian"])
    ap.add_argument("--risk", help="guardian risk name, e.g. jailbreaking, harm")
    ap.add_argument("--model")
    ap.add_argument("--hf")
    ap.add_argument("--name")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else table(a)
