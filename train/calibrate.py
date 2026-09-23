"""Fit (or refit) a checkpoint's temperatures on held-out data, without retraining.

    python calibrate.py --run /runs/my-run                       # its own calibration split
    python calibrate.py --model /runs/my-run/model --calib /data/mine/calib.jsonl [--questions ...]
    python calibrate.py --run /runs/my-run --target soft          # match gold distributions instead

One temperature per (question type, option count) bucket - the table the SDK
applies at inference - with a per-type fallback for buckets under --min-answers.
Values are clamped to the SDK's accepted range [0.5, 5].

--target none resets them to 1.0. --target hard (default) fits the temperature to the gold *label*: afterwards a
stated 80% means right ~80% of the time, which is what "calibrated" usually
means and what ECE / NLL-of-label measure. --target soft fits to the gold
*distribution*; with teacher-sampled soft labels (which are deliberately spread
out) this pushes temperatures above 1 and makes the model under-confident on
hard-label metrics - measured in docs/EXPERIMENTS.md, section 6.

Writes the temperatures into rl_agent_config.json (the previous values are kept
in calibration_history) and records the before/after metrics.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from ckpt import encode_case, load_model, load_tokenizer, read_config
from data import load_splits

MIN_ANSWERS = 30


def fit_temperature(pairs, lo, hi):
    """Temperature minimising cross-entropy of (logits, target distribution) pairs."""
    kmax = max(len(z) for z, _ in pairs)
    Z = torch.full((len(pairs), kmax), -1e4)
    T = torch.zeros((len(pairs), kmax))
    for i, (z, t) in enumerate(pairs):
        Z[i, :len(z)] = torch.tensor(z)
        T[i, :len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss
    opt.step(closure)
    return float(min(hi, max(lo, log_t.exp().item())))


def metrics(pairs, temp_of):
    """pairs: (logits, gold dist, bucket, qtype, label). Hard-label ECE/NLL plus soft NLL."""
    from laya.common import ece_score
    conf, correct, nll, soft_nll = [], [], [], []
    for z, t, key, qt, label in pairs:
        z = np.asarray(z) / temp_of(key, qt)
        p = np.exp(z - z.max())
        p /= p.sum()
        conf.append(p.max())
        correct.append(float(int(p.argmax()) == label))
        nll.append(-float(np.log(max(p[label], 1e-12))))
        soft_nll.append(-float(np.sum(np.asarray(t) * np.log(np.clip(p, 1e-12, 1)))))
    return {"n": len(pairs), "accuracy": round(float(np.mean(correct)), 4),
            "ece": round(ece_score(np.array(conf), np.array(correct)), 4),
            "nll": round(float(np.mean(nll)), 4), "soft_nll": round(float(np.mean(soft_nll)), 4)}


def fit(pairs, target="hard", min_answers=MIN_ANSWERS):
    """-> (per-type temperatures [choice, score, noul], {bucket: temperature}, before, after)."""
    from laya.common import TEMP_MAX, TEMP_MIN

    def tgt(t, label):
        if target == "soft":
            return t
        one = [0.0] * len(t)
        one[label] = 1.0
        return one
    temps, by_bucket = [1.0, 1.0, 1.0], {}
    for qt in range(3):
        sel = [(z, tgt(t, lb)) for z, t, _, q, lb in pairs if q == qt]
        if len(sel) >= min_answers:
            temps[qt] = fit_temperature(sel, TEMP_MIN, TEMP_MAX)
    for key in sorted({p[2] for p in pairs}):
        sel = [(z, tgt(t, lb)) for z, t, kk, _, lb in pairs if kk == key]
        if len(sel) >= min_answers:  # fewer and a fitted temperature is mostly noise
            by_bucket[key] = fit_temperature(sel, TEMP_MIN, TEMP_MAX)
    before = metrics(pairs, lambda key, qt: 1.0)
    after = metrics(pairs, lambda key, qt: by_bucket.get(key, temps[qt]))
    return temps, by_bucket, before, after


@torch.no_grad()
def predict_logits(model, items, pad_id, device, dtype, bs=32):
    from laya.common import collate_items
    model.eval()
    res = [None] * len(items)
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        b = collate_items([[items[i] for i in idx]], pad_id)
        with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
            logits, _ = model(b["input_ids"].to(device), b["attention_mask"].to(device), b["marker_pos"].to(device),
                              b["marker_mask"].to(device), b["qtype"].to(device))
        logits = logits.float().cpu().numpy()
        for r, i in enumerate(idx):
            res[i] = logits[r, :items[i]["k"]].tolist()
    return res


def pairs_for(model, items, tok, device, dtype):
    from laya.common import temp_bucket
    logits = predict_logits(model, items, tok.pad_token_id, device, dtype)
    return [(z, it["target"], temp_bucket(it["qtype"], it["k"]), it["qtype"], it["label"])
            for z, it in zip(logits, items)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="a train.py run dir (uses its model and calibration split)")
    ap.add_argument("--model", help="checkpoint dir to calibrate in place")
    ap.add_argument("--calib", help="JSONL of held-out cases (overrides the run's split)")
    ap.add_argument("--questions")
    ap.add_argument("--target", choices=("hard", "soft", "none"), default="hard",
                    help="none = reset every temperature to 1.0 (an uncalibrated baseline)")
    ap.add_argument("--min-answers", type=int, default=MIN_ANSWERS)
    ap.add_argument("--dry-run", action="store_true", help="report, do not write the config")
    a = ap.parse_args(argv)

    if a.run:
        path = os.path.join(a.run, "model")
        with open(os.path.join(path, "training_summary.json")) as f:
            ar = json.load(f)["args"]
        splits, errors = load_splits(ar["train"], a.calib or ar.get("calib"), ar.get("test"),
                                     a.questions or ar.get("questions"), ar["calib_frac"], ar["test_frac"], ar["seed"])
        cases = splits["calib"]
    elif a.model and a.calib:
        path = a.model
        splits, errors = load_splits(a.calib, None, None, a.questions, 0.0, 0.0)
        cases = splits["train"]
    else:
        sys.exit("give --run, or --model with --calib")
    if errors:
        sys.exit("data errors: %s" % errors[:3])
    if not cases:
        sys.exit("no calibration cases (trained with --calib-frac 0?) - pass --calib")

    cfg = read_config(path)
    tok = load_tokenizer(path)
    items = [it for c in cases for it in encode_case(c, tok, cfg["max_len"], cfg["head_max_len"])[0]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = load_model(path, cfg).to(device)
    pairs = pairs_for(model, items, tok, device, dtype)
    if a.target == "none":
        temps, by_bucket = [1.0, 1.0, 1.0], {}
        before = after = metrics(pairs, lambda key, qt: 1.0)
    else:
        temps, by_bucket, before, after = fit(pairs, a.target, a.min_answers)
    print("Calibration (%s target) on %d held-out answers:" % (a.target, before["n"]))
    for k in ("ece", "nll", "soft_nll"):
        print("  %-8s %.4f -> %.4f" % (k, before[k], after[k]))
    print("  temperatures by type (choice, score, noul): %s" % [round(t, 3) for t in temps])
    print("  by bucket: %s" % {k: round(v, 3) for k, v in by_bucket.items()})
    if a.dry_run:
        return 0
    hist = cfg.get("calibration_history", [])
    hist.append({"temperature": cfg.get("temperature"), "temperature_by_options": cfg.get("temperature_by_options"),
                 "replaced_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    cfg.update(temperature=temps, temperature_by_options=by_bucket, calibration_history=hist,
               calibration={"target": a.target, "n_answers": before["n"], "before": before, "after": after})
    with open(os.path.join(path, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print("Wrote %s/rl_agent_config.json - re-run evaluate.py to see the effect on the test split" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
