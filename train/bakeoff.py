"""Bake-off: Laya vs established approaches on LocalLLaMA/typed-decisions.

    python bakeoff.py laya --name laya-full --model /runs/td-full-cal-hard/model
    python bakeoff.py nli-zeroshot --name deberta-zs --hf MoritzLaurer/deberta-v3-large-zeroshot-v2.0
    python bakeoff.py gliclass --name gliclass-zs --hf knowledgator/gliclass-large-v3.0
    python bakeoff.py heads --name heads-full [--limit 150]          # ModernBERT-large + per-question heads
    python bakeoff.py nli-finetune --name nli-ft-full [--limit 150]   # fine-tune a zero-shot NLI model
    python bakeoff.py table                                            # the comparison table
    python bakeoff.py bootstrap                                        # 95% CIs and paired differences

Everything that could favour one method is shared:
  * splits: typed-decisions train split into 1,075 train / 125 calibration by id hash (seed 0,
    exactly as train.py does), official 400-case test split; --limit takes the same nested
    subsets as train.py's learning curve;
  * labels: the same soft gold distributions (fine-tunes minimise soft cross-entropy);
  * calibration: every method's per-answer log-probabilities get the same hard-label
    temperature fit on the calibration split (per question type x option-count bucket, >= 30
    answers), with the same wide range [0.05, 50] for all. An earlier [0.5, 5] clamp (the range
    the laya SDK allows Laya in serving) was too tight for very over-confident zero-shot models
    and measured the clamp rather than the model. Accuracy never depends on temperature;
  * metrics: evaluate.py's own scoring (accuracy, soft accuracy, Brier, NLL, ECE, score MAE);
  * latency: model time per test case (all its questions), batch of one case, bf16, median.
No method's hyper-parameters were tuned; each uses common defaults (listed per method below).
Results: /runs/bakeoff/<name>.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_splits, option_keys, subsample  # noqa: E402

DATA = "/data/typed-decisions"
OUT = "/runs/bakeoff"
T_MIN, T_MAX = 0.05, 50.0


# --------------------------------------------------------------------------- data
def load(limit=None):
    splits, errors = load_splits(DATA + "/train.jsonl", None, DATA + "/test.jsonl", None, 0.1, 0.1, 0)
    assert not errors, errors[:3]
    return subsample(splits["train"], limit, 0), splits["calib"], splits["test"]


def state_text(state) -> str:
    from laya.common import serialize_state
    return serialize_state(state)


def option_texts(q: dict) -> list[str]:
    """Human-readable option texts, in option_keys order (what the label means)."""
    crit = q.get("criteria")
    if q["type"] == "choice":
        crit = crit if isinstance(crit, dict) else {c: None for c in crit}
        return [k if not v else "%s: %s" % (k, v) for k, v in crit.items()]
    if q["type"] == "score":
        return [str(c) for c in crit]
    crit = crit or {}
    return ["no: " + (crit.get("false") or "the statement does not hold"),
            "yes: " + (crit.get("true") or "the statement holds")]


def answers(cases):
    """One entry per gold answer: the unit every method predicts a distribution for."""
    out = []
    for c in cases:
        for qid, g in c["gold"].items():
            q = c["questions"][qid]
            ins = q["instructions"] if isinstance(q["instructions"], str) else json.dumps(q["instructions"])
            out.append({"case": c["id"], "qid": qid, "q": q, "ins": ins, "type": q["type"],
                        "key": qid + "|" + "|".join(option_keys(q)), "texts": option_texts(q),
                        "dist": g["dist"], "label": g["label"], "workflow": c.get("workflow")})
    return out


def hypothesis(a, text):
    return 'The answer to "%s" is: %s.' % (a["ins"], text)


# --------------------------------------------------------------------------- calibration + scoring
def bucket(a):
    k = len(a["dist"])
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return a["type"] + ":" + size


def fit_temperature(pairs):
    """Hard-label temperature on (log-probs, label) pairs, as calibrate.py fits Laya's."""
    kmax = max(len(z) for z, _ in pairs)
    Z = torch.full((len(pairs), kmax), -1e4)
    for i, (z, _) in enumerate(pairs):
        Z[i, :len(z)] = torch.tensor(z)
    y = torch.tensor([lab for _, lab in pairs])
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Z / log_t.exp(), y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(min(T_MAX, max(T_MIN, log_t.exp().item())))


def calibrate(calib_ans, calib_lp):
    by_bucket, by_type = {}, {}
    for key_fn, store in ((bucket, by_bucket), (lambda a: a["type"], by_type)):
        groups = {}
        for a, lp in zip(calib_ans, calib_lp):
            groups.setdefault(key_fn(a), []).append((lp, a["label"]))
        for k, pairs in groups.items():
            if len(pairs) >= 30:
                store[k] = fit_temperature(pairs)
    return lambda a: by_bucket.get(bucket(a), by_type.get(a["type"], 1.0)), {"buckets": by_bucket, "types": by_type}


def softmax_t(lp, t):
    z = np.asarray(lp, dtype=np.float64) / t
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def score(test_ans, test_lp, temp_of):
    import evaluate as E
    rows = [{"p": softmax_t(lp, temp_of(a)).tolist(), "g": a["dist"], "label": a["label"], "type": a["type"],
             "qid": a["qid"], "workflow": a["workflow"]} for a, lp in zip(test_ans, test_lp)]
    return E.grouped(rows)


def finish(name, method, test_ans, test_lp, calib_ans, calib_lp, extra):
    temp_of, temps = calibrate(calib_ans, calib_lp)
    res = {"name": name, "method": method, "raw": score(test_ans, test_lp, lambda a: 1.0),
           "calibrated": score(test_ans, test_lp, temp_of), "temperatures": temps, **extra,
           "test_logprobs": [list(map(float, lp)) for lp in test_lp],
           "calib_logprobs": [list(map(float, lp)) for lp in calib_lp]}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(res, f, indent=2)
    o = res["calibrated"]["overall"]
    print("%s: accuracy %.3f  soft_acc %.3f  brier %.3f  nll %.3f  ece %.3f  | %s ms/case"
          % (name, o["accuracy"], o["soft_acc"], o["brier"], o["nll"], o["ece"], extra.get("ms_per_case")))
    return res


def per_case(ans):
    groups = {}
    for i, a in enumerate(ans):
        groups.setdefault(a["case"], []).append(i)
    return list(groups.values())


def timed_predict(predict_case, ans, n_timed=60):
    """Run predict_case(list_of_answers) -> list of log-prob vectors, case by case.
    Returns (log-probs aligned with ans, median ms per case over the first n_timed cases)."""
    out = [None] * len(ans)
    times = []
    groups = per_case(ans)
    for idx in groups[:3]:
        predict_case([ans[i] for i in idx])          # warm-up
    for gi, idx in enumerate(groups):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.perf_counter()
        lps = predict_case([ans[i] for i in idx])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if gi < n_timed:
            times.append((time.perf_counter() - t) * 1000)
        for i, lp in zip(idx, lps):
            out[i] = lp
    return out, round(float(np.median(times)), 1)


def log_softmax(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.max() - np.log(np.exp(x - x.max()).sum())).tolist()


# --------------------------------------------------------------------------- methods
def run_laya(a):
    """Laya checkpoint through laya.Agent.system_one (the serving path)."""
    import laya
    _, calib, test = load()
    agent = laya.Agent(a.model, device="cuda")
    state_of = {c["id"]: c["state"] for c in calib + test}

    def predict_case(group):
        qs = {x["qid"]: x["q"] for x in group}
        res = agent.system_one(state_of[group[0]["case"]], qs)["answers"]
        out = []
        for x in group:
            r = res[x["qid"]]
            p = [1 - r["noul"], r["noul"]] if x["type"] == "noul" else [r["probabilities"][k] for k in option_keys(x["q"])]
            out.append(np.log(np.clip(p, 1e-9, 1)).tolist())
        return out
    calib_ans, test_ans = answers(calib), answers(test)
    calib_lp, _ = timed_predict(predict_case, calib_ans, 0)
    test_lp, ms = timed_predict(predict_case, test_ans)
    return finish(a.name, "laya:" + a.model, test_ans, test_lp, calib_ans, calib_lp, {"ms_per_case": ms})


def nli_scorer(model, tok, max_len, ent_idx):
    @torch.no_grad()
    def predict_case(group, state_of):
        premises, hyps, owner = [], [], []
        for gi, x in enumerate(group):
            for t in x["texts"]:
                premises.append(state_of[x["case"]])
                hyps.append(hypothesis(x, t))
                owner.append(gi)
        enc = tok(premises, hyps, truncation="only_first", max_length=max_len, padding=True, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**enc).logits.float()
        ent = (logits[:, ent_idx] - logits[:, 1 - ent_idx]).cpu().numpy()   # binary: entailment vs not
        out, pos = [], 0
        for x in group:
            k = len(x["texts"])
            out.append(log_softmax(ent[pos:pos + k]))
            pos += k
        return out
    return predict_case


def run_nli_zeroshot(a):
    """Off-the-shelf zero-shot NLI classifier: each option as a hypothesis, softmax over
    options of the entailment score (the standard zero-shot-classification recipe)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    _, calib, test = load()
    tok = AutoTokenizer.from_pretrained(a.hf)
    model = AutoModelForSequenceClassification.from_pretrained(a.hf).cuda().eval()
    ent_idx = [i for i, l in model.config.id2label.items() if l.lower().startswith("entail")][0]
    max_len = min(1024, getattr(model.config, "max_position_embeddings", 512))
    state_of = {c["id"]: state_text(c["state"]) for c in calib + test}
    pc = nli_scorer(model, tok, max_len, int(ent_idx))
    calib_ans, test_ans = answers(calib), answers(test)
    calib_lp, _ = timed_predict(lambda g: pc(g, state_of), calib_ans, 0)
    test_lp, ms = timed_predict(lambda g: pc(g, state_of), test_ans)
    return finish(a.name, "nli-zeroshot:" + a.hf, test_ans, test_lp, calib_ans, calib_lp,
                  {"ms_per_case": ms, "max_len": max_len})


def run_gliclass(a):
    """GLiClass zero-shot: text = question + state, labels = option texts, single-label."""
    sys.path.insert(0, "/runs/.pylib")
    from gliclass import GLiClassModel, ZeroShotClassificationPipeline
    from transformers import AutoTokenizer
    _, calib, test = load()
    model = GLiClassModel.from_pretrained(a.hf)
    tok = AutoTokenizer.from_pretrained(a.hf, add_prefix_space=True)
    pipe = ZeroShotClassificationPipeline(model, tok, classification_type="single-label", device="cuda:0")
    state_of = {c["id"]: state_text(c["state"]) for c in calib + test}

    def predict_case(group):
        out = []
        for x in group:
            res = pipe(x["ins"] + "\n" + state_of[x["case"]], x["texts"], threshold=0.0)[0]
            scores = {r["label"]: r["score"] for r in res}
            p = np.array([scores.get(t, 1e-9) for t in x["texts"]], dtype=np.float64)
            out.append(np.log(np.clip(p / p.sum(), 1e-9, 1)).tolist())
        return out
    calib_ans, test_ans = answers(calib), answers(test)
    calib_lp, _ = timed_predict(predict_case, calib_ans, 0)
    test_lp, ms = timed_predict(predict_case, test_ans)
    return finish(a.name, "gliclass:" + a.hf, test_ans, test_lp, calib_ans, calib_lp, {"ms_per_case": ms})


def schedule(opt, steps, warmup=0.1):
    w = max(1, int(steps * warmup))
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / w if s < w else 0.5 * (1 + math.cos(math.pi * (s - w) / max(1, steps - w))))


def run_heads(a):
    """The conventional fine-tune: ModernBERT-large encodes the state once, mean-pooled, and one
    linear head per question (id + option set) predicts its options. Defaults: 4 epochs (as
    Laya), AdamW lr 3e-5 encoder / 1e-3 heads, 10% warmup + cosine, 8 cases per step, max 1024
    tokens, bf16 autocast, gradient checkpointing, soft cross-entropy against the gold dists."""
    from transformers import AutoModel, AutoTokenizer
    torch.manual_seed(0)
    random.seed(0)
    train, calib, test = load(a.limit)
    tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
    enc = AutoModel.from_pretrained("answerdotai/ModernBERT-large", attn_implementation="sdpa").cuda()
    enc.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    keys = sorted({x["key"] for x in answers(train + calib + test)})
    sizes = {x["key"]: len(x["dist"]) for x in answers(train + calib + test)}
    d = enc.config.hidden_size
    heads = torch.nn.ModuleDict({str(i): torch.nn.Linear(d, sizes[k]) for i, k in enumerate(keys)}).cuda()
    head_of = {k: str(i) for i, k in enumerate(keys)}
    drop = torch.nn.Dropout(0.1)

    def encode(cases):
        b = tok([state_text(c["state"]) for c in cases], truncation=True, max_length=1024, padding=True,
                return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = enc(**b).last_hidden_state.float()
        m = b["attention_mask"].unsqueeze(-1).float()
        return (h * m).sum(1) / m.sum(1)

    ans_by_case = {}
    for x in answers(train):
        ans_by_case.setdefault(x["case"], []).append(x)
    epochs, bs = 4, 8
    steps = epochs * math.ceil(len(train) / bs)
    opt = torch.optim.AdamW([{"params": enc.parameters(), "lr": 3e-5}, {"params": heads.parameters(), "lr": 1e-3}],
                            weight_decay=0.01)
    sch = schedule(opt, steps)
    t0 = time.time()
    enc.train()
    for ep in range(epochs):
        order = train[:]
        random.Random(ep).shuffle(order)
        for s in range(0, len(order), bs):
            batch = order[s:s + bs]
            pooled = drop(encode(batch))
            loss, n = 0.0, 0
            for i, c in enumerate(batch):
                for x in ans_by_case[c["id"]]:
                    logits = heads[head_of[x["key"]]](pooled[i])
                    loss = loss - (torch.tensor(x["dist"], device="cuda") * torch.log_softmax(logits, -1)).sum()
                    n += 1
            (loss / n).backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(heads.parameters()), 1.0)
            opt.step()
            sch.step()
            opt.zero_grad(set_to_none=True)
        print("  epoch %d/%d, %.0fs" % (ep + 1, epochs, time.time() - t0), flush=True)
    train_s = time.time() - t0
    enc.eval()
    case_of = {c["id"]: c for c in calib + test}

    @torch.no_grad()
    def predict_case(group):
        pooled = encode([case_of[group[0]["case"]]])[0]
        return [torch.log_softmax(heads[head_of[x["key"]]](pooled), -1).tolist() for x in group]
    calib_ans, test_ans = answers(calib), answers(test)
    calib_lp, _ = timed_predict(predict_case, calib_ans, 0)
    test_lp, ms = timed_predict(predict_case, test_ans)
    return finish(a.name, "heads:ModernBERT-large", test_ans, test_lp, calib_ans, calib_lp,
                  {"ms_per_case": ms, "train_cases": len(train), "train_minutes": round(train_s / 60, 1)})


def run_nli_finetune(a):
    """Fine-tune a zero-shot NLI cross-encoder on the task: every option is a (state, hypothesis)
    pair, softmax over the options' entailment scores, soft cross-entropy against the gold
    dist. Defaults: 2 epochs (each answer costs one pass per option), AdamW lr 2e-5, 10% warmup +
    cosine, 16 answers per step, max 1024 tokens, bf16 autocast, gradient checkpointing."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.manual_seed(0)
    random.seed(0)
    train, calib, test = load(a.limit)
    tok = AutoTokenizer.from_pretrained(a.hf)
    model = AutoModelForSequenceClassification.from_pretrained(a.hf).cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    ent_idx = int([i for i, l in model.config.id2label.items() if l.lower().startswith("entail")][0])
    max_len = min(1024, getattr(model.config, "max_position_embeddings", 512))
    state_of = {c["id"]: state_text(c["state"]) for c in train + calib + test}
    train_ans = answers(train)
    epochs, micro, accum = 2, 4, 4
    steps = epochs * math.ceil(len(train_ans) / (micro * accum))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    sch = schedule(opt, steps)
    t0 = time.time()
    model.train()
    for ep in range(epochs):
        order = train_ans[:]
        random.Random(ep).shuffle(order)
        for s in range(0, len(order), micro):
            group = order[s:s + micro]
            prem, hyp = [], []
            for x in group:
                for t in x["texts"]:
                    prem.append(state_of[x["case"]])
                    hyp.append(hypothesis(x, t))
            b = tok(prem, hyp, truncation="only_first", max_length=max_len, padding=True, return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(**b).logits.float()
            ent = lg[:, ent_idx] - lg[:, 1 - ent_idx]
            loss, pos = 0.0, 0
            for x in group:
                k = len(x["texts"])
                loss = loss - (torch.tensor(x["dist"], device="cuda") * torch.log_softmax(ent[pos:pos + k], -1)).sum()
                pos += k
            (loss / len(group) / accum).backward()
            if (s // micro + 1) % accum == 0 or s + micro >= len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sch.step()
                opt.zero_grad(set_to_none=True)
        print("  epoch %d/%d, %.0fs" % (ep + 1, epochs, time.time() - t0), flush=True)
    train_s = time.time() - t0
    model.eval()
    pc = nli_scorer(model, tok, max_len, ent_idx)
    calib_ans, test_ans = answers(calib), answers(test)
    calib_lp, _ = timed_predict(lambda g: pc(g, state_of), calib_ans, 0)
    test_lp, ms = timed_predict(lambda g: pc(g, state_of), test_ans)
    return finish(a.name, "nli-finetune:" + a.hf, test_ans, test_lp, calib_ans, calib_lp,
                  {"ms_per_case": ms, "train_cases": len(train), "train_minutes": round(train_s / 60, 1)})


def recalibrate(_a):
    """Recompute calibrated metrics for every saved run from its stored log-probs."""
    _, calib, test = load()
    calib_ans, test_ans = answers(calib), answers(test)
    for p in sorted(glob.glob(os.path.join(OUT, "*.json"))):
        r = json.load(open(p))
        if "test_logprobs" not in r:
            print("skip %s: no stored log-probs (re-run it)" % r["name"])
            continue
        temp_of, temps = calibrate(calib_ans, r["calib_logprobs"])
        r["calibrated"], r["temperatures"] = score(test_ans, r["test_logprobs"], temp_of), temps
        json.dump(r, open(p, "w"))
        print("recalibrated %s" % r["name"])


def bootstrap(_a, draws=2000):
    """Accuracy 95% CIs and paired differences, resampling test *cases* (a case's answers stay
    together). Covers test-set sampling only, not training-seed variation."""
    _, _, test = load()
    ans = answers(test)
    cases = sorted({a["case"] for a in ans})
    idx = {c: [i for i, a in enumerate(ans) if a["case"] == c] for c in cases}
    rng = np.random.default_rng(0)
    samples = [np.concatenate([idx[c] for c in rng.choice(cases, len(cases))]) for _ in range(draws)]
    runs = {}
    for p in sorted(glob.glob(os.path.join(OUT, "*.json"))):
        r = json.load(open(p))
        if "test_logprobs" in r:
            runs[r["name"]] = np.array([int(np.argmax(lp)) == a["label"] for lp, a in zip(r["test_logprobs"], ans)], float)
    print("| method | accuracy | 95% CI |\n|---|---|---|")
    for n, c in runs.items():
        s = np.array([c[d].mean() for d in samples])
        print("| %s | %.3f | %.3f-%.3f |" % (n, c.mean(), np.percentile(s, 2.5), np.percentile(s, 97.5)))
    pairs = [("deberta-v3-large-zs", "laya-zeroshot"), ("laya-ft-150", "nli-modernbert-ft-150"),
             ("laya-ft-150", "heads-ft-150"), ("heads-ft-1075", "laya-ft-1075"), ("laya-ft-1075", "nli-modernbert-ft-1075")]
    print("\n| A - B | difference | 95% CI | share of draws A <= B |\n|---|---|---|---|")
    for a, b in pairs:
        if a in runs and b in runs:
            d = np.array([runs[a][x].mean() - runs[b][x].mean() for x in samples])
            print("| %s - %s | %+.3f | %+.3f to %+.3f | %.3f |" % (a, b, runs[a].mean() - runs[b].mean(),
                                                                   np.percentile(d, 2.5), np.percentile(d, 97.5), (d <= 0).mean()))


def table(_a):
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, "*.json"))):
        r = json.load(open(p))
        rows.append(r)
    cols = ["accuracy", "soft_acc", "brier", "nll", "ece", "score_mae"]
    print("| method | train cases | " + " | ".join(cols) + " | ms/case |")
    print("|---" * (len(cols) + 3) + "|")
    for r in rows:
        o = r["calibrated"]["overall"]
        print("| %s | %s | %s | %s |" % (r["name"], r.get("train_cases", "-"),
                                          " | ".join("%.3f" % o[c] for c in cols), r.get("ms_per_case")))
    print("\nAccuracy by question type (calibrated):")
    for r in rows:
        bt = r["calibrated"]["by_type"]
        print("  %-22s %s" % (r["name"], "  ".join("%s %.3f" % (t, m["accuracy"]) for t, m in sorted(bt.items()))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("method", choices=["laya", "nli-zeroshot", "gliclass", "heads", "nli-finetune", "table", "recalibrate",
                                       "bootstrap"])
    ap.add_argument("--name")
    ap.add_argument("--model", help="laya checkpoint dir")
    ap.add_argument("--hf", help="Hugging Face model id")
    ap.add_argument("--limit", type=int, help="train on the nested N-case subset (as train.py --limit)")
    a = ap.parse_args()
    {"laya": run_laya, "nli-zeroshot": run_nli_zeroshot, "gliclass": run_gliclass, "heads": run_heads,
     "nli-finetune": run_nli_finetune, "table": table, "recalibrate": recalibrate, "bootstrap": bootstrap}[a.method](a)
