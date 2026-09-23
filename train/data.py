"""Training data: load, normalise and split Laya fine-tuning cases.

One case = one state + the questions asked about it + the gold answers. Files are
JSON Lines, one case per line:

    {"id": "t-001",                       # optional, used for stable splits
     "state": <string | object | list>,   # exactly what you will send to /v1/decide
     "questions": {qid: {...}},           # optional if --questions gives a shared schema
     "gold": {qid: <answer>}}             # answers; questions without one are skipped

A gold answer can be written in full ...

    {"probabilities": {"billing": 0.7, "technical": 0.3}}          # choice
    {"probabilities": {"0": 0.1, "1": 0.6, "2": 0.3}}              # score (level index)
    {"probabilities": {"false": 0.2, "true": 0.8}}                 # noul

... or as shorthand for a hard label:

    "billing"            choice label
    2                    score level (int)
    true / false         noul
    0.8                  noul P(true)

The full form is what LocalLLaMA/typed-decisions uses; extra keys (label, confidence,
score, noul) are ignored except `label`, which wins ties for the hard label.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

QTYPES = ("choice", "score", "noul")


class DataError(ValueError):
    pass


# --------------------------------------------------------------------------- questions
def check_question(qid: str, q: Any) -> dict:
    if not isinstance(q, dict):
        raise DataError("question %r must be an object" % qid)
    t = q.get("type")
    if t not in QTYPES:
        raise DataError("question %r: type must be one of %s, got %r" % (qid, QTYPES, t))
    if not isinstance(q.get("instructions"), (str, dict, list)) or not q.get("instructions"):
        raise DataError("question %r: instructions are required" % qid)
    crit = q.get("criteria")
    if t == "choice":
        if isinstance(crit, list):
            crit = {c: None for c in crit}
        if not isinstance(crit, dict) or len(crit) < 2:
            raise DataError("question %r: choice needs criteria with >= 2 labels" % qid)
    elif t == "score":
        if not isinstance(crit, list) or len(crit) < 2:
            raise DataError("question %r: score needs criteria as a list of >= 2 levels" % qid)
    elif crit is not None and not isinstance(crit, dict):
        raise DataError("question %r: noul criteria must be {\"true\": ..., \"false\": ...}" % qid)
    return {"type": t, "instructions": q["instructions"], "criteria": crit} if crit is not None else \
        {"type": t, "instructions": q["instructions"]}


def option_keys(q: dict) -> list[str]:
    """Label for each option, in the SDK's option order (render_options)."""
    if q["type"] == "choice":
        return list(q["criteria"].keys())
    if q["type"] == "score":
        return [str(i) for i in range(len(q["criteria"]))]
    return ["false", "true"]


def to_internal(q: dict) -> dict:
    """The SDK's internal question form ({t, ins, crit}), as build_sequence expects."""
    ins = q["instructions"] if isinstance(q["instructions"], str) else json.dumps(q["instructions"])
    return {"t": q["type"], "ins": ins, "crit": q.get("criteria")}


# --------------------------------------------------------------------------- gold
def gold_distribution(qid: str, q: dict, g: Any) -> tuple[list[float], int]:
    """Gold answer -> (target distribution in option order, hard label index)."""
    keys = option_keys(q)
    t = q["type"]
    dist = None
    label = None
    if isinstance(g, dict):
        probs = g.get("probabilities")
        if probs is None and t == "noul" and isinstance(g.get("noul"), (int, float)) and not isinstance(g.get("noul"), bool):
            probs = {"false": 1 - float(g["noul"]), "true": float(g["noul"])}
        if probs is None and "label" in g:
            g = g["label"]  # fall through to shorthand
        else:
            if not isinstance(probs, dict):
                raise DataError("gold %r: expected a `probabilities` object" % qid)
            unknown = set(map(str, probs)) - set(keys)
            if unknown:
                raise DataError("gold %r: probabilities name unknown options %s (options: %s)"
                                % (qid, sorted(unknown), keys))
            dist = [float(probs.get(k) or 0.0) for k in keys]
            if "label" in g and str(g["label"]).lower() in [k.lower() for k in keys]:
                label = [k.lower() for k in keys].index(str(g["label"]).lower())
    if dist is None:
        dist = [0.0] * len(keys)
        if t == "noul" and isinstance(g, bool):
            dist[int(g)] = 1.0
        elif t == "noul" and isinstance(g, str) and g.lower() in ("true", "false", "yes", "no"):
            dist[int(g.lower() in ("true", "yes"))] = 1.0
        elif t == "noul" and isinstance(g, (int, float)) and 0 <= g <= 1:
            dist = [1.0 - float(g), float(g)]
        elif t == "score" and isinstance(g, int) and not isinstance(g, bool) and 0 <= g < len(keys):
            dist[g] = 1.0
        elif t == "score" and isinstance(g, str) and g.isdigit() and int(g) < len(keys):
            dist[int(g)] = 1.0
        elif t == "choice" and isinstance(g, str) and g in keys:
            dist[keys.index(g)] = 1.0
        else:
            raise DataError("gold %r: cannot read %r as a %s answer (options: %s)" % (qid, g, t, keys))
    if any(v < 0 or math.isnan(v) for v in dist):
        raise DataError("gold %r: probabilities must be >= 0" % qid)
    s = sum(dist)
    if s <= 0:
        raise DataError("gold %r: probabilities sum to 0" % qid)
    dist = [v / s for v in dist]
    if label is None:
        label = max(range(len(dist)), key=dist.__getitem__)
    return dist, label


# --------------------------------------------------------------------------- files
def load_cases(path: str, shared_questions: dict | None = None) -> tuple[list[dict], list[str]]:
    """Read a JSONL file. Returns (cases, errors); a case with any error is dropped."""
    cases, errors = [], []
    shared = {k: check_question(k, v) for k, v in (shared_questions or {}).items()}
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            where = "%s:%d" % (path, n)
            try:
                row = json.loads(line)
            except ValueError as e:
                errors.append("%s: not valid JSON (%s)" % (where, e))
                continue
            try:
                cases.append(normalise_case(row, shared, default_id="%s#%d" % (path.rsplit("/", 1)[-1], n)))
            except DataError as e:
                errors.append("%s: %s" % (where, e))
    return cases, errors


def normalise_case(row: dict, shared: dict, default_id: str) -> dict:
    if not isinstance(row, dict):
        raise DataError("each line must be a JSON object")
    # typed-decisions stores these as JSON strings; accept both
    for k in ("state", "questions", "gold"):
        if isinstance(row.get(k), str) and k != "state":
            try:
                row[k] = json.loads(row[k])
            except ValueError:
                raise DataError("%s is a string but not JSON" % k)
    if "state" not in row or row["state"] in (None, "", {}, []):
        raise DataError("state is missing or empty")
    qs = dict(shared)
    qs.update({k: check_question(k, v) for k, v in (row.get("questions") or {}).items()})
    if not qs:
        raise DataError("no questions (give them per line or with --questions)")
    gold_in = row.get("gold")
    if not isinstance(gold_in, dict) or not gold_in:
        raise DataError("gold is missing or empty")
    unknown = set(gold_in) - set(qs)
    if unknown:
        raise DataError("gold answers unknown questions %s" % sorted(unknown))
    gold = {}
    for qid, g in gold_in.items():
        if g is None:
            continue  # explicitly unlabelled
        dist, label = gold_distribution(qid, qs[qid], g)
        gold[qid] = {"dist": dist, "label": label}
    if not gold:
        raise DataError("no usable gold answers")
    return {"id": str(row.get("id") or default_id), "state": row["state"],
            "questions": {k: qs[k] for k in gold}, "all_questions": qs, "gold": gold,
            "workflow": row.get("workflow")}


def split_of(case_id: str, seed: int, calib_frac: float, test_frac: float) -> str:
    """Deterministic split by hashed id, so adding data never reshuffles old cases."""
    h = int(hashlib.sha256(("%d:%s" % (seed, case_id)).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < test_frac:
        return "test"
    if h < test_frac + calib_frac:
        return "calib"
    return "train"


def load_splits(train: str, calib: str | None = None, test: str | None = None, questions: str | None = None,
                calib_frac: float = 0.1, test_frac: float = 0.1, seed: int = 0) -> tuple[dict, list[str]]:
    """Explicit calib/test files win; otherwise carve them out of `train` by id hash."""
    shared = None
    if questions:
        with open(questions, encoding="utf-8") as f:
            shared = json.load(f)
    splits, errors = {"train": [], "calib": [], "test": []}, []
    base, errs = load_cases(train, shared)
    errors += errs
    fracs = (0.0 if calib else calib_frac, 0.0 if test else test_frac)
    for c in base:
        splits[split_of(c["id"], seed, *fracs)].append(c)
    for name, path in (("calib", calib), ("test", test)):
        if path:
            cs, errs = load_cases(path, shared)
            splits[name], errors = cs, errors + errs
    return splits, errors


def subsample(cases: list[dict], n: int | None, seed: int = 0) -> list[dict]:
    """A deterministic, *nested* random subset: the 150-case sample is contained in
    the 300-case one, so learning-curve runs differ only in how much data they see.
    (Taking the first N lines would follow file order, which is often grouped.)"""
    if not n or n >= len(cases):
        return cases
    key = lambda c: hashlib.sha256(("sub:%d:%s" % (seed, c["id"])).encode()).hexdigest()
    return sorted(cases, key=key)[:n]


def schemas(cases: list[dict], examples: list[dict] | None = None) -> dict:
    """The question set(s) a model was trained on, one per workflow (or "default"),
    with an example state - so whoever serves the model can ask it the right questions."""
    out = {}
    for c in cases:
        out.setdefault(c.get("workflow") or "default", {"questions": c["all_questions"], "example_state": c["state"]})
    for c in examples or []:  # prefer a held-out example over a training one
        key = c.get("workflow") or "default"
        if key in out and not out[key].get("_held_out"):
            out[key].update(example_state=c["state"], _held_out=True)
    for v in out.values():
        v.pop("_held_out", None)
    return out


def stats(cases: list[dict]) -> dict:
    per_q: dict[str, dict] = {}
    for c in cases:
        for qid, g in c["gold"].items():
            q = c["questions"][qid]
            s = per_q.setdefault(qid, {"type": q["type"], "n": 0, "soft": 0, "labels": {}})
            s["n"] += 1
            s["soft"] += int(max(g["dist"]) < 0.999)
            key = option_keys(q)[g["label"]]
            s["labels"][key] = s["labels"].get(key, 0) + 1
    return per_q
