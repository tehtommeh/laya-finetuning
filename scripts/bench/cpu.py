"""CPU micro-benchmarks for the request path (docs/PERFORMANCE.md, stage 4).

    bash scripts/bench/run.sh cpu.py T1          # T1..T6; each is single-threaded and CPU-only,
                                                 # so several can run in parallel
"""
import asyncio
import json
import os
import random
import statistics
import sys
import time

sys.path.insert(0, "/app")
sys.path.insert(0, "/tmp/pylib")   # T3 only: pip install --target /tmp/pylib orjson (if not already installed)
import numpy as np
import torch

torch.set_num_threads(1)
from transformers import AutoTokenizer
import laya
from laya.common import QTYPES, build_sequence, clamp_temperature, collate_items, render_options

M = "/models/convaiinnovations__laya"
QS = {"team": {"type": "choice", "instructions": "Which team?",
               "criteria": {"billing": "payments", "technical": "bugs", "sales": "pricing", "other": "else"}},
      "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "medium", "high"]},
      "flag": {"type": "noul", "instructions": "Does it need a human?"}}
rnd = random.Random(0)
SUBJ = ["charged twice", "app crashes on login", "need a quote for 50 seats", "password reset not arriving",
        "cancel my plan", "API returns 500", "invoice has wrong VAT number", "dark mode request"]
SHORT = ["Hi, %s. %s" % (rnd.choice(SUBJ), rnd.choice(["Please help.", "This is urgent!", "Thanks."])) for _ in range(500)]


class LiteAgent:
    """What encode/postprocess need from laya.Agent, without loading the model."""
    _to_internal = staticmethod(laya.Agent._to_internal)

    def __init__(self):
        self.cfg = json.load(open(M + "/rl_agent_config.json"))
        self.tok = AutoTokenizer.from_pretrained(M + "/tokenizer")
        self.temperature = [clamp_temperature(t) for t in self.cfg["temperature"]]
        self.temperature_by_options = {k: clamp_temperature(v) for k, v in self.cfg["temperature_by_options"].items()}


def bench(fn, n=2000, warm=50):
    for _ in range(warm):
        fn()
    t = time.process_time()
    for _ in range(n):
        fn()
    return (time.process_time() - t) / n * 1e6  # CPU µs per call


def T1():
    """Building a pass's tensors: laya.collate_items vs numpy-vectorised."""
    ag = LiteAgent()
    import batching
    for label, states, nrows in (("short, 60 rows", SHORT, 20), ("long-ish, 20 rows", [s * 12 for s in SHORT], 7)):
        items = [it for s in states[:nrows] for it in batching.encode(ag, [s], QS)[0]]
        pad = ag.tok.pad_token_id

        def vec():
            n, L = len(items), max(len(it["ids"]) for it in items)
            kmax = max(len(it["markers"]) for it in items)
            ids = np.full((n, L), pad, dtype=np.int64)
            att = np.zeros((n, L), dtype=np.int64)
            mpos = np.zeros((n, kmax), dtype=np.int64)
            mmask = np.zeros((n, kmax), dtype=bool)
            for i, it in enumerate(items):
                l = len(it["ids"]); ids[i, :l] = it["ids"]; att[i, :l] = 1
                k = len(it["markers"]); mpos[i, :k] = it["markers"]; mmask[i, :k] = True
            return (torch.from_numpy(ids), torch.from_numpy(att), torch.from_numpy(mpos), torch.from_numpy(mmask),
                    torch.tensor([it["qtype"] for it in items]))
        a = collate_items([items], pad); b = vec()
        same = all(torch.equal(a[k], v) for k, v in zip(("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"), b))
        print("T1 collate %-18s laya.collate_items %7.0f µs | numpy %6.0f µs | identical=%s" % (
            label, bench(lambda: collate_items([items], pad), 300), bench(vec, 300), same))


def T2():
    """Encoding one request: current (state tokenised once per question) vs cached question heads."""
    ag = LiteAgent()
    import batching
    tok, max_len, hml = ag.tok, ag.cfg["max_len"], ag.cfg["head_max_len"]
    internal = {q: ag._to_internal(v) for q, v in QS.items()}
    # cache: everything before the state depends only on the question -> tokenise once per question definition
    heads = {}
    for qid, iq in internal.items():
        ids, markers = build_sequence(tok, "", iq, max_len, hml)   # empty state: head + [SEP] + [SEP]
        heads[qid] = (ids[:-1], markers)                            # drop the trailing [SEP] of the empty state

    def cached(state):
        st = tok(laya.common.serialize_state(state).replace(tok.mask_token, " "), add_special_tokens=False)["input_ids"]
        out = []
        for qid, (head, markers) in heads.items():
            room = max(0, max_len - len(head) - 1)
            out.append((head + st[:room] + [tok.sep_token_id])[:max_len])
        return out
    same = all([it["ids"] for it in sorted(batching.encode(ag, [s], QS)[0], key=lambda it: list(QS).index(it["q"]))]
               == cached(s) for s in SHORT[:200] + [SHORT[0] * 200])
    i = iter(range(10 ** 9))
    print("T2 encode  current batching.encode %6.0f µs/request | cached heads + state once %6.0f µs | identical=%s" % (
        bench(lambda: batching.encode(ag, [SHORT[next(i) % 500]], QS)), bench(lambda: cached(SHORT[next(i) % 500])), same))


def T3():
    """Request/response JSON + validation."""
    import orjson
    import app as A
    from fastapi.encoders import jsonable_encoder
    body = json.dumps({"state": SHORT[0], "questions": QS, "model": "english"}).encode()
    resp = {"model": "laya-rl-agent", "answers": {
        "team": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.9, "technical": 0.05, "sales": 0.03, "other": 0.02},
                 "confidence": 0.61, "action": {"act_probability": 0.7}},
        "urgency": {"type": "score", "score": 1.2, "legend": {"0": "low", "1": "medium", "2": "high"},
                    "probabilities": {"0": 0.1, "1": 0.6, "2": 0.3}, "confidence": 0.3, "action": {"act_probability": 0.7}},
        "flag": {"type": "noul", "noul": 0.8, "confidence": 0.8, "action": {"act_probability": 0.7}}},
        "usage": {"input_tokens": 120, "output_tokens": 0}, "latency_ms": 26.1,
        "routing": {"model": "english", "repo": "x", "reason": "English Latin text", "workflow": None,
                    "detection": {"script": "latin", "language": "en", "is_english": True, "language_undecided": False,
                                  "diacritic_rate": 0.0, "non_latin_fraction": 0.0}},
        "timing": {"queue_ms": 1.0, "gpu_passes": 1}}
    req = A.DecideRequest.model_validate_json(body)
    print("T3 json    parse+validate: pydantic model_validate_json %5.0f µs | json.loads+model_validate %5.0f µs" % (
        bench(lambda: A.DecideRequest.model_validate_json(body)), bench(lambda: A.DecideRequest.model_validate(json.loads(body)))))
    print("T3 json    _questions() normalise %5.0f µs" % bench(lambda: A._questions(req.questions)))
    print("T3 json    response: jsonable_encoder+json.dumps (FastAPI default) %5.0f µs | orjson.dumps %5.0f µs" % (
        bench(lambda: json.dumps(jsonable_encoder(resp), ensure_ascii=False, separators=(",", ":")).encode()),
        bench(lambda: orjson.dumps(resp))))


def T4():
    """Post-processing one request's 3 rows."""
    ag = LiteAgent()
    import batching
    items, meta = batching.encode(ag, [SHORT[0]], QS)
    t = batching.Ticket(ag, items, meta)
    for i, it in enumerate(items):
        t.logits[i] = np.random.randn(6); t.act[i] = np.array([0.7, 0.3])
    print("T4 postprocess one request (3 rows) %5.0f µs" % bench(lambda: batching.postprocess(t)))


def T5():
    """Routing: language/script detection."""
    router = laya.Router(models={"english": (M, None), "multilingual": (M, "multilingual"), "typed-decisions": (M, "typed-decisions")})
    long = {"subject": SHORT[1], "body": " ".join(SHORT[:40])}
    print("T5 route   short state %5.0f µs | ~500-token state %5.0f µs" % (
        bench(lambda: router.route(SHORT[3], QS)), bench(lambda: router.route(long, QS), 500)))


def T6():
    """Hand-offs: asyncio.to_thread round trip, and Future -> wrap_future wake-up."""
    import concurrent.futures
    async def main():
        loop = asyncio.get_running_loop()
        async def hop():
            await asyncio.to_thread(lambda: None)
        async def fut():
            f = concurrent.futures.Future()
            w = asyncio.wrap_future(f)
            f.set_result(1)
            await w
        for name, fn in (("to_thread round trip", hop), ("Future->wrap_future", fut)):
            for _ in range(200):
                await fn()
            t, w = time.process_time(), time.perf_counter()
            n = 3000
            for _ in range(n):
                await fn()
            print("T6 hops    %-22s CPU %5.0f µs  wall %5.0f µs" % (name, (time.process_time() - t) / n * 1e6, (time.perf_counter() - w) / n * 1e6))
    asyncio.run(main())


if __name__ == "__main__":
    globals()[sys.argv[1]]()
