#!/usr/bin/env python3
"""CUDA-graph path tests (api/batching.py GraphRunner), on the GPU with real checkpoints.

    docker compose exec -T api python - < scripts/test_cuda_graphs.py      # or: make test-coalescing

For every captured bucket of every checkpoint, with real encoded requests:
  1. graph replay == eager model on the same padded tensors: bit for bit in almost all
     cases; otherwise every answer probability within PROB_TOL. Capture makes the
     encoder's GPU libraries pick slightly different kernels, which can move a value by a
     bf16 rounding step for an unlucky input; it compounds through the layers but stays
     far below PROB_TOL (deterministic; localised to the encoder, not the memory pool or
     streams). A real graph bug - a wrong buffer, another row's data - moves
     probabilities by far more, so this still catches it;
  2. still matching after replaying other buckets in random order first (the buckets
     share one memory pool; this is the safety condition for that);
  3. vs eager on unpadded tensors (what laya does): reports the bf16 padding noise;
  4. passes that do not fit a bucket (too wide, too many rows, >16 options) fall back.
"""
from __future__ import annotations

import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, "/app")
import laya  # noqa: E402
import batching  # noqa: E402

FAILS = []
PROB_TOL = 0.05      # answer-probability difference allowed between graph and eager at the same shape
M = "/models/convaiinnovations__laya"
WORDS = "refund invoice outage login crash quote seats urgent billing cancel password dashboard".split()


def check(name, cond, detail=""):
    print("%s %s%s" % ("PASS" if cond else "FAIL", name, (" - " + detail) if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def questions(rnd, n):
    qs = {}
    for j in range(n):
        t = rnd.choice(["choice", "score", "noul"])
        if t == "choice":
            qs["q%d" % j] = {"type": t, "instructions": "Which area?",
                             "criteria": {"o%d" % k: rnd.choice(WORDS) for k in range(rnd.randint(2, 6))}}
        elif t == "score":
            qs["q%d" % j] = {"type": t, "instructions": "How urgent?", "criteria": ["low", "mid", "high", "now"][:rnd.randint(2, 4)]}
        else:
            qs["q%d" % j] = {"type": t, "instructions": "Does it ask for money back?"}
    return qs


def items_for(ag, rnd, rows, length):
    """Real encoded rows: `rows` of them, each no longer than `length` tokens."""
    out = []
    while len(out) < rows:
        words = rnd.randint(1, max(1, length // 3))
        state = " ".join(rnd.choice(WORDS) for _ in range(words))
        its, _ = batching.encode(ag, [state], questions(rnd, rnd.randint(1, 3)))
        out += [it for it in its if len(it["ids"]) <= length]
    return out[:rows]


@torch.no_grad()
def eager(ag, tensors):
    with torch.autocast("cuda", dtype=ag.dtype):
        logits, act = ag.model(*[torch.as_tensor(t).cuda() for t in tensors])
    return logits.float().cpu().numpy(), torch.softmax(act.float(), -1).cpu().numpy()


def probs(z, k):
    z = z[:k] - z[:k].max()
    p = np.exp(z)
    return p / p.sum()


def main():
    rnd = random.Random(0)
    for name, path, sub in (("english", M, None), ("multilingual", M, "multilingual"), ("td-full", "/finetuned/td-full", None)):
        try:
            ag = laya.Agent(path, device="cuda", subfolder=sub)
        except Exception as e:  # noqa: BLE001 - a fine-tune may not be published
            print("skip %s: %s" % (name, e))
            continue
        batching.precast_weights(ag.model, ag.dtype)
        runner = batching.GraphRunner(ag, max_tokens=4096)   # capture() also limits to buckets that can beat eager
        n = runner.capture()
        keys = list(runner.graphs)
        exact, near, noise, worst_same_shape = 0, 0, [], 0.0
        for key in keys:
            rows, length = key
            for trial in range(3):
                n_rows = rnd.randint(max(1, rows // 2), rows)
                its = items_for(ag, rnd, n_rows, length)
                for other in rnd.sample(keys, min(4, len(keys))):       # replay other buckets first
                    runner.run_bucket(items_for(ag, rnd, 1, other[1]), other)
                got = runner.run_bucket(its, key)
                ref = eager(ag, runner.pad_to(its, key))
                same = np.array_equal(got[0], ref[0][:len(its)]) and np.array_equal(got[1], ref[1][:len(its)])
                exact += same
                pdiff = max(np.abs(probs(got[0][r], len(it["markers"])) - probs(ref[0][r], len(it["markers"]))).max()
                            for r, it in enumerate(its))
                adiff = float(np.abs(got[1] - ref[1][:len(its)]).max())
                worst_same_shape = max(worst_same_shape, pdiff, adiff)
                close = pdiff <= PROB_TOL and adiff <= PROB_TOL
                near += close
                if not close:
                    print("  mismatch at bucket %s: probability diff %.4f, act diff %.4f" % (key, pdiff, adiff))
                    if os.environ.get("DUMP_MISMATCH"):
                        np.savez("/tmp/mismatch_%s.npz" % name, *runner.pad_to(its, key), got=got[0], ref=ref[0])
                unpadded = eager(ag, batching.collate(its, ag.tok.pad_token_id))
                for r, it in enumerate(its):
                    k = len(it["markers"])
                    noise.append(np.abs(probs(got[0][r], k) - probs(unpadded[0][r], k)).max())
        total = len(keys) * 3
        check("%s: %d graphs; replay vs eager on the same padded tensors: %d/%d bit for bit, all within %.2f "
              "(max probability diff %.4f; other buckets replayed in between)" % (name, n, exact, total, PROB_TOL,
                                                                                  worst_same_shape), near == total)
        noise.sort()
        print("     vs unpadded eager (bf16 padding noise): median %.4f  p99 %.4f  max %.4f in probability"
              % (noise[len(noise) // 2], noise[int(len(noise) * .99)], noise[-1]))
        wide = items_for(ag, rnd, 1, runner.max_len)
        wide[0]["ids"] = wide[0]["ids"] + [ag.tok.pad_token_id] * (runner.max_len + 1 - len(wide[0]["ids"]))
        many_opts = dict(items_for(ag, rnd, 1, 64)[0], markers=list(range(1, 18)))
        # routing: every choice must be the faster option by the runner's own measurements
        bad_route = 0
        for n_rows in (1, 3, 5, 8, 16, 30):
            for width in (40, 100, 250, 450):
                if width > runner.max_len:
                    continue
                key = runner.choose(n_rows, width, 4)
                est = runner.eager_estimate_ms(n_rows, width)
                if key is not None and runner.graph_ms[key] >= est:
                    bad_route += 1
        print("     eager floor %.1f ms, %.2f ms per 1k tokens; e.g. 1x40 -> %s, 5x250 -> %s, 16x100 -> %s" % (
            runner.eager_floor_ms, runner.eager_ms_per_token * 1000, runner.choose(1, 40, 4),
            runner.choose(5, 250, 4), runner.choose(16, 100, 4)))
        check("%s: routing picks a graph only when its measured time beats eager" % name, bad_route == 0)
        check("%s: passes that fit no bucket fall back (too wide / too many rows / >16 options)" % name,
              runner.run(wide) is None and runner.run(items_for(ag, rnd, 33, 64)) is None
              and runner.run([many_opts]) is None)
        del runner, ag
        torch.cuda.empty_cache()
    print("\n%s: %d failure(s)" % ("ALL PASSED" if not FAILS else "FAILED", len(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
