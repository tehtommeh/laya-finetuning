#!/usr/bin/env python3
"""Unit tests for the coalescing scheduler (api/batching.py), with a fake model.

    docker compose exec -T api python - < scripts/test_scheduler.py      # or: make test-coalescing

Runs inside the API container (for torch and laya) but never touches the GPU:
the fake forward pass returns each row's own id as its output, so any result
landing in the wrong ticket or position is detected exactly. Covers isolation
under heavy concurrency, grouping by checkpoint, failure bisection, OOM
splitting, the fatal-error path, cancellation, fairness both ways, and
backpressure.
"""
from __future__ import annotations

import random
import sys
import threading
import time

import numpy as np
import torch

sys.path.insert(0, "/app")
import batching  # noqa: E402
from batching import Busy, FatalGPUError, Scheduler  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print("%s %s%s" % ("PASS" if cond else "FAIL", name, (" - " + detail) if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


class FakeAgent:
    def __init__(self, name):
        self.name = name
        self.device = torch.device("cpu")


def rows(ticket_id, n, agent, poison=()):
    return [{"uid": ticket_id * 100000 + i, "agent": agent.name, "ids": [0] * random.randint(20, 400),
             "poison": i in poison} for i in range(n)]


def make_forward(delay=0.0, oom_above=None, calls=None):
    def forward(agent, items):
        if calls is not None:
            calls.append((agent.name, len(items), {it["uid"] // 100000 for it in items}))
        if any(it["agent"] != agent.name for it in items):
            raise AssertionError("rows of another checkpoint were mixed into this pass")
        if oom_above is not None and len(items) > oom_above:
            raise torch.cuda.OutOfMemoryError("fake OOM")
        if any(it["poison"] for it in items):
            raise ValueError("poison row")
        if delay:
            time.sleep(delay)
        logits = np.array([[it["uid"], -it["uid"]] for it in items], dtype=np.float64)
        act = np.array([[it["uid"] % 7, 0] for it in items], dtype=np.float64)
        return logits, act
    return forward


def verify(ticket):
    """Every row's output must be its own id, at its own index."""
    bad = [i for i, it in enumerate(ticket.items)
           if i not in ticket.errors and (ticket.logits[i] is None or ticket.logits[i][0] != it["uid"]
                                          or ticket.act[i][0] != it["uid"] % 7)]
    return bad


def test_isolation_under_concurrency():
    agents = [FakeAgent("english"), FakeAgent("multilingual"), FakeAgent("ft")]
    calls = []
    s = Scheduler(token_budget=8192, queue_max=10 ** 7, forward_fn=make_forward(delay=0.002, calls=calls))
    s.start()
    tickets, lock = [], threading.Lock()

    def client(cid):
        rnd = random.Random(cid)
        for k in range(20):
            agent = rnd.choice(agents)
            n = rnd.choice([1, 3, 3, 5, 10, 60, 300])
            t = s.submit(agent, rows(cid * 1000 + k, n, agent), {})
            t.future.result(timeout=120)
            with lock:
                tickets.append(t)
    threads = [threading.Thread(target=client, args=(c,)) for c in range(200)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    s.stop()
    bad = [(t.items[0]["uid"] // 100000, verify(t)) for t in tickets if verify(t)]
    check("isolation: 4,000 tickets from 200 threads, every row got its own result", not bad, str(bad[:3]))
    check("isolation: requests really were coalesced", s.stats["max_tickets_per_pass"] > 5,
          "max tickets per pass %d" % s.stats["max_tickets_per_pass"])
    owner = {t.items[0]["uid"] // 100000: t.items[0]["agent"] for t in tickets}
    mixed = [c for c in calls if any(owner[tid] != c[0] for tid in c[2])]
    check("grouping: every pass held a single checkpoint (%d passes checked)" % len(calls), not mixed, str(mixed[:2]))
    check("no ticket failed", not any(t.errors or t.dead for t in tickets))


def test_failure_bisection():
    a = FakeAgent("english")
    s = Scheduler(forward_fn=make_forward(delay=0.001))
    s.stats  # noqa: B018
    # hold the worker so everything lands in shared passes
    good = [s.submit(a, rows(i, 5, a), {}) for i in range(1, 30)]
    bad1 = s.submit(a, rows(900, 8, a, poison={3}), {})
    bad2 = s.submit(a, rows(901, 4, a, poison={0, 2}), {})
    s.start()
    for t in good + [bad1, bad2]:
        t.future.result(timeout=30)
    s.stop()
    check("bisection: tickets sharing a pass with a poison row still succeed",
          all(not t.errors and not verify(t) for t in good))
    check("bisection: only the poison rows failed", set(bad1.errors) == {3} and set(bad2.errors) == {0, 2},
          "%s %s" % (sorted(bad1.errors), sorted(bad2.errors)))
    check("bisection: the non-poison rows of a partly failed ticket have results", not verify(bad1) and not verify(bad2))
    # post-processing turns a failed row into a per-state error
    t = bad1
    t.meta = {"qids": ["q"], "internal": {"q": None}, "n_states": 8}
    for i, it in enumerate(t.items):
        it["s"], it["q"], it["markers"] = i, "q", [0, 1]
    orig = batching._answer
    batching._answer = lambda agent, iq, z, a_, k: {"v": float(z[0])}
    out = batching.postprocess(t)
    batching._answer = orig
    check("partial failure: failed state gets {'error'}, others get answers",
          "error" in out[3] and all("answers" in out[i] for i in range(8) if i != 3))


def test_oom_split():
    a = FakeAgent("english")
    s = Scheduler(forward_fn=make_forward(oom_above=4))
    ts = [s.submit(a, rows(i, 7, a), {}) for i in range(1, 6)]
    s.start()
    for t in ts:
        t.future.result(timeout=30)
    s.stop()
    check("OOM: oversized passes are split until they fit, nothing fails",
          all(not t.errors and not verify(t) for t in ts))


def test_fatal():
    a = FakeAgent("english")

    def boom(agent, items):
        raise RuntimeError("CUDA error: device-side assert triggered")
    orig = batching._cuda_usable
    batching._cuda_usable = lambda device: False
    s = Scheduler(forward_fn=boom, exit_on_fatal=False)
    ts = [s.submit(a, rows(i, 3, a), {}) for i in range(1, 5)]
    s.start()
    errs = []
    for t in ts:
        try:
            t.future.result(timeout=10)
        except Exception as e:  # noqa: BLE001
            errs.append(e)
    s.thread.join(5)
    batching._cuda_usable = orig
    check("fatal: every queued request fails with FatalGPUError",
          len(errs) == 4 and all(isinstance(e, FatalGPUError) for e in errs))
    check("fatal: the worker stops", not s.alive)
    try:
        s.submit(a, rows(99, 1, a), {})
        check("fatal: new submissions are refused", False)
    except RuntimeError:
        check("fatal: new submissions are refused", True)


def test_cancellation():
    a = FakeAgent("english")
    calls = []
    s = Scheduler(forward_fn=make_forward(calls=calls))
    keep = s.submit(a, rows(1, 5, a), {})
    drop = s.submit(a, rows(2, 50, a), {})
    drop.dead = True                          # what the API does on timeout
    s.start()
    keep.future.result(timeout=10)
    time.sleep(0.2)
    s.stop()
    ran = set().union(*(c[2] for c in calls))
    check("cancellation: an abandoned ticket's rows never run", 2 not in ran and 1 in ran, str(ran))
    t = batching.Ticket(a, rows(3, 1, a), {})
    t.future.cancel()
    t.resolve()                               # must not raise InvalidStateError
    check("cancellation: resolving a cancelled future does not raise", t.dead)


def test_fairness():
    a = FakeAgent("english")
    s = Scheduler(token_budget=4096, forward_fn=make_forward(delay=0.01))
    s.start()
    big = s.submit(a, rows(1, 3000, a), {})
    time.sleep(0.05)
    waits = []
    for i in range(20):                       # singles arriving while the big batch runs
        t0 = time.perf_counter()
        t = s.submit(a, rows(100 + i, 3, a), {})
        t.future.result(timeout=30)
        waits.append(time.perf_counter() - t0)
    big_done_early = big.future.done()
    big.future.result(timeout=120)
    s.stop()
    check("fairness: singles are not stuck behind a big batch", max(waits) < 0.2 and not big_done_early,
          "max single wait %.3fs, big finished first: %s" % (max(waits), big_done_early))

    s = Scheduler(token_budget=4096, forward_fn=make_forward(delay=0.005))
    s.start()
    big = s.submit(a, rows(2, 400, a), {})
    stop = time.perf_counter() + 5
    flood = threading.Event()

    def flooder(k):
        i = 0
        while not flood.is_set() and time.perf_counter() < stop:
            s.submit(a, rows(1000 + k * 10000 + i, 3, a), {}).future.result(timeout=30)
            i += 1
    threads = [threading.Thread(target=flooder, args=(k,)) for k in range(32)]
    for th in threads:
        th.start()
    ok = True
    try:
        big.future.result(timeout=5)
    except Exception:  # noqa: BLE001
        ok = False
    flood.set()
    for th in threads:
        th.join()
    s.stop()
    check("fairness: a big batch still finishes under a constant flood of singles", ok and not verify(big))


def test_backpressure_and_empty():
    a = FakeAgent("english")
    s = Scheduler(queue_max=100, forward_fn=make_forward())
    s.submit(a, rows(1, 80, a), {})        # worker not started: stays queued
    try:
        s.submit(a, rows(2, 30, a), {})
        check("backpressure: a full queue refuses new work", False)
    except Busy:
        check("backpressure: a full queue refuses new work", True)
    s.start()
    t = s.submit(a, [], {})
    check("empty ticket resolves immediately", t.future.done())
    s.stop()


if __name__ == "__main__":
    random.seed(0)
    for fn in (test_isolation_under_concurrency, test_failure_bisection, test_oom_split, test_fatal,
               test_cancellation, test_fairness, test_backpressure_and_empty):
        fn()
    print("\n%s: %d failure(s)" % ("FAILED" if FAILS else "ALL PASSED", len(FAILS)))
    sys.exit(1 if FAILS else 0)
