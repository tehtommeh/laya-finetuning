#!/usr/bin/env python3
"""Concurrent load test for /v1/decide (stdlib only).

    python3 scripts/load_test.py [--api http://localhost:8001] [--clients 1 4 16 64] [--seconds 15]
                                 [--kind short|long|both] [--model english]

N client threads each send /v1/decide requests back to back (one state, 3
questions) for --seconds. Reports requests/s, p50/p95/p99 latency as seen by
the client, and errors. "short" = one-sentence tickets (~110 tokens/state),
"long" = typed-decisions states (~500 tokens) if data/typed-decisions exists.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import random
import statistics
import threading
import time
import urllib.parse

QUESTIONS = {
    "team": {"type": "choice", "instructions": "Which team?",
             "criteria": {"billing": "payments", "technical": "bugs", "sales": "pricing", "other": "else"}},
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "medium", "high"]},
    "flag": {"type": "noul", "instructions": "Does it need a human?"},
}


def short_states(n=500, seed=0):
    rnd = random.Random(seed)
    subj = ["charged twice", "app crashes on login", "need a quote for 50 seats", "password reset not arriving",
            "cancel my plan", "API returns 500", "invoice has wrong VAT number", "dark mode request",
            "slow dashboard", "refund for last month"]
    return ["Hi, %s. %s" % (rnd.choice(subj), rnd.choice(["Please help.", "This is urgent!", "Thanks.", "Any update?"]))
            for _ in range(n)]


def long_states(seed=0):
    path = os.path.join(os.path.dirname(__file__), "..", "data", "typed-decisions", "test.jsonl")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        states = [json.loads(line)["state"] for line in f]
    random.Random(seed).shuffle(states)
    return states


def run(api, states, clients, seconds, model):
    u = urllib.parse.urlparse(api)
    lat, errors, lock = [], [], threading.Lock()
    stop = time.perf_counter() + seconds

    def client(cid):
        conn = http.client.HTTPConnection(u.hostname, u.port, timeout=300)  # keep-alive per client
        i = cid
        while time.perf_counter() < stop:
            body = json.dumps({"state": states[i % len(states)], "questions": QUESTIONS, "model": model})
            t = time.perf_counter()
            try:
                conn.request("POST", "/v1/decide", body, {"content-type": "application/json"})
                r = conn.getresponse()
                data = r.read()
                ok = r.status == 200 and b'"answers"' in data
            except Exception as e:  # noqa: BLE001
                ok, data = False, str(e).encode()
                conn = http.client.HTTPConnection(u.hostname, u.port, timeout=300)
            dt = (time.perf_counter() - t) * 1000
            with lock:
                (lat if ok else errors).append(dt if ok else data[:120])
            i += clients

    t0 = time.perf_counter()
    threads = [threading.Thread(target=client, args=(c,)) for c in range(clients)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    wall = time.perf_counter() - t0
    lat.sort()
    pct = lambda p: lat[min(len(lat) - 1, int(len(lat) * p))] if lat else float("nan")
    return {"clients": clients, "requests": len(lat), "req_per_s": len(lat) / wall, "p50": pct(0.50),
            "p95": pct(0.95), "p99": pct(0.99), "errors": len(errors), "first_error": errors[0] if errors else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", default="http://localhost:8001")
    ap.add_argument("--clients", type=int, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("--seconds", type=float, default=15)
    ap.add_argument("--kind", choices=("short", "long", "both"), default="both")
    ap.add_argument("--model", default="english")
    a = ap.parse_args()
    kinds = {"short": short_states()}
    if a.kind in ("long", "both"):
        kinds["long"] = long_states()
    if a.kind == "long":
        kinds.pop("short")
    print("%-6s %8s %9s %9s %9s %9s %9s %7s" % ("states", "clients", "requests", "req/s", "p50 ms", "p95 ms", "p99 ms", "errors"))
    for kind, states in kinds.items():
        if not states:
            print("%-6s (no data/typed-decisions; run make example-data)" % kind)
            continue
        run(a.api, states, 4, 2, a.model)  # warm-up
        for c in a.clients:
            r = run(a.api, states, c, a.seconds, a.model)
            print("%-6s %8d %9d %9.1f %9.1f %9.1f %9.1f %7d" % (kind, c, r["requests"], r["req_per_s"], r["p50"],
                                                              r["p95"], r["p99"], r["errors"]))
            if r["first_error"]:
                print("       first error: %s" % r["first_error"])


if __name__ == "__main__":
    main()
