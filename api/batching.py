"""GPU batching and request coalescing for laya Agents.

Every inference request - a single /v1/decide, one checkpoint of /v1/compare,
or a group of /v1/decide/batch - becomes a Ticket: its (state, question)
sequences, encoded with the SDK's own `build_sequence`, plus a Future for the
reply. One worker thread owns the GPU. Each round it takes queued rows for one
checkpoint (up to `round_factor` x the token budget), sorts them by length and
packs them into forward passes of at most the token budget - sorting matters:
coalesced rows range from ~100 to 1,000 tokens and every row in a pass is padded
to the longest (unsorted, 43% of GPU work was padding). Each row goes back to
the ticket it came from; when a ticket's last row arrives its Future resolves
and the waiting request post-processes its own rows.

  * Opportunistic: nothing waits on purpose. An idle GPU runs a lone request at
    once; requests arriving while the GPU is busy are combined next round.
  * Fair: the oldest ticket gets at least `oldest_share` of every round (so a
    big batch always progresses), the rest goes to tickets with the fewest
    remaining rows first (so single requests are not stuck behind a batch).
  * Isolated: rows are written back by (ticket, index) only; a ticket's results
    are assembled from its own rows and nothing else. scripts/test_coalescing.py
    checks this under concurrency against the SDK's own system_one.
  * Failure-contained: a failing forward pass is bisected until the failing
    rows are isolated, so only the states owning them get an error. A fatal
    CUDA error (the context is unusable afterwards) fails everything queued and
    exits the process so Docker's restart policy brings the API back cleanly.

Post-processing (`_answer`) mirrors laya.Agent.system_one line for line (same
temperatures, rounding and output shape). If the laya pin is bumped, re-run
scripts/test_coalescing.py and scripts/test_batch_equivalence.py.
"""
from __future__ import annotations

import collections
import concurrent.futures
import logging
import os
import threading
import time
from typing import Any, Optional

import numpy as np
import torch

from laya.common import (QTYPES, build_sequence, collate_items, confidence_from_probs, render_options,
                         temp_bucket)

log = logging.getLogger("api.batching")


# --------------------------------------------------------------------------- encoding / post-processing
def encode(agent, states: list, questions: dict) -> tuple[list[dict], dict]:
    """(state, question) sequences for many states, sorted by length (less padding when
    a ticket is split across chunks). Raises ValueError, like system_one, if a
    question's options do not fit the checkpoint's head_max_len."""
    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)
    qids = list(questions)
    internal = {qid: agent._to_internal(questions[qid]) for qid in qids}
    items = []
    for si, state in enumerate(states):
        for qid in qids:
            iq = internal[qid]
            seq, markers = build_sequence(agent.tok, state, iq, max_len, head_max_len)
            if len(markers) != len(render_options(iq)):
                raise ValueError("question %r options exceed head_max_len=%d" % (qid, head_max_len))
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[iq["t"]], "s": si, "q": qid})
    items.sort(key=lambda it: len(it["ids"]))
    return items, {"qids": qids, "internal": internal, "n_states": len(states)}


def _answer(agent, iq: dict, logits_row: np.ndarray, act_row: np.ndarray, k: int) -> dict:
    """One question's answer, exactly as laya.Agent.system_one builds it."""
    qt = QTYPES[iq["t"]]
    t_scale = agent.temperature_by_options.get(temp_bucket(qt, k), agent.temperature[qt])
    z = logits_row[:k] / t_scale
    p = np.exp(z - z.max())
    p = p / p.sum()
    conf_score = round(confidence_from_probs(p, k), 4)
    ext = {"act_probability": round(float(act_row[0]), 4)}
    if iq["t"] == "choice":
        keys = list(iq["crit"].keys())
        return {"type": "choice", "choice": keys[int(p.argmax())],
                "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
                "confidence": conf_score, "action": ext}
    if iq["t"] == "score":
        return {"type": "score", "score": round(float((np.arange(k) * p).sum()), 4),
                "legend": {str(i): c for i, c in enumerate(iq["crit"])},
                "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                "confidence": conf_score, "action": ext}
    return {"type": "noul", "noul": round(float(p[1]), 4),
            "confidence": round(max(float(p[1]), 1.0 - float(p[1])), 4), "action": ext}


def postprocess(ticket: "Ticket") -> list[dict]:
    """A finished ticket's rows -> one system_one-shaped result per state, in state order.
    A state any of whose rows failed gets {"error": ...} instead, the others are unaffected."""
    meta, agent = ticket.meta, ticket.agent
    answers: list[dict] = [dict() for _ in range(meta["n_states"])]
    tokens = [0] * meta["n_states"]
    failed: dict[int, str] = {}
    for i, (it, z, a) in enumerate(zip(ticket.items, ticket.logits, ticket.act)):
        if i in ticket.errors:
            failed.setdefault(it["s"], "%s: %s" % (type(ticket.errors[i]).__name__, ticket.errors[i]))
            continue
        answers[it["s"]][it["q"]] = _answer(agent, meta["internal"][it["q"]], z, a, len(it["markers"]))
        tokens[it["s"]] += len(it["ids"])
    return [{"error": failed[s]} if s in failed else
            {"model": "laya-rl-agent", "answers": {q: answers[s][q] for q in meta["qids"]},
             "usage": {"input_tokens": tokens[s], "output_tokens": 0}} for s in range(meta["n_states"])]


class FatalGPUError(RuntimeError):
    """The CUDA context is unusable; only a process restart recovers."""


def _cuda_usable(device) -> bool:
    if device.type != "cuda":
        return True
    try:
        torch.ones(1, device=device).add_(1).item()
        torch.cuda.synchronize(device)
        return True
    except Exception:  # noqa: BLE001
        return False


@torch.no_grad()
def _forward_once(agent, items: list[dict]):
    b = collate_items([items], agent.tok.pad_token_id)
    with torch.autocast(device_type=agent.device.type, dtype=agent.dtype, enabled=agent.device.type == "cuda"):
        logits, act = agent.model(b["input_ids"].to(agent.device), b["attention_mask"].to(agent.device),
                                  b["marker_pos"].to(agent.device), b["marker_mask"].to(agent.device),
                                  b["qtype"].to(agent.device))
    return logits.float().cpu().numpy(), torch.softmax(act.float(), -1).cpu().numpy()


def _forward(agent, items: list[dict], forward_fn=None) -> tuple[list, int]:
    """Run rows, isolating failures. Returns ([(logits_row, act_row) | Exception per row], passes).

    Any failure (including CUDA OOM) is bisected until the failing rows are
    isolated: only they get an Exception, every other row still gets results.
    This replaces the SDK's silent move of the whole model to CPU on OOM.
    Raises FatalGPUError if the CUDA context itself is broken."""
    run = forward_fn or _forward_once
    try:
        logits, act = run(agent, items)
        return [(logits[r], act[r]) for r in range(len(items))], 1
    except Exception as e:  # noqa: BLE001
        if isinstance(e, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
        elif not _cuda_usable(agent.device):
            raise FatalGPUError("CUDA context unusable after %s: %s" % (type(e).__name__, e)) from e
        if len(items) == 1:
            return [e], 1
        half = len(items) // 2
        a, na = _forward(agent, items[:half], forward_fn)
        b, nb = _forward(agent, items[half:], forward_fn)
        return a + b, 1 + na + nb


# --------------------------------------------------------------------------- tickets and the scheduler
class Busy(Exception):
    """The queue is full; the caller should retry later (HTTP 503)."""


class Ticket:
    __slots__ = ("agent", "items", "meta", "future", "next", "done", "logits", "act", "errors", "dead",
                 "t_submit", "t_start", "t_done", "passes")

    def __init__(self, agent, items: list[dict], meta: dict):
        self.agent, self.items, self.meta = agent, items, meta
        self.future: concurrent.futures.Future = concurrent.futures.Future()
        self.next = 0                       # rows handed to the GPU so far
        self.done = 0                       # rows with results back
        self.logits: list = [None] * len(items)
        self.act: list = [None] * len(items)
        self.errors: dict[int, BaseException] = {}   # row index -> why that row failed
        self.dead = False                   # failed or abandoned: skip its remaining rows
        self.t_submit, self.t_start, self.t_done = time.perf_counter(), None, None
        self.passes = 0                     # forward passes this ticket took part in

    @property
    def remaining(self) -> int:
        return len(self.items) - self.next

    def resolve(self):
        self.t_done = time.perf_counter()
        try:
            self.future.set_result(self)
        except concurrent.futures.InvalidStateError:  # the waiter gave up (timeout)
            self.dead = True

    def fail(self, exc: BaseException):
        self.dead = True
        try:
            self.future.set_exception(exc)
        except concurrent.futures.InvalidStateError:
            pass


class Scheduler:
    """Owns the GPU: one worker thread runs every forward pass, combining queued tickets."""

    def __init__(self, token_budget: int = 8192, max_seqs: int = 256, queue_max: int = 50000,
                 oldest_share: float = 0.25, round_factor: int = 4, forward_fn=None, exit_on_fatal: bool = True):
        self.token_budget, self.max_seqs, self.round_factor = token_budget, max_seqs, round_factor
        self.queue_max, self.oldest_share = queue_max, oldest_share
        self.pending: collections.deque[Ticket] = collections.deque()   # FIFO by submission
        self.cond = threading.Condition()
        self.stopping = False
        self.thread = threading.Thread(target=self._run, name="gpu-worker", daemon=True)
        self.stats = {"passes": 0, "rows": 0, "tickets": 0, "max_tickets_per_pass": 0, "row_errors": 0,
                      "tokens": 0, "padded_tokens": 0}
        self.forward_fn = forward_fn          # test hook: replaces the model call
        self.exit_on_fatal = exit_on_fatal

    # ---- lifecycle
    def start(self):
        self.thread.start()

    def stop(self, timeout: float = 10):
        with self.cond:
            self.stopping = True
            self.cond.notify_all()
        self.thread.join(timeout)

    @property
    def alive(self) -> bool:
        return self.thread.is_alive()

    # ---- request side
    def submit(self, agent, items: list[dict], meta: dict) -> Ticket:
        t = Ticket(agent, items, meta)
        if not items:
            t.resolve()
            return t
        with self.cond:  # held only for list operations; the worker never holds it during GPU work
            if self.stopping or (self.thread.ident is not None and not self.alive):  # stopped or died
                raise RuntimeError("GPU worker is not running")
            queued = sum(x.remaining for x in self.pending if not x.dead)
            if queued + len(items) > self.queue_max:
                raise Busy("%d sequences queued (QUEUE_MAX_SEQUENCES=%d)" % (queued, self.queue_max))
            self.pending.append(t)
            self.cond.notify()
        return t

    def queued(self) -> int:
        with self.cond:
            return sum(x.remaining for x in self.pending if not x.dead)

    # ---- worker side
    def _take_round(self) -> tuple[Any, list[list[tuple[Ticket, int]]]]:
        """Pick one checkpoint's rows for the next round and pack them into length-sorted
        passes. Caller holds the lock. Returns (agent, [pass, ...]), pass = [(ticket, row)]."""
        live = [t for t in self.pending if not t.dead and not t.future.cancelled() and t.remaining > 0]
        self.pending = collections.deque(live)
        if not live:
            return None, []
        oldest = live[0]
        agent = oldest.agent
        same = [t for t in live if t.agent is agent]
        round_budget = self.token_budget * self.round_factor
        picked: list[tuple[Ticket, int]] = []
        used = {"tokens": 0}

        def add(t: Ticket, cap: Optional[int]) -> bool:
            """Take t's next rows (real tokens, not padded); False once the round is full."""
            mine = 0
            while t.remaining > 0:
                n = len(t.items[t.next]["ids"])
                if picked and used["tokens"] + n > round_budget:
                    return False
                if cap is not None and mine and mine + n > cap:
                    return True
                picked.append((t, t.next))
                t.next += 1
                used["tokens"] += n
                mine += n
            return True

        if add(oldest, int(round_budget * self.oldest_share)):               # progress guarantee
            for t in sorted(same, key=lambda x: x.remaining):                 # then smallest first
                if not add(t, None):
                    break
        self.pending = collections.deque(t for t in self.pending if t.remaining > 0)

        picked.sort(key=lambda ti: len(ti[0].items[ti[1]]["ids"]))           # similar lengths share a pass
        passes, cur, cur_max = [], [], 0
        for ti in picked:
            n = len(ti[0].items[ti[1]]["ids"])
            if cur and (len(cur) + 1 > self.max_seqs or max(cur_max, n) * (len(cur) + 1) > self.token_budget):
                passes.append(cur)
                cur, cur_max = [], 0
            cur.append(ti)
            cur_max = max(cur_max, n)
        if cur:
            passes.append(cur)
        return agent, passes

    def _run(self):
        while True:
            with self.cond:
                while not self.stopping and not any(not t.dead and t.remaining > 0 for t in self.pending):
                    self.cond.wait()
                if self.stopping:
                    for t in self.pending:
                        t.fail(RuntimeError("server shutting down"))
                    return
                agent, round_passes = self._take_round()
            for plan in round_passes:
                if self._run_pass(agent, plan):
                    return                                        # fatal: worker stops

    def _run_pass(self, agent, plan: list[tuple[Ticket, int]]) -> bool:
        """Run one forward pass and hand each row back to its ticket. True = fatal, stop."""
        tickets = list(dict.fromkeys(t for t, _ in plan))   # ordered, unique
        now = time.perf_counter()
        for t in tickets:
            if t.t_start is None:
                t.t_start = now
            t.passes += 1
        try:
            rows, n = _forward(agent, [t.items[i] for t, i in plan], self.forward_fn)
        except FatalGPUError as e:
            log.critical("%s - failing all queued requests and exiting for a clean restart", e)
            with self.cond:
                self.stopping = True
                doomed = list(dict.fromkeys(tickets + list(self.pending)))
            for t in doomed:
                t.fail(e)
            if self.exit_on_fatal:
                logging.shutdown()
                os._exit(3)   # Docker's restart policy brings the API back with a fresh CUDA context
            return True
        except Exception as e:  # noqa: BLE001 - a bug here must not kill the worker
            log.exception("scheduler error for %d tickets", len(tickets))
            for t in tickets:
                t.fail(e)
            return False
        self.stats["passes"] += n
        self.stats["rows"] += len(plan)
        lens = [len(t.items[i]["ids"]) for t, i in plan]
        self.stats["tokens"] += sum(lens)
        self.stats["padded_tokens"] += max(lens) * len(lens)
        self.stats["tickets"] += len(tickets)
        self.stats["max_tickets_per_pass"] = max(self.stats["max_tickets_per_pass"], len(tickets))
        for (t, i), row in zip(plan, rows):
            if t.dead:
                continue
            if isinstance(row, BaseException):
                t.errors[i] = row
                self.stats["row_errors"] += 1
                log.warning("row %d of a ticket failed: %s: %s", i, type(row).__name__, row)
            else:
                t.logits[i], t.act[i] = row
            t.done += 1
            if t.done == len(t.items):
                t.resolve()
        return False
