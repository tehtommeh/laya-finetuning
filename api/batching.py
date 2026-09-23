"""GPU batching and request coalescing for laya Agents.

Every inference request - a single /v1/decide, one checkpoint of /v1/compare,
or a group of /v1/decide/batch - becomes a Ticket: its (state, question)
sequences (identical to the SDK's `build_sequence`, with each question's part
cached), plus a Future for the reply. One worker thread owns the GPU. Each round it takes queued rows for one
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
make test-coalescing and make test-batch. docs/PERFORMANCE.md has the
measurements behind each design choice here.
"""
from __future__ import annotations

import collections
import concurrent.futures
import json
import logging
import os
import threading
import time
from typing import Any, Optional

import numpy as np
import torch

from laya.common import (QTYPES, build_sequence, confidence_from_probs, render_options, serialize_state,
                         temp_bucket)

log = logging.getLogger("api.batching")


# --------------------------------------------------------------------------- encoding / post-processing
HEAD_CACHE_MAX = 4096   # cached question encodings per checkpoint


def _question_head(agent, iq: dict) -> tuple[list[int], list[int]]:
    """The part of a sequence that depends only on the question: [CLS] type+instructions
    [SEP] [MASK] opt0 [MASK] opt1 ... [SEP]. Cached per checkpoint, since callers repeat the
    same questions. Built by laya's own build_sequence with an empty state, minus the final
    [SEP], so a full sequence is head + state tokens + [SEP] exactly as build_sequence makes it
    (checked by scripts/test_scheduler.py). Raises ValueError if the options do not fit."""
    cache = agent.__dict__.setdefault("_head_cache", {})
    key = json.dumps(iq, sort_keys=True, ensure_ascii=False)
    hit = cache.get(key)
    if hit is None:
        max_len, head_max_len = agent.cfg.get("max_len", 512), agent.cfg.get("head_max_len", 192)
        ids, markers = build_sequence(agent.tok, "", iq, max_len, head_max_len)
        if len(markers) != len(render_options(iq)):
            raise ValueError("options exceed head_max_len=%d" % head_max_len)
        if len(cache) >= HEAD_CACHE_MAX:
            cache.clear()
        hit = cache[key] = (ids[:-1], markers)
    return hit


def encode(agent, states: list, questions: dict) -> tuple[list[dict], dict]:
    """(state, question) sequences for many states, sorted by length (less padding when a
    ticket is split across passes). Identical to laya's build_sequence per pair, but each
    question's part is cached and each state is tokenised once rather than once per question
    (720 -> 57 us for a 3-question request). Raises ValueError, like system_one, if a
    question's options do not fit the checkpoint's head_max_len."""
    max_len = agent.cfg.get("max_len", 512)
    tok = agent.tok
    qids = list(questions)
    internal = {qid: agent._to_internal(questions[qid]) for qid in qids}
    heads = {}
    for qid in qids:
        try:
            heads[qid] = _question_head(agent, internal[qid])
        except ValueError as e:
            raise ValueError("question %r: %s" % (qid, e)) from None
    items = []
    for si, state in enumerate(states):
        st = tok(serialize_state(state).replace(tok.mask_token, " "), add_special_tokens=False)["input_ids"]
        for qid in qids:
            head, markers = heads[qid]
            room = max(0, max_len - len(head) - 1)
            items.append({"ids": (head + st[:room] + [tok.sep_token_id])[:max_len], "markers": markers,
                          "qtype": QTYPES[internal[qid]["t"]], "s": si, "q": qid})
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


def collate(items: list[dict], pad_id: int):
    """The tensors laya.common.collate_items builds, via numpy instead of one torch.tensor per
    row (2.4 ms -> 0.17 ms for a 60-row pass; identical values, checked in test_scheduler.py)."""
    n, width = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = np.full((n, width), pad_id, dtype=np.int64)
    att = np.zeros((n, width), dtype=np.int64)
    mpos = np.zeros((n, kmax), dtype=np.int64)
    mmask = np.zeros((n, kmax), dtype=bool)
    for i, it in enumerate(items):
        length, k = len(it["ids"]), len(it["markers"])
        ids[i, :length] = it["ids"]
        att[i, :length] = 1
        mpos[i, :k] = it["markers"]
        mmask[i, :k] = True
    qtype = np.fromiter((it["qtype"] for it in items), dtype=np.int64, count=n)
    return tuple(torch.from_numpy(x) for x in (ids, att, mpos, mmask, qtype))


def precast_weights(model, dtype) -> int:
    """Store matmul weights (Linear, MultiheadAttention in-projections) in the autocast dtype.

    Under autocast those weights are converted from fp32 on *every* forward pass - ~400
    extra kernel launches and 35% of a small pass's GPU time. Converting once gives the same
    values autocast would (outputs are bit-identical), and halves those weights' memory.
    Norms and embeddings keep their dtype, as autocast would run them. Returns bytes saved."""
    saved = 0

    def cast(t):
        nonlocal saved
        if t is not None and t.dtype == torch.float32:
            saved += t.numel() * (t.element_size() - torch.empty((), dtype=dtype).element_size())
            t.data = t.data.to(dtype)
    for mod in model.modules():
        if isinstance(mod, torch.nn.Linear):
            cast(mod.weight)
            cast(mod.bias)
        elif isinstance(mod, torch.nn.MultiheadAttention):
            cast(mod.in_proj_weight)
            cast(mod.in_proj_bias)
    return saved


# --------------------------------------------------------------------------- CUDA graphs
GRAPH_ROWS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
GRAPH_LENS = (32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024)
GRAPH_MAX_OPTIONS = 16      # passes with more options per question run eagerly
_GRAPH_POOL = None          # one memory pool for every checkpoint's graphs (see GraphRunner)
_WARMUP_STREAM = None       # one side stream for all capture warm-ups: cuBLAS keeps a workspace per stream


def _warmup_stream(device):
    global _WARMUP_STREAM
    if _WARMUP_STREAM is None:
        _WARMUP_STREAM = torch.cuda.Stream(device)
    return _WARMUP_STREAM


def _shared_graph_pool():
    global _GRAPH_POOL
    if _GRAPH_POOL is None:
        _GRAPH_POOL = torch.cuda.graph_pool_handle()
    return _GRAPH_POOL


class GraphRunner:
    """CUDA graphs for small forward passes of one checkpoint.

    A small pass is dispatch-bound: ~950 kernel launches from Python cost ~20 ms whatever
    the batch size, while the GPU work for a few rows takes a few ms. A CUDA graph records
    the launches once and replays them with one call (1-request pass: 20.8 -> 6.7 ms).

    Graphs need fixed shapes, so passes are padded to the smallest captured bucket of
    (rows, length); extra rows are dummies whose outputs are dropped. Padding costs GPU work:
    once a pass is big enough to be GPU-bound, padding it up to a bucket is slower than
    running it eagerly (measured: 5 questions on a 250-token state went 24 -> 45 ms with a
    fixed 2,048-token cap). So each pass is routed by measured cost: at capture time every
    bucket's replay is timed, and the eager path's fixed dispatch floor and per-token rate
    are measured; a pass uses a graph only if its bucket's time beats the eager estimate for
    its real (unpadded) shape. This calibrates itself to the GPU and the checkpoint.

    Replaying at a bucket shape matches the eager model on the same padded tensors: bit for
    bit in ~99.9% of cases, and otherwise by one or two bf16 rounding steps, because capture
    makes the encoder's GPU libraries pick slightly different kernels (deterministic per input;
    scripts/test_cuda_graphs.py). Padding to a bucket rather than to the longest row shifts
    bf16 results within the noise any change of batch shape causes (docs/PERFORMANCE.md).

    Only buckets that can beat eager are captured: the eager floor and per-token rate are
    measured first, and a bucket is captured if its GPU work (tokens x rate) is below the
    floor, i.e. while the pass is still dispatch-bound. `max_tokens` is a hard cap on top.

    All graphs, of every checkpoint, share one memory pool (per-checkpoint pools cost 1.9 GB
    for four checkpoints). That is safe here because replays are serialised on the single GPU
    worker thread and every output is copied to the CPU immediately, before any other graph
    runs.
    """

    def __init__(self, agent, max_tokens: int = 2048):
        self.agent = agent
        self.max_tokens = max_tokens
        self.max_len = agent.cfg.get("max_len", 512)
        self.pool = _shared_graph_pool()
        self.graphs: dict[tuple[int, int], tuple] = {}
        self.lens = [length for length in GRAPH_LENS if length <= self.max_len]
        self.replays = 0
        self.eager_passes = 0
        self.graph_ms: dict[tuple[int, int], float] = {}   # measured replay time per bucket
        self.eager_floor_ms = 0.0                           # eager pass cost that does not scale with size
        self.eager_ms_per_token = 0.0                        # eager cost per padded token once GPU-bound
        self.capture_limit_tokens = 0

    @torch.no_grad()
    def _fwd(self, inputs):
        with torch.autocast(device_type="cuda", dtype=self.agent.dtype, cache_enabled=False):
            logits, act = self.agent.model(*inputs)
        return logits.float(), torch.softmax(act.float(), -1)

    def _dummy_inputs(self, rows: int, length: int) -> list[np.ndarray]:
        """Host-side buffers of a bucket shape, pre-filled as valid dummy rows: one [CLS]
        token attended, one option marker. Real rows overwrite their slice."""
        tok = self.agent.tok
        ids = np.full((rows, length), tok.pad_token_id, dtype=np.int64)
        ids[:, 0] = tok.cls_token_id
        att = np.zeros((rows, length), dtype=np.int64)
        att[:, 0] = 1
        mpos = np.zeros((rows, GRAPH_MAX_OPTIONS), dtype=np.int64)
        mmask = np.zeros((rows, GRAPH_MAX_OPTIONS), dtype=bool)
        mmask[:, 0] = True
        return [ids, att, mpos, mmask, np.zeros(rows, dtype=np.int64)]

    def capture(self) -> int:
        """Measure eager, then capture every bucket that can beat it. Returns the number captured."""
        dev = self.agent.device
        self._calibrate_eager()
        worthwhile = self.eager_floor_ms / max(self.eager_ms_per_token, 1e-9)   # tokens of GPU work = one dispatch floor
        self.capture_limit_tokens = int(min(self.max_tokens, worthwhile))
        for length in self.lens:
            for rows in GRAPH_ROWS:
                if rows * length > self.capture_limit_tokens and not (rows == 1 and length == self.lens[0]):
                    continue
                static = [torch.from_numpy(a).to(dev) for a in self._dummy_inputs(rows, length)]
                side = _warmup_stream(dev)
                side.wait_stream(torch.cuda.current_stream(dev))
                with torch.cuda.stream(side):                       # warm up off the capture stream
                    for _ in range(3):
                        self._fwd(static)
                torch.cuda.current_stream(dev).wait_stream(side)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=self.pool):
                    out = self._fwd(static)
                self.graphs[(rows, length)] = (graph, static, out)
        torch.cuda.synchronize(dev)
        for key, (graph, _, _) in self.graphs.items():
            self.graph_ms[key] = self._time_ms(graph.replay)
        return len(self.graphs)

    @staticmethod
    def _time_ms(fn, reps: int = 7) -> float:
        fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(reps):
            t = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t) * 1000)
        return sorted(times)[len(times) // 2]

    def _calibrate_eager(self):
        """The eager path's floor (a tiny pass) and per-token rate (a large, GPU-bound pass)."""
        dev = self.agent.device
        tiny = [torch.from_numpy(a).to(dev) for a in self._dummy_inputs(1, self.lens[0])]
        big_rows, big_len = 32, min(self.max_len, 512)
        big = [torch.from_numpy(a).to(dev) for a in self._dummy_inputs(big_rows, big_len)]
        big[1].fill_(1)   # attend to every position, like a full batch
        self.eager_floor_ms = self._time_ms(lambda: self._fwd(tiny))
        self.eager_ms_per_token = self._time_ms(lambda: self._fwd(big), reps=3) / (big_rows * big_len)

    def eager_estimate_ms(self, n_rows: int, width: int) -> float:
        return max(self.eager_floor_ms, n_rows * width * self.eager_ms_per_token)

    def bucket(self, n_rows: int, width: int, kmax: int) -> Optional[tuple[int, int]]:
        """The fastest captured bucket that fits, or None if none fits."""
        if kmax > GRAPH_MAX_OPTIONS:
            return None
        fits = [k for k in self.graphs if k[0] >= n_rows and k[1] >= width]
        if not fits:
            return None
        return min(fits, key=lambda k: self.graph_ms.get(k, float(k[0] * k[1])))

    def choose(self, n_rows: int, width: int, kmax: int) -> Optional[tuple[int, int]]:
        """Bucket to replay for this pass, or None to run it eagerly (cheaper by measurement)."""
        key = self.bucket(n_rows, width, kmax)
        if key is None or self.graph_ms.get(key, 0.0) >= self.eager_estimate_ms(n_rows, width):
            return None
        return key

    def pad_to(self, items: list[dict], key: tuple[int, int]) -> list[np.ndarray]:
        rows, length = key
        buf = self._dummy_inputs(rows, length)
        ids, att, mpos, mmask, qtype = buf
        for i, it in enumerate(items):
            n, k = len(it["ids"]), len(it["markers"])
            ids[i, :] = self.agent.tok.pad_token_id
            ids[i, :n] = it["ids"]
            att[i, :] = 0
            att[i, :n] = 1
            mpos[i, :] = 0
            mpos[i, :k] = it["markers"]
            mmask[i, :] = False
            mmask[i, :k] = True
            qtype[i] = it["qtype"]
        return buf

    def run(self, items: list[dict]):
        """(logits, act) as numpy for these rows, or None if no captured bucket fits."""
        width = max(len(it["ids"]) for it in items)
        kmax = max(len(it["markers"]) for it in items)
        key = self.choose(len(items), width, kmax)
        if key is None:
            self.eager_passes += 1
            return None
        return self.run_bucket(items, key)

    def run_bucket(self, items: list[dict], key: tuple[int, int]):
        """Replay the graph of bucket `key` for these rows (they must fit it)."""
        graph, static, (out_logits, out_act) = self.graphs[key]
        for dst, src in zip(static, self.pad_to(items, key)):
            dst.copy_(torch.from_numpy(src))
        graph.replay()
        self.replays += 1
        n = len(items)
        return out_logits[:n].cpu().numpy(), out_act[:n].cpu().numpy()   # copied out before any other replay


@torch.no_grad()
def _forward_once(agent, items: list[dict]):
    runner = getattr(agent, "_graphs", None)
    if runner is not None:
        out = runner.run(items)
        if out is not None:
            return out
    tensors = [t.to(agent.device) for t in collate(items, agent.tok.pad_token_id)]
    with torch.autocast(device_type=agent.device.type, dtype=agent.dtype, enabled=agent.device.type == "cuda"):
        logits, act = agent.model(*tensors)
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
