# Performance: what was optimised, and how much each step gained

Every optimisation made to the serving path, in the order it was done, with the measurement that motivated it
and the result. All numbers are from one RTX 3090 (24 GB, driver 580) with the English base, 3 questions per
request, unless stated otherwise. "Short" = one-sentence tickets (~110 tokens per state across 3 questions);
"long" = typed-decisions states (~500 tokens per state). Reproduce with `make load-test` (N concurrent clients
sending `/v1/decide`) and `make test-batch`. The micro-benchmarks behind stage 4 and the rejected ideas are in
[`scripts/bench/`](../scripts/bench) (`bash scripts/bench/run.sh <script> [args]`): `cpu.py` (T1–T6),
`pass_anatomy.py`, `graph_potential.py` and `replicas.py`. Stage 5 is verified by `scripts/test_cuda_graphs.py`.

## Summary

| stage | short, 128 clients | long, 128 clients | single request | batch of 400 short |
|---|---|---|---|---|
| 0. one request at a time behind a GPU lock | 39 req/s (p50 3.2 s) | 34 req/s | 25 ms | ~39 states/s (loop) |
| 1. true GPU batching for `/v1/decide/batch` | – | – | – | ~330 states/s |
| 2. request coalescing (one GPU worker, shared passes) | 212 req/s | 36 req/s | 26 ms | – |
| 3. length-sorted rounds (padding 43% → 13%) | 207 req/s | 47 req/s | 26 ms | – |
| 4. four CPU-side fixes (below) | 308 req/s (p50 0.4 s) | 55 req/s | 22 ms | 425 states/s |
| 5. CUDA graphs for small passes | **314 req/s** | **53 req/s** | **6.7 ms** | 425 states/s |
| **total gain** | **8.0×** | **1.6×** | **3.7× faster** | **~11×** |

Stage 5 matters most at low to moderate traffic: 1 client 44 → 148 req/s, 4 clients 87 → 194, 16 clients
206 → 265 (short messages).

GPU memory for the four resident checkpoints went from 6.4 GB to 4.0 GB allocated in stage 4. Stage 5 adds
~1.1 GB of device memory for the graphs.

Every stage keeps the answers the SDK would give. Where outputs are claimed identical, a test compares them to
laya's own functions (see [Correctness](#correctness)).

## Baseline decisions (initial build)

- **All checkpoints preloaded and warmed up at startup.** With the SDK's default (`max_loaded=1`), traffic that
  alternates languages rebuilds a model on every switch (7–10 s per the model card). A warm-up call per checkpoint
  moves CUDA kernel initialisation out of the first user request.
- **Inference off the event loop,** so `/health` stays responsive while the GPU is busy.
- **One API process.** Each extra process would load its own copy of every checkpoint onto the same GPU and
  compete for it. This never changed, and no stage needs more processes.

## Stage 1: true GPU batching for `/v1/decide/batch`

**Motivation.** The first batch endpoint looped `system_one` per state: only 1.07–1.10× faster than N separate
calls (it saved HTTP round trips, nothing else).

**Change.** Every (state, question) pair is encoded with laya's own `build_sequence`, sorted by length, and run in
chunks of at most `BATCH_TOKEN_BUDGET` padded tokens. Results are split back per state with laya's post-processing,
reproduced line for line.

**Result.**

| input | N = 10 | N = 100 | N = 400+ |
|---|---|---|---|
| short | 5.9× | 8.1× | 8.7× (~3 ms/state) |
| long | 1.2× | 1.6× | ~1.75× |

Long states gain less because one long state already nearly fills the GPU, which saturates at ~44k tokens/s.

**Token budget sweep** (4k, 8k, 16k, 64k): throughput was flat above 8k, 64k used 4 GB more VRAM, and one big
padded pass lost to several length-sorted smaller ones for small batches. The default is 8,192.

**Bug found on the way:** compose did not pass `BATCH_TOKEN_BUDGET` into the container, so the setting silently
had no effect. It is now passed through, and checked with `printenv` inside the container.

## Stage 2: request coalescing

**Motivation.** Parallel `/v1/decide` calls took turns at the GPU lock: throughput was flat at ~39 req/s at any
concurrency, while latency grew linearly (128 clients: 3.2 s each).

**Change** ([`api/batching.py`](../api/batching.py)). Every request becomes a ticket (its encoded rows plus a
future). One worker thread owns the GPU and combines whatever is queued into shared forward passes:

- opportunistic, with no fixed wait: an idle GPU runs a lone request at once;
- fair: the oldest ticket gets at least 25% of every round, then the tickets with the fewest remaining rows first;
- failure-contained: a failing pass is bisected until the failing rows are isolated.

`/v1/decide/batch` and `/v1/compare` submit tickets to the same queue.

**Result.** Short: 39 → 212 req/s at 128 clients (5.4×), p50 3.2 s → 0.57 s. Long: 34 → 36 req/s (+6%),
which was suspiciously low.

## Stage 3: length-sorted rounds

**Motivation.** Scheduler counters showed **43% of all GPU work was padding** under mixed long traffic. A pass
held rows from many requests (100–630 tokens), each padded to the longest.

**Change.** The worker takes a *round* of rows (up to 4 × the token budget, with the same fairness rules), sorts it
by length, and packs it into passes of similar lengths.

**Result.** Padding fell to 13%. Long: 39 → 50 req/s at 64 clients (+28%). The 1,000-request mixed leak-test
workload went from 45 s to 30 s.

## Stage 4: CPU-side fixes

### How the bottleneck was found

At high concurrency with short messages the GPU flickered between 0% and 100% utilisation, so it was waiting on
the CPU. Per-thread CPU accounting (`/proc/1/task/*/stat` during a load run) located the cost:

| thread | CPU per request (64 clients) |
|---|---|
| GPU worker | 4.0 ms (83% of a core) |
| event loop (HTTP, JSON, validation) | 0.8 ms |
| thread pool (tokenising, post-processing) | ~1.0 ms |

Six CPU micro-benchmarks, run in parallel as independent single-threaded processes, then priced each piece:

| test | measured | candidate | result |
|---|---|---|---|
| T1 build a pass's tensors (`collate_items`) | 2.4 ms per 60-row pass | numpy | 0.17 ms, identical |
| T2 encode one 3-question request | 720 µs | cached question part, state tokenised once | 57 µs, identical |
| T3 response serialisation (FastAPI `jsonable_encoder`) | 94 µs | orjson | 1 µs |
| T4 post-processing | 62 µs | – | fine |
| T5 routing (language detection) | 21 µs short, 444 µs long | – | fine |
| T6 thread-pool hop | ~80 µs | skip for small requests | – |

A GPU pass-anatomy test (CUDA events around each step) showed the forward pass itself is **dispatch-bound when
small**: ~23 ms of CPU to issue it whatever its size, with GPU time equal to that up to ~15 rows.

| rows per pass | CPU dispatch | GPU time |
|---|---|---|
| 3 | 22.9 ms | 22.9 ms |
| 15 | 22.8 ms | 22.8 ms |
| 60 | 23.4 ms | 59.9 ms |
| 120 | 23.4 ms | 108.0 ms |

A torch profile of one pass found 1,366 kernel launches. About 400 of them were **autocast re-converting every
fp32 weight matrix to bf16 on every pass**, which was also 35% of a small pass's GPU time.

### The four fixes

1. **Matmul weights stored in bf16 at load time** (`precast_weights`): the same values autocast would produce, so
   outputs are bit-identical. This removed ~400 launches per pass and 0.74 GB of VRAM per ModernBERT-large
   checkpoint (2.4 GB across four).
2. **Cached question encoding** (`encode`): the question-only part of each sequence is cached per checkpoint, and
   each state is tokenised once rather than once per question. Identical to `build_sequence`, 12× cheaper. Small
   requests are encoded inline on the event loop, saving a thread-pool hop.
3. **Numpy tensor building** (`collate`) in the GPU worker: identical to `collate_items`, 14× cheaper.
4. **orjson responses** for `/v1/decide`, `/v1/compare` and `/v1/decide/batch`, skipping FastAPI's generic encoder.

**Result.**

| | before | after |
|---|---|---|
| single request (p50) | 25.1 ms | 22.0 ms |
| short, 16 / 64 / 128 clients | 155 / 200 / 207 req/s | 206 / 279 / 308 req/s |
| long, 64 / 128 clients | 48 / 47 req/s | 55 / 55 req/s |
| batch of 400 short / long | 330 / 57 states/s | 425 / 61 states/s |
| GPU memory, 4 checkpoints | 6.38 GB | 3.97 GB |

Short requests gained most (+49% at 128 clients): cheaper encoding shrinks the gap in which the GPU waits for the
next round to be prepared.

## Measured and rejected

- **Bigger token budgets** (16k–64k): same throughput, more VRAM, and worse for small batches.
- **Firing parallel `/v1/decide` calls** before coalescing: no gain, because they queued on the lock.
- **More API processes:** duplicate model copies competing for one GPU. The limit is the GPU worker, not HTTP.
- **Loading the model more than once (replicas).** Do small, dispatch-bound passes leave GPU cycles a second copy
  could use? `replicas.py` runs R copies of the English model, each looping forward passes of a fixed size, as
  threads in one process or as separate processes. Total throughput relative to one copy:

  | pass size | 2 threads | 2 processes | 3 processes | 4 processes |
  |---|---|---|---|---|
  | 3 rows (1 request) | 1.29× | **2.02×** | **2.45×** | 2.51× |
  | 15 rows (5 requests) | 1.06× | 1.07× | 1.08× | 1.08× |
  | 60 rows | 1.01× | 0.93× | 0.93× | 0.93× |
  | 120 rows | 1.00× | 0.92× | 0.92× | 0.92× |

  Idle cycles exist, but only in one-request passes. Separate processes fill them (up to ~2.5×); threads barely
  can, because only the GIL holder dispatches kernels. From ~5 requests per pass the GPU is already busy, and larger
  passes lose ~8% to time-slicing between processes. Rejected for this server:
  - coalescing already uses those cycles more cheaply: a 5-request pass costs about the same as a 1-request pass,
    so concurrent requests share a pass instead of each getting its own;
  - under load, passes are large, and replicas would lose throughput;
  - replicas cannot cut a lone request's latency, because it is still one ~22 ms pass;
  - they cost ~1 GB of VRAM per extra copy of each checkpoint, plus a multi-process design.

  CUDA graphs (stage 5) attack the same idle time by removing dispatch overhead, and do reduce latency. NVIDIA
  MPS (concurrent kernels across processes) was not tested.
- **Overlapping pass dispatch (pipelining):** large passes already overlap dispatch with GPU execution inside the
  pass. The measured gap between passes was the CPU preparation that stage 4 removed.
- **Optimising routing, validation or post-processing:** tens of microseconds each (T3–T5).

## Stage 5: CUDA graphs for small passes

**Motivation.** Stage 4 left a lone request at ~22 ms, almost all of it Python issuing the ~950 kernel launches
of a forward pass; the GPU work for a few rows takes a few ms. `graph_potential.py` measured the potential at a
fixed shape, with bit-identical outputs:

| rows per pass | eager | CUDA graph |
|---|---|---|
| 3 (1 request) | 20.8 ms | 6.7 ms |
| 15 (5 requests) | 20.8 ms | 15.7 ms |
| 60 | 57.8 ms | 55.3 ms |
| 120 | 106.3 ms | 103.7 ms |

**What a CUDA graph is.** The GPU is fast, but each of the ~950 operations in a pass has to be issued from Python
one at a time. A graph records the whole sequence once and replays it with a single call. A recorded graph only
works for a fixed input shape, so passes are padded to a set of *buckets*: rows × length, e.g. 4 × 64.

**First version, and what went wrong.** Buckets of {1, 2, 4, 8, 16, 32} rows × {64 … 1024} tokens, capped at 2,048
padded tokens. Single short requests went from 22 to 8 ms. But several cases got slower: 5 questions on a
250-token state went 24 → 45 ms, and one long request 24 → 47 ms. **Padding waste:** once a pass is past a few
hundred tokens the GPU is the bottleneck, so padding it up to a bucket (e.g. 5 × 250 → 8 × 256) cost more GPU
time than the dispatch it saved.

**Final version** ([`api/batching.py`](../api/batching.py), `GraphRunner`):

- **Self-calibrating routing.** At startup, each checkpoint measures its eager dispatch floor (~20 ms English,
  ~16 ms multilingual) and per-token rate (~19 ms per 1k tokens English, ~8.6 multilingual), times every
  captured graph, and routes each pass to a graph only when that bucket's measured time beats the eager estimate
  for the pass's real shape. For example, 1 × 40 tokens → graph; 5 × 250 → eager on English but a graph on the
  cheaper multilingual model.
- **Finer buckets** (rows 1, 2, 3, 4, 6, 8, 12, 16, 24, 32 × lengths 32 … 1024), **captured only where they can
  win**: a bucket's GPU work must be under the eager dispatch floor (~1,000 tokens English, ~1,900
  multilingual). That is 46–65 graphs per checkpoint, ~10 s each to capture; startup goes from ~72 s to ~111 s.
- **Memory.** All checkpoints share one graph memory pool, and captures reuse one warm-up stream (cuBLAS keeps a
  scratch workspace per stream: 200+ warm-up streams had cost 0.7 GB). The graph overhead went from 1.9 GB to
  ~1.1 GB.
- `CUDA_GRAPHS=0` disables graphs (an operational kill switch for driver problems), and `GRAPH_MAX_TOKENS` caps
  the capture size.

**Result.**

| | stage 4 | stage 5 |
|---|---|---|
| single short request (p50) | 22.0 ms | **6.7 ms** |
| 3 / 5 / 10 questions on one ~500-token state | 24 / 24 / 35 ms | **15 / 24 / 36 ms** |
| short, 1 / 4 / 16 clients | 44 / 87 / 206 req/s | **148 / 194 / 265 req/s** |
| short, 64 / 128 clients | 279 / 308 req/s | 289 / 314 req/s |
| long, 16 / 64 / 128 clients (A/B, same session) | 47.7 / 55.0 / 54.7 | 47.4 / 54.1 / 53.8 (−1 to −2%) |

Big passes run eagerly as before; the small cost under heavy long-input load is within 2%.

**Correctness caveat.** Graph replay equals the eager model on the same padded tensors bit for bit in 500 of 501
checks. The exception differs by one bf16 rounding step (0.003 in probability): capture makes the encoder's GPU
libraries pick slightly different kernels. This was localised to the encoder, and ruled out as memory-pool
interference, streams or caching. Padding to a bucket rather than to the longest row moves results within the
usual bf16 batch-shape noise, so a solo request no longer matches `system_one` bit for bit. It stays within
0.041 and is deterministic from call to call (`make test-coalescing`).

## Next steps (not implemented)

- **More of each pass under a graph.** Graphs pay off only while dispatch dominates; heavy-load throughput is
  GPU-bound at ~45k tokens/s, and only more compute (or a smaller model: multilingual is ~2× faster per token)
  moves it.
- **Unpadded attention** (FlashAttention varlen, which ModernBERT supports) would remove padding work from big
  passes entirely. It needs the `flash-attn` package in the image.

## Correctness

Each stage was verified before its numbers were recorded:

- `make test-batch`: batched answers match per-state `/v1/decide`.
- `make test-coalescing`:
  - scheduler unit tests with a fake model that returns each row's own id. They cover isolation across 4,000
    tickets from 200 threads, grouping, bisection, OOM, the fatal path, cancellation, fairness and backpressure;
  - the fast paths checked against laya itself: `encode` == `build_sequence` on ~3,600 sequences from both
    tokenizers (cold and cached), `collate` == `collate_items`, and option overflow raises exactly where laya does;
  - a live leak test with 1,000 concurrent mixed requests, each with unique question ids and labels, checked
    against unmodified SDK agents. Keys, routing and token counts must match exactly, and each state must match its
    own reference better than any sibling's (60,236 comparisons). Requests re-sent one at a time must be
    deterministic and within tolerance of `system_one`. With `CUDA_GRAPHS=0` and `EXACT_SOLO=1` they must match
    exactly: before stage 5 they did (max difference 0.00000, which also confirmed the pre-cast weights are
    bit-identical);
  - `scripts/test_cuda_graphs.py`: for every captured bucket of three checkpoints, graph replay vs the eager model
    on the same padded tensors (after replaying other buckets, to test the shared pool); that routing picks a graph
    only when it is measured to be faster; and that passes fitting no bucket fall back to eager.

Under concurrency, values differ from `system_one` by a median of ~0.001 (max ~0.05) in probability. bf16 kernels
round differently in differently padded batches, and the SDK does the same on its own: one input, alone vs in a
padded batch, moved up to 0.13 for inputs whose probability is spread across neighbouring options.
