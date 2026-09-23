# Performance: what was optimised, and how much each step gained

Every optimisation made to the serving path, in the order it was done, with the measurement that motivated it
and the result. All numbers are from one RTX 3090 (24 GB, driver 580) with the English base, 3 questions per
request, unless stated otherwise. "Short" = one-sentence tickets (~110 tokens per state across 3 questions);
"long" = typed-decisions states (~500 tokens per state). Reproduce with `make load-test` (N concurrent clients
sending `/v1/decide`) and `make test-batch`.

## Summary

| stage | short, 128 clients | long, 128 clients | single request | batch of 400 short |
|---|---|---|---|---|
| 0. one request at a time behind a GPU lock | 39 req/s (p50 3.2 s) | 34 req/s | 25 ms | ~39 states/s (loop) |
| 1. true GPU batching for `/v1/decide/batch` | – | – | – | ~330 states/s |
| 2. request coalescing (one GPU worker, shared passes) | 212 req/s | 36 req/s | 26 ms | – |
| 3. length-sorted rounds (padding 43% → 13%) | 207 req/s | 47 req/s | 26 ms | – |
| 4. four CPU-side fixes (below) | **308 req/s (p50 0.4 s)** | **55 req/s** | **22 ms** | **425 states/s** |
| **total gain** | **7.9×** | **1.6×** | **−12%** | **~11×** |

GPU memory for the four resident checkpoints went from 6.4 GB to 4.0 GB (stage 4).

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
- **Overlapping pass dispatch (pipelining):** large passes already overlap dispatch with GPU execution inside the
  pass. The measured gap between passes was the CPU preparation that stage 4 removed.
- **Optimising routing, validation or post-processing:** tens of microseconds each (T3–T5).

## Next step: CUDA graphs (measured, not implemented)

A CUDA graph records a pass's ~950 kernel launches once and replays them with one call. At a fixed shape, with
pre-cast weights and bit-identical outputs:

| rows per pass | eager | CUDA graph |
|---|---|---|
| 3 (1 request) | 20.8 ms | **6.7 ms** |
| 15 (5 requests) | 20.8 ms | 15.7 ms |
| 60 | 57.8 ms | 55.3 ms |
| 120 | 106.3 ms | 103.7 ms |

This would mainly cut latency at low to moderate traffic (a lone request from ~22 ms to roughly ~10 ms). Large
passes are GPU-bound either way. It needs shape buckets (padding rows and length to fixed sizes), capture at
startup, and some VRAM per graph, so it is its own piece of work.

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
    own reference better than any sibling's (60,236 comparisons). Requests re-sent one at a time must match
    `system_one` exactly (max difference 0.00000, which also confirms the pre-cast weights are bit-identical).

Under concurrency, values differ from `system_one` by a median of ~0.001 (max ~0.05) in probability. bf16 kernels
round differently in differently padded batches, and the SDK does the same on its own: one input, alone vs in a
padded batch, moved up to 0.13 for inputs whose probability is spread across neighbouring options.
