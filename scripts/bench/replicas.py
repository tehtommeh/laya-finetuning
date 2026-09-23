"""R model replicas on one GPU, as threads or processes: aggregate passes/s (docs/PERFORMANCE.md, measured and rejected).
    bash scripts/bench/run.sh replicas.py thread|process R ROWS SECONDS"""
import sys, time, os, threading, multiprocessing as mp
sys.path.insert(0, "/app"); sys.path.insert(0, "/tmp/bench")

def make(rows):
    import torch, laya, batching
    from cpu import SHORT, QS
    ag = laya.Agent("/models/convaiinnovations__laya", device="cuda")
    batching.precast_weights(ag.model, ag.dtype)
    items = []
    i = 0
    while len(items) < rows:
        items += batching.encode(ag, [SHORT[i % len(SHORT)]], QS)[0]; i += 1
    items = items[:rows]
    args = [t.cuda() for t in batching.collate(items, ag.tok.pad_token_id)]
    def step():
        with torch.no_grad(), torch.autocast("cuda", dtype=ag.dtype):
            l, a = ag.model(*args)
        l.float().cpu()          # sync, as the real worker does
    for _ in range(5): step()
    return step

def loop(step, seconds, out, start):
    start.wait()
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        step(); n += 1
    out.append((n, time.perf_counter() - t0))

def proc_main(rows, seconds, q, ready, go):
    try:
        step = make(rows)
    except Exception as e:  # report instead of leaving the parent waiting forever
        q.put(("error", repr(e))); ready.set(); return
    ready.set(); go.wait()
    out = []; loop(step, seconds, out, go)
    q.put(out[0])

if __name__ == "__main__":
    mode, R, rows, seconds = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
    if mode == "thread":
        steps = [make(rows) for _ in range(R)]
        out, go = [], threading.Event()
        th = [threading.Thread(target=loop, args=(s, seconds, out, go)) for s in steps]
        for t in th: t.start()
        go.set()
        for t in th: t.join()
    else:
        ctx = mp.get_context("spawn")
        q, go = ctx.Queue(), ctx.Event()
        readies = [ctx.Event() for _ in range(R)]
        ps = [ctx.Process(target=proc_main, args=(rows, seconds, q, readies[i], go)) for i in range(R)]
        for p in ps: p.start()
        for e in readies:
            if not e.wait(300):
                sys.exit("replica did not become ready within 300 s")
        go.set()
        out = [q.get(timeout=300) for _ in range(R)]
        errs = [o for o in out if o[0] == "error"]
        if errs:
            sys.exit("replica failed: %s" % errs[0][1])
        for p in ps: p.join()
    passes = sum(n for n, _ in out); wall = max(w for _, w in out)
    print("%-7s R=%d rows/pass=%3d  passes/s %6.1f  rows/s %7.0f  (per replica %.1f passes/s)" % (
        mode, R, rows, passes / wall, passes * rows / wall, passes / wall / R))
