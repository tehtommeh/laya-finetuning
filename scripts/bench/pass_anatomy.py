"""GPU pass anatomy: tensor building, H2D, CPU dispatch, GPU time (CUDA events), sync (docs/PERFORMANCE.md, stage 4)."""
import sys, time, json, random, statistics
sys.path.insert(0, "/app"); sys.path.insert(0, "/tmp/bench")
import torch, laya, batching
from laya.common import collate_items
from cpu import SHORT, QS
ag = laya.Agent("/models/convaiinnovations__laya", device="cuda")
dev = ag.device
long_states = [json.loads(l)["state"] for l in open("/tmp/bench/test.jsonl")]
def anatomy(label, states, reps=40):
    items = [it for s in states for it in batching.encode(ag, [s], QS)[0]]
    items.sort(key=lambda it: len(it["ids"]))
    rows = {k: [] for k in ("collate", "h2d", "dispatch_cpu", "gpu", "d2h_sync", "total")}
    for r in range(reps + 5):
        t0 = time.perf_counter()
        b = collate_items([items], ag.tok.pad_token_id); t1 = time.perf_counter()
        args = [b[k].to(dev, non_blocking=False) for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")]
        torch.cuda.synchronize(); t2 = time.perf_counter()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        c0 = time.process_time(); e0.record()
        with torch.no_grad(), torch.autocast("cuda", dtype=ag.dtype):
            logits, act = ag.model(*args)
        e1.record(); t3 = time.perf_counter(); c1 = time.process_time()
        out = logits.float().cpu().numpy(); torch.softmax(act.float(), -1).cpu().numpy(); t4 = time.perf_counter()
        if r >= 5:
            rows["collate"].append((t1 - t0) * 1e3); rows["h2d"].append((t2 - t1) * 1e3)
            rows["dispatch_cpu"].append((t3 - t2) * 1e3); rows["gpu"].append(e0.elapsed_time(e1))
            rows["d2h_sync"].append((t4 - t3) * 1e3); rows["total"].append((t4 - t0) * 1e3)
    m = {k: statistics.median(v) for k, v in rows.items()}
    tok = sum(len(it["ids"]) for it in items); pad = max(len(it["ids"]) for it in items) * len(items)
    print("%-26s rows=%3d tokens=%5d padded=%5d | collate %5.2f  h2d %4.2f  dispatch(CPU) %5.2f  GPU %5.2f  sync+d2h %5.2f  total %5.2f ms"
          % (label, len(items), tok, pad, m["collate"], m["h2d"], m["dispatch_cpu"], m["gpu"], m["d2h_sync"], m["total"]))
for n in (1, 5, 20, 40):
    anatomy("short x%d requests" % n, SHORT[:n])
for n in (1, 5):
    anatomy("~500-token x%d requests" % n, long_states[:n])
