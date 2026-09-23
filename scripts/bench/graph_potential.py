"""Pre-cast weights and CUDA-graph replay vs eager at fixed shapes, with bit-identity checks (docs/PERFORMANCE.md)."""
import sys, time, statistics, copy
sys.path.insert(0, "/app"); sys.path.insert(0, "/tmp/bench")
import torch, laya, batching
from laya.common import collate_items
from cpu import SHORT, QS
ag = laya.Agent("/models/convaiinnovations__laya", device="cuda")
base_model = ag.model

def precast(model):
    """bf16 copies of exactly the weights autocast would cast for matmuls (Linear, MHA in_proj)."""
    m = copy.deepcopy(model)
    for mod in m.modules():
        if isinstance(mod, torch.nn.Linear):
            mod.weight.data = mod.weight.data.to(torch.bfloat16)
            if mod.bias is not None: mod.bias.data = mod.bias.data.to(torch.bfloat16)
        if isinstance(mod, torch.nn.MultiheadAttention) and mod.in_proj_weight is not None:
            mod.in_proj_weight.data = mod.in_proj_weight.data.to(torch.bfloat16)
            if mod.in_proj_bias is not None: mod.in_proj_bias.data = mod.in_proj_bias.data.to(torch.bfloat16)
    return m
fast_model = precast(base_model)

def batch_for(n):
    items = [it for s in SHORT[:n] for it in batching.encode(ag, [s], QS)[0]]
    b = collate_items([items], ag.tok.pad_token_id)
    return [b[k].cuda() for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")]

def run(model, args):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return model(*args)

def timeit(fn, reps=30):
    for _ in range(5): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts)

for n in (1, 5, 20, 40):
    args = batch_for(n)
    l0, a0 = run(base_model, args); l1, a1 = run(fast_model, args)
    same = torch.equal(l0, l1) and torch.equal(a0, a1)
    maxd = (l0 - l1).abs().max().item()
    t_base = timeit(lambda: run(base_model, args)); t_fast = timeit(lambda: run(fast_model, args))
    # CUDA graph of the pre-cast model at this exact shape
    static = [a.clone() for a in args]
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): run(fast_model, static)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        gl, ga = run(fast_model, static)
    g.replay(); torch.cuda.synchronize()
    gsame = torch.equal(gl, l1)
    t_graph = timeit(lambda: g.replay())
    print("rows=%3d  baseline %6.2f ms | pre-cast %6.2f ms (identical=%s, max|d|=%.1e) | +CUDA graph %6.2f ms (identical=%s)" % (
        len(args[0]), t_base, t_fast, same, maxd, t_graph, gsame))
print("GPU memory: baseline params %.2f GB, pre-cast %.2f GB" % (
    sum(p.numel() * p.element_size() for p in base_model.parameters()) / 1e9,
    sum(p.numel() * p.element_size() for p in fast_model.parameters()) / 1e9))
