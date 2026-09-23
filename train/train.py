"""Fine-tune a Laya checkpoint on your own typed decisions (RLCD).

    python train.py --name my-run --train /data/mine/train.jsonl [--base english] [options]
    torchrun --standalone --nproc_per_node=2 train.py ...      # multi-GPU (DDP)

The recipe is the official one (NandhaKishorM/laya, notebooks/
laya_finetune_typed_decisions_2xT4_kaggle.ipynb): full fine-tune of encoder and
head, REINFORCE with a group-mean baseline on a strictly proper scoring reward
plus soft cross-entropy, cosine LR, gradient checkpointing. Differences from
the notebook, each deliberate:

  * bf16 autocast on Ampere+ (the checkpoints were trained in bf16); fp16 +
    GradScaler on older GPUs.
  * choice options are shuffled per example, as in the original training code,
    so answer position carries no signal.
  * temperatures are fitted on a *held-out* calibration split, per
    (question type, option count) bucket - the bucket table is what the SDK
    actually applies - and against the gold *labels*. The notebook fitted per
    type, on training data, to the soft distributions, and left the base
    model's bucket table in place (which overrides its own fit). Fitting to
    soft teacher distributions made the model under-confident on held-out
    data (docs/EXPERIMENTS.md, section 6); --calibrate-on soft restores it.
  * the effective batch (64 sequences) is kept constant whatever the GPU count.

Output: /runs/<name>/model/  - a drop-in checkpoint (same layout as the
shipped ones), plus training_summary.json, train_log.jsonl and a model card.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch

import calibrate
from ckpt import encode_case, load_model, load_tokenizer, read_config, resolve_base, save_checkpoint
from data import load_splits, stats, subsample


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = ap.add_argument_group("data")
    d.add_argument("--train", required=True, help="JSONL of cases")
    d.add_argument("--calib", help="JSONL held out for temperature fitting (default: carve from --train)")
    d.add_argument("--test", help="JSONL held out for evaluation (default: carve from --train)")
    d.add_argument("--questions", help="shared question schema for rows that omit `questions`")
    d.add_argument("--calib-frac", type=float, default=0.1)
    d.add_argument("--test-frac", type=float, default=0.1)
    d.add_argument("--limit", type=int, help="train on a random (nested, seeded) subset of N cases - for learning curves")
    d.add_argument("--seed", type=int, default=0)
    m = ap.add_argument_group("model")
    m.add_argument("--name", required=True, help="run name; output goes to <out-root>/<name>")
    m.add_argument("--base", default="english", help="english | multilingual | typed-decisions | checkpoint dir")
    m.add_argument("--out-root", default="/runs")
    m.add_argument("--max-len", type=int, default=1024, help="tokens per question sequence (notebook: 1024)")
    m.add_argument("--head-max-len", type=int, default=256, help="tokens for instructions + options (notebook: 256)")
    o = ap.add_argument_group("optimisation (defaults = official notebook)")
    o.add_argument("--epochs", type=int, default=4)
    o.add_argument("--batch", type=int, default=8, help="sequences per forward pass per GPU")
    o.add_argument("--effective-batch", type=int, default=64, help="sequences per optimiser step, all GPUs")
    o.add_argument("--lr-encoder", type=float, default=2.5e-5)
    o.add_argument("--lr-head", type=float, default=1.0e-4)
    o.add_argument("--group-size", type=int, default=4, help="noisy samples per item for the GRPO baseline")
    o.add_argument("--sigma-start", type=float, default=0.4)
    o.add_argument("--sigma-end", type=float, default=0.1)
    o.add_argument("--ce-weight", type=float, default=1.0)
    o.add_argument("--no-shuffle-options", action="store_true")
    o.add_argument("--no-grad-checkpointing", action="store_true", help="faster, needs much more VRAM")
    o.add_argument("--calibrate-on", choices=("hard", "soft"), default="hard",
                   help="fit temperatures to gold labels (default) or gold distributions (the notebook)")
    return ap.parse_args(argv)


# --------------------------------------------------------------------------- main
def main(argv=None):
    a = parse_args(argv)
    from laya.common import collate_items, proper_reward

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        import torch.distributed as dist
        dist.init_process_group("nccl")
    if not torch.cuda.is_available():
        sys.exit("No CUDA GPU visible. Fine-tuning a 322-421M encoder on CPU is impractical; see docs/FINETUNING.md.")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    is_main = rank == 0
    log = print if is_main else (lambda *x, **k: None)

    random.seed(a.seed + rank)
    np.random.seed(a.seed + rank)
    torch.manual_seed(a.seed)

    run_dir = os.path.join(a.out_root, a.name)
    out_dir = os.path.join(run_dir, "model")
    if is_main:
        os.makedirs(run_dir, exist_ok=True)

    # ---- data
    splits, errors = load_splits(a.train, a.calib, a.test, a.questions, a.calib_frac, a.test_frac, a.seed)
    if errors:
        for e in errors[:20]:
            log("ERROR", e)
        sys.exit("%d data error(s); run validate.py for the full report" % len(errors))
    splits["train"] = subsample(splits["train"], a.limit, a.seed)
    base = resolve_base(a.base)
    cfg = read_config(base)
    tok = load_tokenizer(base)
    rng = None if a.no_shuffle_options else random.Random(a.seed)
    train_items, problems = [], []
    for c in splits["train"]:
        its, pr = encode_case(c, tok, a.max_len, a.head_max_len, rng)
        train_items += its
        problems += pr
    calib_items = [it for c in splits["calib"] for it in encode_case(c, tok, a.max_len, a.head_max_len)[0]]
    if problems:
        log("WARN %d answers skipped (options do not fit head_max_len); e.g. %s" % (len(problems), problems[0]))
    if not train_items:
        sys.exit("no training items")
    log("Data: %d train cases (%d answers), %d calib cases, %d test cases"
        % (len(splits["train"]), len(train_items), len(splits["calib"]), len(splits["test"])))

    # ---- model
    cc = torch.cuda.get_device_capability(device)
    dtype = torch.bfloat16 if cc[0] >= 8 else torch.float16
    t_load = time.time()
    model = load_model(base, cfg)
    if not a.no_grad_checkpointing:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device).train()
    net = model
    if world > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        net = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    log("Base %s (%s) on %s, %s autocast, loaded in %.1fs"
        % (a.base, cfg["encoder"], torch.cuda.get_device_name(device), str(dtype).replace("torch.", ""),
           time.time() - t_load))

    my_items = train_items[rank::world]
    accum = max(1, round(a.effective_batch / (a.batch * world)))
    steps_per_epoch = math.ceil(len(my_items) / (a.batch * accum))
    total_steps = steps_per_epoch * a.epochs
    enc_params = [p for n, p in net.named_parameters() if "encoder." in n]
    head_params = [p for n, p in net.named_parameters() if "encoder." not in n]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": a.lr_encoder}, {"params": head_params, "lr": a.lr_head}],
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    log("Schedule: %d epochs x %d optimiser steps (batch %d x accum %d x %d GPU = %d sequences/step)"
        % (a.epochs, steps_per_epoch, a.batch, accum, world, a.batch * accum * world))

    log_f = open(os.path.join(run_dir, "train_log.jsonl"), "w") if is_main else None
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    tokens = 0
    step = 0
    for epoch in range(a.epochs):
        random.Random(a.seed + epoch + rank).shuffle(my_items)
        sigma = a.sigma_start + (a.sigma_end - a.sigma_start) * (epoch / max(1, a.epochs - 1))
        ep_loss, ep_reward, n_b = 0.0, 0.0, 0
        opt.zero_grad(set_to_none=True)
        for bi, s in enumerate(range(0, len(my_items), a.batch)):
            b = collate_items([my_items[s:s + a.batch]], tok.pad_token_id)
            with torch.autocast("cuda", dtype=dtype):
                logits, act = net(b["input_ids"].to(device), b["attention_mask"].to(device),
                                  b["marker_pos"].to(device), b["marker_mask"].to(device), b["qtype"].to(device))
            logits = logits.float()
            mask = b["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = b["target"].to(device)
            qtype = b["qtype"].to(device)
            # 1. G zero-mean Gaussian perturbations of the logits (the exploration policy)
            eps = torch.randn((a.group_size,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            # 2. strictly proper reward, advantage against the group mean
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), qtype, mask, w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)
            # 3. REINFORCE + soft cross-entropy guidance
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + a.ce_weight * loss_ce) / accum + 0.0 * act.sum()  # act head stays in the graph for DDP
            scaler.scale(loss).backward()
            tokens += int(b["attention_mask"].sum())
            last = s + a.batch >= len(my_items)
            if (bi + 1) % accum == 0 or last:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if is_main and (step % 10 == 0 or last):
                    rec = {"epoch": epoch + 1, "step": step, "of": total_steps, "loss": round(loss.item() * accum, 4),
                           "ce": round(loss_ce.item(), 4), "reward": round(r.mean().item(), 4),
                           "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0, 1)}
                    log_f.write(json.dumps(rec) + "\n")
                    log_f.flush()
                    eta = (time.time() - t0) / step * (total_steps - step)
                    log("  epoch %d step %4d/%d  loss %.4f  ce %.4f  reward %.3f  eta %dm%02ds"
                        % (epoch + 1, step, total_steps, rec["loss"], rec["ce"], rec["reward"], eta // 60, eta % 60))
            ep_loss += loss.item() * accum
            ep_reward += r.mean().item()
            n_b += 1
        log("epoch %d/%d done: mean loss %.4f, mean reward %.3f, %.0fs elapsed"
            % (epoch + 1, a.epochs, ep_loss / max(1, n_b), ep_reward / max(1, n_b), time.time() - t0))
    train_s = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
    peak_reserved_gb = torch.cuda.max_memory_reserved(device) / 1e9

    if world > 1:
        import torch.distributed as dist
        dist.barrier()
    if not is_main:
        return 0

    # ---- calibration on the held-out split (see calibrate.py for why "hard" is the default)
    model.eval()
    new_cfg = dict(cfg)
    temps, temps_by_bucket, calib_report = [1.0, 1.0, 1.0], {}, None
    if calib_items:
        pairs = calibrate.pairs_for(model, calib_items, tok, device, dtype)
        temps, temps_by_bucket, before, after = calibrate.fit(pairs, a.calibrate_on)
        calib_report = {"target": a.calibrate_on, "before": before, "after": after}
        log("Calibration (%s target) on %d held-out answers: ECE %.3f -> %.3f, NLL %.3f -> %.3f"
            % (a.calibrate_on, before["n"], before["ece"], after["ece"], before["nll"], after["nll"]))
    else:
        log("WARN no calibration split - temperatures left at 1.0 (run calibrate.py later with --calib)")
    new_cfg.update({
        "max_len": a.max_len, "head_max_len": a.head_max_len, "temperature": temps,
        "temperature_by_options": temps_by_bucket, "model_name": a.name, "fine_tuned": True,
        "calibration": calib_report,
        "amp_dtype": "bf16" if dtype == torch.bfloat16 else "fp16",
        "training": {"base": a.base, "base_encoder": cfg["encoder"], "updates": step, "epochs_completed": a.epochs,
                     "hours": round(train_s / 3600, 3), "world_size": world, "fine_tuned_from_checkpoint": True},
    })
    lock = {}
    try:
        with open("/models/download.lock.json") as f:
            lock = json.load(f)
    except (OSError, ValueError):
        pass
    summary = {
        "name": a.name, "base": a.base, "base_encoder": cfg["encoder"], "base_path": base, "base_lock": lock,
        "args": vars(a), "gpu": torch.cuda.get_device_name(device), "world_size": world,
        "precision": str(dtype).replace("torch.", ""),
        "data": {"train_cases": len(splits["train"]), "train_answers": len(train_items),
                 "calib_cases": len(splits["calib"]), "test_cases": len(splits["test"]),
                 "skipped_answers": len(problems), "per_question": stats(splits["train"])},
        "timing": {"train_seconds": round(train_s, 1), "seconds_per_epoch": round(train_s / a.epochs, 1),
                   "tokens_per_second": round(tokens / train_s), "optimiser_steps": step},
        "memory": {"peak_allocated_gb": round(peak_gb, 2), "peak_reserved_gb": round(peak_reserved_gb, 2)},
        "calibration": calib_report, "temperature": temps, "temperature_by_options": temps_by_bucket,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    card = model_card(summary)
    save_checkpoint(model, new_cfg, base, out_dir, {"training_summary.json": summary, "README.md": card})
    log("\nSaved %s  (%.1f min training, peak %.1f GB allocated / %.1f GB reserved)"
        % (out_dir, train_s / 60, peak_gb, peak_reserved_gb))
    log("Next: python evaluate.py --run %s   then   make publish RUN=%s" % (run_dir, a.name))
    return 0


def model_card(s):
    qs = "\n".join("| `%s` | %s | %d | %d |" % (q, v["type"], v["n"], v["soft"])
                   for q, v in sorted(s["data"]["per_question"].items()))
    cal = s.get("calibration") or {}
    cal_line = ("ECE %.3f -> %.3f on %d held-out answers" % (cal["before"]["ece"], cal["after"]["ece"],
                                                               cal["before"]["n"])) if cal else "not calibrated"
    return """# {name}

Laya checkpoint fine-tuned from `{base}` ({enc}) with RLCD.

- Trained: {finished} on {gpu} x{world}, {prec}, {mins:.1f} min, peak {mem} GB
- Data: {tc} training cases / {ta} answers, {cc} calibration cases, {xc} test cases
- Calibration: {cal}
- Sequence budget: max_len {ml}, head_max_len {hml}

Load it like any Laya checkpoint:

```python
import laya
agent = laya.Agent("/path/to/{name}")
agent.predict(state, questions)
```

It was trained on these questions; ask them with the same wording and options:

| question | type | answers | soft answers |
|---|---|---|---|
{qs}

See `training_summary.json` for every hyper-parameter and `eval.json` (after evaluate.py) for held-out metrics.
""".format(name=s["name"], base=s["base"], enc=s["base_encoder"], finished=s["finished_at"],
           gpu=s["gpu"], world=s["world_size"], prec=s["precision"], mins=s["timing"]["train_seconds"] / 60,
           mem=s["memory"]["peak_allocated_gb"], tc=s["data"]["train_cases"], ta=s["data"]["train_answers"],
           cc=s["data"]["calib_cases"], xc=s["data"]["test_cases"], cal=cal_line, ml=s["args"]["max_len"],
           hml=s["args"]["head_max_len"], qs=qs)


if __name__ == "__main__":
    sys.exit(main())
