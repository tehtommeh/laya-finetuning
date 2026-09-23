"""Checkpoint I/O and sequence encoding shared by validate / train / evaluate.

Everything that turns a question into tokens goes through the SDK's own
`laya.common.build_sequence`, so training sees byte-identical inputs to serving.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import tempfile

from data import option_keys, to_internal

CHECKPOINT_FILES = ("rl_agent_config.json", "model.safetensors", "encoder/config.json", "tokenizer/tokenizer.json")
BASES = {
    "english": "/models/convaiinnovations__laya",
    "multilingual": "/models/convaiinnovations__laya/multilingual",
    "typed-decisions": "/models/convaiinnovations__laya/typed-decisions",
}


def resolve_base(name_or_path: str) -> str:
    path = BASES.get(name_or_path, name_or_path)
    missing = [f for f in CHECKPOINT_FILES if not os.path.exists(os.path.join(path, f))]
    if missing:
        raise SystemExit("base checkpoint %r is incomplete at %s (missing %s). Run scripts/download.py first, "
                         "or pass english / multilingual / typed-decisions / a checkpoint directory."
                         % (name_or_path, path, ", ".join(missing)))
    return path


def read_config(path: str) -> dict:
    with open(os.path.join(path, "rl_agent_config.json")) as f:
        return json.load(f)


def load_tokenizer(path: str):
    """The SDK patches some tokenizer configs in place; do it on a private copy
    because the model mount is read-only."""
    from laya.agent import _fix_tokenizer_config
    from transformers import AutoTokenizer
    tmp = tempfile.mkdtemp(prefix="laya-tok-")
    shutil.copytree(os.path.join(path, "tokenizer"), os.path.join(tmp, "tokenizer"))
    _fix_tokenizer_config(tmp)
    return AutoTokenizer.from_pretrained(os.path.join(tmp, "tokenizer"))


def load_model(path: str, cfg: dict):
    from laya.common import build_model
    from safetensors.torch import load_file
    model = build_model(cfg, encoder_dir=os.path.join(path, "encoder"))
    model.load_state_dict(load_file(os.path.join(path, "model.safetensors")), strict=True)
    model.encoder.config.reference_compile = False
    return model


def encode_case(case: dict, tok, max_len: int, head_max_len: int, rng: random.Random | None = None):
    """One case -> one item per answered question. Returns (items, problems).

    With `rng`, choice options are shuffled (the original RLCD training code does
    this too) so the model cannot learn "the first option is usually right".
    Score options keep their order: the levels are ordinal.
    """
    from laya.common import QTYPES, build_sequence, render_options
    items, problems = [], []
    for qid, g in case["gold"].items():
        q = case["questions"][qid]
        iq = to_internal(q)
        k = len(render_options(iq))
        order = list(range(k))
        if rng is not None and q["type"] == "choice":
            rng.shuffle(order)
        ids, markers = build_sequence(tok, case["state"], iq, max_len, head_max_len, option_order=order)
        if len(markers) != k:
            problems.append("%s/%s: %d options do not fit in head_max_len=%d" % (case["id"], qid, k, head_max_len))
            continue
        target = [g["dist"][i] for i in order]
        items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["type"]], "target": target,
                      "label": order.index(g["label"]), "case": case["id"], "qid": qid, "k": k,
                      "truncated": len(ids) >= max_len})
    return items, problems


def save_checkpoint(model, cfg: dict, base_path: str, out: str, extra_files: dict | None = None):
    """Write the same layout as the shipped checkpoints, so laya.Agent / the API
    load it unchanged. The encoder architecture and tokenizer are copied from the
    base verbatim: fine-tuning changes weights, never shapes or vocabulary."""
    from safetensors.torch import save_file
    os.makedirs(out, exist_ok=True)
    sd = {k: v.detach().half().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(sd, os.path.join(out, "model.safetensors"))
    for d in ("encoder", "tokenizer"):
        shutil.rmtree(os.path.join(out, d), ignore_errors=True)
        shutil.copytree(os.path.join(base_path, d), os.path.join(out, d))
    with open(os.path.join(out, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    for name, obj in (extra_files or {}).items():
        with open(os.path.join(out, name), "w") as f:
            if isinstance(obj, str):
                f.write(obj)
            else:
                json.dump(obj, f, indent=2, ensure_ascii=False)


__all__ = ["BASES", "resolve_base", "read_config", "load_tokenizer", "load_model", "encode_case",
           "save_checkpoint", "option_keys"]
