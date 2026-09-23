"""Check a fine-tuning dataset before spending GPU time on it.

    python validate.py --train /data/mine/train.jsonl [--questions /data/mine/questions.json]
                       [--calib ...] [--test ...] [--base english] [--max-len 1024 --head-max-len 256]

Reports: parse errors (with file:line), per-split counts, per-question label
balance and how many answers are soft, questions whose options do not fit the
option token budget, and how often the state gets truncated. Exits 1 on any
error so it can gate `make train`.
"""
from __future__ import annotations

import argparse
import sys

from ckpt import encode_case, load_tokenizer, read_config, resolve_base
from data import load_splits, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--calib")
    ap.add_argument("--test")
    ap.add_argument("--questions", help="shared question schema (JSON object) for rows that omit `questions`")
    ap.add_argument("--base", default="english")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--head-max-len", type=int, default=256)
    ap.add_argument("--calib-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    splits, errors = load_splits(a.train, a.calib, a.test, a.questions, a.calib_frac, a.test_frac, a.seed)
    for e in errors[:50]:
        print("ERROR", e)
    if len(errors) > 50:
        print("ERROR ... and %d more" % (len(errors) - 50))

    base = resolve_base(a.base)
    cfg = read_config(base)
    tok = load_tokenizer(base)
    print("\nBase %s (%s), max_len=%d head_max_len=%d" % (a.base, cfg["encoder"], a.max_len, a.head_max_len))

    fit_problems, n_items, n_trunc = [], 0, 0
    lengths = []
    for name, cases in splits.items():
        for c in cases:
            items, problems = encode_case(c, tok, a.max_len, a.head_max_len)
            fit_problems += problems
            n_items += len(items)
            n_trunc += sum(it["truncated"] for it in items)
            lengths += [len(it["ids"]) for it in items]

    print("\n%-6s %7s %9s" % ("split", "cases", "answers"))
    for name, cases in splits.items():
        print("%-6s %7d %9d" % (name, len(cases), sum(len(c["gold"]) for c in cases)))

    print("\nPer question (train split):")
    print("  %-28s %-6s %6s %6s  %s" % ("question", "type", "n", "soft", "label counts"))
    for qid, s in sorted(stats(splits["train"]).items()):
        labels = ", ".join("%s=%d" % kv for kv in sorted(s["labels"].items(), key=lambda kv: -kv[1]))
        print("  %-28s %-6s %6d %6d  %s" % (qid[:28], s["type"], s["n"], s["soft"], labels[:110]))
        top = max(s["labels"].values())
        if s["n"] >= 20 and top / s["n"] > 0.9:
            print("    WARN %s: %.0f%% of answers are one label - the model can learn to always say it"
                  % (qid, 100 * top / s["n"]))
        if len(s["labels"]) < 2:
            print("    WARN %s: only one label ever appears in train" % qid)

    if lengths:
        lengths.sort()
        print("\nSequence length: median %d, p95 %d, max %d tokens; %d/%d (%.1f%%) hit max_len and truncate the state"
              % (lengths[len(lengths) // 2], lengths[int(len(lengths) * 0.95)], lengths[-1], n_trunc, n_items,
                 100 * n_trunc / max(1, n_items)))
        if n_trunc / max(1, n_items) > 0.05:
            print("  WARN more than 5% of inputs are cut off. Shorten states (strip boilerplate) or raise --max-len.")
    for p in fit_problems[:20]:
        print("ERROR", p)

    n_train = len(splits["train"])
    print()
    if n_train < 100:
        print("WARN only %d training cases. Expect little improvement below a few hundred per workflow "
              "(see docs/FINETUNING.md)." % n_train)
    if not splits["calib"]:
        print("WARN no calibration split: probabilities will not be temperature-fitted.")
    if not splits["test"]:
        print("WARN no test split: evaluate.py will have nothing held out to score.")
    bad = len(errors) + len(fit_problems)
    print("%s: %d error(s)" % ("FAILED" if bad else "OK", bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
