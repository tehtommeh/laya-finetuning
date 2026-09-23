"""Tabulate every evaluated run under /runs - e.g. a learning-curve sweep.

    python summarize.py [--runs /runs] [--prefix td-] [--out /runs/summary.md]

For each run with eval.json: training cases, training time, peak VRAM and the
fine-tuned model's held-out metrics, next to the baselines scored in that eval.
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="/runs")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--out")
    a = ap.parse_args(argv)

    rows, baselines = [], {}
    for ev_path in sorted(glob.glob(os.path.join(a.runs, a.prefix + "*", "eval.json"))):
        run = os.path.dirname(ev_path)
        try:
            with open(os.path.join(run, "model", "training_summary.json")) as f:
                s = json.load(f)
            with open(ev_path) as f:
                ev = json.load(f)
        except (OSError, ValueError):
            continue
        ft = next((r for n, r in ev["results"].items() if n.endswith("(fine-tuned)")), None)
        for n, r in ev["results"].items():
            if not n.endswith("(fine-tuned)"):
                baselines.setdefault(n, r["overall"])
        if ft:
            rows.append((s["data"]["train_cases"], s["name"], s, ft["overall"], ev["test_cases"]))
    rows.sort()
    cols = ["accuracy", "soft_acc", "brier", "nll", "ece", "score_mae"]
    out = ["| run | train cases | answers | train time | peak VRAM | " + " | ".join(cols) + " |",
           "|---" * (len(cols) + 5) + "|"]
    for name, o in baselines.items():
        out.append("| %s | - | - | - | - | %s |" % (name, " | ".join("%.3f" % o[c] if c in o else "-" for c in cols)))
    for n, name, s, o, _ in rows:
        out.append("| `%s` | %d | %d | %.1f min | %.1f GB | %s |" % (
            name, n, s["data"]["train_answers"], s["timing"]["train_seconds"] / 60, s["memory"]["peak_allocated_gb"],
            " | ".join("%.3f" % o[c] if c in o else "-" for c in cols)))
    md = "\n".join(out)
    if rows:
        md += "\n\nAll scored on the same %d held-out test cases." % rows[0][4]
    print(md)
    if a.out:
        with open(a.out, "w") as f:
            f.write(md + "\n")


if __name__ == "__main__":
    main()
