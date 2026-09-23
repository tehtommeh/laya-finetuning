"""Fetch LocalLLaMA/typed-decisions (Apache-2.0) as a ready-made example dataset.

    python prepare_example.py [--out /data/typed-decisions] [--workflow all|customer_service|...]

Writes train.jsonl (1,200 cases) and test.jsonl (400 cases) in this project's
case format, plus SOURCE.json with the dataset revision. It is the dataset the
official typed-decisions checkpoint was fine-tuned on, so a run on it
reproduces that result end to end and proves the pipeline works on your GPU
before you label anything yourself.

Each case: 5 questions over one JSON state; gold answers are full probability
distributions from repeated teacher samples (soft labels).
"""
from __future__ import annotations

import argparse
import json
import os

REPO = "LocalLLaMA/typed-decisions"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/data/typed-decisions")
    ap.add_argument("--workflow", default="all")
    ap.add_argument("--revision", default=None, help="dataset commit to pin (default: latest)")
    a = ap.parse_args(argv)

    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().dataset_info(REPO, revision=a.revision)
    os.makedirs(a.out, exist_ok=True)
    counts = {}
    for split in ("train", "test"):
        path = hf_hub_download(REPO, "%s/%s-00000-of-00001.parquet" % (a.workflow, split), repo_type="dataset",
                               revision=info.sha)
        rows = pq.read_table(path, columns=["id", "workflow", "state", "questions", "gold"]).to_pylist()
        with open(os.path.join(a.out, "%s.jsonl" % split), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"id": r["id"], "workflow": r["workflow"], "state": json.loads(r["state"]),
                                    "questions": json.loads(r["questions"]), "gold": json.loads(r["gold"])},
                                   ensure_ascii=False) + "\n")
        counts[split] = len(rows)
    with open(os.path.join(a.out, "SOURCE.json"), "w") as f:
        json.dump({"dataset": REPO, "revision": info.sha, "workflow": a.workflow, "license": "apache-2.0",
                   "cases": counts}, f, indent=2)
    print("Wrote %s: %s (revision %s)" % (a.out, counts, info.sha[:12]))


if __name__ == "__main__":
    main()
