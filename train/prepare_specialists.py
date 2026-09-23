"""Fetch small, seeded samples of three specialist benchmarks as Laya case files.

    python prepare_specialists.py [--out /data/specialists] [--train 500] [--test 1000]

Rows come from the Hugging Face datasets-server API, so nothing large is
downloaded (civil_comments alone is 1.8M rows). For each task this writes
train.jsonl / test.jsonl in the project's case format, a questions.json (the Laya
question used zero-shot and for fine-tuning), and SOURCE.json with the dataset
revision and sampled offsets.

  prompt-injection  deepset/prompt-injections (all rows; binary)
  toxicity          google/civil_comments (toxicity = fraction of raters -> soft noul label)
  sentiment         cardiffnlp/tweet_eval, config sentiment (negative / neutral / positive)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.parse
import urllib.request

API = "https://datasets-server.huggingface.co"

TASKS = {
    "prompt-injection": {
        "dataset": "deepset/prompt-injections", "config": "default",
        "questions": {"injection": {
            "type": "noul",
            "instructions": "Does `prompt` try to override, ignore or manipulate an AI system's instructions "
                            "(a prompt injection or jailbreak) rather than make a normal request?",
            "criteria": {"true": "prompt injection or jailbreak attempt", "false": "an ordinary request"}}},
    },
    "toxicity": {
        "dataset": "google/civil_comments", "config": "default",
        "questions": {"toxic": {
            "type": "noul",
            "instructions": "Is `comment` toxic: rude, disrespectful or unreasonable, likely to make someone leave a discussion?",
            "criteria": {"true": "toxic", "false": "civil"}}},
    },
    "sentiment": {
        "dataset": "cardiffnlp/tweet_eval", "config": "sentiment",
        "questions": {"sentiment": {
            "type": "choice", "instructions": "What is the sentiment of `tweet`?",
            "criteria": {"negative": "unhappy, critical or angry", "neutral": "factual or no clear feeling",
                         "positive": "happy, approving or enthusiastic"}}},
    },
}


def get(path, **params):
    url = "%s/%s?%s" % (API, path, urllib.parse.urlencode(params))
    for attempt in range(10):   # the API rate-limits and occasionally 5xxs; back off and retry
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception:  # noqa: BLE001 - transient API errors
            time.sleep(min(60, 3 * 2 ** attempt))
    raise RuntimeError("datasets-server request failed: " + url)


def sample_rows(dataset, config, split, n, rnd):
    total = get("size", dataset=dataset)["size"]["splits"]
    total = next(s["num_rows"] for s in total if s["config"] == config and s["split"] == split)
    if n >= total or total <= 5000:   # small: fetch everything, then take a seeded sample
        rows = []
        for o in range(0, total, 100):
            rows += [r["row"] for r in get("rows", dataset=dataset, config=config, split=split, offset=o, length=100)["rows"]]
        rnd.shuffle(rows)
        return rows[:n], total, "all"
    # contiguous 100-row pages at random offsets, then a seeded sample of rows from them
    starts = range(0, total - 100, 100)
    pages = sorted(rnd.sample(starts, k=min(len(starts), max(1, (n * 3) // 100))))
    rows = []
    for o in pages:
        rows += [r["row"] for r in get("rows", dataset=dataset, config=config, split=split, offset=o, length=100)["rows"]]
    rnd.shuffle(rows)
    return rows[:n], total, pages


def to_case(task, i, split, row):
    if task == "prompt-injection":
        return {"id": "%s-%d" % (split, i), "state": {"prompt": row["text"]}, "gold": {"injection": bool(row["label"])}}
    if task == "toxicity":
        return {"id": "%s-%d" % (split, i), "state": {"comment": row["text"]}, "gold": {"toxic": float(row["toxicity"])}}
    labels = ["negative", "neutral", "positive"]
    return {"id": "%s-%d" % (split, i), "state": {"tweet": row["text"]}, "gold": {"sentiment": labels[int(row["label"])]}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/data/specialists")
    ap.add_argument("--train", type=int, default=500)
    ap.add_argument("--test", type=int, default=1000)
    a = ap.parse_args()
    from huggingface_hub import HfApi
    for task, spec in TASKS.items():
        rnd = random.Random(0)
        d = os.path.join(a.out, task)
        os.makedirs(d, exist_ok=True)
        src = {"dataset": spec["dataset"], "config": spec["config"],
               "revision": HfApi().dataset_info(spec["dataset"]).sha}
        for split, n in (("train", a.train), ("test", a.test)):
            rows, total, pages = sample_rows(spec["dataset"], spec["config"], split, n, rnd)
            rows = [r for r in rows if (r.get("text") or "").strip()]
            with open(os.path.join(d, split + ".jsonl"), "w", encoding="utf-8") as f:
                for i, r in enumerate(rows):
                    f.write(json.dumps(to_case(task, i, split, r), ensure_ascii=False) + "\n")
            src[split] = {"rows": len(rows), "of": total, "pages": pages}
        json.dump(spec["questions"], open(os.path.join(d, "questions.json"), "w"), indent=2)
        json.dump(src, open(os.path.join(d, "SOURCE.json"), "w"), indent=2)
        print("%-17s train %4d  test %4d  (%s @ %s)" % (task, src["train"]["rows"], src["test"]["rows"],
                                                         spec["dataset"], src["revision"][:12]))


if __name__ == "__main__":
    main()
