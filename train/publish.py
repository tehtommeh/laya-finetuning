"""Copy a finished run's checkpoint to ./finetuned/<name>/, where the API serves it.

    python publish.py /runs/my-run [--as my-model] [--force]

Only complete checkpoints are published. The API loads every directory under
./finetuned that holds an rl_agent_config.json at startup, so restart it
afterwards:  docker compose restart api
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys

from ckpt import CHECKPOINT_FILES
from data import load_splits, schemas

RESERVED = {"auto", "laya", "english", "multilingual", "typed-decisions", "en", "multi", "ml", "typed",
            "typed_decisions", "decisions", "default", "laya-multilingual", "laya-typed-decisions"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--as", dest="name")
    ap.add_argument("--dest", default="/finetuned")
    ap.add_argument("--force", action="store_true", help="replace an existing published model of that name")
    a = ap.parse_args(argv)

    src = os.path.join(a.run, "model")
    missing = [f for f in CHECKPOINT_FILES if not os.path.exists(os.path.join(src, f))]
    if missing:
        sys.exit("%s is not a complete checkpoint (missing %s) - did training finish?" % (src, ", ".join(missing)))
    name = a.name or os.path.basename(os.path.normpath(a.run))
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", name):
        sys.exit("name %r: use lowercase letters, digits, '.', '_' or '-'" % name)
    if name in RESERVED:
        sys.exit("name %r is reserved for the shipped checkpoints; pick another with --as" % name)
    if not os.path.exists(os.path.join(src, "eval.json")):
        print("WARN no eval.json - run evaluate.py first so the UI can show held-out metrics")
    dst = os.path.join(a.dest, name)
    if os.path.exists(dst):
        if not a.force:
            sys.exit("%s already exists; pass --force to replace it" % dst)
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    # Record the trained question schemas so the UI can offer them as presets.
    try:
        with open(os.path.join(src, "training_summary.json")) as f:
            ar = json.load(f)["args"]
        splits, _ = load_splits(ar["train"], ar.get("calib"), ar.get("test"), ar.get("questions"),
                                ar["calib_frac"], ar["test_frac"], ar["seed"])
        sch = schemas(splits["train"], splits["test"] or splits["calib"])
        with open(os.path.join(dst, "schemas.json"), "w") as f:
            json.dump(sch, f, indent=2, ensure_ascii=False)
        print("Wrote %d question schema(s): %s" % (len(sch), ", ".join(sch)))
    except (OSError, KeyError, ValueError) as e:
        print("WARN could not write schemas.json (%s); the UI will not offer this model's questions" % e)
    print("Published %s -> %s\nNow: docker compose restart api   (then pick %r in the UI's checkpoint menu)"
          % (src, dst, name))


if __name__ == "__main__":
    main()
