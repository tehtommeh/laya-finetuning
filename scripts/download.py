#!/usr/bin/env python3
"""Download model weights into the project and track the exact revision.

Weights live under ./models/<org>__<name>/ as a plain directory so containers can
load them by path with no Hub access at runtime. A download.lock.json records the
commit SHA, which is what makes "has this model changed upstream?" answerable
without re-downloading anything.

Usage:
    python3 download.py Qwen/Qwen3-8B                 # download (resumes if partial)
    python3 download.py --check                       # any upstream changes?
    python3 download.py --update Qwen/Qwen3-8B        # pull the new revision
    python3 download.py Qwen/Qwen3-8B --revision <sha>  # pin exactly
    python3 download.py Qwen/Qwen3-8B --all           # include onnx/duplicate formats
    python3 download.py --verify                      # re-check local files against the lock
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: E402

LOCK_NAME = "download.lock.json"

# Repos routinely ship the same tensors several ways. Pulling all of them can
# triple the download for no benefit, so prefer one format unless --all is given.
DUPLICATE_FORMATS = [
    "*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*",
    "*.onnx", "*.onnx_data", "onnx/*", "openvino/*", "*.tflite",
]
ALWAYS_SKIP = [".gitattributes", "*.lock"]


def models_dir(root):
    return os.path.join(root, "models")


def local_path(root, repo):
    return os.path.join(models_dir(root), repo.replace("/", "__"))


def lock_path(root):
    return os.path.join(models_dir(root), LOCK_NAME)


def read_lock(root):
    p = lock_path(root)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {"models": {}}


def write_lock(root, lock):
    os.makedirs(models_dir(root), exist_ok=True)
    with open(lock_path(root), "w") as f:
        json.dump(lock, f, indent=2)


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return "{:.1f}{}".format(n, unit)
        n /= 1024
    return "{:.1f}PB".format(n)


def pick_patterns(files, include_all):
    """Choose which files to fetch, preferring safetensors over legacy formats."""
    if include_all:
        return None, list(ALWAYS_SKIP)
    names = [f.lower() for f in files]
    ignore = list(DUPLICATE_FORMATS) + list(ALWAYS_SKIP)
    has_safetensors = any(n.endswith(".safetensors") for n in names)
    if has_safetensors:
        # Only drop pytorch_model*.bin when safetensors genuinely replace them;
        # some repos keep unrelated .bin assets that the model still needs.
        ignore += ["pytorch_model*.bin", "pytorch_model.bin.index.json"]
    return None, ignore


def remote_info(repo, revision, token):
    _bootstrap.ensure(["huggingface_hub"])
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    try:
        return api.model_info(repo, revision=revision, files_metadata=True)
    except Exception as e:
        msg = str(e)
        hint = ""
        if "401" in msg or "403" in msg or "gated" in msg.lower():
            hint = ("\nThis repo looks gated or private. Accept its licence on the model page, "
                    "then authenticate with 'hf auth login' or set HF_TOKEN.")
        elif "404" in msg:
            hint = "\nCheck the repo id - it may have been renamed or moved."
        sys.exit("Could not reach '{}': {}{}".format(repo, e, hint))


def do_download(root, repo, revision, token, include_all, force):
    _bootstrap.ensure(["huggingface_hub"])
    from huggingface_hub import snapshot_download

    info = remote_info(repo, revision, token)
    files = [s.rfilename for s in info.siblings]
    sizes = {s.rfilename: (s.size or 0) for s in info.siblings}
    allow, ignore = pick_patterns(files, include_all)

    dest = local_path(root, repo)
    os.makedirs(dest, exist_ok=True)

    total = sum(sizes.values())
    print("Repo      {} @ {}".format(repo, (info.sha or "?")[:12]))
    print("Files     {} ({} on the Hub)".format(len(files), human(total)))
    print("Target    {}".format(dest))
    if ignore:
        print("Skipping  {}".format(", ".join(ignore[:6]) + (" ..." if len(ignore) > 6 else "")))
    print("Downloading (resumes automatically if interrupted)...")

    t0 = time.time()
    snapshot_download(
        repo_id=repo,
        revision=revision or info.sha,
        local_dir=dest,
        allow_patterns=allow,
        ignore_patterns=ignore,
        token=token,
        force_download=force,
        max_workers=int(os.environ.get("LMS_DL_WORKERS", "8")),
    )
    elapsed = time.time() - t0

    on_disk = {}
    for dirpath, _, names in os.walk(dest):
        for n in names:
            fp = os.path.join(dirpath, n)
            rel = os.path.relpath(fp, dest)
            if rel.startswith(".cache"):
                continue
            try:
                on_disk[rel] = os.path.getsize(fp)
            except OSError:
                pass

    lock = read_lock(root)
    lock["models"][repo] = {
        "repo": repo,
        "revision": info.sha,
        "revision_pinned": bool(revision),
        "path": os.path.relpath(dest, root),
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "file_count": len(on_disk),
        "bytes_on_disk": sum(on_disk.values()),
        "files": on_disk,
        "last_modified": str(getattr(info, "lastModified", "")),
    }
    write_lock(root, lock)

    print("\nDone in {:.0f}s - {} across {} files".format(
        elapsed, human(sum(on_disk.values())), len(on_disk)))
    print("Recorded revision {} in {}".format((info.sha or "?")[:12], LOCK_NAME))
    return dest


def do_check(root, token, repo_filter=None):
    lock = read_lock(root)
    entries = lock.get("models", {})
    if not entries:
        print("Nothing downloaded yet (no {} found).".format(LOCK_NAME))
        return 0
    stale = 0
    for repo, rec in entries.items():
        if repo_filter and repo != repo_filter:
            continue
        if rec.get("revision_pinned"):
            print("{:45} pinned to {} - skipping".format(repo, (rec["revision"] or "?")[:12]))
            continue
        info = remote_info(repo, None, token)
        local_rev, remote_rev = rec.get("revision"), info.sha
        if local_rev == remote_rev:
            print("{:45} up to date ({})".format(repo, (local_rev or "?")[:12]))
        else:
            stale += 1
            print("{:45} UPDATE AVAILABLE".format(repo))
            print("{:45}   local  {}".format("", (local_rev or "?")[:12]))
            print("{:45}   remote {}  ({})".format("", (remote_rev or "?")[:12],
                                                   getattr(info, "lastModified", "")))
            print("{:45}   pull with: python3 download.py --update {}".format("", repo))
    if stale == 0:
        print("\nAll models are current.")
    return stale


def do_verify(root):
    """Confirm what is on disk still matches the lock.

    Interrupted downloads and half-deleted directories otherwise surface much
    later as an opaque container crash loop.
    """
    lock = read_lock(root)
    entries = lock.get("models", {})
    if not entries:
        print("Nothing to verify.")
        return 0
    problems = 0
    for repo, rec in entries.items():
        dest = os.path.join(root, rec["path"])
        if not os.path.isdir(dest):
            print("{:45} MISSING directory {}".format(repo, dest))
            problems += 1
            continue
        missing, wrong = [], []
        for rel, size in rec.get("files", {}).items():
            fp = os.path.join(dest, rel)
            if not os.path.exists(fp):
                missing.append(rel)
            elif size and os.path.getsize(fp) != size:
                wrong.append(rel)
        if missing or wrong:
            problems += 1
            print("{:45} {} missing, {} wrong size".format(repo, len(missing), len(wrong)))
            for f in (missing + wrong)[:5]:
                print("{:45}   {}".format("", f))
            print("{:45}   repair with: python3 download.py {}".format("", repo))
        else:
            print("{:45} OK ({} files, {})".format(
                repo, rec["file_count"], human(rec["bytes_on_disk"])))
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo", nargs="?", help="HF repo id, e.g. Qwen/Qwen3-8B")
    ap.add_argument("--root", default=".", help="project root containing models/")
    ap.add_argument("--revision", help="pin to a branch, tag or commit SHA")
    ap.add_argument("--check", action="store_true", help="report upstream changes, download nothing")
    ap.add_argument("--update", action="store_true", help="re-download at the latest revision")
    ap.add_argument("--verify", action="store_true", help="check local files against the lock")
    ap.add_argument("--all", action="store_true", help="include onnx/duplicate weight formats")
    ap.add_argument("--force", action="store_true", help="re-fetch even if files look complete")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if args.verify:
        sys.exit(1 if do_verify(root) else 0)
    if args.check:
        sys.exit(0 if do_check(root, token, args.repo) == 0 else 2)
    if not args.repo:
        ap.error("a repo id is required unless --check or --verify is used")

    if args.update:
        lock = read_lock(root)
        rec = lock.get("models", {}).get(args.repo)
        if rec:
            print("Updating {} (currently {})".format(args.repo, (rec.get("revision") or "?")[:12]))

    do_download(root, args.repo, args.revision, token, args.all, args.force)


if __name__ == "__main__":
    main()
