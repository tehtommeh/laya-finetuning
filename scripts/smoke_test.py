#!/usr/bin/env python3
"""Run the checks declared in stack.json against a running stack.

Uses only the standard library so it works on any host and inside any container
without an install step. The scaffolder writes stack.json; this turns "I think
it's up" into a pass/fail table you can paste into a report.

Usage:
    python3 smoke_test.py                      # uses ./stack.json
    python3 smoke_test.py --wait 300           # wait for health before testing
    python3 smoke_test.py --json
    python3 smoke_test.py --only "chat"        # run matching checks only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
    GREEN = RED = YELLOW = DIM = RESET = ""


def resolve(template, services):
    out = template
    for name, base in services.items():
        out = out.replace("{" + name + "}", base.rstrip("/"))
    return out


def dig_path(obj, path):
    """Walk a dotted path like 'choices.0.message.content'."""
    cur = obj
    for part in path.split("."):
        if part == "":
            continue
        if isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return None, False
        elif isinstance(cur, dict):
            if part not in cur:
                return None, False
            cur = cur[part]
        else:
            return None, False
    return cur, True


def request(url, method="GET", payload=None, headers=None, timeout=60):
    data = None
    hdrs = dict(headers or {})
    if payload is not None:
        data = json.dumps(payload).encode()
        hdrs.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
            return r.status, body, time.time() - t0, None
    except urllib.error.HTTPError as e:
        return e.code, e.read(), time.time() - t0, None
    except Exception as e:
        return None, b"", time.time() - t0, "{}: {}".format(type(e).__name__, e)


def wait_for_health(stack, budget):
    """Poll health endpoints until they answer or the budget runs out.

    Large models can take minutes to load; failing the suite because the server
    was still warming up produces false alarms and wasted debugging.
    """
    services = stack["services"]
    targets = [(n, resolve(u, services)) for n, u in stack.get("health", {}).items()]
    if not targets:
        return True
    deadline = time.time() + budget
    pending = dict(targets)
    print("Waiting up to {}s for: {}".format(budget, ", ".join(pending)))
    last_note = {}
    while pending and time.time() < deadline:
        for name, url in list(pending.items()):
            status, _, _, err = request(url, timeout=10)
            if status and 200 <= status < 300:
                print("  {}{} ready{} after {:.0f}s".format(
                    GREEN, name, RESET, budget - (deadline - time.time())))
                pending.pop(name)
            elif err and last_note.get(name) != err:
                last_note[name] = err
        if pending:
            time.sleep(3)
    for name in pending:
        print("  {}{} never became healthy{} (last: {})".format(
            RED, name, RESET, last_note.get(name, "no response")))
    return not pending


def evaluate(check, status, body):
    """Return (ok, detail). Checks are deliberately shallow but specific."""
    expect = check.get("expect_status", 200)
    if status != expect:
        snippet = body[:200].decode("utf-8", "replace").replace("\n", " ")
        return False, "HTTP {} (wanted {}) {}".format(status, expect, snippet)

    parsed = None
    if body:
        try:
            parsed = json.loads(body)
        except ValueError:
            parsed = None

    path = check.get("expect_json_path")
    value = None
    if path:
        if parsed is None:
            return False, "response was not JSON"
        value, found = dig_path(parsed, path)
        if not found:
            return False, "missing json path '{}'".format(path)
        if value in (None, "", [], {}):
            return False, "json path '{}' was empty".format(path)
        if "expect_equals" in check and value != check["expect_equals"]:
            return False, "json path '{}' was {!r}, wanted {!r}".format(path, value, check["expect_equals"])

    needle = check.get("expect_contains")
    if needle:
        hay = json.dumps(parsed) if parsed is not None else body.decode("utf-8", "replace")
        if needle.lower() not in hay.lower():
            return False, "response did not contain '{}'".format(needle)

    ctype = check.get("expect_content_type")
    if ctype and parsed is None and not body:
        return False, "empty body, expected {}".format(ctype)

    if value is not None:
        text = str(value).replace("\n", " ")
        return True, text[:90] + ("..." if len(text) > 90 else "")
    if parsed is not None:
        keys = list(parsed)[:5] if isinstance(parsed, dict) else "list[{}]".format(len(parsed))
        return True, "ok {}".format(keys)
    return True, "ok ({} bytes)".format(len(body))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stack", default="stack.json")
    ap.add_argument("--wait", type=int, default=0, help="seconds to wait for health first")
    ap.add_argument("--only", help="run only checks whose name contains this")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.stack):
        sys.exit("No {} found - run this from the project root.".format(args.stack))
    with open(args.stack) as f:
        stack = json.load(f)

    services = stack["services"]
    if args.wait and not wait_for_health(stack, args.wait):
        print("\n{}Services never became healthy.{} Inspect logs with: docker compose logs".format(RED, RESET))
        sys.exit(1)

    checks = stack.get("checks", [])
    if args.only:
        checks = [c for c in checks if args.only.lower() in c["name"].lower()]
    if not checks:
        sys.exit("No checks to run.")

    print("\n{:<34} {:>7} {:>8}  {}".format("CHECK", "STATUS", "TIME", "DETAIL"))
    print("-" * 100)

    results, failures = [], 0
    for check in checks:
        url = resolve(check["url"], services)
        status, body, elapsed, err = request(
            url,
            method=check.get("method", "GET"),
            payload=check.get("json"),
            headers=check.get("headers"),
            timeout=check.get("timeout", 60),
        )
        if err:
            ok, detail = False, err
        else:
            ok, detail = evaluate(check, status, body)

        optional = check.get("optional", False)
        if not ok and optional:
            mark, colour = "SKIP", YELLOW
        elif ok:
            mark, colour = "PASS", GREEN
        else:
            mark, colour = "FAIL", RED
            failures += 1

        print("{:<34} {}{:>7}{} {:>7.2f}s  {}{}{}".format(
            check["name"][:34], colour, mark, RESET, elapsed,
            DIM if ok else "", detail[:60], RESET))
        results.append({"name": check["name"], "url": url, "status": status,
                        "ok": ok, "optional": optional, "seconds": round(elapsed, 2),
                        "detail": detail})

    passed = sum(1 for r in results if r["ok"])
    print("-" * 100)
    print("{}/{} passed{}".format(passed, len(results),
                                  ", {} failed".format(failures) if failures else ""))

    if args.json:
        print(json.dumps({"passed": passed, "total": len(results),
                          "failed": failures, "results": results}, indent=2))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
