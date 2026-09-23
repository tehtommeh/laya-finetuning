"""Find an interpreter that has the required packages, without touching system Python.

Hosts differ wildly: some have huggingface_hub globally, some only inside a uv
tool venv, some are PEP 668 "externally managed" where pip install just fails.
Rather than make the caller solve that, a script calls ensure() and gets
re-executed under a suitable interpreter if the current one falls short.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys


def _have(mods):
    return all(importlib.util.find_spec(m) is not None for m in mods)


def ensure(modules, pip_names=None):
    """Guarantee `modules` are importable, re-executing this process if needed.

    Returns once the modules are available in the running interpreter.
    """
    if _have(modules):
        return
    pip_names = pip_names or modules
    if os.environ.get("_LMS_BOOTSTRAPPED") == "1":
        sys.exit(
            "Missing Python packages: {}\n"
            "Install them with one of:\n"
            "  uv pip install --system {}\n"
            "  pipx runpip <tool> install {}\n"
            "  python3 -m pip install --user {}".format(
                ", ".join(pip_names), " ".join(pip_names),
                " ".join(pip_names), " ".join(pip_names))
        )

    env = dict(os.environ, _LMS_BOOTSTRAPPED="1")

    # uv is the fastest path and needs no pre-existing venv.
    uv = shutil.which("uv")
    if uv:
        # "python" must stay literal: uv resolves it to the interpreter inside the
        # ephemeral venv it just built. An absolute path would bypass that venv
        # and land back on the interpreter that is missing the packages.
        cmd = [uv, "run", "--quiet", "--no-project"]
        for name in pip_names:
            cmd += ["--with", name]
        cmd += ["python", os.path.abspath(sys.argv[0])] + sys.argv[1:]
        try:
            rc = subprocess.call(cmd, env=env)
            if rc == 0:
                sys.exit(0)
        except OSError:
            pass

    # Otherwise look for any interpreter on PATH that already has the modules,
    # including the venvs behind installed CLI tools like `hf`.
    candidates = []
    for exe in ("python3", "python"):
        p = shutil.which(exe)
        if p:
            candidates.append(p)
    for tool in ("hf", "huggingface-cli"):
        p = shutil.which(tool)
        if p:
            venv_py = os.path.join(os.path.dirname(os.path.realpath(p)), "python")
            if os.path.exists(venv_py):
                candidates.append(venv_py)
    for base in ("~/.local/share/uv/tools", "~/.local/pipx/venvs"):
        base = os.path.expanduser(base)
        if os.path.isdir(base):
            for entry in sorted(os.listdir(base)):
                venv_py = os.path.join(base, entry, "bin", "python")
                if os.path.exists(venv_py):
                    candidates.append(venv_py)

    probe = "import importlib.util,sys; sys.exit(0 if all(importlib.util.find_spec(m) for m in {!r}) else 1)".format(list(modules))
    seen = set()
    for py in candidates:
        rp = os.path.realpath(py)
        if rp in seen or rp == os.path.realpath(sys.executable):
            continue
        seen.add(rp)
        try:
            if subprocess.call([py, "-c", probe], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
                sys.exit(subprocess.call([py, sys.argv[0]] + sys.argv[1:], env=env))
        except OSError:
            continue

    sys.exit(
        "Missing Python packages: {}\n"
        "No interpreter on this host has them and uv is not installed. Fix with:\n"
        "  python3 -m pip install --user {}\n"
        "  (or install uv: curl -LsSf https://astral.sh/uv/install.sh | sh)".format(
            ", ".join(pip_names), " ".join(pip_names))
    )
