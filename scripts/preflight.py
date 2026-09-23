#!/usr/bin/env python3
"""Probe the host for everything the stack depends on, and emit a machine profile.

Everything downstream (serving engine, dtype, quantization, compose syntax, GPU
wiring) is derived from this profile rather than assumed, so the same skill
produces a working stack on a 3090 box, a dual-A100 server, a Mac, or a
CPU-only VM.

Usage:
    python3 preflight.py                 # human-readable report
    python3 preflight.py --json          # machine-readable profile
    python3 preflight.py --json -o host_profile.json
    python3 preflight.py --require-gpu   # exit 1 if no usable GPU
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys

# Compute capability -> what the hardware can actually do. Getting this wrong is
# the single most common cause of "works on my machine" failures, because vLLM
# and transformers both happily accept flags the GPU cannot execute.
CC_FEATURES = [
    # (min_cc, name, bf16, fp8_native, flash_attn, marlin_int4, vllm_ok)
    (9.0, "Hopper/Blackwell", True, True, True, True, True),
    (8.9, "Ada Lovelace", True, True, True, True, True),
    (8.0, "Ampere", True, False, True, True, True),
    (7.5, "Turing", False, False, True, True, True),
    (7.0, "Volta", False, False, False, False, True),
    (6.0, "Pascal", False, False, False, False, False),
    (0.0, "pre-Pascal", False, False, False, False, False),
]


def run(cmd, timeout=25):
    """Run a command, returning (ok, stdout). Never raises."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode == 0, (p.stdout or "") + (p.stderr or "" if p.returncode else "")
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False, ""


def cc_features(cc: float) -> dict:
    for min_cc, name, bf16, fp8, fa, marlin, vllm_ok in CC_FEATURES:
        if cc >= min_cc:
            return {
                "arch_name": name,
                "bf16": bf16,
                "fp8_native": fp8,
                "flash_attention": fa,
                "marlin_int4": marlin,
                "vllm_supported": vllm_ok,
            }
    return {}


def probe_nvidia() -> list:
    ok, out = run([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,compute_cap,driver_version",
        "--format=csv,noheader,nounits",
    ])
    if not ok or not out.strip():
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            cc = float(parts[4])
        except ValueError:
            cc = 0.0
        total = int(float(parts[2]))
        used = int(float(parts[3]))
        gpu = {
            "index": int(parts[0]),
            "name": parts[1],
            "vram_total_mb": total,
            "vram_used_mb": used,
            "vram_free_mb": total - used,
            "compute_capability": cc,
            "driver_version": parts[5],
        }
        gpu.update(cc_features(cc))
        gpus.append(gpu)
    return gpus


def probe_rocm() -> list:
    """AMD GPUs. vLLM has ROCm builds but images differ, so flag it clearly."""
    if not shutil.which("rocm-smi"):
        return []
    ok, out = run(["rocm-smi", "--showproductname", "--csv"])
    if not ok:
        return []
    gpus = []
    for line in out.strip().splitlines()[1:]:
        if not line.strip() or "card" not in line.lower():
            continue
        gpus.append({"name": line.split(",")[-1].strip(), "vendor": "amd"})
    return gpus


def probe_cuda_runtime() -> str | None:
    """Max CUDA version the driver supports - caps which base images can run."""
    ok, out = run(["nvidia-smi"])
    if ok:
        m = re.search(r"CUDA Version:\s*([0-9.]+)", out)
        if m:
            return m.group(1)
    return None


def probe_docker() -> dict:
    info = {
        "installed": False,
        "version": None,
        "daemon_running": False,
        "compose_command": None,
        "compose_version": None,
        "gpu_runtime": False,
        "gpu_runtime_verified": False,
        "notes": [],
    }
    ok, out = run(["docker", "--version"])
    if not ok:
        info["notes"].append("docker CLI not found")
        return info
    info["installed"] = True
    m = re.search(r"([0-9]+\.[0-9]+\.[0-9]+)", out)
    info["version"] = m.group(1) if m else out.strip()

    ok, _ = run(["docker", "info"], timeout=30)
    info["daemon_running"] = ok
    if not ok:
        info["notes"].append("docker daemon not reachable (permissions or not started)")

    # Compose v2 plugin is the norm; fall back to the legacy standalone binary.
    ok, out = run(["docker", "compose", "version"])
    if ok:
        info["compose_command"] = "docker compose"
        info["compose_version"] = out.strip().splitlines()[0] if out.strip() else None
    elif shutil.which("docker-compose"):
        ok, out = run(["docker-compose", "version"])
        if ok:
            info["compose_command"] = "docker-compose"
            info["compose_version"] = out.strip().splitlines()[0] if out.strip() else None
            info["notes"].append("legacy docker-compose v1: use 'deploy.resources' GPU syntax, not 'gpus:'")
    if not info["compose_command"]:
        info["notes"].append("no compose found - install the docker compose plugin")

    if info["daemon_running"]:
        ok, out = run(["docker", "info", "--format", "{{json .Runtimes}}"])
        if ok and "nvidia" in out:
            info["gpu_runtime"] = True
        else:
            ok2, out2 = run(["docker", "info"], timeout=30)
            info["gpu_runtime"] = ok2 and "nvidia" in out2.lower()
        if not info["gpu_runtime"]:
            info["notes"].append(
                "nvidia container runtime not registered - install nvidia-container-toolkit "
                "then 'sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker'"
            )
    return info


def verify_gpu_in_docker(cuda_version: str | None) -> tuple[bool, str]:
    """Actually run a container and ask it for the GPU.

    A registered runtime is not proof it works; driver/toolkit mismatches only
    show up at container start, and finding out now is far cheaper than finding
    out after a 30GB download.
    """
    tag = "12.4.1-base-ubuntu22.04"
    if cuda_version:
        try:
            if float(cuda_version.split(".")[0]) < 12:
                tag = "11.8.0-base-ubuntu22.04"
        except ValueError:
            pass
    image = f"nvidia/cuda:{tag}"
    ok, out = run(
        ["docker", "run", "--rm", "--gpus", "all", image, "nvidia-smi", "-L"],
        timeout=300,
    )
    if ok:
        return True, out.strip()
    ok, out = run(
        ["docker", "run", "--rm", "--runtime", "nvidia", image, "nvidia-smi", "-L"],
        timeout=300,
    )
    return ok, out.strip()[:400]


def probe_ports(ports) -> dict:
    """Find which of the wanted ports are free, and suggest alternatives.

    A port collision surfaces only at `docker compose up`, after the images are
    built and the weights downloaded, so it is worth catching up front.
    """
    import socket

    def free(port):
        # Check both stacks: a v6-only listener still blocks a v4 bind.
        for family, addr in ((socket.AF_INET, "0.0.0.0"), (socket.AF_INET6, "::")):
            s = socket.socket(family, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((addr, port))
            except OSError:
                s.close()
                return False
            except Exception:
                pass
            finally:
                try:
                    s.close()
                except OSError:
                    pass
        return True

    result = {}
    for name, port in ports.items():
        port = int(port)
        if free(port):
            result[name] = {"port": port, "available": True}
            continue
        holder = None
        ok, out = run(["docker", "ps", "--format", "{{.Names}} {{.Ports}}"])
        if ok:
            for line in out.splitlines():
                if ":{}->".format(port) in line:
                    holder = line.split()[0]
                    break
        alt = next((p for p in range(port + 1, port + 60) if free(p)), None)
        result[name] = {"port": port, "available": False,
                        "held_by": holder, "suggested": alt}
    return result


def probe_disk(path: str) -> dict:
    target = os.path.abspath(path)
    while not os.path.exists(target) and target != "/":
        target = os.path.dirname(target)
    usage = shutil.disk_usage(target)
    return {
        "checked_path": target,
        "free_gb": round(usage.free / 1e9, 1),
        "total_gb": round(usage.total / 1e9, 1),
    }


def probe_hf() -> dict:
    info = {"cli": shutil.which("hf") or shutil.which("huggingface-cli"), "token": False, "user": None}
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        info["token"] = True
    token_file = os.path.expanduser(
        os.path.join(os.environ.get("HF_HOME", "~/.cache/huggingface"), "token")
    )
    if os.path.exists(os.path.expanduser(token_file)):
        info["token"] = True
    if info["cli"] and info["token"]:
        ok, out = run([info["cli"], "auth", "whoami"], timeout=20)
        if ok:
            info["user"] = out.strip().splitlines()[0] if out.strip() else None
    info["python_hub"] = False
    try:
        import huggingface_hub  # noqa: F401
        info["python_hub"] = True
    except ImportError:
        pass
    return info


def build_profile(workdir: str, verify_docker_gpu: bool, ports=None) -> dict:
    gpus = probe_nvidia()
    amd = probe_rocm()
    cuda = probe_cuda_runtime()
    docker = probe_docker()

    if verify_docker_gpu and gpus and docker["daemon_running"] and docker["gpu_runtime"]:
        ok, detail = verify_gpu_in_docker(cuda)
        docker["gpu_runtime_verified"] = ok
        if not ok:
            docker["notes"].append(f"GPU passthrough test failed: {detail}")

    total_vram = sum(g["vram_total_mb"] for g in gpus)
    free_vram = sum(g["vram_free_mb"] for g in gpus)
    homogeneous = len({g["name"] for g in gpus}) <= 1

    profile = {
        "host": {
            "os": platform.system(),
            "release": platform.release(),
            "arch": platform.machine(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "gpu": {
            "vendor": "nvidia" if gpus else ("amd" if amd else "none"),
            "count": len(gpus) or len(amd),
            "devices": gpus or amd,
            "total_vram_mb": total_vram,
            "free_vram_mb": free_vram,
            "homogeneous": homogeneous,
            "driver_cuda_version": cuda,
            # Tensor parallelism requires a power-of-two device count in vLLM,
            # and identical cards - anything else needs pipeline parallel or one GPU.
            "tensor_parallel_max": (
                max([n for n in (1, 2, 4, 8) if n <= len(gpus)], default=1)
                if homogeneous else 1
            ),
        },
        "docker": docker,
        "disk": probe_disk(workdir),
        "huggingface": probe_hf(),
        "ports": probe_ports(ports or {"api": 8000, "frontend": 7860}),
    }

    caps = profile["gpu"]
    dev0 = (gpus or [{}])[0]
    profile["capabilities"] = {
        "can_run_gpu_containers": bool(
            gpus and docker["daemon_running"] and docker["gpu_runtime"]
        ),
        "vllm_viable": bool(gpus and dev0.get("vllm_supported") and platform.machine() in ("x86_64", "aarch64")),
        "bf16": bool(dev0.get("bf16")),
        "fp8_native": bool(dev0.get("fp8_native")),
        "flash_attention": bool(dev0.get("flash_attention")),
        "int4_marlin": bool(dev0.get("marlin_int4")),
        "recommended_dtype": (
            "bfloat16" if dev0.get("bf16") else ("float16" if gpus else "float32")
        ),
        "tensor_parallel_size": caps["tensor_parallel_max"],
    }

    blockers = []
    if not docker["installed"]:
        blockers.append("Docker is not installed")
    elif not docker["daemon_running"]:
        blockers.append("Docker daemon is not reachable")
    if not docker["compose_command"]:
        blockers.append("Docker Compose is not available")
    if not gpus and not amd:
        blockers.append("No GPU detected - the stack will run on CPU and be very slow")
    elif gpus and not docker["gpu_runtime"]:
        blockers.append("GPU present but Docker cannot use it (nvidia-container-toolkit missing)")
    for name, info in profile["ports"].items():
        if not info["available"]:
            blockers.append(
                "Port {} ({}) is in use{} - use {} instead".format(
                    info["port"], name,
                    " by " + info["held_by"] if info["held_by"] else "",
                    info["suggested"]))
    if profile["disk"]["free_gb"] < 30:
        blockers.append(f"Only {profile['disk']['free_gb']}GB free at {profile['disk']['checked_path']}")
    profile["blockers"] = blockers
    return profile


def report(p: dict) -> str:
    L = []
    h, g, d = p["host"], p["gpu"], p["docker"]
    L.append(f"Host       {h['os']} {h['release']} ({h['arch']}), {h['cpu_count']} cores")
    if g["devices"] and g["vendor"] == "nvidia":
        for dev in g["devices"]:
            L.append(
                f"GPU {dev['index']}      {dev['name']} - {dev['vram_total_mb']}MB "
                f"({dev['vram_free_mb']}MB free), CC {dev['compute_capability']} "
                f"[{dev.get('arch_name','?')}], driver {dev['driver_version']}"
            )
        L.append(f"CUDA       driver supports up to {g['driver_cuda_version']}")
        c = p["capabilities"]
        feats = [k for k in ("bf16", "fp8_native", "flash_attention", "int4_marlin") if c[k]]
        L.append(f"Features   {', '.join(feats) or 'none'} | dtype={c['recommended_dtype']} | TP={c['tensor_parallel_size']}")
    elif g["vendor"] == "amd":
        L.append(f"GPU        {g['count']} AMD device(s) - needs ROCm images, not the CUDA path")
    else:
        L.append("GPU        none detected")
    L.append(
        f"Docker     {d['version'] or 'missing'} | daemon={'up' if d['daemon_running'] else 'down'} "
        f"| {d['compose_command'] or 'no compose'} | gpu-runtime="
        f"{'verified' if d['gpu_runtime_verified'] else ('registered' if d['gpu_runtime'] else 'no')}"
    )
    L.append(f"Disk       {p['disk']['free_gb']}GB free at {p['disk']['checked_path']}")
    port_bits = []
    for name, info in p["ports"].items():
        port_bits.append("{}={} {}".format(
            name, info["port"],
            "free" if info["available"] else "TAKEN->try {}".format(info["suggested"])))
    L.append("Ports      " + ", ".join(port_bits))
    hf = p["huggingface"]
    hf_cli = "yes" if hf["cli"] else "no"
    if hf["token"]:
        hf_tok = "yes ({})".format(hf["user"]) if hf["user"] else "yes"
    else:
        hf_tok = "no"
    L.append("HF         cli={} token={}".format(hf_cli, hf_tok))
    for note in d["notes"]:
        L.append(f"  note: {note}")
    if p["blockers"]:
        L.append("")
        L.append("BLOCKERS:")
        for b in p["blockers"]:
            L.append(f"  - {b}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    ap.add_argument("-o", "--output", help="also write the JSON profile here")
    ap.add_argument("--workdir", default=".", help="path whose filesystem to check for free space")
    ap.add_argument("--no-docker-gpu-test", action="store_true",
                    help="skip actually starting a container to verify GPU passthrough")
    ap.add_argument("--require-gpu", action="store_true", help="exit non-zero if no usable GPU")
    ap.add_argument("--ports", default="api=8000,frontend=7860",
                    help="comma-separated name=port pairs to check for conflicts")
    args = ap.parse_args()

    ports = {}
    for pair in args.ports.split(","):
        if "=" in pair:
            name, _, port = pair.partition("=")
            ports[name.strip()] = int(port)

    p = build_profile(args.workdir, verify_docker_gpu=not args.no_docker_gpu_test, ports=ports)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(p, f, indent=2)
    print(json.dumps(p, indent=2) if args.json else report(p))

    if args.require_gpu and not p["capabilities"]["can_run_gpu_containers"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
