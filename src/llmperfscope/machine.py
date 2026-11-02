"""Machine state around a measurement. A number without its load context cannot be trusted later, so every run stores
this, and A/B arms re-check it. Thresholds scale with the core count: a daemon at 20% of one core means nothing on 256
cores and a lot on 8."""

from __future__ import annotations

import ctypes as C
import getpass
import hashlib
import mmap
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import procfs, system
from .gpu import THROTTLE_HOT, G_MEM, G_TEMP, G_THROTTLE, G_UTIL, throttle_names

BUILD_TOOLS = {"cmake", "make", "gmake", "ninja", "cc1", "cc1plus", "cicc", "ptxas", "nvcc", "clang", "clang++", "rustc",
               "cargo", "lld", "ld", "gcc", "g++", "fatbinary"}


def meminfo():
    """kB, by the names of /proc/meminfo."""
    if sys.platform == "darwin":
        return {key: value * 1024 for key, value in system.meminfo().items()}
    out = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                key, _, value = line.partition(":")
                parts = value.split()
                if parts:
                    out[key] = int(parts[0])
    except OSError:
        pass
    return out


def _cpu_scan():
    """pid -> (ticks, comm, user id)."""
    if sys.platform == "darwin":
        from . import darwin
        return darwin.foreign_cpu_scan()
    out = {}
    for pid in procfs.all_pids():
        st = procfs.stat(pid)
        if st:
            out[pid] = (st["utime"] + st["stime"], st["comm"], procfs.owner(pid))
    return out


def _ancestors():
    pids = set()
    pid = os.getpid()
    while pid > 1:
        pids.add(pid)
        st = procfs.stat(pid)
        if not st:
            break
        pid = st["ppid"]
    return pids


def cpu_load(seconds=2.0, ignore=()):
    """Foreign CPU use in cores over `seconds`, biggest first."""
    skip = _ancestors() | set(ignore)
    first = _cpu_scan()
    t0 = time.monotonic()
    time.sleep(seconds)
    second = _cpu_scan()
    dt = time.monotonic() - t0
    rows = []
    for pid, (ticks, comm, uid) in second.items():
        if pid in skip:
            continue
        before = first.get(pid)
        delta = ticks - (before[0] if before else 0) if before else 0
        cores = delta / procfs.CLK_TCK / dt
        if cores > 0:
            rows.append({"pid": pid, "comm": comm, "uid": uid, "cores": round(cores, 3)})
    rows.sort(key=lambda r: -r["cores"])
    return rows


def model_cached_fraction(path, samples=64, window=16 * 1024 * 1024):
    """How much of a file is in the page cache, estimated from evenly spaced windows. mincore does not fault anything in.
    This is what tells a cold load from a warm one."""
    try:
        size = os.path.getsize(path)
        if size == 0:
            return None
        libc = C.CDLL(None, use_errno=True)
        libc.mmap.restype = C.c_void_p
        libc.mmap.argtypes = [C.c_void_p, C.c_size_t, C.c_int, C.c_int, C.c_int, C.c_long]
        libc.munmap.argtypes = [C.c_void_p, C.c_size_t]
        libc.mincore.argtypes = [C.c_void_p, C.c_size_t, C.c_void_p]
        page = mmap.PAGESIZE
        window = min(window, size)
        step = max((size - window) // max(samples - 1, 1), 1)
        resident = total = 0
        fd = os.open(path, os.O_RDONLY)
        try:
            for i in range(samples):
                offset = min(i * step, size - window) // page * page
                length = min(window, size - offset)
                base = libc.mmap(None, length, mmap.PROT_READ, mmap.MAP_SHARED, fd, offset)
                if base is None or base == C.c_void_p(-1).value:
                    return None
                try:
                    pages = (length + page - 1) // page
                    vector = (C.c_ubyte * pages)()
                    if libc.mincore(base, length, vector) != 0:
                        return None
                    resident += sum(v & 1 for v in vector)
                    total += pages
                finally:
                    libc.munmap(base, length)
                if size <= window:
                    break
        finally:
            os.close(fd)
        return round(resident / total, 3) if total else None
    except (OSError, ValueError, AttributeError):
        return None


def evict_from_page_cache(path):
    """Asks the kernel to drop a file's clean pages, so the next load is a cold one. Needs no root, unlike drop_caches, and
    touches nothing but this file. Pages another process has mapped stay. Returns the cached fraction afterwards."""
    if sys.platform == "darwin":
        from . import darwin
        darwin.evict_from_page_cache(path)
        return model_cached_fraction(path)
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    return model_cached_fraction(path)


def host_info():
    info = {"hostname": platform.node(), "kernel": platform.release(), "cpus": os.cpu_count(), "user": getpass.getuser()}
    if sys.platform == "darwin":
        from . import darwin
        info["os"] = f"macOS {platform.mac_ver()[0]}"
        info["cpu"] = darwin.sysctl_text("machdep.cpu.brand_string")
        info["performance_cores"] = darwin.sysctl_int("hw.perflevel0.logicalcpu")
        info["efficiency_cores"] = darwin.sysctl_int("hw.perflevel1.logicalcpu")
        return info
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        info["numa_nodes"] = sum(1 for d in os.listdir("/sys/devices/system/node") if d.startswith("node"))
    except OSError:
        pass
    for key, path in (("governor", "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
                      ("thp", "/sys/kernel/mm/transparent_hugepage/enabled")):
        try:
            with open(path) as f:
                info[key] = f.read().strip()
        except OSError:
            pass
    return info


def snapshot(backend, cpu_seconds=2.0, model=None, ignore=(), gpus=None):
    """The state before a run, with a verdict: `clean` plus the reasons when it is not."""
    cpus = os.cpu_count() or 1
    # A Mac in a desktop session never gets below these: the window server, the terminal, an animated wallpaper.
    desktop = sys.platform == "darwin"
    single_limit = max(0.25 if desktop else 0.10, 0.01 * cpus)
    total_limit = max(1.0 if desktop else 0.20, 0.02 * cpus)
    load = cpu_load(cpu_seconds, ignore) if cpu_seconds > 0 else []
    total = sum(r["cores"] for r in load)
    me = os.getuid()
    builds = [r for r in load if r["comm"] in BUILD_TOOLS and r["uid"] == me]
    mem = meminfo()
    state = {
        "t_wall": time.time(),
        "foreign_cpu_cores": round(total, 3),
        "top_processes": load[:8],
        "mem_available_mib": mem.get("MemAvailable", 0) // 1024,
        "mem_total_mib": mem.get("MemTotal", 0) // 1024,
        "page_cache_mib": mem.get("Cached", 0) // 1024,
        "swap_used_mib": (mem.get("SwapTotal", 0) - mem.get("SwapFree", 0)) // 1024,
        "loadavg": os.getloadavg() if hasattr(os, "getloadavg") else None,
    }
    reasons = []
    if builds:
        reasons.append("build running: " + ", ".join(sorted({r["comm"] for r in builds})))
    big = [r for r in load if r["cores"] > single_limit]
    if big:
        reasons.append(f"foreign process above {single_limit:.2f} cores: " +
                       ", ".join(f'{r["comm"]}={r["cores"]:.2f}' for r in big[:4]))
    if total > total_limit:
        reasons.append(f"foreign CPU {total:.2f} cores above {total_limit:.2f}")
    if state["mem_total_mib"] and state["mem_available_mib"] < 0.2 * state["mem_total_mib"]:
        reasons.append(f'host memory available {state["mem_available_mib"]} MiB is under 20%')

    rows = backend.fast()
    slow = backend.slow()
    state["gpus"] = []
    for i, row in enumerate(rows):
        foreign = {pid: mib for pid, mib in slow[i]["procs"].items() if pid not in ignore}
        entry = {
            "index": backend.devices[i]["index"],
            "mem_used_mib": row[G_MEM],
            "util": row[G_UTIL],
            "temp_c": row[G_TEMP],
            "throttle": throttle_names(row[G_THROTTLE] or 0),
            "compute_processes": foreign,
        }
        state["gpus"].append(entry)
        if gpus is not None and backend.devices[i]["index"] not in gpus:
            continue
        if backend.name == "apple":
            # Everything that draws on the screen holds the GPU open, so what counts here is who is using it.
            busy = {pid: share for pid, share in backend.foreign_busy().items() if pid not in ignore} if cpu_seconds > 0 else {}
            entry["foreign_gpu_pct"] = busy
            # An animated wallpaper alone takes 13%.
            if sum(busy.values()) > 25:
                reasons.append(f"GPU {entry['index']} {sum(busy.values()):.0f}% busy with processes {sorted(busy, key=busy.get, reverse=True)[:4]}")
        else:
            if foreign:
                reasons.append(f"GPU {entry['index']} has compute processes: {sorted(foreign)}")
            if row[G_UTIL] and row[G_UTIL] > 5:
                reasons.append(f"GPU {entry['index']} busy at {row[G_UTIL]}%")
        if row[G_THROTTLE] and row[G_THROTTLE] & THROTTLE_HOT:
            reasons.append(f"GPU {entry['index']} throttled: {throttle_names(row[G_THROTTLE])}")
    if model:
        state["model_page_cache_fraction"] = model_cached_fraction(model)
        state["model_storage"] = system.file_location(model)
    state["clean"] = not reasons
    state["reasons"] = reasons
    return state


def wait_until_quiet(backend, max_wait_s=3600, poll_s=20.0, model=None, ignore=(), gpus=None, log=None):
    """Waits for two consecutive clean samples, as the timing rule asks. Returns the last state; it may still be dirty."""
    deadline = time.monotonic() + max_wait_s
    clean_in_a_row = 0
    last_message = None
    while True:
        state = snapshot(backend, model=model, ignore=ignore, gpus=gpus)
        if state["clean"]:
            clean_in_a_row += 1
            if clean_in_a_row >= 2:
                return state
            continue
        clean_in_a_row = 0
        if time.monotonic() > deadline:
            return state
        message = "machine busy, waiting: " + "; ".join(state["reasons"])
        if log and message != last_message:
            log(message)
            last_message = message
        time.sleep(poll_s)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def binary_identity(exe):
    """Which code ran: hash of the executable and of the libraries next to it, plus the git commit when there is one."""
    resolved = shutil.which(exe) or exe
    path = Path(resolved)
    if not path.exists():
        return {"exe": exe}
    path = path.resolve()
    out = {"exe": str(path), "sha256": _sha256(path)}
    libs = {}
    for lib in sorted(path.parent.glob("lib*.so*")):
        if lib.is_file() and not lib.is_symlink():
            libs[lib.name] = _sha256(lib)
    if libs:
        out["libs"] = libs
    try:
        commit = subprocess.run(["git", "-C", str(path.parent), "rev-parse", "HEAD"], capture_output=True, text=True,
                                timeout=20)
        if commit.returncode == 0:
            out["git_commit"] = commit.stdout.strip()
            dirty = subprocess.run(["git", "-C", str(path.parent), "status", "--porcelain"], capture_output=True, text=True,
                                   timeout=60)
            out["git_dirty_files"] = len(dirty.stdout.splitlines()) if dirty.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        pass
    return out


ENV_PREFIXES = ("NCCL_", "GGML_", "LLAMA_", "CUDA_", "OMP_", "KMP_", "MKL_", "VLLM_", "SGLANG_", "OLLAMA_", "PYTORCH_",
                "TORCH_", "HF_HUB_OFFLINE", "TOKENIZERS_PARALLELISM", "LD_PRELOAD")
_SECRET_WORDS = ("TOKEN", "KEY", "SECRET", "PASSWORD", "CREDENTIAL")


def relevant_env(env):
    """The environment that changes engine behaviour. Anything that looks like a credential stays out of the report."""
    return {k: v for k, v in sorted(env.items())
            if k.startswith(ENV_PREFIXES) and not any(word in k.upper() for word in _SECRET_WORDS)}


def redact_command(cmd):
    out = []
    hide_next = False
    for token in cmd:
        if hide_next:
            out.append("<redacted>")
            hide_next = False
            continue
        lower = token.lower()
        if lower.startswith("--") and any(word.lower() in lower for word in _SECRET_WORDS):
            if "=" in token:
                out.append(token.split("=", 1)[0] + "=<redacted>")
            else:
                out.append(token)
                hide_next = True
            continue
        out.append(token)
    return out
