"""GPU timeline from NVIDIA Nsight Systems: which kernels ran, how much of each GPU's time they used, what the host was
waiting on. Works for any CUDA application; tracing adds overhead, so these runs are marked `perturbed` and never used for
A/B speed numbers. Needs no admin rights (CUPTI activity tracing)."""

from __future__ import annotations

import bisect
import glob
import os
import re
import shutil
import sqlite3
import subprocess
from collections import Counter, defaultdict

# Collectives across GPUs, so a report can tell computing from communicating.
DEFAULT_COMM_REGEX = r"nccl|allreduce|all_reduce|ggml_cuda_ar_|ar_add_kernel|ucx|ompi"


def find_nsys():
    found = shutil.which("nsys")
    if found:
        return found
    candidates = []
    for variable in ("CUDA_HOME", "CUDA_PATH"):
        if os.environ.get(variable):
            candidates.append(os.path.join(os.environ[variable], "bin", "nsys"))
    candidates += sorted(glob.glob("/usr/local/cuda*/bin/nsys"), reverse=True)
    candidates += sorted(glob.glob("/opt/nvidia/nsight-systems/*/bin/nsys"), reverse=True)
    for path in candidates:
        if os.access(path, os.X_OK):
            return path
    return None


def wrap(cmd, exe_index, run_dir, extra_args=()):
    """nsys goes right before the engine, after wrappers like taskset, so it inherits their pinning."""
    tool = find_nsys()
    if not tool:
        raise FileNotFoundError("nsys not found: put it on PATH or set CUDA_HOME")
    prefix = [tool, "profile", "-t", "cuda,nvtx", "--cuda-graph-trace=node", "--cuda-memory-usage=true", "--sample=none",
              "--cpuctxsw=none", "-f", "true", "-o", str(run_dir / "trace"), *extra_args]
    return cmd[:exe_index] + prefix + cmd[exe_index:]


def finish(run_dir, phases, wall_t0_ns, log=None):
    report = run_dir / "trace.nsys-rep"
    if not report.exists():
        return {"error": "nsys wrote no report"}
    sqlite_path = run_dir / "trace.sqlite"
    if log:
        log("exporting the nsys report")
    done = subprocess.run([find_nsys(), "export", "--type", "sqlite", "--force-overwrite", "true", "-o", str(sqlite_path),
                           str(report)], capture_output=True, text=True, check=False)
    if done.returncode != 0 or not sqlite_path.exists():
        return {"error": f"nsys export failed: {done.stderr[-400:]}"}
    return analyze(sqlite_path, phases, wall_t0_ns)


def _tables(db):
    return {row[0] for row in db.execute("select name from sqlite_master where type='table'")}


def _short(name, limit=56):
    return re.sub(r"[<(].*", "", name)[:limit]


class _DeviceTrack:
    """Busy time as the union of intervals: streams on one GPU can overlap."""

    def __init__(self):
        self.busy_end = None
        self.busy_ns = 0
        self.kernel_ns = 0
        self.comm_ns = 0
        self.copy_ns = 0
        self.kernels = 0
        self.comm_kernels = 0
        self.first = None
        self.last = None
        self.gaps = Counter()
        self.gap_ns = Counter()
        self.gap_pairs = Counter()
        self.previous = None

    def add(self, start, end, name, kind):
        if self.first is None:
            self.first = start
        self.last = max(self.last or end, end)
        if self.busy_end is None or start >= self.busy_end:
            if self.busy_end is not None:
                gap = start - self.busy_end
                bucket = "<1us" if gap < 1_000 else "1-10us" if gap < 10_000 else "10-100us" if gap < 100_000 else ">100us"
                self.gaps[bucket] += 1
                self.gap_ns[bucket] += gap
                if gap >= 100_000:
                    self.gap_pairs[f"{self.previous} -> {name}"] += gap
            self.busy_ns += end - start
            self.busy_end = end
        elif end > self.busy_end:
            self.busy_ns += end - self.busy_end
            self.busy_end = end
        self.previous = name
        if kind == "kernel":
            self.kernel_ns += end - start
            self.kernels += 1
        elif kind == "comm":
            self.comm_ns += end - start
            self.comm_kernels += 1
        else:
            self.copy_ns += end - start


def analyze(sqlite_path, phases=None, wall_t0_ns=None, comm_regex=DEFAULT_COMM_REGEX):
    """Per phase group and per device: kernels, collectives, copies, idle, biggest gaps, top kernels, host API time.
    Without phases the whole trace is one group."""
    db = sqlite3.connect(str(sqlite_path))
    db.text_factory = lambda b: b.decode("utf-8", "replace")
    tables = _tables(db)
    if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:
        return {"error": "no kernels in the trace"}
    names = dict(db.execute("select id, value from StringIds"))
    comm = re.compile(comm_regex, re.I)
    is_comm = {}

    # Phase boundaries in the trace's clock: session-relative ns, anchored to wall time.
    boundaries, groups, tokens_by_group = [], [], Counter()
    session_utc = None
    if "TARGET_INFO_SESSION_START_TIME" in tables:
        row = db.execute("select utcEpochNs from TARGET_INFO_SESSION_START_TIME").fetchone()
        session_utc = row[0] if row else None
    if phases and wall_t0_ns and session_utc:
        for phase in phases:
            boundaries.append(wall_t0_ns + int(phase["start_s"] * 1e9) - session_utc)
            groups.append(phase["group"])
            if phase.get("tokens"):
                tokens_by_group[phase["group"]] += phase["tokens"]
    wall_by_group = Counter()
    if phases:
        for phase in phases:
            wall_by_group[phase["group"]] += phase["end_s"] - phase["start_s"]

    def group_of(t):
        if not boundaries:
            return "all"
        i = bisect.bisect_right(boundaries, t) - 1
        return groups[i] if i >= 0 else "before"

    tracks = defaultdict(_DeviceTrack)  # (group, device) -> track
    kernel_time = defaultdict(lambda: Counter())  # group -> name -> ns
    kernel_count = defaultdict(lambda: Counter())

    events = []
    for device, start, end, name_id in db.execute(
            "select deviceId, start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL"):
        events.append((device, start, end, name_id, 0))
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        for device, start, end, copy_kind in db.execute(
                "select deviceId, start, end, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY"):
            events.append((device, start, end, copy_kind, 1))
    events.sort(key=lambda e: e[1])
    # Exact bytes between tiers, unlike the sampled PCIe rates: RAM to GPU, GPU to RAM, GPU to GPU.
    copies = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        kinds = dict(db.execute("select id, label from ENUM_CUDA_MEMCPY_OPER")) if _has_table(db, "ENUM_CUDA_MEMCPY_OPER") else {}
        for start, end, size, kind in db.execute("select start, end, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY"):
            entry = copies[group_of(start)][kinds.get(kind, str(kind))]
            entry[0] += size
            entry[1] += end - start
            entry[2] += 1

    for device, start, end, ident, is_copy in events:
        group = group_of(start)
        track = tracks[(group, device)]
        if is_copy:
            track.add(start, end, f"copy{ident}", "copy")
            continue
        comm_kernel = is_comm.get(ident)
        full = names.get(ident, str(ident))
        if comm_kernel is None:
            comm_kernel = is_comm[ident] = bool(comm.search(full))
        short = _short(full)
        track.add(start, end, short, "comm" if comm_kernel else "kernel")
        kernel_time[group][short] += end - start
        kernel_count[group][short] += 1

    out = {"groups": {}, "tables": sorted(t for t in tables if t.startswith("CUPTI_ACTIVITY"))[:12]}
    for group in sorted({g for g, _ in tracks}):
        devices = []
        for (g, device), track in sorted(tracks.items()):
            if g != group:
                continue
            span = (track.last - track.first) if track.first is not None else 0
            devices.append({
                "device": device,
                "span_ms": round(span / 1e6, 3),
                "kernel_ms": round(track.kernel_ns / 1e6, 3),
                "comm_ms": round(track.comm_ns / 1e6, 3),
                "copy_ms": round(track.copy_ns / 1e6, 3),
                "busy_ms": round(track.busy_ns / 1e6, 3),
                "idle_ms": round((span - track.busy_ns) / 1e6, 3),
                "kernels": track.kernels,
                "comm_kernels": track.comm_kernels,
                "gaps": {k: {"n": track.gaps[k], "ms": round(track.gap_ns[k] / 1e6, 3)} for k in track.gaps},
                "largest_gaps": [{"between": k, "ms": round(v / 1e6, 3)} for k, v in track.gap_pairs.most_common(5)],
            })
        entry = {"devices": devices}
        tokens = tokens_by_group.get(group)
        if tokens:
            entry["tokens"] = tokens
            per_device = len(devices) or 1
            entry["per_token_ms"] = {
                key: round(sum(d[key] for d in devices) / per_device / tokens, 4)
                for key in ("kernel_ms", "comm_ms", "copy_ms", "idle_ms")
            }
            entry["per_token_kernels"] = round(sum(d["kernels"] for d in devices) / per_device / tokens, 1)
        total = sum(kernel_time[group].values()) or 1
        entry["top_kernels"] = [
            {"name": name, "ms": round(ns / 1e6, 3), "share": round(ns / total, 4), "n": kernel_count[group][name],
             "avg_us": round(ns / 1e3 / max(kernel_count[group][name], 1), 2)}
            for name, ns in kernel_time[group].most_common(16)
        ]
        if group in copies:
            entry["copies"] = {kind: {"mib": round(v[0] / (1 << 20), 2), "ms": round(v[1] / 1e6, 3), "n": v[2],
                                      "mib_per_token": round(v[0] / (1 << 20) / tokens, 4) if tokens else None}
                               for kind, v in sorted(copies[group].items())}
        if group in wall_by_group:
            entry["phase_wall_s"] = round(wall_by_group[group], 4)
        out["groups"][group] = entry

    # Host side: time inside CUDA API calls, per group and thread. This shows what the CPU is blocked on.
    api = defaultdict(lambda: defaultdict(Counter))
    api_calls = defaultdict(Counter)
    for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        if table in tables:
            for tid, start, end, name_id in db.execute(f"select globalTid, start, end, nameId from {table}"):
                group = group_of(start)
                name = names.get(name_id, str(name_id)).split("_v")[0]
                api[group][tid & 0xffffff][name] += end - start
                api_calls[group][name] += 1
    for group, threads in api.items():
        if group not in out["groups"]:
            continue
        totals = Counter()
        for counts in threads.values():
            totals.update(counts)
        out["groups"][group]["host_api"] = {
            "threads": len(threads),
            "total_ms": round(sum(totals.values()) / 1e6, 3),
            "top": [{"name": name, "ms": round(ns / 1e6, 3), "n": api_calls[group][name]}
                    for name, ns in totals.most_common(8)],
        }

    if "CUDA_GPU_MEMORY_USAGE_EVENTS" in tables:
        out["gpu_allocations"] = _allocations(db, names)
    db.close()
    return out


def _allocations(db, names):
    """Peak bytes the application asked CUDA for, per device, and the largest single allocations."""
    columns = {row[1] for row in db.execute("pragma table_info(CUDA_GPU_MEMORY_USAGE_EVENTS)")}
    if not {"bytes", "memKind", "memoryOperationType", "deviceId"} <= columns:
        return {"note": "unexpected table layout"}
    kinds = dict(db.execute("select id, label from ENUM_CUDA_MEM_KIND")) if _has_table(db, "ENUM_CUDA_MEM_KIND") else {}
    ops = dict(db.execute("select id, label from ENUM_CUDA_DEV_MEM_EVENT_OPER")) if _has_table(
        db, "ENUM_CUDA_DEV_MEM_EVENT_OPER") else {}
    current, peak, sizes = Counter(), Counter(), Counter()
    for device, size, kind, op in db.execute(
            "select deviceId, bytes, memKind, memoryOperationType from CUDA_GPU_MEMORY_USAGE_EVENTS order by start"):
        label = str(ops.get(op, op)).lower()
        key = (device, kinds.get(kind, str(kind)))
        if "dealloc" in label or "free" in label or label == "1":
            current[key] -= size
        else:
            current[key] += size
            peak[key] = max(peak[key], current[key])
            sizes[size] += 1
    return {
        "peak_mib": [{"device": k[0], "kind": k[1], "mib": round(v / (1 << 20), 1)} for k, v in sorted(peak.items())],
        "largest_allocations_mib": [{"mib": round(size / (1 << 20), 1), "n": n}
                                    for size, n in sorted(sizes.items(), reverse=True)[:8]],
    }


def _has_table(db, name):
    return db.execute("select 1 from sqlite_master where type='table' and name=?", (name,)).fetchone() is not None
