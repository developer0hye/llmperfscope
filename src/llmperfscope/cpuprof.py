"""Host CPU hot spots without perf: builds and loads native/cpuprof.c, then symbolizes its samples per phase group."""

from __future__ import annotations

import bisect
import glob
import os
import shutil
import struct
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

NATIVE = Path(__file__).parent / "native"
MAGIC_NO_WEIGHT = 0x4c5053434f504531
MAGIC = 0x4c5053434f504532
# Below this share of the measured CPU time, the profile is missing something worth saying.
GOOD_COVERAGE = 0.9


def library_path():
    """Builds the shim next to the package, or in the user's cache when the package directory is read-only."""
    source = NATIVE / "cpuprof.c"
    compiler = None
    for directory in (NATIVE, Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "llmperfscope"):
        target = directory / "libllmperfscope_cpu.so"
        if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
            return target
        compiler = compiler or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
        if not compiler:
            raise FileNotFoundError("the CPU profiler needs a C compiler to build its shim once")
        try:
            directory.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=directory, suffix=".so", delete=False) as tmp:
                temporary = tmp.name
        except OSError:
            continue
        subprocess.run([compiler, "-O2", "-g", "-shared", "-fPIC", "-pthread", "-o", temporary, str(source), "-ldl", "-lrt"], check=True)
        os.replace(temporary, target)
        return target
    raise OSError("no writable directory for the CPU profiler shim")


def prepare_env(env, run_dir, hz):
    env = dict(env)
    library = str(library_path())
    existing = env.get("LD_PRELOAD")
    env["LD_PRELOAD"] = f"{library}:{existing}" if existing else library
    env["LLMPERFSCOPE_CPUPROF_OUT"] = str(Path(run_dir).resolve() / "cpuprof")
    env["LLMPERFSCOPE_CPUPROF_HZ"] = str(hz)
    return env


class _Symbols:
    """Address -> function for one mapped file, from its ELF symbol tables."""

    def __init__(self, path):
        self.path = path
        self.starts = []
        self.ends = []
        self.names = []
        self.segments = []
        if os.path.exists(path):
            self.segments = _load_segments(path)
            pairs = {}
            nm = shutil.which("nm")
            if nm:
                for extra in ([], ["-D"]):
                    done = subprocess.run([nm, "-C", "-S", "--defined-only", *extra, path], capture_output=True, text=True,
                                          check=False, errors="replace")
                    for line in done.stdout.splitlines():
                        parts = line.split(None, 3)
                        if len(parts) == 4 and parts[2] in "tTwWiI":
                            try:
                                pairs.setdefault(int(parts[0], 16), (int(parts[1], 16), parts[3]))
                            except ValueError:
                                pass
            for address in sorted(pairs):
                self.starts.append(address)
                self.ends.append(address + pairs[address][0])
                self.names.append(pairs[address][1])

    def lookup(self, file_offset):
        """Only an address inside a symbol's extent gets its name. In a stripped library such as the GPU driver, the nearest
        exported symbol before an address says nothing about the code there, and printing it would mislead."""
        address = file_offset
        for p_offset, p_vaddr, p_filesz in self.segments:
            if p_offset <= file_offset < p_offset + p_filesz:
                address = file_offset - p_offset + p_vaddr
                break
        i = bisect.bisect_right(self.starts, address) - 1
        if i >= 0 and address < self.ends[i]:
            return self.names[i]
        return None


def _load_segments(path):
    """PT_LOAD program headers of a 64-bit ELF: maps file offsets to the addresses nm prints."""
    try:
        with open(path, "rb") as f:
            header = f.read(64)
            if header[:4] != b"\x7fELF" or header[4] != 2:
                return []
            e_phoff = struct.unpack_from("<Q", header, 32)[0]
            e_phentsize, e_phnum = struct.unpack_from("<HH", header, 54)
            f.seek(e_phoff)
            table = f.read(e_phentsize * e_phnum)
    except OSError:
        return []
    segments = []
    for i in range(e_phnum):
        base = i * e_phentsize
        p_type = struct.unpack_from("<I", table, base)[0]
        if p_type == 1:
            p_offset, p_vaddr = struct.unpack_from("<QQ", table, base + 8)
            p_filesz = struct.unpack_from("<Q", table, base + 32)[0]
            segments.append((p_offset, p_vaddr, p_filesz))
    return segments


def _load_maps(path):
    ranges = []
    with open(path) as f:
        for line in f:
            parts = line.split(None, 5)
            if len(parts) == 6 and "x" in parts[1]:
                start, end = (int(x, 16) for x in parts[0].split("-"))
                ranges.append((start, end, int(parts[2], 16), parts[5].strip()))
    ranges.sort()
    return ranges


def _read_samples(path):
    """[(time ns, tid, thread name, periods it stands for, frames)], dropped count, samples per CPU second."""
    with open(path, "rb") as f:
        magic, count, dropped, max_frames, hz = struct.unpack("<5Q", f.read(40))
        if magic not in (MAGIC, MAGIC_NO_WEIGHT):
            raise ValueError("not a llmperfscope CPU profile")
        weighted = magic == MAGIC
        layout = f"<QIIII16s{max_frames}Q" if weighted else f"<QII16s{max_frames}Q"
        size = struct.calcsize(layout)
        first_frame = 6 if weighted else 4
        samples = []
        for _ in range(count):
            raw = f.read(size)
            if len(raw) < size:
                break
            values = struct.unpack(layout, raw)
            comm = values[first_frame - 1].split(b"\0", 1)[0].decode("utf-8", "replace")
            samples.append((values[0], values[1], comm, values[3] if weighted else 1, values[first_frame:first_frame + values[2]]))
    return samples, dropped, hz


def summarize(run_dir, phases, t_baseline_ns, top=20):
    """Per phase group: where the CPU samples are, by module, function and thread. For the hot functions it also says how many threads
    ran them, which is what tells serial code from parallel code."""
    files = sorted(glob.glob(str(Path(run_dir) / "cpuprof.*.bin")))
    if not files:
        return {"error": "no samples: the process was killed before it could write them, or ran no CPU"}
    starts = [t_baseline_ns + int(p["start_s"] * 1e9) for p in phases]
    groups = [p["group"] for p in phases]
    wall_by_group = Counter()
    for phase in phases:
        wall_by_group[phase["group"]] += phase["end_s"] - phase["start_s"]
    symbols = {}
    per_group = defaultdict(lambda: {"n": 0, "raw": 0, "modules": Counter(), "self": Counter(), "inclusive": Counter(), "threads": Counter(),
                                     "per_tid": Counter(), "self_tids": defaultdict(Counter), "tid_names": {}})
    total = dropped_total = 0
    period_s = 0.005
    for path in files:
        ranges = _load_maps(path[:-4] + ".maps")
        range_starts = [r[0] for r in ranges]
        samples, dropped, hz = _read_samples(path)
        dropped_total += dropped
        period_s = 1.0 / hz if hz else period_s

        def resolve(address):
            i = bisect.bisect_right(range_starts, address) - 1
            if i < 0 or address >= ranges[i][1]:
                return "[unknown]", "[unknown]"
            start, _end, offset, module_path = ranges[i]
            module = os.path.basename(module_path) or "[anon]"
            if not module_path.startswith("/"):
                return module, module
            table = symbols.get(module_path)
            if table is None:
                table = symbols[module_path] = _Symbols(module_path)
            name = table.lookup(address - start + offset)
            return module, name if name else f"{module}!?"

        cache = {}
        for t, tid, comm, weight, frames in samples:
            if not frames:
                continue
            total += weight
            # The sample arrives when a long system call returns, so it belongs to the phase where the call began. The difference only
            # matters for a call that crosses a phase boundary.
            i = bisect.bisect_right(starts, t) - 1
            bucket = per_group[groups[i] if i >= 0 else "before"]
            bucket["n"] += weight
            bucket["raw"] += 1
            bucket["threads"][_thread_label(comm)] += weight
            bucket["per_tid"][tid] += weight
            bucket["tid_names"][tid] = comm
            seen = set()
            for depth, address in enumerate(frames):
                resolved = cache.get(address)
                if resolved is None:
                    resolved = cache[address] = resolve(address)
                module, name = resolved
                if depth == 0:
                    bucket["self"][name] += weight
                    bucket["modules"][module] += weight
                    bucket["self_tids"][name][tid] += weight
                if name not in seen:
                    seen.add(name)
                    bucket["inclusive"][name] += weight
    out = {"samples": total, "dropped": dropped_total, "period_s": period_s, "groups": {}}
    for group, bucket in per_group.items():
        n = bucket["n"] or 1
        wall = wall_by_group.get(group, 0.0)
        self_rows = []
        for name, count in bucket["self"].most_common(top):
            tids = bucket["self_tids"][name]
            busiest = max(tids.values())
            row = {"name": name[:96], "share": round(count / n, 4), "cpu_s": round(count * period_s, 3), "threads": len(tids),
                   # With all threads in this function at once, the time it holds up the run is the longest any one thread spent in it.
                   "busiest_thread_s": round(busiest * period_s, 3)}
            if wall > 0:
                row["pct_of_wall_on_busiest_thread"] = round(100 * busiest * period_s / wall, 1)
            self_rows.append(row)
        out["groups"][group] = {
            "samples": bucket["n"],
            "raw_samples": bucket["raw"],
            "cpu_s": round(bucket["n"] * period_s, 3),
            "modules": [{"name": k, "share": round(v / n, 4)} for k, v in bucket["modules"].most_common(8)],
            "self": self_rows,
            "inclusive": [{"name": k[:96], "share": round(v / n, 4)} for k, v in bucket["inclusive"].most_common(top)],
            "threads": [{"name": k, "share": round(v / n, 4)} for k, v in bucket["threads"].most_common(6)],
            "busiest_threads": [{"tid": tid, "name": bucket["tid_names"][tid], "cpu_s": round(count * period_s, 3)}
                                for tid, count in bucket["per_tid"].most_common(8)],
        }
    return out


def coverage(profile, groups):
    """How much of the CPU time measured from outside the profile accounts for: the check that it is not missing something. Sets it per
    group and returns the whole-run figure, or None with too little CPU time to judge."""
    profiled = measured = 0.0
    for name, entry in (profile.get("groups") or {}).items():
        host = (groups.get(name) or {}).get("host") or {}
        cpu = (host.get("user_s") or 0.0) + (host.get("sys_s") or 0.0)
        if cpu >= 0.5:
            entry["coverage"] = round(entry["cpu_s"] / cpu, 3)
        profiled += entry["cpu_s"]
        measured += cpu
    return round(profiled / measured, 3) if measured >= 0.5 else None


def _thread_label(name):
    return "".join("#" if c.isdigit() else c for c in name)
