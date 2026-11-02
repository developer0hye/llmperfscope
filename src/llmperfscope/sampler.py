"""The sampling threads: GPU telemetry, the target's process tree and system-wide data movement, written to a gzip JSONL file as it
goes.

One tick per line: {"t": CLOCK_MONOTONIC ns, "g": [per-device fast row], "h": [host row], "s": {slow data, on slow ticks}}.
Cumulative host counters only ever grow: a process that has exited keeps its last values, so nothing is lost to it.
"""

from __future__ import annotations

import gzip
import json
import threading
import time

from . import procfs
from .system import SystemSampler

# Columns of the host row.
H_ANON, H_FILE, H_SHMEM, H_HWM, H_SWAP, H_PIN, H_THREADS, H_PROCS, H_UTICKS, H_STICKS, H_MAJFLT, H_MINFLT = range(12)


class BlockingQueries(threading.Thread):
    """Driver queries that block: the energy counter (milliseconds, up to 250 ms on an idle GPU) and PCIe throughput (20 ms each). They
    get a thread of their own so the fast samples keep their rate, and once the target's GPUs are known only those are read."""

    def __init__(self, backend, period_s=0.5, pcie=True):
        super().__init__(name="llmperfscope-driver-queries", daemon=True)
        self.backend = backend
        self.period_s = period_s
        self.want_pcie = pcie
        self.used = set()
        self.pcie = [[None, None] for _ in backend.devices]
        self._energy = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def drain_energy(self):
        with self._lock:
            out, self._energy = self._energy, []
        return out

    def read_energy(self):
        for index in sorted(self.used) or range(len(self.backend.devices)):
            before = time.monotonic_ns()
            value = self.backend.energy_mj(index)
            if value is not None:
                with self._lock:
                    self._energy.append([index, (before + time.monotonic_ns()) // 2, value])

    def run(self):
        cycle = 0
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            self.read_energy()
            if self.want_pcie and cycle % 2 == 0:
                for index in sorted(self.used) or range(len(self.backend.devices)):
                    for direction in (0, 1):
                        if self._stop_event.is_set():
                            return
                        self.pcie[index][direction] = self.backend.pcie_kbps(index, direction)
            cycle += 1
            self._stop_event.wait(max(0.0, self.period_s - (time.monotonic() - t0)))


class Sampler(threading.Thread):
    def __init__(self, path, backend, hz=50.0, slow_hz=2.0, pcie=True):
        super().__init__(name="llmperfscope-sampler", daemon=True)
        self.path = path
        self.backend = backend
        self.period_ns = int(1e9 / hz)
        self.slow_period_ns = int(1e9 / slow_hz)
        self.roots = set()
        self.pids = set()
        self._stop_event = threading.Event()
        self._last_cpu = {}  # pid -> (utime, stime, majflt, minflt), kept after exit
        self._last_io = {}  # pid -> (read_bytes, write_bytes, rchar, wchar), kept after exit
        self._last_threads = {}  # tid -> ticks
        self.system = SystemSampler()
        self.queries = BlockingQueries(backend, pcie=pcie) if backend.devices else None
        self.want_pcie = pcie
        self.latest = None  # last tick, read by the watchdog
        self.ticks = 0
        self.max_gap_ns = 0
        self.t_first = self.t_last = None
        self.cpu_seconds = 0.0
        self.error = None

    def set_roots(self, roots):
        self.roots = set(roots)
        self.pids = procfs.tree(self.roots)

    def add_root(self, pid):
        self.roots = self.roots | {pid}

    def stop(self):
        self._stop_event.set()
        if self.queries:
            self.queries.stop()

    def _host_row(self):
        anon = file_ = shmem = hwm = swap = pin = n_threads = n_procs = 0
        for pid in list(self.pids):
            st = procfs.status(pid)
            if not st:
                continue
            n_procs += 1
            anon += st.get("anon", 0)
            file_ += st.get("file", 0)
            shmem += st.get("shmem", 0)
            hwm += st.get("hwm", 0)
            swap += st.get("swap", 0)
            pin += st.get("pin", 0)
            n_threads += st.get("threads", 0)
            cpu = procfs.stat(pid)
            if cpu:
                self._last_cpu[pid] = (cpu["utime"], cpu["stime"], cpu["majflt"], cpu["minflt"])
        totals = [sum(v[i] for v in self._last_cpu.values()) for i in range(4)]
        return [anon, file_, shmem, hwm, swap, pin, n_threads, n_procs] + totals

    def _slow(self):
        self.pids = procfs.tree(self.roots) if self.roots else set()
        self.backend.set_target(self.pids)
        rows = self.backend.slow()
        out = {
            "self_mib": [sum(mib for pid, mib in row["procs"].items() if pid in self.pids) for row in rows],
            "foreign_mib": [sum(mib for pid, mib in row["procs"].items() if pid not in self.pids) for row in rows],
            "sys": self.system.read(),
        }
        if any(row["nvlink"] is not None for row in rows):
            out["nvlink"] = [row["nvlink"] for row in rows]
        if rows and "gpu_ns" in rows[0]:
            out["gpu_ns"] = [sum(row["gpu_ns"].values()) for row in rows]
        util = self.backend.util_samples()
        if any(util):
            offset = time.time_ns() - time.monotonic_ns()
            out["util"] = [[[t - offset, value] for t, value in samples] for samples in util]
        if self.queries:
            self.queries.used = {i for i, mib in enumerate(out["self_mib"]) if mib} | self.queries.used
            energy = self.queries.drain_energy()
            if energy:
                out["energy"] = energy
            if self.want_pcie:
                out["pcie"] = [list(row) for row in self.queries.pcie]
        for pid in self.pids:
            counters = procfs.io(pid)
            if counters:
                self._last_io[pid] = (counters.get("read_bytes", 0), counters.get("write_bytes", 0),
                                      counters.get("rchar", 0), counters.get("wchar", 0))
        if self._last_io:
            out["io"] = [sum(v[i] for v in self._last_io.values()) for i in range(4)]
        changed = {}
        for pid in self.pids:
            for tid, comm, ticks in procfs.threads(pid):
                if ticks != self._last_threads.get(tid):
                    self._last_threads[tid] = ticks
                    changed[tid] = [comm, ticks]
        if changed:
            out["threads"] = changed
        return out

    def run(self):
        try:
            if self.queries:
                self.queries.start()
            self._loop()
        except Exception as e:  # noqa: BLE001 - a dead sampler must not take the benchmark down, the report says so
            self.error = repr(e)

    def _loop(self):
        cpu0 = time.thread_time()
        with gzip.open(self.path, "wt", compresslevel=1) as f:
            next_tick = time.monotonic_ns()
            next_slow = next_tick
            next_flush = next_tick + 5_000_000_000
            prev = None
            while True:
                now = time.monotonic_ns()
                stopping = self._stop_event.is_set()
                if stopping and self.queries:
                    # One last reading of the counters, so they reach the end of the run.
                    self.queries.join(timeout=5)
                    self.queries.read_energy()
                    now = time.monotonic_ns()
                tick = {"t": now, "g": self.backend.fast(), "h": self._host_row()}
                if now >= next_slow or stopping:
                    tick["s"] = self._slow()
                    next_slow = now + self.slow_period_ns
                f.write(json.dumps(tick, separators=(",", ":")) + "\n")
                self.latest = tick
                self.ticks += 1
                if prev is not None:
                    if not stopping:
                        self.max_gap_ns = max(self.max_gap_ns, now - prev)
                else:
                    self.t_first = now
                prev = self.t_last = now
                if now >= next_flush:
                    f.flush()
                    next_flush = now + 5_000_000_000
                if stopping:
                    break
                next_tick += self.period_ns
                delay = next_tick - time.monotonic_ns()
                if delay > 0:
                    self._stop_event.wait(delay / 1e9)
                else:
                    next_tick = time.monotonic_ns()  # fell behind: do not burst to catch up
        self.cpu_seconds = time.thread_time() - cpu0


def load_ticks(path):
    """All ticks of a sample file. A file cut short by a crash still gives the ticks before the cut."""
    ticks = []
    try:
        with gzip.open(path, "rt") as f:
            for line in f:
                try:
                    ticks.append(json.loads(line))
                except ValueError:
                    break
    except (EOFError, OSError):
        pass
    return ticks


def energy_series(ticks, device):
    """(times ns, joules) of one device's cumulative counter, however it was recorded."""
    points = []
    for tick in ticks:
        value = tick["g"][device][2]
        if value is not None:
            points.append((tick["t"], value / 1000))
        for index, t, mj in (tick.get("s") or {}).get("energy", []):
            if index == device:
                points.append((t, mj / 1000))
    points.sort()
    return [p[0] for p in points], [p[1] for p in points]
