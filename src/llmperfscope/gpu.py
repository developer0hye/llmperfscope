"""GPU telemetry. NVML today; the interface is small so other vendors can plug in.

Everything here works without admin rights. Hardware counters (cache hit rate, per-kernel DRAM traffic) need
RmProfilingAdminOnly=0 or root and go through `nsys --gpu-metrics-devices` instead; see capabilities.py.
"""

from __future__ import annotations

import ctypes as C
import os
import sys
import time

# Columns of one device in the fast sample.
G_MEM, G_POWER, G_ENERGY, G_SM, G_MEMCLK, G_UTIL, G_UTILMEM, G_THROTTLE, G_TEMP = range(9)

THROTTLE_BITS = {
    0x1: "idle",
    0x2: "app_clocks",
    0x4: "sw_power_cap",
    0x8: "hw_slowdown",
    0x10: "sync_boost",
    0x20: "sw_thermal",
    0x40: "hw_thermal",
    0x80: "hw_power_brake",
    0x100: "display_clocks",
}
# Reasons that mean the GPU was held below what the work wanted.
THROTTLE_BAD = 0x4 | 0x8 | 0x20 | 0x40 | 0x80
# Reasons that are real even on an idle GPU. The software power cap bit is not: idle GPUs report it with nothing to cap.
THROTTLE_HOT = 0x8 | 0x20 | 0x40 | 0x80


def throttle_names(bits, include_idle=False):
    return [name for bit, name in THROTTLE_BITS.items() if (bits & bit) and (include_idle or bit != 0x1)]


class GpuBackend:
    """No GPU telemetry: runs still get host metrics and the phase timeline."""

    name = "none"

    def __init__(self, reason="no backend"):
        self.reason = reason
        self.devices = []

    def fast(self):
        """Cheap gauges, per device: mem MiB, W, mJ (None here, see energy_mj), SM MHz, mem MHz, util %, mem util %, throttle bits, C."""
        return []

    def energy_mj(self, index):
        """The device's cumulative energy counter. Can block far longer than the gauges, so it has its own thread."""
        return None

    def slow(self):
        """Costlier reads, per device: {"procs": {pid: MiB}, "nvlink": (tx KiB, rx KiB) or None}."""
        return []

    def set_target(self, pids):
        """The processes being measured, for a backend that reports something per process."""

    def util_samples(self):
        """Per device, new utilization samples as (wall ns, %). Empty when the backend keeps no history."""
        return [[] for _ in self.devices]

    def pcie_kbps(self, index, direction):
        return None

    def driver_version(self):
        return None

    def close(self):
        pass


class _Memory(C.Structure):
    _fields_ = [("total", C.c_ulonglong), ("free", C.c_ulonglong), ("used", C.c_ulonglong)]


class _Utilization(C.Structure):
    _fields_ = [("gpu", C.c_uint), ("memory", C.c_uint)]


class _ProcessInfo(C.Structure):
    _fields_ = [("pid", C.c_uint), ("used", C.c_ulonglong), ("gi", C.c_uint), ("ci", C.c_uint)]


class _FieldUnion(C.Union):
    _fields_ = [("d", C.c_double), ("ui", C.c_uint), ("ul", C.c_ulong), ("ull", C.c_ulonglong), ("sll", C.c_longlong)]


class _Sample(C.Structure):
    _fields_ = [("timeStamp", C.c_ulonglong), ("value", C.c_ulonglong)]


class _FieldValue(C.Structure):
    _fields_ = [
        ("fieldId", C.c_uint),
        ("scopeId", C.c_uint),
        ("timestamp", C.c_longlong),
        ("latencyUsec", C.c_longlong),
        ("valueType", C.c_int),
        ("nvmlReturn", C.c_int),
        ("value", _FieldUnion),
    ]


_FI_NVLINK_TX = 138  # NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX, KiB
_FI_NVLINK_RX = 139  # NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX, KiB
_ALL_LINKS = 0xFFFFFFFF
_NOT_AVAILABLE = 0xFFFFFFFFFFFFFFFF
_MAX_PROCS = 256
_MIB = 1024 * 1024


class NvmlBackend(GpuBackend):
    name = "nvml"

    def __init__(self, library="libnvidia-ml.so.1"):
        super().__init__(reason="")
        self.lib = C.CDLL(library)
        rc = self.lib.nvmlInit_v2()
        if rc != 0:
            raise OSError(f"nvmlInit_v2 returned {rc}")
        self.handles = []
        count = C.c_uint()
        self.lib.nvmlDeviceGetCount_v2(C.byref(count))
        for i in range(count.value):
            handle = C.c_void_p()
            if self.lib.nvmlDeviceGetHandleByIndex_v2(i, C.byref(handle)) != 0:
                continue
            self.handles.append(handle)
            self.devices.append(self._describe(i, handle))

        self._throttle = None
        for fn in ("nvmlDeviceGetCurrentClocksEventReasons", "nvmlDeviceGetCurrentClocksThrottleReasons"):
            if hasattr(self.lib, fn):
                self._throttle = getattr(self.lib, fn)
                break
        self._procs = None
        for fn in ("nvmlDeviceGetComputeRunningProcesses_v3", "nvmlDeviceGetComputeRunningProcesses_v2"):
            if hasattr(self.lib, fn):
                self._procs = getattr(self.lib, fn)
                break
        self._proc_buffer = (_ProcessInfo * _MAX_PROCS)()
        self._mem = _Memory()
        self._util = _Utilization()
        self._uint = C.c_uint()
        self._ull = C.c_ulonglong()
        self._nvlink_ok = [self._probe_nvlink(h) for h in self.handles]
        for dev, ok in zip(self.devices, self._nvlink_ok):
            dev["nvlink"] = ok
        self._samples = (_Sample * 512)()
        self._util_seen = [0 for _ in self.handles]

    def driver_version(self):
        text = C.create_string_buffer(96)
        if self.lib.nvmlSystemGetDriverVersion(text, 96) == 0:
            return text.value.decode()
        return None

    def _describe(self, index, handle):
        name = C.create_string_buffer(96)
        uuid = C.create_string_buffer(96)
        self.lib.nvmlDeviceGetName(handle, name, 96)
        self.lib.nvmlDeviceGetUUID(handle, uuid, 96)
        mem = _Memory()
        self.lib.nvmlDeviceGetMemoryInfo(handle, C.byref(mem))
        limit = C.c_uint()
        power_limit = self.lib.nvmlDeviceGetEnforcedPowerLimit(handle, C.byref(limit)) == 0
        return {
            "index": index,
            "name": name.value.decode(),
            "uuid": uuid.value.decode(),
            "memory_total_mib": mem.total // _MIB,
            "power_limit_w": limit.value / 1000 if power_limit else None,
        }

    def _read_nvlink(self, handle):
        fields = (_FieldValue * 2)()
        fields[0].fieldId = _FI_NVLINK_TX
        fields[1].fieldId = _FI_NVLINK_RX
        fields[0].scopeId = fields[1].scopeId = _ALL_LINKS
        if self.lib.nvmlDeviceGetFieldValues(handle, 2, fields) != 0:
            return None
        if fields[0].nvmlReturn != 0 or fields[1].nvmlReturn != 0:
            return None
        return (fields[0].value.ull, fields[1].value.ull)

    def _probe_nvlink(self, handle):
        return self._read_nvlink(handle) is not None

    def _get_uint(self, fn, handle, *args):
        return self._uint.value if fn(handle, *args, C.byref(self._uint)) == 0 else None

    def energy_mj(self, index):
        # Measured on 8 RTX A6000: 2.5 ms per GPU, about 1 us for the other queries. Read at 50 Hz on 8 GPUs it cost 3.4% of prefill speed
        # and, worse, under-reported: 818 W where the power gauge said 1,395 W. About twice a second it agrees with the gauge (1,602 W against
        # 1,591 W). So it is read slowly, and every report compares the two.
        value = C.c_ulonglong()
        if self.lib.nvmlDeviceGetTotalEnergyConsumption(self.handles[index], C.byref(value)) == 0:
            return value.value
        return None

    def fast(self):
        lib = self.lib
        rows = []
        for handle in self.handles:
            used = self._mem.used // _MIB if lib.nvmlDeviceGetMemoryInfo(handle, C.byref(self._mem)) == 0 else None
            power = self._get_uint(lib.nvmlDeviceGetPowerUsage, handle)
            energy = None
            sm_clock = self._get_uint(lib.nvmlDeviceGetClockInfo, handle, 1)
            mem_clock = self._get_uint(lib.nvmlDeviceGetClockInfo, handle, 2)
            if lib.nvmlDeviceGetUtilizationRates(handle, C.byref(self._util)) == 0:
                util, util_mem = self._util.gpu, self._util.memory
            else:
                util = util_mem = None
            throttle = None
            if self._throttle is not None and self._throttle(handle, C.byref(self._ull)) == 0:
                throttle = self._ull.value
            temp = self._get_uint(lib.nvmlDeviceGetTemperature, handle, 0)
            rows.append([used, power / 1000 if power is not None else None, energy, sm_clock, mem_clock, util, util_mem,
                         throttle, temp])
        return rows

    def slow(self):
        rows = []
        for handle, nvlink_ok in zip(self.handles, self._nvlink_ok):
            procs = {}
            if self._procs is not None:
                count = C.c_uint(_MAX_PROCS)
                if self._procs(handle, C.byref(count), self._proc_buffer) == 0:
                    for i in range(min(count.value, _MAX_PROCS)):
                        info = self._proc_buffer[i]
                        if info.used != _NOT_AVAILABLE:
                            procs[info.pid] = info.used // _MIB
            rows.append({"procs": procs, "nvlink": self._read_nvlink(handle) if nvlink_ok else None})
        return rows

    def util_samples(self):
        """New utilization samples per device since the last call, as (wall ns, %). The driver keeps a history at a finer
        grain than the value a single query returns, so short phases are not judged by one stale reading."""
        rows = []
        for i, handle in enumerate(self.handles):
            count = C.c_uint(len(self._samples))
            value_type = C.c_int()
            new = []
            rc = self.lib.nvmlDeviceGetSamples(handle, 1, C.c_ulonglong(self._util_seen[i]), C.byref(value_type), C.byref(count),
                                               self._samples)  # NVML_GPU_UTILIZATION_SAMPLES
            if rc == 0:
                for k in range(count.value):
                    sample = self._samples[k]
                    # The value is a union, and a utilization only sets its low 32 bits.
                    new.append((sample.timeStamp * 1000, sample.value & 0xFFFFFFFF))
                if new:
                    self._util_seen[i] = new[-1][0] // 1000
            rows.append(new)
        return rows

    def pcie_kbps(self, index, direction):
        """PCIe throughput in KB/s, direction 0 = GPU to host, 1 = host to GPU. The driver measures over 20 ms and blocks for
        that long, so this only ever runs on its own thread, one query at a time."""
        value = C.c_uint()
        if self.lib.nvmlDeviceGetPcieThroughput(self.handles[index], direction, C.byref(value)) == 0:
            return value.value
        return None

    def close(self):
        try:
            self.lib.nvmlShutdown()
        except OSError:
            pass


class AppleBackend(GpuBackend):
    """Apple Silicon, from IOKit and IOReport, with no root needed. There is one pool of memory, so GPU memory is what the driver has in use of it, and that is everything on the
    machine: macOS does not say which process holds it, though it does say which process used how much GPU time. No temperature, no PCIe."""

    name = "apple"

    def __init__(self):
        super().__init__(reason="")
        from . import darwin
        self.darwin = darwin
        self.accelerator = darwin.Accelerator()
        self.energy = darwin.EnergyCounter()
        self.clock = darwin.GpuClock()
        self.mhz = None
        self.devices = [{"index": 0, "name": self.accelerator.name(), "uuid": "apple-gpu", "memory_total_mib": darwin.gpu_working_set_mib() or (darwin.sysctl_int("hw.memsize") or 0) >> 20,
                         "memory_total_is_working_set_limit": True, "power_limit_w": None, "nvlink": False, "unified_memory": True}]
        self.target = set()
        self.client_ns = {}  # GPU client -> (pid, GPU time seen so far). A client that is gone keeps its last value, or a process that exits would take its time with it.
        self.target_ns = None
        self.t_wall = None
        self.samples = []

    def fast(self):
        memory, util = self.accelerator.read() or (None, None)
        # Moderate thermal pressure is when macOS starts holding clocks down, which a fanless Mac reaches within minutes of full load.
        throttle = 0x20 if (self.darwin.thermal_level() or 0) >= 1 else 0
        return [[memory, None, None, self.mhz, None, util, None, throttle, None]]

    def energy_mj(self, index):
        """The clock is an average since its last read, so it goes with the other reads on the slow thread."""
        self.mhz = self.clock.read()
        return self.energy.millijoules()

    def set_target(self, pids):
        self.target = set(pids)

    def slow(self):
        """The target's share of GPU time, as this GPU's utilization: the machine-wide figure includes the display. Reading it on slow ticks is enough, because it is a
        cumulative counter."""
        clients = self.accelerator.clients()
        if not any(ns for _pid, ns in clients.values()):
            # The display always has some, so no time anywhere means this macOS does not publish it. Then GPU numbers are the machine's, and memory growth says what was used.
            return [{"procs": {}, "nvlink": None}]
        for client, (pid, ns) in clients.items():
            _old_pid, old = self.client_ns.get(client, (pid, 0))
            self.client_ns[client] = (pid, max(ns, old))
        now = time.time_ns()
        total = sum(ns for pid, ns in self.client_ns.values() if pid in self.target)
        if self.target and self.target_ns is not None and now > self.t_wall:
            self.samples.append(((now + self.t_wall) // 2, min(100.0, round(100 * (total - self.target_ns) / (now - self.t_wall), 1))))
        if self.target:
            self.target_ns, self.t_wall = total, now
        return [{"procs": {}, "nvlink": None, "gpu_ns": {pid: ns for pid, ns in self.client_ns.values() if pid in self.target}}]

    def foreign_busy(self, seconds=0.5):
        """(process id, share of the GPU in %) of whoever else used the GPU over a short wait, for the check before a run."""
        def per_process():
            out = {}
            for pid, ns in self.accelerator.clients().values():
                out[pid] = out.get(pid, 0) + ns
            return out
        before, t0 = per_process(), time.monotonic()
        time.sleep(seconds)
        after, wall = per_process(), time.monotonic() - t0
        return {pid: round(100 * (ns - before.get(pid, ns)) / (wall * 1e9), 1) for pid, ns in after.items() if ns > before.get(pid, ns)}

    def util_samples(self):
        out, self.samples = self.samples, []
        return [out]

    def close(self):
        self.accelerator.close()


def open_backend():
    """The best backend on this machine. Never raises: no GPU still gives a host-only profile."""
    if os.environ.get("LLMPERFSCOPE_NO_GPU"):
        return GpuBackend(reason="LLMPERFSCOPE_NO_GPU is set")
    if sys.platform == "darwin":
        try:
            return AppleBackend()
        except (OSError, AttributeError) as e:
            return GpuBackend(reason=f"GPU statistics not available: {e}")
    try:
        return NvmlBackend()
    except (OSError, AttributeError) as e:
        return GpuBackend(reason=f"NVML not available: {e}")
