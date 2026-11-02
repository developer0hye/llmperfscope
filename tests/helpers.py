"""Test doubles with known answers. An instrument is only trustworthy if it reports what is known to have happened."""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from llmperfscope import procfs  # noqa: E402
from llmperfscope.gpu import GpuBackend  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CAN_MEASURE = sys.platform.startswith("linux") or sys.platform == "darwin"
# On GitHub's macOS virtual machines a sleep of 20 ms takes over 100 ms, and a process can take 30 s to start, so a known time cannot be checked there. A real Mac is fine.
UNRELIABLE_TIMERS = sys.platform == "darwin" and bool(os.environ.get("CI"))


class ScriptedGpu(GpuBackend):
    """GPUs whose power and memory are set by the test. Energy is the exact integral of the constant power."""

    name = "scripted"

    def __init__(self, count=2, power_w=100.0, used_mib=1000, baseline_mib=100, total_mib=8192, util=90, throttle=0):
        super().__init__(reason="")
        self.devices = [{"index": i, "name": "scripted", "uuid": f"GPU-{i}", "memory_total_mib": total_mib, "power_limit_w": 300.0,
                         "nvlink": False} for i in range(count)]
        self.power_w = power_w
        self.used_mib = used_mib
        self.baseline_mib = baseline_mib
        self.util = util
        self.throttle = throttle
        self.t0 = time.monotonic()
        self.children = False

    def _running(self):
        # The target is a child of the test process, as it would be of the llmperfscope command.
        for pid in procfs.children_of(os.getpid()):
            st = procfs.stat(pid)
            if st and st["state"] != "Z":
                return pid
        return None

    def energy_mj(self, index):
        return int(self.power_w * (time.monotonic() - self.t0) * 1000)

    def fast(self):
        running = self._running() is not None
        mem = self.baseline_mib + (self.used_mib if running else 0)
        return [[mem, self.power_w, None, 1800, 7000, self.util if running else 0, 10, self.throttle, 50] for _ in self.devices]

    def slow(self):
        pid = self._running()
        return [{"procs": {pid: self.used_mib} if pid else {}, "nvlink": None} for _ in self.devices]


def temp_dir():
    return tempfile.mkdtemp(prefix="llmperfscope-test-", dir=os.environ.get("LLMPERFSCOPE_TEST_DIR"))


def group(**kwargs):
    """A phase group as analyze.aggregate_group would produce it, with just the fields a rule looks at."""
    out = {"wall_s": 10.0, "gpu": {}, "host": {}, "flows": {}, "pressure": {}}
    for key, value in kwargs.items():
        if key in out and isinstance(out[key], dict):
            out[key].update(value)
        else:
            out[key] = value
    return out
