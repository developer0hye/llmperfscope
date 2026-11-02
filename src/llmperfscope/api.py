"""Using llmperfscope from Python, for engines that are libraries (transformers, vLLM's LLM class, ...).

The recommended form keeps the sampler in another process, so it cannot disturb the engine:

    # script.py, started with: llmperfscope run -- python script.py
    import llmperfscope
    with llmperfscope.phase("load"):
        model = AutoModelForCausalLM.from_pretrained(...)
    timer = llmperfscope.TokenTimer()
    with llmperfscope.phase("decode"):
        model.generate(**inputs, max_new_tokens=128, streamer=timer)
    llmperfscope.mark_tokens(timer)

`Profile` does the same from inside one process, without the command line.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from . import __version__, analyze, diagnose, machine, markers
from .gpu import open_backend
from .sampler import Sampler, load_ticks


def _torch_memory():
    """PyTorch's own view of GPU memory, when PyTorch is already loaded: what its tensors use against what its allocator holds.
    The difference cannot be seen from outside the process. Importing it here would be wrong: this must not load anything."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        mib = 1 << 20
        return {
            "allocated_mib": round(sum(torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())) / mib, 1),
            "reserved_mib": round(sum(torch.cuda.memory_reserved(i) for i in range(torch.cuda.device_count())) / mib, 1),
            "max_allocated_mib": round(sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / mib, 1),
            "max_reserved_mib": round(sum(torch.cuda.max_memory_reserved(i) for i in range(torch.cuda.device_count())) / mib, 1),
        }
    except Exception:  # noqa: BLE001 - a stats call must never break the program being measured
        return None


def _attach_torch(meta):
    stats = _torch_memory()
    if stats:
        meta["torch"] = stats
    return meta


@contextmanager
def phase(name, group=None, tokens=None, next_phase="idle", **meta):
    """Everything inside is phase `name`. On leaving, the time goes to `next_phase` so a gap is not counted as this phase."""
    markers.mark(name, group=group, tokens=tokens, **_attach_torch(dict(meta)))
    try:
        yield
    finally:
        markers.mark(next_phase, group=next_phase, **_attach_torch({}))


class TokenTimer:
    """Pass as `streamer=` to `model.generate` in transformers. Records when each token arrives, with no dependency on transformers:
    it only has the `put` and `end` methods a streamer needs. Note that using a streamer has a small cost of its own."""

    def __init__(self):
        self.times = []
        self.batches = 0

    def put(self, value):
        self.batches += 1
        # The first call hands over the prompt, the later ones one new token each.
        if self.batches > 1:
            self.times.append(time.monotonic())

    def end(self):
        pass

    @property
    def tokens(self):
        return len(self.times)

    def summary(self):
        if not self.times:
            return {"tokens": 0}
        gaps = [b - a for a, b in zip(self.times, self.times[1:])]
        ordered = sorted(gaps)
        out = {"tokens": len(self.times)}
        if gaps:
            out.update({"tokens_per_s": round(len(gaps) / sum(gaps), 3),
                        "itl_ms_p50": round(ordered[len(ordered) // 2] * 1000, 3),
                        "itl_ms_p99": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))] * 1000, 3)})
        return out


def mark_tokens(timer, group="decode"):
    """Attaches a TokenTimer's result to the phase that just ran."""
    markers.mark("token_stats", group="idle", decode_group=group, token_stats=timer.summary())


class Profile:
    """Samples from a thread of this process. Fine for finding bottlenecks; for speed numbers prefer `llmperfscope run`."""

    def __init__(self, out, label="", hz=50.0, slow_hz=2.0, baseline_s=0.5):
        self.dir = Path(out)
        self.label = label
        self.hz = hz
        self.slow_hz = slow_hz
        self.baseline_s = baseline_s
        self.events = []
        self.lock = threading.Lock()
        self.report = None

    def _on_mark(self, event):
        phase_name = event["phase"]
        record = {"name": phase_name, "group": event.get("group", phase_name), "t": event["t"]}
        if event.get("tokens") is not None:
            record["tokens"] = event["tokens"]
        if event.get("meta"):
            record["meta"] = event["meta"]
        with self.lock:
            self.events.append(record)

    def __enter__(self):
        if not sys.platform.startswith("linux"):
            raise OSError(f"measuring needs Linux for now, and this is {sys.platform}")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.backend = open_backend()
        self.before = machine.snapshot(self.backend, cpu_seconds=0.0, ignore={os.getpid()})
        self.sampler = Sampler(self.dir / "samples.jsonl.gz", self.backend, hz=self.hz, slow_hz=self.slow_hz)
        self.sampler.set_roots({os.getpid()})
        self.t0 = time.monotonic_ns()
        self.events.append({"name": "baseline", "group": "baseline", "t": self.t0})
        self.sampler.start()
        time.sleep(self.baseline_s)
        self.events.append({"name": "run", "group": "run", "t": time.monotonic_ns()})
        markers.add_listener(self._on_mark)
        return self

    def mark(self, phase_name, group=None, tokens=None, **meta):
        markers.mark(phase_name, group=group, tokens=tokens, **meta)

    def __exit__(self, exc_type, exc, tb):
        markers.remove_listener(self._on_mark)
        self.events.append({"name": "tail", "group": "tail", "t": time.monotonic_ns()})
        time.sleep(0.2)
        self.sampler.stop()
        self.sampler.join(timeout=30)
        t_end = time.monotonic_ns()
        after = machine.snapshot(self.backend, cpu_seconds=0.0, ignore={os.getpid()})
        devices = self.backend.devices
        self.backend.close()
        ticks = load_ticks(self.dir / "samples.jsonl.gz")
        phases = sorted(self.events, key=lambda p: p["t"])
        sections = analyze.summarize(ticks, phases, t_end) if ticks else {}
        torch_stats = _collect_torch(phases)
        for item in phases:
            item["start_s"] = round((item.pop("t") - self.t0) / 1e9, 6)
            item["end_s"] = round((item.pop("t_end", t_end) - self.t0) / 1e9, 6)
        self.report = {
            "schema": "llmperfscope.run/1",
            "tool": {"name": "llmperfscope", "version": __version__},
            "label": self.label,
            "status": "failed" if exc_type else "ok",
            "perturbed": False,
            "adapter": "in-process",
            "host": machine.host_info(),
            "devices": devices,
            "disks": self.sampler.system.disks,
            "machine_before": self.before,
            "machine_after": after,
            "sampler": {"hz_target": self.hz, "max_gap_ms": round(self.sampler.max_gap_ns / 1e6, 1),
                        "cpu_seconds": round(self.sampler.cpu_seconds, 3), "error": self.sampler.error},
            "notes": ["sampled from inside the profiled process"],
            "phases": phases,
        }
        if torch_stats:
            self.report["torch_memory"] = torch_stats
        self.report.update(sections)
        derived = analyze.throughput_from_phases(sections.get("groups") or {})
        if derived:
            self.report["throughput"] = derived
        self.report["diagnosis"] = diagnose.diagnose(self.report)
        with open(self.dir / "run.json", "w") as f:
            json.dump(self.report, f, indent=1)
        return False


def _collect_torch(phases):
    """The highest PyTorch memory figures reported by any mark, for the memory findings."""
    out = {}
    for item in phases:
        stats = (item.get("meta") or {}).get("torch")
        if stats:
            for key, value in stats.items():
                out[key] = max(out.get(key, 0.0), value)
    return out
