"""`llmperfscope check`: what can be measured on this machine, what cannot, and what it would take."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys

from . import cpuprof, machine, nsys
from .gpu import open_backend


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _driver_param(name):
    text = _read("/proc/driver/nvidia/params") or ""
    for line in text.splitlines():
        if line.startswith(name + ":"):
            return line.split(":", 1)[1].strip()
    return None


def _collect_macos(add):
    add("process memory by kind, CPU, disk reads (libproc, Mach)", "yes")
    add("disk traffic, memory compressor and pressure, network", "yes")
    backend = open_backend()
    if backend.name == "none":
        add("GPU memory, utilization, energy", "no", backend.reason)
    else:
        memory_mib = backend.devices[0]["memory_total_mib"]
        add("GPU memory and utilization (IOKit)", "yes", f"{backend.devices[0]['name']}, one pool of memory, macOS recommends the GPU hold at most {memory_mib:,} MiB")
        add("GPU time per process (IOKit)", "yes" if backend.accelerator.clients() else "no", "" if backend.accelerator.clients() else "the registry lists no GPU clients")
        add("GPU energy counter (IOReport, undocumented)", "yes" if backend.energy_mj(0) is not None else "no", "the GPU only: IOReport's CPU channels do not count up here")
        add("GPU clock (IOReport, undocumented)", "yes" if backend.clock.frequencies else "no")
        add("thermal pressure", "yes")
        backend.close()
    add("per-thread CPU time", "yes")
    add("model page cache state, --cold-start", "yes")
    add("GPU timeline (--nsys)", "no", "NVIDIA only. On a Mac: Instruments' Metal System Trace, which comes with Xcode")
    add("host CPU hot spots (--cpu-profile)", "no", "needs Linux for now. On a Mac: `sample PID 10`, which needs no root")
    add("containers", "no", "they run in a virtual machine on a Mac, out of sight of the host")


def collect():
    items = []

    def add(what, status, detail=""):
        items.append({"what": what, "status": status, "detail": detail})

    if sys.platform == "darwin":
        _collect_macos(add)
        add("figures", "yes" if importlib.util.find_spec("matplotlib") else "no", "" if importlib.util.find_spec("matplotlib") else 'pip install "llmperfscope[figures]"')
        return {"host": machine.host_info(), "items": items}

    if not sys.platform.startswith("linux"):
        add("measuring: run, serve, attach, matrix", "no", f"needs Linux or macOS, this is {sys.platform}")
        add("reading runs: show, html, anonymize, diff", "yes")
        add("figures", "yes" if importlib.util.find_spec("matplotlib") else "no", "" if importlib.util.find_spec("matplotlib") else 'pip install "llmperfscope[figures]"')
        return {"host": machine.host_info(), "items": items}

    add("process memory, CPU, disk reads (/proc)", "yes" if os.path.exists("/proc/self/status") else "no",
        "" if os.path.exists("/proc/self/status") else "needs Linux")
    backend = open_backend()
    if backend.name == "none":
        add("GPU memory, power, energy, clocks, throttling", "no", backend.reason)
    else:
        rows = backend.fast()
        slow = backend.slow()
        names = sorted({d["name"] for d in backend.devices})
        add("GPU memory, power, clocks, utilization", "yes", f"{len(rows)} x {', '.join(names)}")
        add("GPU energy counter", "yes" if backend.devices and backend.energy_mj(0) is not None else "no")
        add("per-process GPU memory", "yes" if slow else "no")
        nvlink = [d["index"] for d in backend.devices if d.get("nvlink")]
        add("NVLink traffic", "yes" if nvlink else "no", f"GPUs {nvlink}" if nvlink else "no NVLink or not reported")
    backend.close()
    add("per-thread CPU time", "yes")

    tool = nsys.find_nsys()
    add("GPU timeline: kernels, copies, host API (nsys)", "yes" if tool else "no",
        tool or "install Nsight Systems or set CUDA_HOME")
    admin_only = _driver_param("RmProfilingAdminOnly")
    if admin_only == "0":
        add("GPU hardware counters (nsys --gpu-metrics-devices, ncu)", "yes")
    elif admin_only is None:
        add("GPU hardware counters (nsys --gpu-metrics-devices, ncu)", "unknown", "no NVIDIA driver parameters found")
    else:
        add("GPU hardware counters (nsys --gpu-metrics-devices, ncu)", "needs admin",
            "RmProfilingAdminOnly=1: load the driver with NVreg_RestrictProfilingToAdminUsers=0 or run as root")

    paranoid = _read("/proc/sys/kernel/perf_event_paranoid")
    if paranoid is not None and int(paranoid) <= 2:
        add("perf record", "yes", f"perf_event_paranoid={paranoid}")
    else:
        add("perf record", "needs admin", f"perf_event_paranoid={paranoid}: `sysctl kernel.perf_event_paranoid=2`")
    try:
        library = cpuprof.library_path()
        add("host CPU hot spots without perf (--cpu-profile)", "yes", str(library))
    except (OSError, subprocess.SubprocessError) as e:
        add("host CPU hot spots without perf (--cpu-profile)", "no", str(e))
    add("model page cache state", "yes")
    add("file tools", "yes" if shutil.which("nm") else "partly", "" if shutil.which("nm") else "nm missing: no CPU symbols")
    return {"host": machine.host_info(), "items": items}


def render(info):
    lines = [f"{info['host'].get('hostname')}: {info['host'].get('cpus')} CPUs", "", "| measurement | available | |", "|---|---|---|"]
    for item in info["items"]:
        lines.append(f"| {item['what']} | {item['status']} | {item['detail']} |")
    return "\n".join(lines) + "\n"
