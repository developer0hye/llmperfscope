"""Turns ticks and phases into per-phase numbers. Runs after the target has exited, so its cost does not matter."""

from __future__ import annotations

import bisect
import re

from .gpu import G_MEM, G_MEMCLK, G_POWER, G_SM, G_TEMP, G_THROTTLE, G_UTIL, G_UTILMEM, THROTTLE_BAD, throttle_names
from .procfs import CLK_TCK
from .sampler import H_ANON, H_FILE, H_HWM, H_MAJFLT, H_MINFLT, H_PIN, H_SHMEM, H_STICKS, H_SWAP, H_THREADS, H_UTICKS, energy_series
from .system import SECTOR

PAGE_KB = 4
# The energy counter is read about twice a second, so energy per token means nothing over less time than this.
MIN_WALL_FOR_ENERGY_PER_TOKEN_S = 2.0


def _interp(times, values, t):
    """Value of a cumulative counter at time t. Clamps at the ends."""
    if not times:
        return None
    i = bisect.bisect_left(times, t)
    if i <= 0:
        return values[0]
    if i >= len(times):
        return values[-1]
    t0, t1 = times[i - 1], times[i]
    if t1 == t0:
        return values[i]
    return values[i - 1] + (values[i] - values[i - 1]) * (t - t0) / (t1 - t0)


def _delta(times, values, t0, t1):
    if not times:
        return None
    return _interp(times, values, t1) - _interp(times, values, t0)


def _kb_to_mib(kb):
    return round(kb / 1024, 1)


class Series:
    """Column access over the ticks of one run."""

    def __init__(self, ticks):
        self.ticks = ticks
        self.t = [tick["t"] for tick in ticks]
        self.n_gpus = len(ticks[0]["g"]) if ticks else 0
        self.energy = [energy_series(ticks, dev) for dev in range(self.n_gpus)]
        self.cpu = {}
        for key, col in (("user", H_UTICKS), ("sys", H_STICKS), ("majflt", H_MAJFLT), ("minflt", H_MINFLT)):
            self.cpu[key] = [tick["h"][col] for tick in ticks]
        slow = [tick for tick in ticks if "s" in tick]
        self.slow_t = [tick["t"] for tick in slow]
        self.slow = slow
        self.io_t = [tick["t"] for tick in slow if "io" in tick["s"]]
        self.io = [tick["s"]["io"] for tick in slow if "io" in tick["s"]]
        self.nvlink_t = [tick["t"] for tick in slow if tick["s"].get("nvlink")]
        self.nvlink = [tick["s"]["nvlink"] for tick in slow if tick["s"].get("nvlink")]
        self.util = [([], []) for _ in range(self.n_gpus)]
        for tick in slow:
            for dev, samples in enumerate(tick["s"].get("util", [])):
                for t, value in samples:
                    self.util[dev][0].append(t)
                    self.util[dev][1].append(value)
        self.system = [tick["s"]["sys"] for tick in slow if tick["s"].get("sys")]
        self.system_t = [tick["t"] for tick in slow if tick["s"].get("sys")]

    def window(self, t0, t1):
        """Indices [i, j) of the ticks inside the window. Never empty if there are ticks: a phase shorter than the sample
        period still gets the nearest tick for its gauges."""
        i = bisect.bisect_left(self.t, t0)
        j = bisect.bisect_left(self.t, t1)
        if i >= len(self.t):
            i = len(self.t) - 1
            j = len(self.t)
        if j <= i:
            j = i + 1
        return i, j

    def system_delta(self, getter, t0, t1):
        """Change of a system counter over a window, or None when it was never read."""
        times, values = [], []
        for t, row in zip(self.system_t, self.system):
            value = getter(row)
            if value is not None:
                times.append(t)
                values.append(value)
        return _delta(times, values, t0, t1) if times else None

    def system_gauges(self, getter, t0, t1):
        """Readings of a system gauge inside a window; the nearest one when none falls inside."""
        if not self.system_t:
            return []
        i = bisect.bisect_left(self.system_t, t0)
        j = bisect.bisect_left(self.system_t, t1)
        rows = self.system[i:j] or [self.system[min(i, len(self.system) - 1)]]
        return [v for v in (getter(row) for row in rows) if v is not None]

    def pcie_average(self, device, direction, t0, t1):
        """Average KB/s of the slow PCIe samples inside a window, or None when no sample fell inside."""
        values = []
        i = bisect.bisect_left(self.slow_t, t0)
        j = bisect.bisect_left(self.slow_t, t1)
        for tick in self.slow[i:j]:
            rows = tick["s"].get("pcie")
            if rows and rows[device][direction] is not None:
                values.append(rows[device][direction])
        return sum(values) / len(values) if values else None


def used_devices(series, baseline_mem, min_above_baseline=64):
    """GPUs this run touched: any process of its tree holds memory there. When that is not visible (other PID namespace,
    NVML without process info) fall back to memory growth over the baseline."""
    used = set()
    for tick in series.slow:
        for dev, mib in enumerate(tick["s"].get("self_mib", [])):
            if mib:
                used.add(dev)
    if used:
        return sorted(used)
    for dev in range(series.n_gpus):
        peak = max((tick["g"][dev][G_MEM] or 0) for tick in series.ticks)
        if peak - baseline_mem[dev] >= min_above_baseline:
            used.add(dev)
    return sorted(used)


def _flows(series, t0, t1, devices, wall):
    """Bytes moved between tiers in a window: disk, RAM (swap, page cache), GPU (PCIe, NVLink), network."""
    if not series.system:
        return {}, {}, {}
    flows = {}
    disks = {}
    names = set()
    for row in series.system:
        names |= set(row.get("disk", {}))
    for name in sorted(names):
        read = series.system_delta(lambda r, n=name: r.get("disk", {}).get(n, [None])[0], t0, t1)
        write = series.system_delta(lambda r, n=name: (r.get("disk", {}).get(n) or [None, None])[1], t0, t1)
        busy = series.system_delta(lambda r, n=name: (r.get("disk", {}).get(n) or [None, None, None])[2], t0, t1)
        if read is None:
            continue
        if read > 0 or write > 0 or (busy or 0) > 0:
            disks[name] = {
                "read_mib": round(read * SECTOR / (1 << 20), 1),
                "write_mib": round((write or 0) * SECTOR / (1 << 20), 1),
                "busy_pct": round(min(100.0, (busy or 0) / (wall * 1000) * 100), 1) if wall > 0 else None,
            }
    if disks:
        flows["disks"] = disks

    def vm(key):
        return series.system_delta(lambda r: r.get("vm", {}).get(key), t0, t1)

    swap_in, swap_out = vm("pswpin"), vm("pswpout")
    if swap_in is not None:
        flows["swap_in_mib"] = round(swap_in * PAGE_KB / 1024, 1)
        flows["swap_out_mib"] = round((swap_out or 0) * PAGE_KB / 1024, 1)
    refault = vm("workingset_refault_file")
    if refault is not None:
        flows["page_cache_refault_mib"] = round(refault * PAGE_KB / 1024, 1)
    scanned = vm("pgscan_direct")
    if scanned is not None:
        flows["direct_reclaim_mib"] = round(scanned * PAGE_KB / 1024, 1)
    cached = series.system_gauges(lambda r: r.get("mem", {}).get("Cached"), t0, t1)
    if cached:
        flows["page_cache_change_mib"] = cached[-1] - cached[0]
    available = series.system_gauges(lambda r: r.get("mem", {}).get("MemAvailable"), t0, t1)
    if available:
        flows["mem_available_min_mib"] = min(available)
    swap_free = series.system_gauges(lambda r: r.get("mem", {}).get("SwapFree"), t0, t1)
    swap_total = series.system_gauges(lambda r: r.get("mem", {}).get("SwapTotal"), t0, t1)
    if swap_free and swap_total:
        flows["swap_used_max_mib"] = swap_total[-1] - min(swap_free)
    rx = series.system_delta(lambda r: (r.get("net") or [None])[0], t0, t1)
    tx = series.system_delta(lambda r: (r.get("net") or [None, None])[1], t0, t1)
    if rx is not None:
        flows["network_rx_mib"] = round(rx / (1 << 20), 1)
        flows["network_tx_mib"] = round((tx or 0) / (1 << 20), 1)
    to_gpu = from_gpu = None
    for dev in devices:
        value = series.pcie_average(dev, 1, t0, t1)
        if value is not None:
            to_gpu = (to_gpu or 0.0) + value * wall / 1024
        value = series.pcie_average(dev, 0, t0, t1)
        if value is not None:
            from_gpu = (from_gpu or 0.0) + value * wall / 1024
    # Estimates: PCIe traffic is sampled, not counted.
    if to_gpu is not None:
        flows["pcie_to_gpu_mib_est"] = round(to_gpu, 1)
    if from_gpu is not None:
        flows["pcie_from_gpu_mib_est"] = round(from_gpu, 1)

    pressure = {}
    for key in ("cpu_some", "memory_some", "memory_full", "io_some", "io_full"):
        value = series.system_delta(lambda r, k=key: r.get("psi", {}).get(k), t0, t1)
        if value is not None and wall > 0:
            pressure[key + "_pct"] = round(min(100.0, value / (wall * 1e6) * 100), 1)

    system_cpu = {}
    first = series.system_gauges(lambda r: r.get("cpu"), t0, t1)
    if len(series.system_t) >= 2:
        deltas = [series.system_delta(lambda r, c=c: (r.get("cpu") or [None] * 8)[c], t0, t1) for c in range(8)]
        if first and all(d is not None for d in deltas):
            total = sum(deltas)
            if total > 0:
                system_cpu = {
                    "busy_pct": round((deltas[0] + deltas[1] + deltas[2] + deltas[5] + deltas[6]) / total * 100, 1),
                    "iowait_pct": round(deltas[4] / total * 100, 2),
                    "steal_pct": round(deltas[7] / total * 100, 2),
                }
    return flows, pressure, system_cpu


def phase_metrics(series, t0, t1, devices, baseline_mem):
    i, j = series.window(t0, t1)
    ticks = series.ticks[i:j]
    wall = (t1 - t0) / 1e9
    out = {"wall_s": round(wall, 6)}
    if not ticks:
        return out

    rows = []
    total_j = 0.0
    have_energy = bool(devices)
    for dev in devices:
        values = [tick["g"][dev] for tick in ticks]
        mem = [v[G_MEM] for v in values if v[G_MEM] is not None]
        energy = _delta(*series.energy[dev], t0, t1)
        if energy is None:
            have_energy = False
        else:
            total_j += energy
        sm = [v[G_SM] for v in values if v[G_SM] is not None]
        # The driver's own history when it has samples in this window, else the instantaneous reading.
        times, history = series.util[dev]
        a, b = bisect.bisect_left(times, t0), bisect.bisect_left(times, t1)
        util = history[a:b] or [v[G_UTIL] for v in values if v[G_UTIL] is not None]
        util_mem = [v[G_UTILMEM] for v in values if v[G_UTILMEM] is not None]
        power = [v[G_POWER] for v in values if v[G_POWER] is not None]
        throttle = 0
        throttled_ticks = 0
        for v in values:
            bits = v[G_THROTTLE] or 0
            throttle |= bits
            if bits & THROTTLE_BAD:
                throttled_ticks += 1
        rows.append({
            "index": dev,
            "mem_start_mib": mem[0] if mem else None,
            "mem_end_mib": mem[-1] if mem else None,
            "mem_peak_mib": max(mem) if mem else None,
            "energy_j": round(energy, 3) if energy is not None else None,
            "power_avg_w": round(energy / wall, 2) if energy is not None and wall > 0 else (
                round(sum(power) / len(power), 2) if power else None),
            "power_gauge_w": round(sum(power) / len(power), 2) if power else None,
            "sm_mhz_avg": round(sum(sm) / len(sm)) if sm else None,
            "sm_mhz_min": min(sm) if sm else None,
            "mem_mhz_avg": round(sum(v[G_MEMCLK] or 0 for v in values) / len(values)),
            "util_avg": round(sum(util) / len(util), 1) if util else None,
            "util_mem_avg": round(sum(util_mem) / len(util_mem), 1) if util_mem else None,
            "temp_c_max": max((v[G_TEMP] or 0) for v in values),
            "throttle": throttle_names(throttle),
            # A stray tick is not a limit: a phase counts as throttled when it spent a real part of its time that way.
            "throttled": throttled_ticks >= max(2, 0.1 * len(values)),
            "throttled_pct": round(100 * throttled_ticks / len(values), 1),
        })
    if devices:
        sums = [sum(tick["g"][dev][G_MEM] or 0 for dev in devices) for tick in ticks]
        baseline = sum(baseline_mem[dev] for dev in devices)
        utils = [r["util_avg"] for r in rows if r["util_avg"] is not None]
        out["gpu"] = {
            "mem_start_mib": sums[0] - baseline,
            "mem_end_mib": sums[-1] - baseline,
            "mem_peak_mib": max(sums) - baseline,
            "mem_peak_max_device_mib": max((r["mem_peak_mib"] or 0) - baseline_mem[r["index"]] for r in rows),
            "energy_j": round(total_j, 3) if have_energy else None,
            "power_avg_w": round(total_j / wall, 2) if have_energy and wall > 0 else None,
            "power_gauge_w": round(sum(r["power_gauge_w"] or 0 for r in rows), 2),
            "sm_mhz_avg": _avg(rows, "sm_mhz_avg"),
            "sm_mhz_min": min((r["sm_mhz_min"] for r in rows if r["sm_mhz_min"] is not None), default=None),
            "util_avg": _avg(rows, "util_avg"),
            "util_min_device": min(utils) if utils else None,
            "util_max_device": max(utils) if utils else None,
            "util_mem_avg": _avg(rows, "util_mem_avg"),
            "temp_c_max": max(r["temp_c_max"] for r in rows),
            "throttled": any(r["throttled"] for r in rows),
            "throttle": sorted({name for r in rows for name in r["throttle"]}),
            "devices": rows,
        }

    anon = [tick["h"][H_ANON] for tick in ticks]
    file_ = [tick["h"][H_FILE] for tick in ticks]
    shmem = [tick["h"][H_SHMEM] for tick in ticks]
    host = {
        "anon_start_mib": _kb_to_mib(anon[0]),
        "anon_end_mib": _kb_to_mib(anon[-1]),
        "anon_peak_mib": _kb_to_mib(max(anon)),
        "file_start_mib": _kb_to_mib(file_[0]),
        "file_end_mib": _kb_to_mib(file_[-1]),
        "file_peak_mib": _kb_to_mib(max(file_)),
        "shmem_peak_mib": _kb_to_mib(max(shmem)),
        "pinned_peak_mib": _kb_to_mib(max(tick["h"][H_PIN] for tick in ticks)),
        "swap_peak_mib": _kb_to_mib(max(tick["h"][H_SWAP] for tick in ticks)),
        "threads_max": max(tick["h"][H_THREADS] for tick in ticks),
        "user_s": round(_delta(series.t, series.cpu["user"], t0, t1) / CLK_TCK, 3),
        "sys_s": round(_delta(series.t, series.cpu["sys"], t0, t1) / CLK_TCK, 3),
        "major_faults": round(_delta(series.t, series.cpu["majflt"], t0, t1)),
    }
    host["busy_cores"] = round((host["user_s"] + host["sys_s"]) / wall, 2) if wall > 0 else None
    if series.io_t:
        host["disk_read_mib"] = round(_delta(series.io_t, [v[0] for v in series.io], t0, t1) / (1 << 20), 1)
        host["disk_write_mib"] = round(_delta(series.io_t, [v[1] for v in series.io], t0, t1) / (1 << 20), 1)
    out["host"] = host
    flows, pressure, system_cpu = _flows(series, t0, t1, devices, wall)
    if series.nvlink_t:
        tx = 0.0
        for dev in devices:
            usable = [(t, v[dev]) for t, v in zip(series.nvlink_t, series.nvlink) if v[dev] is not None]
            if usable:
                tx += _delta([u[0] for u in usable], [u[1][0] for u in usable], t0, t1)
        # Every byte one GPU sends another receives, so transmit alone is the traffic.
        flows["nvlink_mib"] = round(tx / 1024, 1)
    if flows:
        out["flows"] = flows
    if pressure:
        out["pressure"] = pressure
    if system_cpu:
        out["system_cpu"] = system_cpu
    return out


def _avg(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return round(sum(values) / len(values), 1) if values else None


def thread_cpu_by_group(series, phases):
    """CPU seconds per thread per phase group, from the per-thread snapshots on slow ticks. Numbers in thread names are dropped
    so the `cuda-EvtHandlr` threads of all GPUs add up; the busiest single thread is kept apart, because one saturated thread
    is a bottleneck that an average over many threads hides."""
    if not phases:
        return {}
    starts = [p["t"] for p in phases]
    last = {}
    by_name = {}
    by_tid = {}
    for tick in series.slow:
        threads = tick["s"].get("threads")
        if not threads:
            continue
        i = bisect.bisect_right(starts, tick["t"]) - 1
        group = phases[i]["group"] if i >= 0 else "startup"
        for tid, (comm, ticks) in threads.items():
            delta = ticks - last.get(tid, 0)
            last[tid] = ticks
            if delta > 0:
                name = re.sub(r"[0-9]+", "#", comm)
                bucket = by_name.setdefault(group, {})
                bucket[name] = bucket.get(name, 0.0) + delta / CLK_TCK
                per_tid = by_tid.setdefault(group, {})
                entry = per_tid.setdefault(tid, [comm, 0.0])
                entry[1] += delta / CLK_TCK
    out = {}
    for group, names in by_name.items():
        top = max(by_tid[group].values(), key=lambda entry: entry[1])
        out[group] = {
            "threads": {name: round(seconds, 2) for name, seconds in sorted(names.items(), key=lambda kv: -kv[1])[:8]},
            "busiest": {"name": top[0], "cpu_s": round(top[1], 2)},
        }
    return out


def _sum_flows(items):
    """Adds the moved bytes of many phases. Busy percentages become time-weighted, gauges keep their extreme."""
    out = {}
    wall_total = sum(wall for wall, _ in items) or 1.0
    disks = {}
    for wall, flows in items:
        for name, disk in flows.get("disks", {}).items():
            entry = disks.setdefault(name, {"read_mib": 0.0, "write_mib": 0.0, "busy_pct": 0.0})
            entry["read_mib"] += disk["read_mib"]
            entry["write_mib"] += disk["write_mib"]
            entry["busy_pct"] += (disk["busy_pct"] or 0.0) * wall / wall_total
    if disks:
        out["disks"] = {name: {k: round(v, 1) for k, v in entry.items()} for name, entry in disks.items()}
    for key in ("swap_in_mib", "swap_out_mib", "page_cache_refault_mib", "direct_reclaim_mib", "page_cache_change_mib",
                "network_rx_mib", "network_tx_mib", "pcie_to_gpu_mib_est", "pcie_from_gpu_mib_est", "nvlink_mib"):
        values = [flows[key] for _, flows in items if key in flows]
        if values:
            out[key] = round(sum(values), 1)
    values = [flows["mem_available_min_mib"] for _, flows in items if "mem_available_min_mib" in flows]
    if values:
        out["mem_available_min_mib"] = min(values)
    values = [flows["swap_used_max_mib"] for _, flows in items if "swap_used_max_mib" in flows]
    if values:
        out["swap_used_max_mib"] = max(values)
    return out


def aggregate_group(metrics_list):
    """Sums and peaks over all phases of one group."""
    out = {"phases": len(metrics_list), "wall_s": round(sum(m["wall_s"] for m in metrics_list), 6)}
    tokens = [m["tokens"] for m in metrics_list if m.get("tokens")]
    if tokens:
        out["tokens"] = sum(tokens)
    wall = out["wall_s"]
    gpu = [m["gpu"] for m in metrics_list if m.get("gpu")]
    if gpu:
        energies = [g["energy_j"] for g in gpu]
        energy = sum(energies) if all(e is not None for e in energies) else None
        out["gpu"] = {
            "mem_start_mib": gpu[0]["mem_start_mib"],
            "mem_end_mib": gpu[-1]["mem_end_mib"],
            "mem_peak_mib": max(g["mem_peak_mib"] for g in gpu),
            "mem_peak_max_device_mib": max(g["mem_peak_max_device_mib"] for g in gpu),
            "energy_j": round(energy, 3) if energy is not None else None,
            "power_avg_w": round(energy / wall, 2) if energy is not None and wall > 0 else None,
            "power_gauge_w": _weighted(metrics_list, "power_gauge_w"),
            "sm_mhz_avg": _weighted(metrics_list, "sm_mhz_avg"),
            "sm_mhz_min": min((g["sm_mhz_min"] for g in gpu if g["sm_mhz_min"] is not None), default=None),
            "util_avg": _weighted(metrics_list, "util_avg"),
            "util_min_device": _weighted(metrics_list, "util_min_device"),
            "util_max_device": _weighted(metrics_list, "util_max_device"),
            "util_mem_avg": _weighted(metrics_list, "util_mem_avg"),
            "temp_c_max": max(g["temp_c_max"] for g in gpu),
            "throttled": any(g["throttled"] for g in gpu),
            "throttle": sorted({name for g in gpu for name in g["throttle"]}),
        }
        if energy is not None and out.get("tokens"):
            out["gpu"]["j_per_token"] = round(energy / out["tokens"], 4)
    host = [m["host"] for m in metrics_list if m.get("host")]
    if host:
        user = sum(h["user_s"] for h in host)
        system = sum(h["sys_s"] for h in host)
        out["host"] = {
            "anon_start_mib": host[0]["anon_start_mib"],
            "anon_end_mib": host[-1]["anon_end_mib"],
            "anon_peak_mib": max(h["anon_peak_mib"] for h in host),
            "file_start_mib": host[0]["file_start_mib"],
            "file_end_mib": host[-1]["file_end_mib"],
            "file_peak_mib": max(h["file_peak_mib"] for h in host),
            "shmem_peak_mib": max(h["shmem_peak_mib"] for h in host),
            "pinned_peak_mib": max(h["pinned_peak_mib"] for h in host),
            "swap_peak_mib": max(h["swap_peak_mib"] for h in host),
            "threads_max": max(h["threads_max"] for h in host),
            "user_s": round(user, 3),
            "sys_s": round(system, 3),
            "busy_cores": round((user + system) / wall, 2) if wall > 0 else None,
            "major_faults": sum(h["major_faults"] for h in host),
        }
        for key in ("disk_read_mib", "disk_write_mib"):
            values = [h[key] for h in host if key in h]
            if values:
                out["host"][key] = round(sum(values), 1)
    flows = [(m["wall_s"], m["flows"]) for m in metrics_list if m.get("flows")]
    if flows:
        out["flows"] = _sum_flows(flows)
    pressures = [(m["wall_s"], m["pressure"]) for m in metrics_list if m.get("pressure")]
    if pressures:
        total = sum(w for w, _ in pressures) or 1.0
        keys = {k for _, p in pressures for k in p}
        out["pressure"] = {k: round(sum(p.get(k, 0.0) * w for w, p in pressures) / total, 1) for k in sorted(keys)}
    system_cpu = [(m["wall_s"], m["system_cpu"]) for m in metrics_list if m.get("system_cpu")]
    if system_cpu:
        total = sum(w for w, _ in system_cpu) or 1.0
        keys = {k for _, p in system_cpu for k in p}
        out["system_cpu"] = {k: round(sum(p.get(k, 0.0) * w for w, p in system_cpu) / total, 2) for k in sorted(keys)}
    return out


def _weighted(metrics_list, key):
    total = weight = 0.0
    for m in metrics_list:
        gpu = m.get("gpu")
        if gpu and gpu.get(key) is not None:
            total += gpu[key] * m["wall_s"]
            weight += m["wall_s"]
    return round(total / weight, 1) if weight else None


GROUP_ORDER = ["baseline", "startup", "run", "download", "load", "setup", "warmup", "prefill", "decode", "serve", "observe",
               "teardown", "idle", "tail"]


def energy_self_check(groups):
    """Power from the energy counter against the mean of the power gauge, two independent readings, over the longest working phase. The
    counter has been seen to under-report by half when read too often, and nothing else in a report would show that."""
    candidates = [g for name, g in groups.items() if name in ("decode", "prefill", "serve", "run", "observe") and g.get("wall_s", 0) >= MIN_WALL_FOR_ENERGY_PER_TOKEN_S]
    candidates = [g for g in candidates if (g.get("gpu") or {}).get("power_avg_w") and (g.get("gpu") or {}).get("power_gauge_w")]
    if not candidates:
        return None
    gpu = max(candidates, key=lambda g: g["wall_s"])["gpu"]
    return round(gpu["power_avg_w"] / gpu["power_gauge_w"], 3)


def throughput_from_phases(groups):
    """Tokens per second from marked phases, for engines that report nothing themselves. It includes whatever else happens inside
    the phase, which is why an engine's own timers are preferred when there are any."""
    out = {}
    for name in ("prefill", "decode"):
        group = groups.get(name) or {}
        if group.get("tokens") and group.get("wall_s"):
            out[name] = {"tokens_per_s": round(group["tokens"] / group["wall_s"], 3), "source": "phase time"}
    return out


def summarize(ticks, phases, t_end):
    """phases: [{"name", "group", "t", ...}] sorted by t. Adds t_end and metrics in place, returns the report sections."""
    series = Series(ticks)
    for i, phase in enumerate(phases):
        phase["t_end"] = phases[i + 1]["t"] if i + 1 < len(phases) else t_end

    baseline = [p for p in phases if p["group"] == "baseline"]
    baseline_mem = [0] * series.n_gpus
    if baseline and ticks:
        i, j = series.window(baseline[0]["t"], baseline[0]["t_end"])
        for dev in range(series.n_gpus):
            values = [tick["g"][dev][G_MEM] or 0 for tick in ticks[i:j]]
            baseline_mem[dev] = round(sum(values) / len(values)) if values else 0
    devices = used_devices(series, baseline_mem)

    groups = {}
    for phase in phases:
        phase["metrics"] = phase_metrics(series, phase["t"], phase["t_end"], devices, baseline_mem)
        if phase.get("tokens") is not None:
            phase["metrics"]["tokens"] = phase["tokens"]
        groups.setdefault(phase["group"], []).append(phase["metrics"])
    aggregated = {group: aggregate_group(metrics) for group, metrics in groups.items()}
    for group, threads in thread_cpu_by_group(series, phases).items():
        if group in aggregated:
            host = aggregated[group].setdefault("host", {})
            host["threads_cpu_s"] = threads["threads"]
            host["busiest_thread"] = threads["busiest"]
            wall = aggregated[group]["wall_s"]
            if wall > 0:
                host["busiest_thread"]["busy"] = round(min(1.0, threads["busiest"]["cpu_s"] / wall), 3)
    ordered = {group: aggregated[group] for group in GROUP_ORDER if group in aggregated}
    ordered.update({group: value for group, value in aggregated.items() if group not in ordered})

    self_peak = [0] * series.n_gpus
    for tick in series.slow:
        for dev, mib in enumerate(tick["s"].get("self_mib", [])):
            self_peak[dev] = max(self_peak[dev], mib)
    gpu_devices = []
    for dev in devices:
        peak = max((tick["g"][dev][G_MEM] or 0) for tick in ticks)
        gpu_devices.append({
            "index": dev,
            "baseline_mib": baseline_mem[dev],
            "peak_mib": peak,
            "peak_above_baseline_mib": peak - baseline_mem[dev],
            "process_peak_mib": self_peak[dev],
        })
    tail = [p for p in phases if p["group"] == "tail"]
    if tail and ticks:
        last = ticks[-1]["g"]
        for entry in gpu_devices:
            entry["after_exit_mib"] = last[entry["index"]][G_MEM]
    memory = {
        "gpu": {
            "devices": gpu_devices,
            "peak_above_baseline_total_mib": sum(d["peak_above_baseline_mib"] for d in gpu_devices),
            "peak_above_baseline_max_device_mib": max((d["peak_above_baseline_mib"] for d in gpu_devices), default=0),
        },
    }
    if ticks:
        memory["host"] = {
            "anon_peak_mib": _kb_to_mib(max(t["h"][H_ANON] for t in ticks)),
            "file_peak_mib": _kb_to_mib(max(t["h"][H_FILE] for t in ticks)),
            "shmem_peak_mib": _kb_to_mib(max(t["h"][H_SHMEM] for t in ticks)),
            "pinned_peak_mib": _kb_to_mib(max(t["h"][H_PIN] for t in ticks)),
            "swap_peak_mib": _kb_to_mib(max(t["h"][H_SWAP] for t in ticks)),
            "vm_hwm_sum_mib": _kb_to_mib(max(t["h"][H_HWM] for t in ticks)),
        }
    energy = {}
    measured_groups = [g for g in ordered if g not in ("baseline", "tail")]
    totals = [ordered[g].get("gpu", {}).get("energy_j") for g in measured_groups]
    if totals and all(v is not None for v in totals):
        energy["total_j"] = round(sum(totals), 3)
    for group in ("prefill", "decode", "serve"):
        j_per_token = ordered.get(group, {}).get("gpu", {}).get("j_per_token")
        if j_per_token is not None and ordered[group]["wall_s"] >= MIN_WALL_FOR_ENERGY_PER_TOKEN_S:
            energy[f"{group}_j_per_token"] = j_per_token
    return {"groups": ordered, "memory": memory, "energy": energy, "devices_used": devices}
