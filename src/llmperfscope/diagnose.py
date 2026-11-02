"""Reads the numbers and says what they mean: what limits each phase, whether memory is being overused, and where data is
travelling between disk, RAM and GPU that it should not have to.

Every statement carries the numbers it rests on, and the report shows them. A rule only fires on evidence that survives short
runs and shared machines; when a phase is too short to judge it says so instead of guessing.
"""

from __future__ import annotations

from . import system

MIN_WALL_FOR_UTILIZATION_S = 1.0
ACTIVE_GROUPS = ("load", "setup", "warmup", "prefill", "decode", "serve", "run", "observe")
WORK_GROUPS = ("prefill", "decode", "serve", "run", "observe")


def _rate(mib, wall):
    return mib / wall if wall > 0 else 0.0


def _gib(mib):
    return f"{mib / 1024:,.2f} GiB"


def _disk_kinds(run):
    return run.get("disks") or {}


def _busiest_disk(group, disks):
    best = None
    for name, disk in (group.get("flows", {}).get("disks") or {}).items():
        if best is None or (disk.get("busy_pct") or 0) > (best[1].get("busy_pct") or 0):
            best = (name, disk)
    if not best:
        return None
    info = disks.get(best[0], {})
    return {"name": best[0], "kind": info.get("kind", "disk"), **best[1]}


def really_throttled(gpu, run):
    """NVML sets the software power cap bit on GPUs far below their limit, so that bit alone needs the power to back it up. The
    thermal and hardware reasons mean what they say."""
    if not gpu.get("throttled"):
        return False
    reasons = set(gpu.get("throttle") or [])
    if reasons & {"hw_slowdown", "sw_thermal", "hw_thermal", "hw_power_brake"}:
        return True
    if "sw_power_cap" not in reasons:
        return False
    used = run.get("devices_used") or []
    limits = [d.get("power_limit_w") for d in run.get("devices") or [] if d.get("index") in used and d.get("power_limit_w")]
    if not limits or not gpu.get("power_avg_w"):
        return False
    return gpu["power_avg_w"] / len(limits) >= 0.9 * (sum(limits) / len(limits))


def _on_macos(run):
    return str((run.get("host") or {}).get("os", "")).startswith("macOS")


def _unified_memory(run):
    return any(d.get("unified_memory") for d in run.get("devices") or [])


def _gpu_tool(run):
    """Where to look next for what the GPU does."""
    return "Instruments' Metal System Trace shows which shaders take the time" if _on_macos(run) else "Use --nsys to see which kernels take the time"


def _host_tool(run):
    """Where to look next for what the host does."""
    return "`sample` shows what it does" if _on_macos(run) else "use --cpu-profile to see what it does"


def classify_phase(name, group, run):
    """What limited this phase. Returns {"limit", "confidence", "why": text, "evidence": {...}}."""
    wall = group.get("wall_s", 0.0)
    gpu = group.get("gpu") or {}
    host = group.get("host") or {}
    flows = group.get("flows") or {}
    pressure = group.get("pressure") or {}
    disks = _disk_kinds(run)
    evidence = {}

    util = gpu.get("util_avg")
    util_max = gpu.get("util_max_device")
    util_min = gpu.get("util_min_device")
    if util is not None:
        evidence["gpu_util_pct"] = util
    busy_cores = host.get("busy_cores")
    if busy_cores is not None:
        evidence["cpu_cores"] = busy_cores
    busiest = host.get("busiest_thread") or {}
    thread_busy = busiest.get("busy")
    if thread_busy is not None:
        evidence["busiest_thread"] = f"{busiest.get('name')} {thread_busy * 100:.0f}%"
    for key in ("io_some_pct", "memory_some_pct", "memory_full_pct", "cpu_some_pct"):
        if key in pressure:
            evidence[f"stall_{key[:-4]}_pct"] = pressure[key]

    swap_rate = _rate(flows.get("swap_in_mib", 0.0) + flows.get("swap_out_mib", 0.0), wall)
    memory_stall = pressure.get("memory_some_pct", 0.0)
    refault_rate = _rate(flows.get("page_cache_refault_mib", 0.0), wall)
    thrashing = refault_rate > 50.0 and memory_stall > 2.0
    if _on_macos(run):
        # macOS swaps and compresses other programs' idle memory as a matter of course, and it has no measure of tasks waiting on memory. Its critical pressure level says that RAM
        # is short. Short of that, RAM is only the limit if nothing else is busy.
        compressor_rate = _rate(flows.get("compressed_mib", 0.0) + flows.get("decompressed_mib", 0.0), wall)
        critical = pressure.get("memory_full_pct", 0.0) > 5.0
        squeezed = memory_stall > 2.0 and (swap_rate > 1.0 or compressor_rate > 50.0) and (util is None or util < 50) and (busy_cores or 0.0) < 2.0
        if critical or squeezed:
            causes = [f"{swap_rate:,.0f} MiB/s of swap" if swap_rate > 1.0 else None,
                      f"{compressor_rate:,.0f} MiB/s through the memory compressor" if compressor_rate > 1.0 else None,
                      f"macOS's memory pressure at warning or above {memory_stall:.0f}% of the time"]
            evidence["swap_mib_per_s"], evidence["compressor_mib_per_s"] = round(swap_rate, 1), round(compressor_rate, 1)
            return {"limit": "ram", "confidence": "high" if critical else "medium", "evidence": evidence, "why": "Host RAM is short: " + ", ".join(c for c in causes if c) + "."}
    elif swap_rate > 1.0 or pressure.get("memory_full_pct", 0.0) > 5.0 or thrashing:
        causes = []
        if swap_rate > 1.0:
            causes.append(f"{swap_rate:,.0f} MiB/s of swap")
            evidence["swap_mib_per_s"] = round(swap_rate, 1)
        if refault_rate > 1.0:
            causes.append(f"{refault_rate:,.0f} MiB/s of file pages dropped and read again")
            evidence["refault_mib_per_s"] = round(refault_rate, 1)
        causes.append(f"tasks stalled on memory {memory_stall:.0f}% of the time")
        return {"limit": "ram", "confidence": "high", "evidence": evidence, "why": "Host RAM is short: " + ", ".join(causes) + "."}

    disk = _busiest_disk(group, disks)
    own_read = host.get("disk_read_mib")
    read_mib = own_read if own_read is not None else (disk or {}).get("read_mib", 0.0)
    io_stall = pressure.get("io_some_pct", 0.0)
    if disk:
        evidence["disk"] = f"{disk['name']} ({disk['kind']}) {disk.get('busy_pct', 0):.0f}% busy, {_rate(disk['read_mib'], wall):.0f} MiB/s read"
    if read_mib and _rate(read_mib, wall) > 5.0 and (io_stall > 25.0 or (disk and (disk.get("busy_pct") or 0) > 60.0)):
        kind = disk["kind"] if disk else "disk"
        return {"limit": f"disk ({kind})", "confidence": "high", "evidence": evidence,
                "why": f"Waiting on {kind} reads: {_rate(read_mib, wall):.0f} MiB/s, " + (f"I/O stall {io_stall:.0f}% of the time." if "io_some_pct" in pressure else f"the disk {(disk or {}).get('busy_pct', 0):.0f}% busy.")}

    if wall < MIN_WALL_FOR_UTILIZATION_S:
        return {"limit": "too short to judge", "confidence": "none", "evidence": evidence,
                "why": f"{wall:.2f} s is too short for the GPU's utilization counter. Use a longer run."}

    if util is not None and util > 50 and really_throttled(gpu, run):
        reasons = "macOS's thermal pressure" if _on_macos(run) else ", ".join(gpu.get("throttle", []))
        clock = f", {gpu['sm_mhz_avg']:.0f} MHz average" if gpu.get("sm_mhz_avg") else ""
        return {"limit": "gpu (throttled)", "confidence": "high", "evidence": evidence,
                "why": f"GPU clocks were held down by {reasons} at {util:.0f}% utilization{clock}."}

    if util is not None and util_max is not None and util_min is not None and util_max - util_min > 25 and util_max >= 80:
        return {"limit": "gpu imbalance", "confidence": "high", "evidence": evidence,
                "why": f"Work is uneven across GPUs: {util_min:.0f}% to {util_max:.0f}%. The busiest one sets the pace."}
    if util is not None and util >= 85:
        return {"limit": "gpu", "confidence": "high", "evidence": evidence,
                "why": f"GPU compute is the limit: {util:.0f}% busy. {_gpu_tool(run)}."}

    cpu_stall = pressure.get("cpu_some_pct", 0.0)
    if cpu_stall > 30.0 and (run.get("host", {}).get("cpus") or 0) > 0:
        return {"limit": "cpu contention", "confidence": "medium", "evidence": evidence,
                "why": f"Runnable tasks waited for a CPU {cpu_stall:.0f}% of the time. Something else is competing for the cores."}

    gpu_percent = util if util is not None else 0
    if (util is None or util < 30) and busy_cores is not None and busy_cores >= 2.0:
        return {"limit": "cpu (many threads)", "confidence": "high", "evidence": evidence,
                "why": f"{busy_cores:.1f} CPU cores are busy while the GPU is {gpu_percent:.0f}% busy: the work is on the CPU."}

    if (util is None or util < 30) and thread_busy is not None and thread_busy >= 0.7:
        return {"limit": "cpu (one thread)", "confidence": "high", "evidence": evidence,
                "why": f"Thread {busiest.get('name')} runs at {thread_busy * 100:.0f}% while the GPU is {gpu_percent:.0f}% busy. Everything waits on "
                       f"this one thread; {_host_tool(run)}."}

    # The GPU does a real part of the work and also sits idle: the idle part goes to host work, or to the two waiting for each other.
    if util is not None and util >= 30:
        spinning = thread_busy is not None and thread_busy >= 0.7
        detail = f"GPU {util:.0f}% busy with nothing else saturated, so the rest of the time is host work or synchronization."
        if spinning:
            detail = (f"GPU {util:.0f}% busy while thread {busiest.get('name')} runs at {thread_busy * 100:.0f}%: the host is working or spinning "
                      "on a wait during the GPU's idle part.")
        return {"limit": "gpu + host wait", "confidence": "medium", "evidence": evidence,
                "why": detail + (" Instruments' Metal System Trace shows the gaps, `sample` what the host does." if _on_macos(run) else " --nsys shows the gaps, --cpu-profile what the host does.")}

    if busy_cores is not None and busy_cores >= 0.7:
        return {"limit": "cpu", "confidence": "low", "evidence": evidence,
                "why": f"{busy_cores:.1f} CPU cores busy and the GPU {gpu_percent:.0f}% busy: host work, spread over threads that take turns."}

    if name in ("serve", "observe"):
        return {"limit": "client or idle", "confidence": "low", "evidence": evidence,
                "why": "Nothing is saturated: the server has spare capacity at this load."}
    return {"limit": "none visible", "confidence": "low", "evidence": evidence,
            "why": "No GPU, CPU, disk or memory limit is visible. The wait is somewhere these counters do not see."}


def memory_findings(run):
    """Is memory being overused? Each finding says how much, where, and when."""
    findings = []
    memory = run.get("memory") or {}
    groups = run.get("groups") or {}
    devices = {d["index"]: d for d in run.get("devices") or []}
    gpu_devices = (memory.get("gpu") or {}).get("devices") or []

    def add(severity, what, text, **evidence):
        findings.append({"severity": severity, "what": what, "text": text, "evidence": evidence})

    for entry in gpu_devices:
        device = devices.get(entry["index"], {})
        total = device.get("memory_total_mib")
        if total and entry["peak_mib"] >= 0.95 * total:
            if device.get("memory_total_is_working_set_limit"):
                add("warning", "gpu_memory_nearly_full",
                    f"The GPU reached {entry['peak_mib']:,} MiB of the {total:,} MiB macOS recommends it hold. Past that, Metal starts paging and inference slows down sharply. "
                    "`sudo sysctl iogpu.wired_limit_mb=N` raises the limit until the next restart, at the expense of everything else in RAM.", peak_mib=entry["peak_mib"], total_mib=total)
            else:
                add("warning", "gpu_memory_nearly_full",
                    f"GPU {entry['index']} reached {entry['peak_mib']:,} of {total:,} MiB. There is no headroom left: a longer prompt or more "
                    "parallel requests can run out of memory.", peak_mib=entry["peak_mib"], total_mib=total)
    peaks = [e["peak_above_baseline_mib"] for e in gpu_devices]
    if len(peaks) >= 2 and min(peaks) > 0 and max(peaks) > 1.25 * min(peaks) and max(peaks) - min(peaks) > 256:
        add("info", "gpu_memory_uneven",
            f"GPUs are filled unevenly: {min(peaks):,} to {max(peaks):,} MiB. The fullest one decides how large a model or context fits.",
            min_mib=min(peaks), max_mib=max(peaks))

    steady_group = next((groups[g] for g in ("decode", "serve", "observe") if g in groups), None)
    if steady_group and steady_group.get("gpu"):
        peak_all = max((g.get("gpu", {}).get("mem_peak_mib", 0) for g in groups.values()), default=0)
        steady = steady_group["gpu"].get("mem_end_mib", 0)
        peak_group = max(groups, key=lambda g: groups[g].get("gpu", {}).get("mem_peak_mib", 0))
        if peak_all - steady > max(256, 0.05 * max(steady, 1)):
            add("info", "gpu_transient_peak",
                f"GPU memory peaks {peak_all - steady:,} MiB above its steady level, during {peak_group}. That peak decides what fits, "
                "not the steady level.", peak_mib=peak_all, steady_mib=steady, phase=peak_group)

    for name in ("decode", "serve"):
        group = groups.get(name) or {}
        host = group.get("host") or {}
        gpu = group.get("gpu") or {}
        anon = host.get("anon_peak_mib", 0)
        gpu_mem = gpu.get("mem_peak_mib", 0)
        # In one pool of memory the two are the same pages, not a copy.
        separate = not _unified_memory(run)
        if separate and anon > 2048 and gpu_mem > 0 and anon > 0.25 * gpu_mem:
            add("info", "host_ram_held_while_on_gpu",
                f"The process holds {_gib(anon)} of its own host RAM during {name}, next to {_gib(gpu_mem)} on the GPU. If that is a copy of "
                "the weights it can usually be released after upload.", host_anon_mib=anon, gpu_mib=gpu_mem, phase=name)
        file_mib = host.get("file_peak_mib", 0)
        if separate and file_mib > 4096 and gpu_mem > 0:
            add("info", "weights_mapped_in_page_cache",
                f"{_gib(file_mib)} of mapped files stay in RAM during {name}. These pages are reclaimable page cache, not a leak, but they "
                "are why a later run loads fast and why the machine shows little free RAM.", file_mib=file_mib, phase=name)
        start, end = host.get("anon_start_mib", 0), host.get("anon_end_mib", 0)
        if end - start > max(128, 0.05 * max(start, 1)):
            # One run cannot tell a leak from a cache filling up once, so only large growth is worth a warning.
            add("warning" if end - start > max(1024, 0.2 * max(start, 1)) else "info", "host_memory_growing",
                f"Host memory grew {end - start:,.0f} MiB during {name}, from {_gib(start)} to {_gib(end)}. If it keeps growing with more "
                "requests it is a leak or an unbounded cache; if it levels off, a cache filling up.", start_mib=start, end_mib=end, phase=name)
        start, end = gpu.get("mem_start_mib", 0), gpu.get("mem_end_mib", 0)
        if end - start > max(256, 0.05 * max(start, 1)):
            add("info", "gpu_memory_growing",
                f"GPU memory grew {end - start:,} MiB during {name}. Buffers sized on demand can do this once; if it continues, look for growth "
                "with context length.", start_mib=start, end_mib=end, phase=name)

    host_memory = memory.get("host") or {}
    if host_memory.get("swap_peak_mib", 0) > 64:
        add("warning", "process_swapped",
            f"{host_memory['swap_peak_mib']:,.0f} MiB of the process was compressed by macOS. Touching those pages later means decompressing them, or reading them from disk if they "
            "went on to swap, in the middle of inference." if _on_macos(run) else
            f"{host_memory['swap_peak_mib']:,.0f} MiB of the process was in swap. Touching those pages later means disk reads in the middle of "
            "inference.", swap_mib=host_memory["swap_peak_mib"])
    lowest = None
    total_ram = (run.get("machine_before") or {}).get("mem_total_mib")
    for group in groups.values():
        value = (group.get("flows") or {}).get("mem_available_min_mib")
        if value is not None and (lowest is None or value < lowest):
            lowest = value
    if lowest is not None and total_ram and lowest < 0.05 * total_ram:
        add("warning", "host_ram_nearly_full",
            f"Available host RAM fell to {lowest:,} MiB of {total_ram:,}. The kernel starts dropping cache and swapping around here.",
            available_mib=lowest, total_mib=total_ram)

    for row in memory.get("breakdown_mib") or []:
        if row.get("kind") == "host" and (row.get("model") or 0) > 1024 and gpu_devices:
            add("info", "weights_on_host",
                f"{_gib(row['model'])} of the model stays in host buffers, not on a GPU. CPU work reads it from there, so with mmap it depends on the "
                "page cache, and the disk when the cache has been evicted.", host_model_mib=row["model"])
        if row.get("kind") == "device" and row.get("self"):
            # llama.cpp's table for its tensor-split device has the whole group's total but one GPU's share of the buffers. Taking that
            # at face value would "find" most of the memory unaccounted for.
            if str(row.get("name", "")).startswith("Meta"):
                continue
            unaccounted = row.get("unaccounted") or 0
            if unaccounted > 0.15 * row["self"] and unaccounted > 256:
                add("info", "gpu_memory_unaccounted",
                    f"{row.get('name')}: {unaccounted:,} MiB in use that the engine cannot account for, against {row['self']:,} MiB it can. "
                    "Other processes or driver overhead hold it.", unaccounted_mib=unaccounted, self_mib=row["self"])
            compute = row.get("compute") or 0
            if compute > 0.3 * row["self"] and compute > 256:
                add("info", "compute_buffers_large",
                    f"{row.get('name')}: compute buffers take {compute:,} MiB of {row['self']:,} MiB. They scale with batch size, so a smaller "
                    "batch frees memory.", compute_mib=compute, self_mib=row["self"])

    torch_stats = run.get("torch_memory") or {}
    reserved, allocated = torch_stats.get("max_reserved_mib"), torch_stats.get("max_allocated_mib")
    if reserved and allocated and reserved > 1.3 * allocated and reserved - allocated > 512:
        add("info", "torch_reserved_above_allocated",
            f"PyTorch reserved {reserved:,.0f} MiB but its tensors peaked at {allocated:,.0f} MiB. The difference sits in its allocator cache.",
            reserved_mib=reserved, allocated_mib=allocated)

    # The fullest pool decides. Engines publish several related gauges, and one finding about them is enough.
    pools = [(name.split("{")[0], peak) for name, peak in ((run.get("workload") or {}).get("server_gauges_peak") or {}).items()
             if peak is not None and any(word in name.lower().split("{")[0] for word in ("kv_cache_usage", "gpu_cache_usage", "token_usage"))]
    if pools:
        name, peak = max(pools, key=lambda item: item[1] / 100 if item[1] > 1.0 else item[1])
        fraction = peak / 100 if peak > 1.0 else peak
        if fraction < 0.25:
            add("info", "kv_cache_mostly_unused",
                f"The server's cache pool peaked at {fraction * 100:.0f}% ({name}). Most of the memory reserved for it was never used at this "
                "load; a smaller reservation would fit, unless a heavier load is coming.", metric=name, peak=peak)
        elif fraction > 0.95:
            add("warning", "kv_cache_full",
                f"The server's cache pool reached {fraction * 100:.0f}% ({name}). Requests are probably waiting or being preempted.",
                metric=name, peak=peak)

    # After attaching, the process is still running when the profile ends, so it still holding its memory says nothing.
    tail_group = groups.get("tail") or {}
    if tail_group and not run.get("attached"):
        for entry in gpu_devices:
            after = entry.get("after_exit_mib")
            if after is not None and after - entry["baseline_mib"] > 128:
                add("info", "gpu_memory_after_exit",
                    f"GPU {entry['index']} still holds {after - entry['baseline_mib']:,} MiB more than before the run, 1 s after it ended.",
                    after_mib=after, baseline_mib=entry["baseline_mib"])
    return findings


def movement_findings(run):
    """Data going back and forth between tiers."""
    findings = []
    groups = run.get("groups") or {}
    before = run.get("machine_before") or {}
    disks = _disk_kinds(run)

    def add(severity, what, text, **evidence):
        findings.append({"severity": severity, "what": what, "text": text, "evidence": evidence})

    for name in ("startup", "load", "warmup"):
        group = groups.get(name) or {}
        cores = (group.get("host") or {}).get("busy_cores")
        busy = (group.get("system_cpu") or {}).get("busy_pct")
        if group.get("wall_s", 0.0) >= 2.0 and cores is not None and cores < 0.1 and busy is not None and busy > 25.0:
            # Measured: Metal compiled llama.cpp's shaders in a system process for 8.7 s on a first run, and took milliseconds on the next.
            add("info", "work_in_another_process",
                f"During {name} the process used {cores:.2f} CPU cores for {group['wall_s']:,.1f} s while the machine's CPUs were {busy:.0f}% busy, so another process is doing the work or "
                "competing with it." + (" On a Mac this is usually the Metal shader compiler on first use. macOS keeps the result, so the next run does not pay for it." if _on_macos(run) else ""),
                wall_s=group["wall_s"], cores=cores, system_busy_pct=busy, phase=name)

    warmup = groups.get("warmup") or {}
    if warmup.get("wall_s", 0.0) > 5.0:
        grown = (warmup.get("gpu") or {}).get("mem_end_mib", 0) - (warmup.get("gpu") or {}).get("mem_start_mib", 0)
        add("info", "slow_first_use",
            f"The first use took {warmup['wall_s']:,.1f} s, and GPU memory grew {grown:,} MiB in it. Engines that load models on demand or compile "
            "kernels on first use do this: the first user of a fresh server waits this long.", wall_s=warmup["wall_s"], gpu_growth_mib=grown)

    load = groups.get("load") or {}
    location = before.get("model_storage")
    kind = system.storage_kind(location)
    if kind in ("hdd", "network", "disk"):
        # An array of rotating disks reports as rotational too and can be fast, so the speed actually measured decides how much this matters.
        host_load = load.get("host") or {}
        read_rate = _rate(host_load.get("disk_read_mib") or 0.0, load.get("wall_s", 0.0))
        measured = (host_load.get("disk_read_mib") or 0.0) > 256
        name = location.get("path")
        if measured and read_rate >= 500:
            add("info", "model_on_slow_storage", f"The model is on {kind} storage ({name}), which delivered {read_rate:,.0f} MiB/s here.", storage=kind, read_mib_per_s=round(read_rate))
        elif measured:
            add("warning", "model_on_slow_storage", f"The model is on {kind} storage ({name}) at only {read_rate:,.0f} MiB/s. Loading depends on it, and so does inference "
                "if pages get evicted.", storage=kind, read_mib_per_s=round(read_rate))
        else:
            add("info", "model_on_slow_storage", f"The model is on {kind} storage ({name}). This run did not read it from disk; --cold-start measures what a first load costs.",
                storage=kind)

    if load:
        host = load.get("host") or {}
        read_mib = host.get("disk_read_mib")
        wall = load.get("wall_s", 0.0)
        cached = before.get("model_page_cache_fraction")
        disk = _busiest_disk(load, disks)
        # The busiest disk may be busy with something else. If the model's own disk was read at all, that is the one.
        for candidate in (location or {}).get("disks") or []:
            traffic = ((load.get("flows") or {}).get("disks") or {}).get(candidate.get("name"))
            if traffic and traffic.get("read_mib"):
                disk = {"name": candidate["name"], "kind": candidate.get("kind", "disk"), **traffic}
        if read_mib and read_mib > 256:
            where = f" from {disk['name']} ({disk['kind']})" if disk else ""
            add("info", "load_read_from_disk",
                f"Load read {_gib(read_mib)}{where} at {_rate(read_mib, wall):,.0f} MiB/s" +
                (f"; {cached * 100:.0f}% of the model was cached before the run." if cached is not None else "."),
                read_mib=read_mib, cached_fraction=cached)
        elif cached is not None and cached > 0.9:
            add("info", "load_from_page_cache",
                f"The model was {cached * 100:.0f}% in the page cache, so this load time is a warm one. A cold start reads it from disk and "
                "takes longer.", cached_fraction=cached)

    for name in ("prefill", "decode", "serve"):
        group = groups.get(name) or {}
        if not group:
            continue
        wall = group.get("wall_s", 0.0)
        host = group.get("host") or {}
        flows = group.get("flows") or {}
        tokens = group.get("tokens") or 0
        read_mib = host.get("disk_read_mib") or 0.0
        faults = host.get("major_faults") or 0
        if read_mib > 16 or faults > 256:
            add("warning", "disk_reads_during_inference",
                f"{_gib(read_mib)} read from disk and {faults:,} major page faults during {name}. The weights or cache are not staying in RAM: "
                "evicted pages, mmap of tensors used on the CPU, or an engine offloading to disk.",
                read_mib=read_mib, major_faults=faults, phase=name)
        written = host.get("disk_write_mib") or 0.0
        if written > 64:
            add("info", "disk_writes_during_inference",
                f"{_gib(written)} written to disk during {name}: a cache being offloaded to disk, or logging. Either way those writes share the "
                "disk with any reads the engine needs.", write_mib=written, phase=name)
        compressed = flows.get("decompressed_mib", 0.0) + flows.get("compressed_mib", 0.0)
        if compressed > 256 and _rate(compressed, wall) > 50.0:
            add("info", "memory_compressor_busy",
                f"{compressed:,.0f} MiB went through macOS's memory compressor during {name}, {_rate(compressed, wall):,.0f} MiB/s" +
                (f", with memory pressure at warning or above {(group.get('pressure') or {})['memory_some_pct']:.0f}% of the time" if (group.get("pressure") or {}).get("memory_some_pct") else "") +
                ". That is CPU time spent on memory, and other programs being squeezed to make room: closing them gives a steadier measurement.", compressed_mib=compressed, phase=name)
        swap_in = flows.get("swap_in_mib", 0.0)
        swap_out = flows.get("swap_out_mib", 0.0)
        if swap_in + swap_out > 16:
            add("warning", "swapping_during_inference",
                f"The system swapped {swap_in:,.0f} MiB in and {swap_out:,.0f} MiB out during {name}.", swap_in_mib=swap_in, swap_out_mib=swap_out,
                phase=name)
        pcie_in = flows.get("pcie_to_gpu_mib_est")
        pcie_out = flows.get("pcie_from_gpu_mib_est")
        # A sample every second or so says nothing about a short phase, and an idle GPU's first reading can be far off. In a mixed serve
        # phase the prompts dominate, so a per-token figure only means something for pure decode.
        if name == "decode" and tokens and pcie_in is not None and pcie_out is not None and wall >= 3.0 and ((group.get("gpu") or {}).get("util_avg") or 0) >= 10:
            per_token = (pcie_in + pcie_out) / tokens
            if per_token > 8.0:
                add("info", "heavy_host_gpu_traffic",
                    f"About {per_token:,.1f} MiB crosses PCIe per token during {name}: {pcie_in:,.0f} MiB toward the GPUs and {pcie_out:,.0f} MiB back. "
                    "That is tensors moving between RAM and GPU (partial offload, CPU-side layers), a collective going through host memory, or a GPU "
                    "polling host memory. --nsys gives the exact bytes of real copies, and whatever remains is the other two.",
                    mib_per_token=round(per_token, 2), phase=name)
        nvlink = flows.get("nvlink_mib")
        if tokens and nvlink and name != "serve":
            add("info", "gpu_to_gpu_traffic", f"{nvlink / tokens:,.2f} MiB per token crosses NVLink during {name}.",
                mib_per_token=round(nvlink / tokens, 3), phase=name)
        gpu = group.get("gpu") or {}
        util = gpu.get("util_avg")
        cores = host.get("busy_cores") or 0
        if util is not None and wall >= MIN_WALL_FOR_UTILIZATION_S and util < 50 and cores >= 3.0:
            add("info", "cpu_gpu_split",
                f"During {name} the GPU is {util:.0f}% busy while {cores:.1f} CPU cores work. Part of the model is probably running on the CPU.",
                gpu_util=util, cpu_cores=cores, phase=name)
    return findings


def timeline_findings(run):
    """From a traced run: how each GPU's time divides, and what the host does meanwhile."""
    findings = []
    nsys = run.get("nsys") or {}
    for name in ("prefill", "decode", "serve", "run"):
        group = (nsys.get("groups") or {}).get(name)
        if not group or not group.get("devices"):
            continue
        spans = [d["span_ms"] for d in group["devices"] if d["span_ms"] > 0]
        if not spans:
            continue
        n = len(group["devices"])
        span = sum(spans) / n
        kernel = sum(d["kernel_ms"] for d in group["devices"]) / n
        comm = sum(d["comm_ms"] for d in group["devices"]) / n
        copy = sum(d["copy_ms"] for d in group["devices"]) / n
        idle = sum(d["idle_ms"] for d in group["devices"]) / n
        top = group.get("top_kernels") or []
        top_text = ", ".join(f"{k['name']} {k['share'] * 100:.0f}%" for k in top[:3])
        text = (f"{name}: each GPU spends {kernel / span * 100:.0f}% of its time in kernels, {comm / span * 100:.0f}% in collectives, "
                f"{copy / span * 100:.0f}% in copies and {idle / span * 100:.0f}% idle (tracing slows the host, so idle reads high). Top kernels: {top_text}.")
        findings.append({"severity": "info", "what": "gpu_time_split", "text": text,
                         "evidence": {"phase": name, "kernel_ms": round(kernel, 2), "comm_ms": round(comm, 2), "copy_ms": round(copy, 2),
                                      "idle_ms": round(idle, 2)}})
        host_api = group.get("host_api") or {}
        if host_api.get("top"):
            top_api = ", ".join(f"{a['name']} {a['ms']:,.0f} ms" for a in host_api["top"][:3])
            findings.append({"severity": "info", "what": "host_api_time", "text": f"{name}: the host spent its CUDA calls in {top_api}.",
                             "evidence": {"phase": name}})
        copies = group.get("copies") or {}
        moved = ", ".join(f"{kind} {c['mib']:,.1f} MiB" + (f" ({c['mib_per_token']:.3f} per token)" if c.get("mib_per_token") else "")
                          for kind, c in copies.items())
        if moved:
            findings.append({"severity": "info", "what": "exact_copies", "text": f"{name}: copies traced: {moved}.", "evidence": {"phase": name}})
    return findings


_DRIVER_MODULES = ("libcuda", "libcudart", "libnvidia", "[vdso]", "libnccl")


def cpu_findings(run):
    """From a CPU-sampled run: how much host time is inside the GPU driver, and which of the engine's own functions take the rest. The
    self time is what a speed-up can act on; the callers around the main loop are in every sample and say nothing."""
    findings = []
    profile = run.get("cpu_profile") or {}
    for name in ("prefill", "decode", "serve", "run", "load"):
        group = (profile.get("groups") or {}).get(name)
        if not group or group.get("samples", 0) < 50:
            continue
        in_driver = sum(m["share"] for m in group["modules"] if m["name"].startswith(_DRIVER_MODULES))
        own = [x for x in group["self"] if not x["name"].endswith("!?") and x["share"] >= 0.03 and (_is_long_system_call(x["name"]) or not x["name"].startswith(("[", "__", "ioctl", "cu")))]
        top = ", ".join(_describe_function(x) for x in own[:4])
        text = f"{name}: {in_driver * 100:.0f}% of host CPU samples are inside the GPU driver (submitting, or spinning on a wait)."
        if top:
            text += f" Hottest functions outside it: {top}."
        findings.append({"severity": "info", "what": "host_cpu_split", "text": text, "evidence": {"phase": name, "driver_share": round(in_driver, 3)}})
        wait = group.get("gpu_wait") or {}
        if wait.get("share", 0) >= 0.05 and wait.get("sites"):
            sites = ", ".join(f"{x['name'][:48]} {x['share'] * 100:.0f}%" for x in wait["sites"][:4])
            findings.append({"severity": "info", "what": "host_waits_for_gpu", "evidence": {"phase": name, "share": wait["share"], "sites": wait["sites"][:4]},
                             "text": f"{name}: {wait['share'] * 100:.0f}% of host CPU samples wait for the GPU, in {sites}. That is not work to speed up. "
                                     "But a wait that comes before host work the next GPU step needs keeps the GPU idle during that work: ask of each "
                                     "place whether what follows it could have run while waiting."})
    return findings


def _is_long_system_call(name):
    """libc's wrappers around calls where a phase can spend real time in the kernel. Other names starting with __ are libc internals."""
    return name.lstrip("_").split("@")[0] in ("munmap", "mmap", "mmap64", "madvise", "mprotect", "read", "pread", "pread64", "write", "pwrite", "pwrite64", "fsync")


def _describe_function(row):
    """A function's share, how many threads ran it, and how much of the phase it holds up: the part a speed-up could win back."""
    threads = row.get("threads", 1)
    where = "on 1 thread" if threads == 1 else f"on {threads} threads"
    name = row["name"].split("@")[0].lstrip("_") if _is_long_system_call(row["name"]) else row["name"]
    text = f"{name[:48]} {row['share'] * 100:.0f}% {where}"
    held = row.get("pct_of_wall_on_busiest_thread")
    if held is not None:
        text += f" (at most {held:.0f}% of the phase)"
    return text


def serving_findings(run):
    """How a server behaves as load rises: does more concurrency buy more throughput, and what does waiting cost?"""
    findings = []
    first = next((t for t in sorted((t for t in (run.get("throughput") or {}).values() if isinstance(t, dict) and "concurrency" in t),
                                    key=lambda t: t["concurrency"])), {})
    fit = first.get("first_token_fit") or {}
    if (fit.get("fixed_share_at_median") or 0) >= 0.5 and fit.get("requests", 0) >= 20:
        findings.append({"severity": "info", "what": "first_token_fixed_part", "evidence": fit,
                         "text": f"First token = {fit['fixed_ms']:.0f} ms + {fit['per_token_ms']:.2f} ms per prompt token: for the median prompt of "
                                 f"{fit['median_prompt_tokens']} tokens {fit['fixed_share_at_median'] * 100:.0f}% of it does not depend on its length. "
                                 "That is work per request, not per token, and benchmarks with long prompts do not show it."})
    levels = sorted(((t["concurrency"], name, t) for name, t in (run.get("throughput") or {}).items()
                     if isinstance(t, dict) and "concurrency" in t and t.get("output_tokens_per_s")), key=lambda item: item[0])
    for (c0, name0, t0), (c1, name1, t1) in zip(levels, levels[1:]):
        gain = t1["output_tokens_per_s"] / t0["output_tokens_per_s"]
        ttft0 = (t0.get("ttft_ms") or {}).get("p50")
        ttft1 = (t1.get("ttft_ms") or {}).get("p50")
        evidence = {"from": c0, "to": c1, "throughput_gain": round(gain, 2)}
        latency = ""
        if ttft0 and ttft1:
            latency = f", and the median first token moved from {ttft0:,.0f} to {ttft1:,.0f} ms"
            evidence.update(ttft_from_ms=ttft0, ttft_to_ms=ttft1)
        if gain < 1 + 0.2 * (c1 / c0 - 1):
            findings.append({"severity": "warning", "what": "no_parallel_scaling",
                             "text": f"Going from {c0} to {c1} concurrent requests raised throughput only {gain:.2f} times{latency}. Requests are queueing: "
                                     "the server is handling them about one at a time, so check its parallelism setting.", "evidence": evidence})
        else:
            findings.append({"severity": "info", "what": "parallel_scaling",
                             "text": f"Going from {c0} to {c1} concurrent requests raised throughput {gain:.2f} times{latency}.", "evidence": evidence})
    failed = [(name, t) for _, name, t in levels if t.get("failed")]
    for name, t in failed:
        findings.append({"severity": "warning", "what": "failed_requests",
                         "text": f"{t['failed']} of {t['requests']} requests failed in {name}: {t.get('first_error', '')[:160]}", "evidence": {"phase": name}})
    return findings


def next_steps(run, bottlenecks):
    """Which deeper tool answers the question this run raised."""
    steps = []
    limits = {b["limit"] for b in bottlenecks.values()}
    if not run.get("perturbed"):
        macos = _on_macos(run)
        if limits & {"gpu", "gpu + host wait", "gpu imbalance"}:
            steps.append("Instruments' Metal System Trace, which comes with Xcode: which shaders take the GPU's time, and what it waits on in the gaps" if macos else
                         "--nsys: which kernels take the GPU's time, and what it waits on in the gaps")
        if limits & {"cpu (one thread)", "cpu (many threads)", "gpu + host wait"}:
            steps.append("`sample PID 10` while it runs: which functions the host threads spend their time in" if macos else
                         "--cpu-profile: which functions the host threads spend their time in")
    return steps


def diagnose(run):
    groups = run.get("groups") or {}
    bottlenecks = {}
    for name in ACTIVE_GROUPS:
        if name in groups:
            bottlenecks[name] = classify_phase(name, groups[name], run)
    # A server can hit a different limit at each load level, so each level gets its own verdict.
    levels = [p for p in run.get("phases") or [] if p.get("group") == "serve" and p.get("metrics")]
    if len(levels) > 1:
        del bottlenecks["serve"]
        for level in levels:
            bottlenecks[level["name"]] = classify_phase("serve", level["metrics"], run)
    return {
        "bottlenecks": bottlenecks,
        "memory": memory_findings(run),
        "movement": movement_findings(run),
        "serving": serving_findings(run),
        "timeline": timeline_findings(run) + cpu_findings(run),
        "next_steps": next_steps(run, bottlenecks),
    }
