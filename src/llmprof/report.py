"""Human-readable report of a run."""

from __future__ import annotations


def _num(value, digits=1):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    return f"{value:,}"


def _table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" if i == 0 else "---:" for i in range(len(headers))) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _gib(mib):
    return "-" if mib is None else f"{mib / 1024:,.2f}"


def _diagnosis(run):
    diagnosis = run.get("diagnosis") or {}
    out = []
    bottlenecks = diagnosis.get("bottlenecks") or {}
    if bottlenecks:
        out.append("\n## What limits each phase\n")
        judged = {name: b for name, b in bottlenecks.items() if b["confidence"] != "none"}
        if judged:
            rows = [[name, b["limit"], b["confidence"], b["why"]] for name, b in judged.items()]
            lines = ["| phase | limit | confidence | why |", "|---|---|---|---|"]
            lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
            out.append("\n".join(lines))
        skipped = [name for name, b in bottlenecks.items() if b["confidence"] == "none"]
        if skipped:
            out.append(f"\nToo short to judge (under 1 s): {', '.join(skipped)}.")
    for title, key in (("Memory", "memory"), ("Data movement", "movement"), ("Serving", "serving"), ("GPU and host time", "timeline")):
        findings = diagnosis.get(key) or []
        if findings:
            out.append(f"\n## {title} findings\n")
            for finding in sorted(findings, key=lambda f: f["severity"] != "warning"):
                mark = "**warning**: " if finding["severity"] == "warning" else ""
                out.append(f"- {mark}{finding['text']}")
    steps = diagnosis.get("next_steps") or []
    if steps:
        out.append("\nTo look deeper, run again with:\n")
        out.extend(f"- {step}" for step in steps)
    return out


def _movement_table(run):
    """Bytes between tiers, per phase: disk to RAM, RAM to GPU, GPU to GPU, swap."""
    rows = []
    names = set()
    for name, group in (run.get("groups") or {}).items():
        if name in ("baseline", "tail"):
            continue
        flows = group.get("flows") or {}
        host = group.get("host") or {}
        read = host.get("disk_read_mib")
        system_read = sum(d.get("read_mib", 0.0) for d in (flows.get("disks") or {}).values())
        write = sum(d.get("write_mib", 0.0) for d in (flows.get("disks") or {}).values())
        names |= set(flows.get("disks") or {})
        cells = [read, system_read, write, flows.get("swap_in_mib"), flows.get("swap_out_mib"), flows.get("page_cache_change_mib"),
                 flows.get("pcie_to_gpu_mib_est"), flows.get("pcie_from_gpu_mib_est"), flows.get("nvlink_mib"), host.get("major_faults")]
        if any(cell for cell in cells):
            rows.append([name] + [_num(float(c), 0) if isinstance(c, (int, float)) and not isinstance(c, bool) and i < 9 else _num(c)
                                  for i, c in enumerate(cells)])
    if not rows:
        return []
    out = ["\n## Data movement (MiB)\n"]
    out.append(_table(["phase", "target disk read", "all disk read", "all disk write", "swap in", "swap out", "page cache +/-",
                       "RAM to GPU (est)", "GPU to RAM (est)", "GPU to GPU", "major faults"], rows))
    disks = run.get("disks") or {}
    active = [f"{name}: {disks.get(name, {}).get('kind', 'disk')}" for name in sorted(names)]
    if active:
        out.append("\nActive disks: " + ", ".join(active) + ". `all disk` counts every process on the machine.")
    return out


def render(run):
    out = []
    title = run.get("label") or run.get("adapter") or "run"
    out.append(f"# {title}: {run['status']}")
    if run.get("perturbed"):
        out.append("\nTraced or sampled run: use it to find where time goes, not for speed numbers.")
    identity = run.get("identity") or {}
    facts = [f"adapter `{run.get('adapter')}`"]
    if identity.get("git_commit"):
        dirty = identity.get("git_dirty_files")
        facts.append(f"commit `{identity['git_commit'][:10]}`" + (f" (+{dirty} dirty files)" if dirty else ""))
    elif identity.get("sha256"):
        facts.append(f"binary `{identity['sha256'][:12]}`")
    host = run.get("host") or {}
    if host.get("hostname"):
        facts.append(f"host `{host['hostname']}`")
    before = run.get("machine_before") or {}
    if "clean" in before:
        facts.append("machine clean" if before["clean"] else "**machine NOT clean**")
    if before.get("model_page_cache_fraction") is not None:
        facts.append(f"model {before['model_page_cache_fraction'] * 100:.0f}% in page cache")
    out.append("\n" + ", ".join(facts))

    out.extend(_diagnosis(run))

    groups = run.get("groups") or {}
    rows = []
    for name, group in groups.items():
        if name in ("baseline", "tail") or group.get("wall_s", 0.0) < 0.005:
            continue
        gpu = group.get("gpu") or {}
        host_row = group.get("host") or {}
        rows.append([
            name,
            _num(group.get("wall_s"), 2),
            _num(group.get("tokens")) if group.get("tokens") else "-",
            _gib(gpu.get("mem_end_mib")),
            _gib(gpu.get("mem_peak_mib")),
            _gib(host_row.get("anon_peak_mib")),
            _gib(host_row.get("file_peak_mib")),
            _num(gpu.get("energy_j"), 0),
            _num(gpu.get("power_avg_w"), 0),
            _num(gpu.get("sm_mhz_avg"), 0),
            _num(gpu.get("util_avg"), 0),
            _num(host_row.get("busy_cores"), 2),
        ])
    if rows:
        out.append("\n## Lifecycle\n")
        out.append(_table(["phase", "wall s", "tokens", "GPU GiB end", "GPU GiB peak", "host anon GiB", "host file GiB", "GPU J",
                           "GPU W", "SM MHz", "util %", "CPU cores"], rows))
        out.append("\nGPU memory is above the idle baseline, over the GPUs this run used.")

    throughput = run.get("throughput") or {}
    if throughput:
        out.append("\n## Speed\n")
        rows = []
        energy = run.get("energy") or {}
        for name in ("prefill", "decode"):
            t = throughput.get(name)
            if not t:
                continue
            if "tests" in t:
                for test in t["tests"]:
                    rows.append([f"{name} {test['tokens_per_run']}", _num(test["tokens_per_s"], 2), _num(test["stddev"], 2), "-"])
            else:
                rows.append([name, _num(t.get("tokens_per_s"), 2), _num(t.get("stddev"), 2),
                             _num(energy.get(f"{name}_j_per_token"), 3)])
        if rows:
            out.append(_table(["", "tokens/s", "stddev", "GPU J/token"], rows))
        for name, t in throughput.items():
            if name.startswith("serve"):
                out.append(f"\n**{name}**: {_num(t.get('output_tokens_per_s'), 2)} output tokens/s, "
                           f"{t.get('requests')} requests ({t.get('failed')} failed)")
                if t.get("ttft_ms"):
                    d = t["ttft_ms"]
                    out.append(f"- time to first token ms: mean {d['mean']}, p50 {d['p50']}, p90 {d['p90']}, p99 {d['p99']}")
                if t.get("itl_ms"):
                    d = t["itl_ms"]
                    out.append(f"- between tokens ms: mean {d['mean']}, p50 {d['p50']}, p90 {d['p90']}, p99 {d['p99']}")
                if t.get("prefill_tokens_per_s_mean") or t.get("decode_tokens_per_s_mean"):
                    out.append(f"- per request: prefill {_num(t.get('prefill_tokens_per_s_mean'), 1)} tokens/s, "
                               f"decode {_num(t.get('decode_tokens_per_s_mean'), 1)} tokens/s")
                if t.get("j_per_output_token") is not None:
                    out.append(f"- GPU energy: {t['j_per_output_token']:.3f} J per output token")

    memory = run.get("memory") or {}
    gpu = memory.get("gpu") or {}
    if gpu.get("devices"):
        out.append("\n## Memory\n")
        rows = [[d["index"], _num(d["baseline_mib"]), _num(d["peak_mib"]), _num(d["peak_above_baseline_mib"]),
                 _num(d["process_peak_mib"])] for d in gpu["devices"]]
        out.append(_table(["GPU", "idle MiB", "peak MiB", "above idle MiB", "process peak MiB"], rows))
    host_memory = memory.get("host") or {}
    if host_memory:
        out.append(f"\nHost: anon peak {_gib(host_memory.get('anon_peak_mib'))} GiB, file-backed peak "
                   f"{_gib(host_memory.get('file_peak_mib'))} GiB, pinned peak {_gib(host_memory.get('pinned_peak_mib'))} GiB, "
                   f"kernel max RSS {_gib(host_memory.get('max_rss_kernel_mib'))} GiB.")
    breakdown = memory.get("breakdown_mib")
    if breakdown:
        out.append("\nEngine's own breakdown (MiB):\n")
        rows = []
        for r in breakdown:
            # llama.cpp's tensor-split row: buffers are one GPU's share, while its total and free are for the whole group.
            split_row = str(r.get("name", "")).startswith("Meta")
            rows.append([f"{r.get('name')} (one GPU's share)" if split_row else r.get("name"), _num(r.get("self")), _num(r.get("model")),
                         _num(r.get("context")), _num(r.get("compute")), "-" if split_row else _num(r.get("unaccounted"))])
        out.append(_table(["buffer", "self", "model", "context", "compute", "unaccounted"], rows))

    gauges = (run.get("workload") or {}).get("server_gauges_peak") or {}
    if gauges:
        # Labels are dropped for display: the name is what a reader knows the metric by.
        shown = {}
        for name, peak in gauges.items():
            shown[name.split("{")[0]] = max(shown.get(name.split("{")[0], peak), peak)
        out.append("\nServer's own gauges, peak: " + ", ".join(f"`{name}` {peak:,.3g}" for name, peak in sorted(shown.items())))

    out.extend(_movement_table(run))

    hot = []
    for name in ("decode", "prefill", "serve", "load"):
        threads = (groups.get(name, {}).get("host") or {}).get("threads_cpu_s")
        if threads:
            top = ", ".join(f"{k} {v:.1f}s" for k, v in list(threads.items())[:5])
            hot.append(f"- {name}: {top}")
    if hot:
        out.append("\n## Host threads (CPU seconds)\n")
        out.extend(hot)

    nsys = run.get("nsys") or {}
    if nsys.get("groups"):
        out.append("\n## GPU timeline (nsys)\n")
        rows = []
        for name, group in nsys["groups"].items():
            per = group.get("per_token_ms")
            if per:
                rows.append([name, group.get("tokens"), _num(per["kernel_ms"], 3), _num(per["comm_ms"], 3), _num(per["copy_ms"], 3),
                             _num(per["idle_ms"], 3), _num(group.get("per_token_kernels"), 0)])
        if rows:
            out.append(_table(["phase", "tokens", "kernels ms/tok", "collectives ms/tok", "copies ms/tok", "idle ms/tok",
                               "kernels/tok"], rows))
        for name in ("decode", "prefill"):
            top = (nsys["groups"].get(name) or {}).get("top_kernels")
            if top:
                out.append(f"\nTop kernels, {name}: " + ", ".join(f"{k['name']} {k['share'] * 100:.0f}%" for k in top[:6]))
    if nsys.get("error"):
        out.append(f"\nnsys: {nsys['error']}")

    cpu_profile = run.get("cpu_profile") or {}
    if cpu_profile.get("groups"):
        out.append("\n## Host CPU hot spots\n")
        for name in ("decode", "prefill", "serve", "load", "setup", "teardown"):
            group = cpu_profile["groups"].get(name)
            if not group:
                continue
            top = ", ".join(f"{x['name'][:48]} {x['share'] * 100:.0f}% ({x.get('threads', 1)} thr)" for x in group["self"][:6])
            modules = ", ".join(f"{x['name']} {x['share'] * 100:.0f}%" for x in group["modules"][:4])
            busiest = ", ".join(f"{x['name']} {x['cpu_s']:.2f}s" for x in group.get("busiest_threads", [])[:4])
            out.append(f"- **{name}** ({group['samples']} samples): {modules}\n  - {top}\n  - busiest threads: {busiest}")
    if cpu_profile.get("error"):
        out.append(f"\nCPU profile: {cpu_profile['error']}")

    consistency = run.get("consistency") or {}
    if consistency.get("phase_wall_over_engine_time"):
        out.append(f"\nSelf-check: phase time / engine's own timer = {consistency['phase_wall_over_engine_time']:.4f} (should be near 1).")
    if consistency.get("energy_over_power_gauge"):
        out.append(f"\nSelf-check: power from the energy counter / power gauge = {consistency['energy_over_power_gauge']:.3f} (should be near 1).")
    sampler = run.get("sampler") or {}
    if sampler.get("hz_achieved"):
        pinned = f", on CPU {sampler['pinned_cpu']}" if sampler.get("pinned_cpu") is not None else ""
        out.append(f"Sampler: {sampler['hz_achieved']} Hz of {sampler['hz_target']}, largest gap {sampler['max_gap_ms']} ms, "
                   f"{sampler['cpu_seconds']} CPU s{pinned}.")
    failure = run.get("failure") or {}
    if failure:
        out.append("\n## Why it was stopped\n")
        for thread in failure.get("threads", [])[:6]:
            out.append(f"- {thread['comm']} (tid {thread['tid']}): {thread['cpu_s']} CPU s, in {thread['wchan']}")
        for line in failure.get("stderr_tail", [])[-4:]:
            out.append(f"    {line}")
    for note in run.get("notes") or []:
        out.append(f"\nNote: {note}")
    return "\n".join(out) + "\n"
