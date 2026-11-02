"""`llmperfscope kernels`: which GPU kernels are worth speeding up, from a run with --nsys.

A trace says how long each kernel took. It does not say how much of a token that is, because tracing slows the host and stretches the
gaps. Kernel durations hold up under tracing, so they are compared with the time per token of a run without tracing.

A collective runs for as long as its slowest arrival: on the GPU that gets there first, most of it is waiting. The GPU that spends the least
time in collectives shows what they really cost, and the rest of what the others spend is reported as waiting.
"""

from __future__ import annotations

import re

from . import nsys

# What a kernel does for the model, from its name. The names are ggml's for CUDA; another engine's kernels land in `other`.
# A matrix multiply cannot be told apart by name: an expert's and a dense layer's run the same kernel, and so does the output head.
COMPONENTS = [
    ("attention", r"flash_attn|fattn|soft_max|softmax|gated_linear_attn|diag_mask|rope|lightning_indexer|top_k_mask|wkv"),
    ("matrix multiply, dense or expert", r"mul_mat|mmvq|mmvf|\bmmq\b|mmf|quantize|gemm|cublas"),
    ("expert routing and sums", r"top_k|topk|argsort|moe|add_id"),
    ("mixing between streams", r"dsv4_hc|\bhc_"),
    ("normalization", r"norm"),
    ("table lookups", r"get_rows"),
    ("KV cache writes", r"set_rows|\bset\b|set_f"),
    ("element-wise", r"\badd|\bsub|\bmul\b|\bdiv|scale|unary|clamp|sqr|bin_bcast|softcap|\bsum|cumsum|mean"),
    ("copies and reshapes", r"cpy|concat|convert|\bpad|dequantize|memset|repeat|fill"),
]


def component_of(name):
    if nsys.is_collective(name):
        return "collectives"
    for component, pattern in COMPONENTS:
        if re.search(pattern, name, re.I):
            return component
    return "other"


def by_component(rows):
    """Time per token by what the kernels do for the model, waiting for other GPUs left out."""
    totals = {}
    for row in rows:
        component = component_of(row["name"])
        totals[component] = totals.get(component, 0.0) + row["cost_ms_per_token"]
    return sorted(totals.items(), key=lambda item: -item[1])


def _phase_tables(run, phases):
    groups = (run.get("nsys") or {}).get("groups") or {}
    return [(name, groups[name]) for name in phases if name in groups and groups[name].get("tokens") and groups[name].get("devices")]


def per_token(group):
    """Time per token and per GPU, by kernel. For balanced GPUs this is what one GPU spends, and so what a token waits for."""
    devices = group["devices"]
    tokens = group["tokens"]
    count = len(devices)
    comm = [d["comm_ms"] / tokens for d in devices]
    comm_mean = sum(comm) / count
    comm_true = min(comm)
    rows = []
    for kernel in group.get("top_kernels") or []:
        collective = nsys.is_collective(kernel["name"])
        ms = kernel["ms"] / count / tokens
        rows.append({
            "name": kernel["name"],
            "collective": collective,
            "ms_per_token": ms,
            # What it costs without the waiting, shared out over the collectives by their time.
            "cost_ms_per_token": ms * comm_true / comm_mean if collective and comm_mean else ms,
            "calls_per_token": kernel["n"] / count / tokens,
            "us_per_call": kernel["avg_us"],
            "share": kernel["share"],
        })
    return {
        "kernels": rows,
        "kernel_ms": sum(d["kernel_ms"] for d in devices) / count / tokens,
        "collective_ms": comm_true,
        "waiting_ms": comm_mean - comm_true,
        "copy_ms": sum(d["copy_ms"] for d in devices) / count / tokens,
        "gpus": count,
        "tokens": tokens,
    }


def _speed_of(run, phase):
    """ms per token without tracing."""
    tokens_per_s = ((run.get("throughput") or {}).get(phase) or {}).get("tokens_per_s")
    return 1000.0 / tokens_per_s if tokens_per_s else None


def gain_if_free(token_ms, cost_ms):
    """How much faster a token gets, at most, if this took no time at all. Real gains are smaller: other GPUs and other work still run."""
    left = token_ms - cost_ms
    if left <= 0:
        return None
    return token_ms / left - 1.0


def render(run, speed=None, phases=("decode", "prefill"), top=12):
    tables = _phase_tables(run, phases)
    if not tables:
        return "No kernel times here: this needs a run with --nsys and phases with tokens.\n"
    out = []
    for name, group in tables:
        data = per_token(group)
        token_ms = _speed_of(speed, name) if speed else None
        title = f"## {name}: {data['tokens']:,} tokens traced, {data['gpus']} GPU{'s' if data['gpus'] > 1 else ''}"
        if token_ms:
            title += f", {token_ms:.2f} ms per token without tracing"
        out += [title, ""]
        if token_ms:
            rest = token_ms - data["kernel_ms"] - data["collective_ms"] - data["copy_ms"]
            out += ["| where the token's time goes | ms/token | share |", "|---|---:|---:|"]
            for label, value in (("kernels", data["kernel_ms"]), ("collectives, without waiting", data["collective_ms"]), ("copies", data["copy_ms"]),
                                 ("gaps, host work and waiting between kernels", max(rest, 0.0))):
                out.append(f"| {label} | {value:.2f} | {100 * value / token_ms:.0f}% |")
            out.append("")
        components = by_component(data["kernels"])
        if components:
            spent = sum(value for _, value in components) or 1.0
            out += ["| what the kernels do for the model | ms/token | share of kernel time |", "|---|---:|---:|"]
            out += [f"| {component} | {value:.2f} | {100 * value / spent:.0f}% |" for component, value in components]
            out.append("")
        headers = ["kernel", "ms/token", "calls/token", "us/call"]
        if token_ms:
            headers += ["share of token", "speed-up if free, at most"]
        else:
            headers += ["share of kernel time"]
        out += ["| " + " | ".join(headers) + " |", "|---|" + "---:|" * (len(headers) - 1)]
        rows = sorted(data["kernels"], key=lambda r: -r["cost_ms_per_token"])[:top]
        for row in rows:
            label = f"{row['name']} (collective)" if row["collective"] else row["name"]
            cells = [label, f"{row['cost_ms_per_token']:.3f}", f"{row['calls_per_token']:.1f}", f"{row['us_per_call']:.1f}"]
            if token_ms:
                gain = gain_if_free(token_ms, row["cost_ms_per_token"])
                cells += [f"{100 * row['cost_ms_per_token'] / token_ms:.1f}%", "-" if gain is None else f"+{100 * gain:.1f}%"]
            else:
                cells += [f"{100 * row['share']:.0f}%"]
            out.append("| " + " | ".join(cells) + " |")
        out.append("")
        if any(row["collective"] for row in rows):
            out.append(f"Collectives: {data['waiting_ms']:.2f} ms per token on average is waiting for other GPUs to arrive, left out above. Tracing makes that wait longer, so it is an upper bound.")
            out.append("")
    if not speed:
        out.append("Add --speed RUN, the same command without --nsys, for each kernel's share of a token and what it could win.")
    elif speed.get("perturbed"):
        out.append("Note: the --speed run was itself traced or sampled, so its time per token is a little long.")
    if speed and any((((speed.get("throughput") or {}).get(name) or {}).get("tokens_per_s")) is None for name, _ in tables):
        out.append("Note: the --speed run has no speed for some of these phases, so they show no shares.")
    out.append("This says where the time goes and what it could win. It does not say how far a kernel is from what the hardware can do: that takes GPU hardware counters, which need admin rights.")
    return "\n".join(out) + "\n"


def render_diff(run_a, run_b, name_a="A", name_b="B", phases=("decode", "prefill"), top=14):
    """Kernel by kernel, what changed between two traced runs: for telling a change in the kernels from a change in everything around them."""
    tables_a = dict(_phase_tables(run_a, phases))
    tables_b = dict(_phase_tables(run_b, phases))
    shared = [name for name in phases if name in tables_a and name in tables_b]
    if not shared:
        return "No phase with kernel times in both runs: this needs two runs with --nsys.\n"
    out = []
    for phase in shared:
        data_a, data_b = per_token(tables_a[phase]), per_token(tables_b[phase])
        by_a = {row["name"]: row for row in data_a["kernels"]}
        by_b = {row["name"]: row for row in data_b["kernels"]}
        names = sorted(set(by_a) | set(by_b), key=lambda n: -abs((by_b.get(n) or {}).get("ms_per_token", 0.0) - (by_a.get(n) or {}).get("ms_per_token", 0.0)))[:top]
        out += [f"## {phase}", "", f"| | {name_a} ms/token | {name_b} ms/token | change |", "|---|---:|---:|---:|"]
        for label, a, b in (("all kernels", data_a["kernel_ms"], data_b["kernel_ms"]), ("collectives, without waiting", data_a["collective_ms"], data_b["collective_ms"]),
                            ("copies", data_a["copy_ms"], data_b["copy_ms"])):
            out.append(f"| **{label}** | {a:.3f} | {b:.3f} | {b - a:+.3f} |")
        for name in names:
            a, b = (by_a.get(name) or {}).get("ms_per_token"), (by_b.get(name) or {}).get("ms_per_token")
            cells = ["-" if a is None else f"{a:.3f}", "-" if b is None else f"{b:.3f}", f"{(b or 0.0) - (a or 0.0):+.3f}"]
            out.append(f"| {name} | " + " | ".join(cells) + " |")
        out.append("")
    out.append("Collectives in the kernel rows include waiting for other GPUs. One traced run of each is a guide, so check a small difference with more than one.")
    return "\n".join(out) + "\n"
