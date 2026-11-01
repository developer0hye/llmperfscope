"""A/B comparison. `diff` compares finished runs, `matrix` runs both sides itself in alternating order across many cases.

Speed, memory and energy are judged together, so a change cannot look like a win while it costs memory. A difference only
counts when the arms do not overlap and it is larger than the noise floor, because a single ratio from a few runs is mostly
noise: the same binary against itself has shown ratios up to 1.006 at 8 GPUs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from . import runner

# name -> (getter, direction): +1 means bigger is better, -1 smaller is better.
METRICS = {
    "prefill_tps": (lambda r: _get(r, "throughput", "prefill", "tokens_per_s"), +1),
    "decode_tps": (lambda r: _get(r, "throughput", "decode", "tokens_per_s"), +1),
    "gpu_peak_mib": (lambda r: _get(r, "memory", "gpu", "peak_above_baseline_total_mib"), -1),
    "gpu_peak_max_device_mib": (lambda r: _get(r, "memory", "gpu", "peak_above_baseline_max_device_mib"), -1),
    "host_anon_peak_mib": (lambda r: _get(r, "memory", "host", "anon_peak_mib"), -1),
    "load_s": (lambda r: _get(r, "groups", "load", "wall_s"), -1),
    "prefill_j_per_token": (lambda r: _get(r, "energy", "prefill_j_per_token"), -1),
    "decode_j_per_token": (lambda r: _get(r, "energy", "decode_j_per_token"), -1),
}
# Smallest difference that can mean anything, per metric: relative, then absolute. Memory rounds to whole MiB and allocators
# differ by a little from run to run; load time depends on the page cache; energy has its own counter resolution.
FLOORS = {
    "prefill_tps": (0.003, 0.0),
    "decode_tps": (0.003, 0.0),
    "gpu_peak_mib": (0.002, 16.0),
    "gpu_peak_max_device_mib": (0.002, 16.0),
    "host_anon_peak_mib": (0.02, 64.0),
    "load_s": (0.05, 0.5),
    "prefill_j_per_token": (0.05, 0.0),
    "decode_j_per_token": (0.03, 0.0),
}


def _get(run, *keys):
    node = run
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def judge(metric, a_values, b_values, noise=0.0):
    """Verdict for B against A: better, worse or same. `noise` is a relative floor measured with a null comparison."""
    a_values = [v for v in a_values if v is not None]
    b_values = [v for v in b_values if v is not None]
    if not a_values or not b_values:
        return {"verdict": "missing"}
    a_mean = sum(a_values) / len(a_values)
    b_mean = sum(b_values) / len(b_values)
    direction = METRICS[metric][1]
    ratio = b_mean / a_mean if a_mean else None
    relative_floor, absolute_floor = FLOORS[metric]
    floor = max(relative_floor, noise) * abs(a_mean)
    floor = max(floor, absolute_floor)
    difference = (b_mean - a_mean) * direction  # positive: B is better
    separated = min(b_values) > max(a_values) or max(b_values) < min(a_values)
    verdict = "same"
    if abs(difference) > floor and separated:
        verdict = "better" if difference > 0 else "worse"
    return {"a": round(a_mean, 4), "b": round(b_mean, 4), "ratio": round(ratio, 4) if ratio is not None else None,
            "verdict": verdict, "a_runs": a_values, "b_runs": b_values}


def diff(runs_a, runs_b, noise=None):
    """Runs of side A against runs of side B. Failed runs and traced runs do not count; they are listed instead."""
    usable_a = [r for r in runs_a if r.get("status") == "ok" and not r.get("perturbed")]
    usable_b = [r for r in runs_b if r.get("status") == "ok" and not r.get("perturbed")]
    out = {"metrics": {}, "dropped_runs": (len(runs_a) - len(usable_a)) + (len(runs_b) - len(usable_b))}
    dirty = sum(1 for r in usable_a + usable_b if not (r.get("machine_before") or {"clean": True}).get("clean", True))
    if dirty:
        out["dirty_runs"] = dirty
    for metric, (getter, _direction) in METRICS.items():
        result = judge(metric, [getter(r) for r in usable_a], [getter(r) for r in usable_b], (noise or {}).get(metric, 0.0))
        if result["verdict"] != "missing":
            out["metrics"][metric] = result
    worse = [m for m, v in out["metrics"].items() if v["verdict"] == "worse"]
    better = [m for m, v in out["metrics"].items() if v["verdict"] == "better"]
    out["worse"], out["better"] = worse, better
    if worse and better:
        out["verdict"] = "tradeoff"
    else:
        out["verdict"] = "regression" if worse else "improvement" if better else "neutral"
    return out


def load_run(path):
    path = Path(path)
    if path.is_dir():
        path = path / "run.json"
    with open(path) as f:
        return json.load(f)


def render_diff(result, a_label="A", b_label="B"):
    lines = [f"| metric | {a_label} | {b_label} | ratio | |", "|---|---:|---:|---:|---|"]
    for metric, v in result["metrics"].items():
        mark = {"better": "better", "worse": "**worse**", "same": ""}[v["verdict"]]
        lines.append(f"| {metric} | {v['a']:,} | {v['b']:,} | {v['ratio']} | {mark} |")
    lines.append(f"\nVerdict: **{result['verdict']}**")
    if result["dropped_runs"]:
        lines.append(f"{result['dropped_runs']} failed or traced runs were left out.")
    if result.get("dirty_runs"):
        lines.append(f"{result['dirty_runs']} runs started on a machine that was not clean.")
    return "\n".join(lines) + "\n"


def load_cases(path):
    with open(path) as f:
        spec = json.load(f)
    defaults = spec.get("defaults", {})
    cases = []
    for case in spec["cases"]:
        merged = {
            "label": case["label"],
            "binary": case.get("binary", defaults.get("binary", "llama-bench")),
            "prefix": case.get("prefix", defaults.get("prefix", [])),
            "args": list(defaults.get("args", [])) + list(case.get("args", [])),
            "env": {**defaults.get("env", {}), **case.get("env", {})},
            "model": case.get("model"),
            "gpus": case.get("gpus"),
        }
        cases.append(merged)
    return cases


def _command(case, bin_dir):
    command = list(case["prefix"]) + [str(Path(bin_dir) / case["binary"])]
    if case["model"]:
        command += ["-m", case["model"]]
    return command + case["args"]


def _tag(label):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label)


def run_matrix(cases, a, b, out_dir, order="ABBA", timeout_s=1800.0, a_env=None, b_env=None, only=None, noise=None, log=print,
               precheck="wait"):
    """a and b are (name, bin_dir). Fresh process per arm, alternating order, machine re-checked before every arm."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = {"a": a[0], "b": b[0], "order": order, "cases": []}
    for case in cases:
        if only and not re.search(only, case["label"]):
            continue
        if case["model"] and not Path(case["model"]).exists():
            log(f"{case['label']}: model missing, skipped")
            table["cases"].append({"label": case["label"], "skipped": "model missing"})
            continue
        runs = {"A": [], "B": []}
        for i, side in enumerate(order):
            name, bin_dir = a if side == "A" else b
            env = dict(case["env"])
            env.update((a_env if side == "A" else b_env) or {})
            if case["gpus"] is not None:
                env["CUDA_VISIBLE_DEVICES"] = case["gpus"]
            options = runner.Options(out=str(out_dir / _tag(case["label"]) / f"{i}_{side}_{name}"), label=f"{case['label']} [{name}]",
                                     env=env, timeout_s=timeout_s, quiet=True, precheck=precheck)
            log(f"{case['label']}: arm {i + 1}/{len(order)} ({name})")
            try:
                report = runner.Run(_command(case, bin_dir), options).execute()
            except runner.Refused as e:
                log(f"  refused: {e}")
                continue
            if report["status"] != "ok":
                log(f"  arm ended with status {report['status']}")
            runs[side].append(report)
        result = diff(runs["A"], runs["B"], (noise or {}).get(case["label"]))
        result["label"] = case["label"]
        table["cases"].append(result)
        with open(out_dir / "matrix.json", "w") as f:
            json.dump(table, f, indent=1)
        write_table(table, out_dir / "table.md")
        log(f"  -> {result['verdict']}: " + ", ".join(f"{m} {v['ratio']}" for m, v in result["metrics"].items()
                                                      if m in ("prefill_tps", "decode_tps")))
    return table


def write_table(table, path):
    columns = [("prefill_tps", "pp"), ("decode_tps", "tg"), ("gpu_peak_mib", "GPU mem"), ("host_anon_peak_mib", "host anon"),
               ("load_s", "load"), ("decode_j_per_token", "tg J/tok")]
    header = "| case | " + " | ".join(f"{title} {table['b']}/{table['a']}" for _, title in columns) + " | verdict |"
    lines = [header, "|---|" + "---:|" * len(columns) + "---|"]
    for case in table["cases"]:
        if "skipped" in case:
            lines.append(f"| {case['label']} | {case['skipped']} |")
            continue
        cells = []
        for metric, _ in columns:
            value = case["metrics"].get(metric)
            if not value:
                cells.append("-")
                continue
            cell = f"{value['ratio']:.3f}" if value["ratio"] is not None else "-"
            if value["verdict"] == "worse":
                cell = f"**{cell}** worse"
            elif value["verdict"] == "better":
                cell = f"{cell} better"
            cells.append(cell)
        extra = f" ({case['dropped_runs']} arms failed)" if case["dropped_runs"] else ""
        if case.get("worse"):
            extra += ": costs " + ", ".join(case["worse"])
        lines.append(f"| {case['label']} | " + " | ".join(cells) + f" | {case['verdict']}{extra} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def noise_from_null(matrix_path):
    """Noise floors from a null comparison (same binary on both sides): per case and metric, the spread that appeared with
    nothing changed. Pass the file to later matrices with --noise."""
    with open(matrix_path) as f:
        table = json.load(f)
    out = {}
    for case in table["cases"]:
        if "metrics" not in case:
            continue
        floors = {}
        for metric, value in case["metrics"].items():
            values = value["a_runs"] + value["b_runs"]
            mean = sum(values) / len(values)
            if mean:
                floors[metric] = round((max(values) - min(values)) / abs(mean), 5)
        out[case["label"]] = floors
    return out
