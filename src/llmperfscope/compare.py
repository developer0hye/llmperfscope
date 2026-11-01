"""A/B comparison. `diff` compares finished runs, `matrix` runs both sides itself in alternating order across many cases.

Speed, memory and energy are judged together, so a change cannot look like a win while it costs memory. A difference only
counts when the arms do not overlap and it is larger than the noise floor, because a single ratio from a few runs is mostly
noise: the same binary against itself has shown ratios up to 1.006 at 8 GPUs.

Arms that do not overlap are only evidence with enough of them: with 2 per side they miss each other by chance 1 time in 3. A
difference that rests on so little is reported as a hint, marked `?`, and `matrix` runs more arms before it calls anything worse.
"""

from __future__ import annotations

import json
import re
from math import factorial
from pathlib import Path

from . import runner

DEFAULT_ORDER = "ABBABAAB"
# Above this chance of the arms separating by luck, a difference is a hint and not a result.
WEAK_ABOVE = 0.05
# A difference this many times what each side varies by is real even with few runs, such as memory that is the same in every arm.
CLEAR_EFFECT = 5.0

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


def separation_chance(runs_a, runs_b):
    """How often every run of one side beats every run of the other when nothing changed: 1 run per side always, 2 one time in 3,
    3 one in 10, 4 one in 35."""
    if not runs_a or not runs_b:
        return 1.0
    return min(1.0, 2.0 * factorial(runs_a) * factorial(runs_b) / factorial(runs_a + runs_b))


def judge(metric, a_values, b_values, noise=0.0):
    """Verdict for B against A: better, worse or same, with `weak` set when too few runs back it. `noise` is a relative floor measured
    with a null comparison."""
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
    weak = False
    if abs(difference) > floor and separated:
        verdict = "better" if difference > 0 else "worse"
        spread = max(max(a_values) - min(a_values), max(b_values) - min(b_values))
        clear = min(len(a_values), len(b_values)) >= 2 and abs(difference) > CLEAR_EFFECT * spread
        weak = separation_chance(len(a_values), len(b_values)) > WEAK_ABOVE and not clear
    return {"a": round(a_mean, 4), "b": round(b_mean, 4), "ratio": round(ratio, 4) if ratio is not None else None,
            "verdict": verdict, "weak": weak, "a_runs": a_values, "b_runs": b_values}


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
    def named(verdict, weak):
        return [m for m, v in out["metrics"].items() if v["verdict"] == verdict and v["weak"] == weak]

    worse, better = named("worse", False), named("better", False)
    out["worse"], out["better"] = worse, better
    out["maybe_worse"], out["maybe_better"] = named("worse", True), named("better", True)
    if worse and better:
        out["verdict"] = "tradeoff"
    elif worse or better:
        out["verdict"] = "regression" if worse else "improvement"
    else:
        out["verdict"] = "inconclusive" if out["maybe_worse"] or out["maybe_better"] else "neutral"
    return out


def _mark(value):
    """How a metric's verdict reads in a table. `?` means too few runs to tell."""
    mark = {"better": "better", "worse": "worse", "same": ""}[value["verdict"]]
    if value.get("weak"):
        return f"{mark}?"
    return f"**{mark}**" if value["verdict"] == "worse" else mark


def _weak_note(runs_per_side):
    chance = separation_chance(runs_per_side, runs_per_side)
    odds = "every time" if chance >= 1.0 else f"1 time in {round(1 / chance)}"
    return (f"`?`: too few runs to tell. With {runs_per_side} per side, arms miss each other by chance {odds} when nothing changed. "
            "Run more, or use `llmperfscope matrix`.")


def load_run(path):
    path = Path(path)
    if path.is_dir():
        path = path / "run.json"
    with open(path) as f:
        return json.load(f)


def render_diff(result, a_label="A", b_label="B"):
    lines = [f"| metric | {a_label} | {b_label} | ratio | |", "|---|---:|---:|---:|---|"]
    for metric, v in result["metrics"].items():
        lines.append(f"| {metric} | {v['a']:,} | {v['b']:,} | {v['ratio']} | {_mark(v)} |")
    lines.append(f"\nVerdict: **{result['verdict']}**")
    if result["maybe_worse"] or result["maybe_better"]:
        weak = next(v for v in result["metrics"].values() if v.get("weak"))
        lines.append(_weak_note(min(len(weak["a_runs"]), len(weak["b_runs"]))))
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


def run_matrix(cases, a, b, out_dir, order=DEFAULT_ORDER, timeout_s=1800.0, a_env=None, b_env=None, only=None, noise=None, log=print,
               precheck="wait", recheck=1):
    """a and b are (name, bin_dir). Fresh process per arm, alternating order, machine re-checked before every arm. A case that looks worse
    gets the same number of arms again, in mirrored order, up to `recheck` times: among many cases and metrics some will look worse by
    chance, and calling a change a regression should take more than that."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = {"a": a[0], "b": b[0], "order": order, "recheck": recheck, "cases": []}
    for case in cases:
        if only and not re.search(only, case["label"]):
            continue
        if case["model"] and not Path(case["model"]).exists():
            log(f"{case['label']}: model missing, skipped")
            table["cases"].append({"label": case["label"], "skipped": "model missing"})
            continue
        runs = {"A": [], "B": []}
        arms_done = 0

        def run_arms(sequence):
            nonlocal arms_done
            for side in sequence:
                name, bin_dir = a if side == "A" else b
                env = dict(case["env"])
                env.update((a_env if side == "A" else b_env) or {})
                if case["gpus"] is not None:
                    env["CUDA_VISIBLE_DEVICES"] = case["gpus"]
                options = runner.Options(out=str(out_dir / _tag(case["label"]) / f"{arms_done}_{side}_{name}"), label=f"{case['label']} [{name}]",
                                         env=env, timeout_s=timeout_s, quiet=True, precheck=precheck)
                arms_done += 1
                log(f"{case['label']}: arm {arms_done} ({name})")
                try:
                    report = runner.Run(_command(case, bin_dir), options).execute()
                except runner.Refused as e:
                    log(f"  refused: {e}")
                    continue
                if report["status"] != "ok":
                    log(f"  arm ended with status {report['status']}")
                runs[side].append(report)

        floors = (noise or {}).get(case["label"])
        run_arms(order)
        result = diff(runs["A"], runs["B"], floors)
        mirrored = order.translate(str.maketrans("AB", "BA"))
        for _ in range(recheck):
            suspects = result["worse"] + result["maybe_worse"]
            if not suspects:
                break
            log(f"  looks worse in {', '.join(suspects)}: {len(order)} more arms before calling it")
            run_arms(mirrored)
            result = diff(runs["A"], runs["B"], floors)
        result["label"] = case["label"]
        result["arms"] = {"a": len(runs["A"]), "b": len(runs["B"])}
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
            mark = _mark(value)
            if value["verdict"] == "worse" and not value.get("weak"):
                cell = f"**{cell}** worse"
            elif mark:
                cell = f"{cell} {mark}"
            cells.append(cell)
        extra = f" ({case['dropped_runs']} arms failed)" if case["dropped_runs"] else ""
        if case.get("worse"):
            extra += ": costs " + ", ".join(case["worse"])
        arms = case.get("arms")
        if arms and arms["a"] != len(table.get("order", "")) / 2:
            extra += f" ({arms['a']} + {arms['b']} arms)"
        lines.append(f"| {case['label']} | " + " | ".join(cells) + f" | {case['verdict']}{extra} |")
    weak = [v for case in table["cases"] for v in case.get("metrics", {}).values() if v.get("weak")]
    if weak:
        lines += ["", _weak_note(min(min(len(v["a_runs"]), len(v["b_runs"])) for v in weak))]
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
