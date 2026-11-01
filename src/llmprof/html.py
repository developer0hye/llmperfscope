"""One self-contained HTML page per run: the lifecycle on a shared time axis, with the diagnosis on top. No scripts, no network,
nothing to install, so the file can be mailed or attached to a pull request."""

from __future__ import annotations

import html as html_lib
import json
from pathlib import Path

from .gpu import G_MEM, G_SM, G_UTIL
from .sampler import H_ANON, H_FILE, H_STICKS, H_UTICKS, energy_series, load_ticks
from .system import SECTOR
from .procfs import CLK_TCK

WIDTH = 1120
PLOT_LEFT = 74
PLOT_RIGHT = 16
ROW_HEIGHT = 118
GROUP_COLORS = {
    "startup": "#cbd5e1", "download": "#d8b4fe", "load": "#fcd34d", "setup": "#fdba74", "warmup": "#fda4af", "prefill": "#7dd3fc",
    "decode": "#86efac", "serve": "#86efac", "run": "#86efac", "observe": "#86efac", "teardown": "#cbd5e1", "idle": "#e2e8f0",
}
SERIES_COLORS = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2", "#db2777", "#65a30d"]


def _escape(text):
    return html_lib.escape(str(text))


def _downsample(points, limit=WIDTH):
    if len(points) <= limit:
        return points
    step = len(points) / limit
    return [points[int(i * step)] for i in range(limit)]


def _scale(values, floor=None):
    """A round upper bound for an axis."""
    top = max((v for v in values if v is not None), default=0)
    if floor is not None:
        top = max(top, floor)
    if top <= 0:
        return 1.0
    magnitude = 10 ** (len(str(int(top))) - 1) if top >= 1 else 0.1
    for factor in (1, 2, 2.5, 5, 10):
        if top <= factor * magnitude:
            return factor * magnitude
    return top


def _format_axis(value):
    if value >= 1000:
        return f"{value / 1000:g}k"
    return f"{value:g}"


def _chart(title, unit, series, phases, t_max, floor=None):
    """series: [(label, [(t, value), ...])]. Colored bands behind the lines are the phases."""
    all_values = [v for _, points in series for _, v in points]
    if not all_values:
        return ""
    y_max = _scale(all_values, floor)
    plot_width = WIDTH - PLOT_LEFT - PLOT_RIGHT
    x = lambda t: PLOT_LEFT + plot_width * t / max(t_max, 1e-9)  # noqa: E731
    y = lambda v: 10 + (ROW_HEIGHT - 28) * (1 - min(v, y_max) / y_max)  # noqa: E731
    parts = [f'<svg viewBox="0 0 {WIDTH} {ROW_HEIGHT}" role="img" aria-label="{_escape(title)}">']
    for phase in phases:
        color = GROUP_COLORS.get(phase["group"])
        if color and phase["group"] not in ("baseline", "tail"):
            x0, x1 = x(phase["start_s"]), x(phase["end_s"])
            parts.append(f'<rect x="{x0:.1f}" y="10" width="{max(x1 - x0, 0.5):.1f}" height="{ROW_HEIGHT - 28}" fill="{color}" opacity="0.28"/>')
    for fraction in (0, 0.5, 1):
        level = y(y_max * fraction)
        parts.append(f'<line x1="{PLOT_LEFT}" x2="{WIDTH - PLOT_RIGHT}" y1="{level:.1f}" y2="{level:.1f}" stroke="var(--grid)" stroke-width="1"/>')
        parts.append(f'<text x="{PLOT_LEFT - 8}" y="{level + 4:.1f}" text-anchor="end" class="tick">{_format_axis(y_max * fraction)}</text>')
    parts.append(f'<text x="{PLOT_LEFT}" y="{ROW_HEIGHT - 6}" class="tick">{_escape(title)} ({_escape(unit)})</text>')
    legend_x = WIDTH - PLOT_RIGHT
    for index, (label, points) in reversed(list(enumerate(series))):
        color = SERIES_COLORS[index % len(SERIES_COLORS)]
        points = _downsample(points)
        path = " ".join(f"{x(t):.1f},{y(v):.1f}" for t, v in points)
        parts.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="1.5"/>')
        if len(series) > 1 or label:
            parts.append(f'<text x="{legend_x}" y="{ROW_HEIGHT - 6}" text-anchor="end" class="tick" fill="{color}" style="fill:{color}">{_escape(label)}</text>')
            legend_x -= 8 * len(str(label)) + 16
    parts.append("</svg>")
    return "".join(parts)


def _rates(ticks, getter, t0, min_dt_s=0.0):
    """A cumulative counter as a rate per second. Counters that update in steps (the GPU energy counter, CPU ticks) need a window
    wider than a step, or the rate jumps between zero and a multiple of its true value."""
    out = []
    previous = None
    for tick in ticks:
        value = getter(tick)
        if value is None:
            continue
        if previous is None:
            previous = (tick["t"], value)
            continue
        dt = (tick["t"] - previous[0]) / 1e9
        if dt > 0 and dt >= min_dt_s:
            out.append(((tick["t"] - t0) / 1e9, max(0.0, (value - previous[1]) / dt)))
            previous = (tick["t"], value)
    return out


def _energy_rate(ticks, device, t0):
    """Average power between readings of the energy counter."""
    times, joules = energy_series(ticks, device)
    points = []
    anchor = None
    for t, value in zip(times, joules):
        if anchor is None:
            anchor = (t, value)
        elif t - anchor[0] >= 300_000_000:
            points.append(((t - t0) / 1e9, max(0.0, (value - anchor[1]) / ((t - anchor[0]) / 1e9))))
            anchor = (t, value)
    return points


def _table(headers, rows):
    head = "".join(f"<th>{_escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{_escape(c)}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _gib(mib):
    return "-" if mib is None else f"{mib / 1024:,.2f}"


def _value(value, digits=1):
    if value is None:
        return "-"
    return f"{value:,.{digits}f}" if isinstance(value, float) else f"{value:,}"


def render(run, ticks):
    phases = run.get("phases") or []
    t0 = ticks[0]["t"] if ticks else 0
    t_max = (ticks[-1]["t"] - t0) / 1e9 if ticks else 1.0
    devices = run.get("devices_used") or []
    out = []
    title = run.get("label") or run.get("adapter") or "llmprof run"
    out.append(f"<h1>{_escape(title)}</h1>")
    facts = [f"status: {run.get('status')}", f"adapter: {run.get('adapter')}"]
    host = (run.get("host") or {}).get("hostname")
    if host:
        facts.append(f"host: {host}")
    before = run.get("machine_before") or {}
    if "clean" in before:
        facts.append("machine clean" if before["clean"] else "machine NOT clean")
    if before.get("model_page_cache_fraction") is not None:
        facts.append(f"model {before['model_page_cache_fraction'] * 100:.0f}% in page cache")
    storage = before.get("model_storage") or {}
    kinds = sorted({d.get("kind", "?") for d in storage.get("disks", [])})
    if kinds:
        facts.append(f"model on {'/'.join(kinds)} ({storage.get('fstype')})")
    out.append(f'<p class="facts">{" · ".join(_escape(f) for f in facts)}</p>')
    if run.get("perturbed"):
        out.append('<p class="warn">Traced or sampled run: use it to find where time goes, not for speed numbers.</p>')

    diagnosis = run.get("diagnosis") or {}
    judged = {name: b for name, b in (diagnosis.get("bottlenecks") or {}).items() if b["confidence"] != "none"}
    if judged:
        out.append("<h2>What limits each phase</h2>")
        out.append(_table(["phase", "limit", "confidence", "why"], [[n, b["limit"], b["confidence"], b["why"]] for n, b in judged.items()]))
    for heading, key in (("Memory", "memory"), ("Data movement", "movement"), ("Serving", "serving"), ("GPU and host time", "timeline")):
        findings = diagnosis.get(key) or []
        if findings:
            out.append(f"<h2>{heading}</h2><ul>")
            for finding in sorted(findings, key=lambda f: f["severity"] != "warning"):
                css = ' class="warn"' if finding["severity"] == "warning" else ""
                out.append(f"<li{css}>{_escape(finding['text'])}</li>")
            out.append("</ul>")

    out.append("<h2>Lifecycle</h2>")
    used_groups = []
    for phase in phases:
        if phase["group"] not in used_groups and phase["group"] in GROUP_COLORS:
            used_groups.append(phase["group"])
    legend = "".join(f'<span class="swatch" style="background:{GROUP_COLORS[g]}"></span>{_escape(g)} ' for g in used_groups)
    out.append(f'<p class="legend">{legend}</p>')

    if ticks:
        t = lambda tick: (tick["t"] - t0) / 1e9  # noqa: E731
        slow = [tick for tick in ticks if "s" in tick]
        charts = []
        if devices:
            charts.append(_chart("GPU memory", "MiB", [(f"GPU {d}", [(t(k), k["g"][d][G_MEM] or 0) for k in ticks]) for d in devices], phases, t_max))
            charts.append(_chart("GPU power", "W", [(f"GPU {d}", _energy_rate(ticks, d, t0)) for d in devices], phases, t_max))
            charts.append(_chart("GPU utilization", "%", [(f"GPU {d}", _util_points(slow, d, ticks, t0)) for d in devices], phases, t_max, floor=100))
            charts.append(_chart("GPU clock", "MHz", [(f"GPU {d}", [(t(k), k["g"][d][G_SM] or 0) for k in ticks]) for d in devices], phases, t_max))
        charts.append(_chart("Host memory of the process", "GiB", [
            ("anonymous", [(t(k), k["h"][H_ANON] / (1 << 20)) for k in ticks]),
            ("file-backed", [(t(k), k["h"][H_FILE] / (1 << 20)) for k in ticks])], phases, t_max))
        charts.append(_chart("Host CPU of the process", "cores", [("", _rates(ticks, lambda k: (k["h"][H_UTICKS] + k["h"][H_STICKS]) / CLK_TCK, t0, 0.25))], phases, t_max))
        disks = sorted({name for k in slow for name in k["s"].get("sys", {}).get("disk", {})})
        active = []
        for name in disks:
            read = _rates(slow, lambda k, n=name: (k["s"].get("sys", {}).get("disk", {}).get(n) or [None])[0], t0)
            points = [(p[0], p[1] * SECTOR / (1 << 20)) for p in read]
            if any(v > 0.5 for _, v in points):
                kind = (run.get("disks") or {}).get(name, {}).get("kind", "disk")
                active.append((f"{name} ({kind})", points))
        if active:
            charts.append(_chart("Disk reads, all processes", "MiB/s", active, phases, t_max))
        pressure = []
        for key, label in (("io_some", "I/O"), ("memory_some", "memory"), ("cpu_some", "CPU")):
            points = [(p[0], min(100.0, p[1] / 1e4)) for p in _rates(slow, lambda k, c=key: k["s"].get("sys", {}).get("psi", {}).get(c), t0)]
            if any(v > 1 for _, v in points):
                pressure.append((label, points))
        if pressure:
            charts.append(_chart("Time tasks stalled, all processes", "%", pressure, phases, t_max, floor=100))
        pcie = []
        for direction, label in ((1, "RAM to GPU"), (0, "GPU to RAM")):
            points = []
            for k in slow:
                rows = k["s"].get("pcie")
                if rows:
                    values = [rows[d][direction] for d in devices if rows[d][direction] is not None]
                    if values:
                        points.append((t(k), sum(values) / 1024))
            if points:
                pcie.append((label, points))
        if pcie:
            charts.append(_chart("PCIe traffic, sampled", "MiB/s", pcie, phases, t_max))
        nvlink = []
        for d in devices:
            points = _rates([k for k in slow if k["s"].get("nvlink") and k["s"]["nvlink"][d]], lambda k, d=d: k["s"]["nvlink"][d][0] / 1024, t0)
            if any(v > 1 for _, v in points):
                nvlink.append((f"GPU {d}", points))
        if nvlink:
            charts.append(_chart("NVLink traffic sent", "MiB/s", nvlink, phases, t_max))
        out.extend(chart for chart in charts if chart)
        ticks_text = "".join(f'<span style="left:{PLOT_LEFT / WIDTH * 100 + (WIDTH - PLOT_LEFT - PLOT_RIGHT) / WIDTH * 100 * s / max(t_max, 1e-9):.2f}%">{s:g} s</span>'
                             for s in _time_marks(t_max))
        out.append(f'<div class="timeaxis">{ticks_text}</div>')

    rows = []
    for name, group in (run.get("groups") or {}).items():
        if name in ("baseline", "tail") or group.get("wall_s", 0.0) < 0.005:
            continue
        gpu = group.get("gpu") or {}
        host_row = group.get("host") or {}
        rows.append([name, _value(group.get("wall_s"), 2), _value(group.get("tokens")), _gib(gpu.get("mem_end_mib")), _gib(gpu.get("mem_peak_mib")),
                     _gib(host_row.get("anon_peak_mib")), _gib(host_row.get("file_peak_mib")), _value(gpu.get("energy_j"), 0), _value(gpu.get("util_avg"), 0),
                     _value(host_row.get("busy_cores"), 2)])
    if rows:
        out.append("<h2>Phases</h2>")
        out.append(_table(["phase", "wall s", "tokens", "GPU GiB end", "GPU GiB peak", "host anon GiB", "host file GiB", "GPU J", "GPU util %", "CPU cores"], rows))

    throughput = run.get("throughput") or {}
    energy = run.get("energy") or {}
    rows = []
    for name, entry in throughput.items():
        if "tokens_per_s" in entry:
            rows.append([name, _value(entry["tokens_per_s"], 2), _value(entry.get("stddev"), 2), _value(energy.get(f"{name}_j_per_token"), 3), "-", "-"])
        elif "output_tokens_per_s" in entry:
            rows.append([name, _value(entry.get("output_tokens_per_s"), 2), "-", _value(entry.get("j_per_output_token"), 3),
                         _value((entry.get("ttft_ms") or {}).get("p50"), 1), _value((entry.get("itl_ms") or {}).get("p50"), 1)])
    if rows:
        out.append("<h2>Speed</h2>")
        out.append(_table(["", "tokens/s", "stddev", "GPU J/token", "first token ms p50", "between tokens ms p50"], rows))

    return PAGE.format(title=_escape(title), body="\n".join(out))


def _util_points(slow, device, ticks, t0):
    """The driver's utilization history when there is one, otherwise the instantaneous reading."""
    points = []
    for tick in slow:
        rows = tick["s"].get("util")
        if rows:
            points += [((sample_t - t0) / 1e9, value) for sample_t, value in rows[device]]
    if points:
        return sorted(p for p in points if p[0] >= 0)
    return [((tick["t"] - t0) / 1e9, tick["g"][device][G_UTIL] or 0) for tick in ticks]


def _time_marks(t_max):
    step = _scale([t_max / 6])
    marks = []
    value = 0.0
    while value <= t_max:
        marks.append(round(value, 6))
        value += step
    return marks


def write(run_dir):
    run_dir = Path(run_dir)
    with open(run_dir / "run.json") as f:
        run = json.load(f)
    ticks = load_ticks(run_dir / "samples.jsonl.gz")
    target = run_dir / "report.html"
    target.write_text(render(run, ticks), encoding="utf-8")
    return target


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ --bg:#fff; --fg:#111827; --muted:#6b7280; --grid:#e5e7eb; --warn:#b91c1c; --head:#f3f4f6; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0f172a; --fg:#e5e7eb; --muted:#94a3b8; --grid:#1f2937; --warn:#fca5a5; --head:#1e293b; }} }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.5 -apple-system, "Segoe UI", Roboto, sans-serif; margin:0 auto; padding:24px 16px; max-width:1160px; }}
h1 {{ font-size:24px; margin:0 0 4px; }} h2 {{ font-size:17px; margin:28px 0 8px; }}
.facts {{ color:var(--muted); margin:0; }} .legend {{ color:var(--muted); margin:0 0 6px; font-size:13px; }}
.swatch {{ display:inline-block; width:12px; height:12px; border-radius:2px; margin:0 4px 0 10px; vertical-align:-1px; opacity:.75; }}
.warn {{ color:var(--warn); }} ul {{ padding-left:20px; margin:0; }} li {{ margin:4px 0; }}
table {{ border-collapse:collapse; width:100%; font-size:13.5px; display:block; overflow-x:auto; }} th, td {{ padding:5px 10px; text-align:left; border-bottom:1px solid var(--grid); }}
th {{ background:var(--head); font-weight:600; white-space:nowrap; }}
svg {{ display:block; width:100%; height:auto; }} .tick {{ font-size:11px; fill:var(--muted); }}
.timeaxis {{ position:relative; height:20px; font-size:11px; color:var(--muted); }} .timeaxis span {{ position:absolute; transform:translateX(-50%); white-space:nowrap; }}
</style></head><body>
{body}
</body></html>
"""
