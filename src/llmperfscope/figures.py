"""Figures for papers and reports: vector PDF and SVG, high-resolution PNG, the data behind each figure as CSV, and a suggested caption.

Made to go into a paper as they are. Fonts are embedded as TrueType (venues reject Type 3), sizes match one and two column widths,
colors stay distinct under color vision deficiency, and `--print-safe` adds line styles for grayscale print. Identical input gives
identical SVG and PDF, so a figure can be regenerated and diffed.

Needs matplotlib, which the measuring side of llmperfscope does not: pip install "llmperfscope[figures]".
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from . import __version__, nsys
from .gpu import G_MEM, G_POWER, G_SM, G_UTIL
from .procfs import CLK_TCK
from .sampler import H_ANON, H_FILE, H_STICKS, H_UTICKS, energy_series, load_ticks
from .system import SECTOR

SINGLE_COLUMN_IN = 3.5
DOUBLE_COLUMN_IN = 7.16

# Categorical order validated for color vision deficiency on a white surface; its order is part of what makes it safe, so it is
# assigned in sequence and never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQUENTIAL = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
LINE_STYLES = ["-", "--", ":", "-."]
HATCHES = ["", "////", "\\\\\\\\", "xxxx", "", "////", "\\\\\\\\", "xxxx"]
PANEL_NAMES = {"gpu_memory": "GPU memory", "power": "GPU power", "util": "GPU utilization", "clock": "GPU clock", "host_memory": "host memory", "cpu": "host CPU",
               "disk": "disk traffic", "stall": "task stalls", "pcie": "PCIe traffic", "nvlink": "NVLink traffic"}
GOOD, CRITICAL = "#0ca30c", "#d03b3b"
INK, INK2, MUTED, GRID, AXIS, BAND = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#f4f3ef"
NEUTRAL_FILL = "#c3c2b7"
TITLE_PAD = 4.0

PHASE_ORDER = ["startup", "run", "download", "load", "setup", "warmup", "prefill", "decode", "serve", "observe", "teardown", "idle"]
RUN_KINDS = ("lifecycle", "phases", "memory", "movement", "serving", "gpu-time", "hotspots")
COMPARE_KINDS = ("speed", "memory", "energy", "phases", "serving")
ALL_PANELS = ("gpu_memory", "power", "util", "clock", "host_memory", "cpu", "disk", "stall", "pcie", "nvlink")
DEFAULT_PANELS = ("gpu_memory", "power", "util", "host_memory", "cpu", "disk", "stall", "pcie", "nvlink")


@dataclass
class Style:
    width: float = DOUBLE_COLUMN_IN
    height: float = 0.0
    formats: tuple = ("pdf", "png", "svg")
    dpi: int = 300
    font: str = "sans-serif"
    font_size: float = 7.5
    print_safe: bool = False
    title: str = ""
    letters: bool = False


def parse_width(value):
    return {"single": SINGLE_COLUMN_IN, "double": DOUBLE_COLUMN_IN}.get(str(value), None) or float(value)


def _matplotlib():
    try:
        import matplotlib
    except ImportError as error:
        raise SystemExit('Figures need matplotlib: pip install "llmperfscope[figures]"') from error
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _rc(style):
    size = style.font_size
    return {
        # TrueType in PDF and real text in SVG: venues reject Type 3 fonts, and text should stay selectable and editable.
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "llmperfscope",
        "font.family": style.font,
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
        "font.size": size, "axes.titlesize": size, "axes.labelsize": size, "xtick.labelsize": size - 0.5, "ytick.labelsize": size - 0.5,
        "legend.fontsize": size - 0.5, "figure.titlesize": size + 1.5,
        "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.5, "xtick.color": AXIS, "ytick.color": AXIS,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.0, "ytick.major.size": 0.0,
        "xtick.major.pad": 2.0, "ytick.major.pad": 2.5, "axes.labelpad": 2.0, "axes.titlepad": TITLE_PAD,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titlelocation": "left",
        "grid.color": GRID, "grid.linewidth": 0.4, "axes.axisbelow": True,
        "lines.linewidth": 1.1, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False, "legend.borderpad": 0.0, "legend.borderaxespad": 0.0, "legend.handlelength": 1.5, "legend.columnspacing": 1.0, "legend.handletextpad": 0.4,
        "figure.constrained_layout.use": True, "figure.constrained_layout.h_pad": 0.02, "figure.constrained_layout.w_pad": 0.02,
        "figure.constrained_layout.hspace": 0.03, "figure.constrained_layout.wspace": 0.04,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
        "path.simplify": True,
    }


class Output:
    """Writes figures and their data, and collects captions."""

    def __init__(self, directory, style):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.style = style
        self.written = []
        self.captions = []

    def save(self, fig, stem, header, rows, caption):
        software = f"llmperfscope {__version__}"
        # The layout engine moves things a little on every draw. Fixed once, the files do not depend on which formats were asked for.
        fig.canvas.draw()
        fig.set_layout_engine("none")
        for fmt in self.style.formats:
            path = self.dir / f"{stem}.{fmt}"
            # No dates in the files: the same run gives the same figure.
            metadata = {"pdf": {"Creator": software, "Producer": software, "CreationDate": None},
                        "svg": {"Creator": software, "Date": None},
                        "png": {"Software": software}}.get(fmt)
            fig.savefig(path, format=fmt, dpi=self.style.dpi if fmt == "png" else None, metadata=metadata)
            self.written.append(path)
        with open(self.dir / f"{stem}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        self.written.append(self.dir / f"{stem}.csv")
        self.captions.append((stem, caption))

    def finish(self, method):
        lines = ["# Suggested captions", "", "Every number here comes from the run itself. Edit them into your own words.", ""]
        for stem, caption in self.captions:
            lines += [f"**{stem}**: {caption}", ""]
        if method:
            lines += ["## How it was measured", "", method, ""]
        path = self.dir / "captions.md"
        path.write_text("\n".join(lines))
        self.written.append(path)


class RunData:
    """One run directory, with its time axis starting when the process started."""

    def __init__(self, path, label=None):
        self.dir = Path(path)
        with open(self.dir / "run.json") as f:
            self.report = json.load(f)
        self.label = label or self.report.get("label") or self.dir.name
        self.ticks = load_ticks(self.dir / "samples.jsonl.gz")
        self.devices = self.report.get("devices_used") or []
        self.phases = self.report.get("phases") or []
        first = self.ticks[0]["t"] if self.ticks else 0
        self.t0 = self.report.get("t0_monotonic_ns") or first
        started = [p["start_s"] for p in self.phases if p["group"] != "baseline"]
        self.spawn_s = started[0] if started else 0.0
        self.groups = self.report.get("groups") or {}

    def seconds(self, t_ns):
        return (t_ns - self.t0) / 1e9 - self.spawn_s

    def slow(self):
        return [tick for tick in self.ticks if "s" in tick]

    def bands(self):
        """Contiguous phases of one group, as (group, start, end). Zero-length phases do not break them up."""
        out = []
        for phase in self.phases:
            group = phase["group"]
            if group in ("baseline", "tail") or phase["end_s"] - phase["start_s"] < 0.001:
                continue
            start, end = phase["start_s"] - self.spawn_s, phase["end_s"] - self.spawn_s
            if out and out[-1][0] == group:
                out[-1] = (group, out[-1][1], end)
            else:
                out.append((group, start, end))
        return out

    def working_groups(self):
        return [g for g in PHASE_ORDER if g in self.groups and self.groups[g].get("wall_s", 0.0) >= 0.005] + [
            g for g in self.groups if g not in PHASE_ORDER and g not in ("baseline", "tail") and self.groups[g].get("wall_s", 0.0) >= 0.005]

    def serve_levels(self):
        out = [(name, t) for name, t in (self.report.get("throughput") or {}).items()
               if isinstance(t, dict) and "concurrency" in t and t.get("output_tokens_per_s")]
        return sorted(out, key=lambda item: item[1]["concurrency"])

    def requests(self):
        path = self.dir / "requests.jsonl"
        rows = []
        if path.exists():
            with open(path) as f:
                for line in f:
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        pass
        return rows

    def hardware(self):
        names = {}
        for device in self.report.get("devices") or []:
            if device.get("index") in self.devices:
                names[device.get("name")] = names.get(device.get("name"), 0) + 1
        return ", ".join(f"{count} x {name}" for name, count in names.items())


def _decimate(np, x, y, limit=1600):
    """Keeps peaks: each bucket contributes its lowest and highest point. A plain stride would lose exactly the spikes worth seeing."""
    if len(x) <= limit:
        return x, y
    edges = np.linspace(0, len(x), limit // 2 + 1, dtype=int)
    keep = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            segment = y[a:b]
            keep += sorted({a + int(np.argmin(segment)), a + int(np.argmax(segment))})
    return x[keep], y[keep]


def _rate(np, times, values, min_dt_s):
    """A cumulative counter as a rate per second, over windows at least min_dt_s wide."""
    xs, ys = [], []
    anchor = None
    for t, value in zip(times, values):
        if value is None:
            continue
        if anchor is None:
            anchor = (t, value)
        elif t - anchor[0] >= min_dt_s:
            xs.append((t + anchor[0]) / 2)
            ys.append(max(0.0, (value - anchor[1]) / (t - anchor[0])))
            anchor = (t, value)
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def _panels(np, run, wanted, per_gpu):
    """Each panel: key, y label, [(series label, x, y)], fixed top or None. Optional panels only appear when they have something to show."""
    panels = {}
    ticks = run.ticks
    if not ticks:
        return []
    t = np.array([run.seconds(k["t"]) for k in ticks])
    slow = run.slow()
    slow_t = np.array([run.seconds(k["t"]) for k in slow])
    devices = run.devices
    count = len(devices)
    suffix = f", {count} GPUs" if count > 1 else ""

    if devices:
        memory = np.array([[(k["g"][d][G_MEM] or 0) for d in devices] for k in ticks], dtype=float) / 1024
        if per_gpu and count > 1:
            panels["gpu_memory"] = ("gpu_memory", "GPU memory (GiB)", [(f"GPU {d}", t, memory[:, i]) for i, d in enumerate(devices)], None)
        else:
            panels["gpu_memory"] = ("gpu_memory", f"GPU memory{suffix} (GiB)", [("", t, memory.sum(axis=1))], None)

        grid_end = t[-1]
        power_series = []
        grid = np.arange(t[0], grid_end, 0.5)
        cumulative = np.zeros_like(grid)
        have_energy = True
        for d in devices:
            times, joules = energy_series(ticks, d)
            if len(times) < 2:
                have_energy = False
                break
            seconds = np.array([run.seconds(x) for x in times])
            joules = np.array(joules)
            cumulative = cumulative + np.interp(grid, seconds, joules)
            power_series.append((f"GPU {d}", (grid[1:] + grid[:-1]) / 2, np.maximum(0.0, np.diff(np.interp(grid, seconds, joules)) / np.diff(grid))))
        if have_energy and len(grid) > 2:
            if per_gpu and count > 1:
                panels["power"] = ("power", "GPU power (W)", power_series, None)
            else:
                panels["power"] = ("power", f"GPU power{suffix} (W)", [("", (grid[1:] + grid[:-1]) / 2, np.maximum(0.0, np.diff(cumulative) / np.diff(grid)))], None)

        if "power" not in panels:
            gauge = np.array([[(k["g"][d][G_POWER] or 0) for d in devices] for k in ticks], dtype=float)
            panels["power"] = ("power", f"GPU power{suffix} (W)", [("", t, gauge.sum(axis=1))], None)

        history = {d: ([], []) for d in devices}
        for tick in slow:
            for d, samples in enumerate(tick["s"].get("util", [])):
                if d in history:
                    for sample_t, value in samples:
                        history[d][0].append(run.seconds(sample_t))
                        history[d][1].append(value)
        util_rows = []
        for d in devices:
            times, values = history[d]
            if len(times) >= 2:
                order = np.argsort(times)
                util_rows.append(np.interp(t, np.array(times)[order], np.array(values, dtype=float)[order]))
            else:
                util_rows.append(np.array([(k["g"][d][G_UTIL] or 0) for k in ticks], dtype=float))
        util_rows = np.array(util_rows)
        if per_gpu and count > 1:
            panels["util"] = ("util", "GPU utilization (%)", [(f"GPU {d}", t, util_rows[i]) for i, d in enumerate(devices)], 100.0)
        else:
            panels["util"] = ("util", f"GPU utilization{', mean' if count > 1 else ''} (%)", [("", t, util_rows.mean(axis=0))], 100.0)
        clock = np.array([[(k["g"][d][G_SM] or 0) for d in devices] for k in ticks], dtype=float)
        panels["clock"] = ("clock", "GPU clock (MHz)", [("", t, clock.mean(axis=1))], None)

    panels["host_memory"] = ("host_memory", "Host memory (GiB)", [
        ("anonymous", t, np.array([k["h"][H_ANON] for k in ticks]) / (1 << 20)),
        ("file-backed", t, np.array([k["h"][H_FILE] for k in ticks]) / (1 << 20))], None)
    x, y = _rate(np, t, [(k["h"][H_UTICKS] + k["h"][H_STICKS]) / CLK_TCK for k in ticks], 0.25)
    if len(x):
        panels["cpu"] = ("cpu", "Host CPU (cores)", [("", x, y)], None)

    if len(slow) >= 2:
        def system_rate(getter, scale=1.0, min_dt=0.4):
            values = [getter(k["s"].get("sys") or {}) for k in slow]
            x_rate, y_rate = _rate(np, slow_t, values, min_dt)
            return x_rate, y_rate * scale

        def disk_total(index):
            def getter(row):
                disks = row.get("disk") or {}
                return sum(v[index] for v in disks.values()) if disks else None
            return getter

        mib = SECTOR / (1 << 20)
        disk = []
        for label, index in (("read", 0), ("written", 1)):
            x, y = system_rate(disk_total(index), mib)
            if len(y) and y.max() > 10.0:
                disk.append((label, x, y))
        if disk:
            panels["disk"] = ("disk", "Disk, all processes (MiB/s)", disk, None)

        stall = []
        for label, key in (("I/O", "io_some"), ("memory", "memory_some"), ("CPU", "cpu_some")):
            x, y = system_rate(lambda row, k=key: (row.get("psi") or {}).get(k), 100 / 1e6)
            if len(y) and y.max() > 5.0:
                stall.append((label, x, np.minimum(y, 100.0)))
        if stall:
            panels["stall"] = ("stall", "Tasks stalled, system (%)", stall, 100.0)

        pcie = []
        for label, direction in (("RAM to GPU", 1), ("GPU to RAM", 0)):
            xs, ys = [], []
            for tick, when in zip(slow, slow_t):
                rows = tick["s"].get("pcie")
                if rows:
                    values = [rows[d][direction] for d in devices if rows[d][direction] is not None]
                    if values:
                        xs.append(when)
                        ys.append(sum(values) / 1024)
            if ys and max(ys) > 10:
                pcie.append((label, np.array(xs), np.array(ys)))
        if pcie:
            panels["pcie"] = ("pcie", "PCIe, sampled (MiB/s)", pcie, None)

        link_ticks = [k for k in slow if k["s"].get("nvlink")]
        if len(link_ticks) >= 2 and devices:
            when = np.array([run.seconds(k["t"]) for k in link_ticks])
            sent = [sum((k["s"]["nvlink"][d] or [0, 0])[0] for d in devices) / 1024 for k in link_ticks]
            x, y = _rate(np, when, sent, 0.4)
            if len(y) and y.max() > 1.0:
                panels["nvlink"] = ("nvlink", "GPU to GPU, NVLink (MiB/s)", [("", x, y)], None)
    return [panels[key] for key in wanted if key in panels]


def _panel_grid(plt, count, style, row_height, narrow_columns=1):
    """One row in a wide figure. In a narrow one, stacked: three panels side by side in 3.5 inches would run into each other. Returns the
    figure, the axes, and whether each one carries the row labels."""
    columns = count if style.width >= 5 else min(narrow_columns, count)
    rows = math.ceil(count / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(style.width, style.height or rows * row_height), squeeze=False)
    flat = list(axes.flatten())
    for ax in flat[count:]:
        ax.set_visible(False)
    return fig, flat[:count], [i % columns == 0 for i in range(count)]


def _figure_legend(fig, handles, labels, where, columns):
    """Above the panels. On saving, matplotlib anchors a figure's legend to the corner of the cropped image, not of the figure, so its own
    margin is the only one it gets."""
    fig.legend(handles, labels, loc=f"outside upper {where}", ncol=columns, borderaxespad=0.4)


def _letter(ax, index, style):
    if style.letters:
        # On the title's baseline. That clears a tick label sitting at the very top of the axis.
        ax.annotate(f"({chr(97 + index)})", xy=(0, 1), xycoords="axes fraction", xytext=(-2.5, TITLE_PAD), textcoords="offset points", ha="right", va="bottom",
                    fontweight="bold", fontsize=style.font_size)


def _compact(value, _position=None):
    """Clean axis numbers: 1,600 in full, 50K compact, one style per axis."""
    if abs(value) >= 10000:
        return f"{value / 1000:,g}K"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:g}"


def lifecycle(plt, np, run, out, wanted=DEFAULT_PANELS, per_gpu=False, xlim=None):
    style = out.style
    panels = _panels(np, run, wanted, per_gpu)
    if not panels:
        return False
    bands = run.bands()
    per_panel = 0.8 if style.width > 5 else 0.68
    height = style.height or (0.4 + per_panel * len(panels))
    fig, axes = plt.subplots(len(panels) + 1, 1, sharex=True, figsize=(style.width, height), height_ratios=[0.12] + [1.0] * len(panels))
    ribbon, axes = axes[0], list(axes[1:])
    x_max = max((b[2] for b in bands), default=run.seconds(run.ticks[-1]["t"]))
    x_range = tuple(xlim) if xlim else (0.0, x_max)

    ribbon.set_axis_off()
    ribbon.set_ylim(0, 1)
    width_in = style.width * 0.9
    span = max(x_range[1] - x_range[0], 1e-9)
    for index, (group, start, end) in enumerate(bands):
        ribbon.axvspan(start, end, color=BAND if index % 2 == 0 else "#ffffff", linewidth=0, zorder=0)
        visible = max(0.0, min(end, x_range[1]) - max(start, x_range[0]))
        needed_in = len(group) * style.font_size * 0.56 / 72
        if visible / span * width_in >= needed_in:
            middle = (max(start, x_range[0]) + min(end, x_range[1])) / 2
            ribbon.text(middle, 0.45, group, ha="center", va="center", color=INK2, fontsize=style.font_size - 0.5)

    rows = []
    for index, (ax, (key, ylabel, series, top)) in enumerate(zip(axes, panels)):
        for k, (group, start, end) in enumerate(bands):
            if k % 2 == 0:
                ax.axvspan(start, end, color=BAND, linewidth=0, zorder=0)
        for k, (label, x, y) in enumerate(series):
            for a, b in zip(x, y):
                rows.append([key, label, round(float(a), 3), round(float(b), 3)])
            px, py = _decimate(np, x, y)
            linestyle = LINE_STYLES[k % len(LINE_STYLES)] if style.print_safe else "-"
            ax.plot(px, py, color=SERIES[k % len(SERIES)], label=label or None, linestyle=linestyle, zorder=3)
        peak = max((float(np.max(y)) for _, _, y in series if len(y)), default=1.0)
        ax.set_ylim(0, top if top else max(peak * 1.12, 1e-6))
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=3, min_n_ticks=2))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(_compact))
        ax.grid(axis="y")
        ax.spines["left"].set_visible(False)
        ax.set_title(ylabel)
        labelled = [s for s in series if s[0]]
        if len(labelled) >= 2:
            ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=len(labelled))
        _letter(ax, index, style)
    axes[-1].set_xlim(*x_range)
    axes[-1].set_xlabel("Time since process start (s)")
    if style.title:
        fig.suptitle(style.title, x=0.0, ha="left")

    names = [PANEL_NAMES[p[0]] for p in panels]
    phase_text = ", ".join(f"{group} {run.groups[group]['wall_s']:.1f} s" for group in run.working_groups() if group not in ("idle", "startup"))
    out.save(fig, "lifecycle", ["panel", "series", "time_s", "value"], rows,
             f"Lifecycle of {run.label}{_on(run)}: {', '.join(names)}, from process start. Bands mark phases ({phase_text}).")
    plt.close(fig)
    return True


def _on(run):
    hardware = run.hardware()
    return f" on {hardware}" if hardware else ""


def _categories(ax, labels, show=True, rows=None):
    """Rows from the top down. A chart with few rows keeps its bars thin instead of letting them fill the panel. With `rows`, panels side
    by side get the same rows from the top, and so bars of the same thickness."""
    ax.set_yticks(range(len(labels)), labels if show else [""] * len(labels))
    if rows:
        ax.set_ylim(rows - 0.5, -0.5)
        return
    extra = max(0.0, (4 - len(labels)) / 2)
    ax.set_ylim(len(labels) - 0.5 + extra, -0.5 - extra)


def _log_ticks(axis, low, high):
    """Numbers on a log axis even when it spans less than a decade, where the default would label nothing."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator
    decades = math.log10(max(high, 1e-12) / max(low, 1e-12))
    subs = (1.0,) if decades > 2 else (1.0, 2.0, 5.0) if decades > 0.8 else (1.0, 1.5, 2.0, 3.0, 5.0, 7.0)
    axis.set_major_locator(LogLocator(base=10, subs=subs, numticks=12))
    axis.set_major_formatter(FuncFormatter(_compact))
    axis.set_minor_locator(NullLocator())


def _hbars(ax, labels, values, formatter, title, color=SERIES[0], errors=None, first=True):
    """Horizontal bars with the value at the tip. Every value is labeled, so the value axis would only repeat it."""
    positions = list(range(len(labels)))
    numeric = [v if v is not None else 0.0 for v in values]
    # One repetition has a standard deviation of zero, and a bar would end in a stray tick.
    errors = errors if errors and any(errors) else None
    bars = {"xerr": errors, "error_kw": {"elinewidth": 0.6, "ecolor": INK2, "capsize": 1.2, "capthick": 0.6}} if errors else {}
    ax.barh(positions, numeric, height=0.62, color=color, linewidth=0, **bars)
    _categories(ax, labels, first)
    top = max(numeric + [v + (e or 0) for v, e in zip(numeric, errors or [0] * len(numeric))]) or 1.0
    for position, value, number in zip(positions, values, numeric):
        offset = (errors[position] if errors else 0.0) or 0.0
        ax.text(number + offset + top * 0.02, position, "-" if value is None else formatter(value), va="center", ha="left", color=INK, fontsize=ax.xaxis.label.get_size() - 0.5)
    ax.set_xlim(0, top * 1.28)
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)


def _fmt(value, digits=None):
    if value is None:
        return "-"
    if digits is not None:
        return f"{value:,.{digits}f}"
    if value >= 100:
        return f"{value:,.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def phases_figure(plt, run, out):
    style = out.style
    groups = run.working_groups()
    if not groups:
        return False
    metrics = [
        ("Wall time (s)", lambda g: g.get("wall_s")),
        ("GPU utilization (%)", lambda g: (g.get("gpu") or {}).get("util_avg")),
        ("Host CPU (cores)", lambda g: (g.get("host") or {}).get("busy_cores")),
        ("GPU energy (kJ)", lambda g: None if (g.get("gpu") or {}).get("energy_j") is None else g["gpu"]["energy_j"] / 1000),
    ]
    metrics = [m for m in metrics if any(m[1](run.groups[g]) is not None for g in groups)]
    fig, axes, labelled = _panel_grid(plt, len(metrics), style, 0.32 + 0.17 * len(groups), narrow_columns=2)
    rows = []
    for index, (ax, (title, getter)) in enumerate(zip(axes, metrics)):
        values = [getter(run.groups[g]) for g in groups]
        _hbars(ax, groups, values, _fmt, title, first=labelled[index])
        rows += [[title, g, v] for g, v in zip(groups, values)]
        _letter(ax, index, style)
    out.save(fig, "phases", ["metric", "phase", "value"], rows,
             f"Per-phase summary of {run.label}{_on(run)}: wall time, GPU utilization, host CPU and GPU energy.")
    plt.close(fig)
    return True


def memory_figure(plt, run, out):
    style = out.style
    groups = [g for g in run.working_groups() if g != "idle"]
    if not groups:
        return False
    memory = run.report.get("memory") or {}
    columns = [
        ("GPU memory, peak (GiB)", lambda g: None if (g.get("gpu") or {}).get("mem_peak_mib") is None else g["gpu"]["mem_peak_mib"] / 1024),
        ("Host anonymous, peak (GiB)", lambda g: None if (g.get("host") or {}).get("anon_peak_mib") is None else g["host"]["anon_peak_mib"] / 1024),
        ("Host file-backed, peak (GiB)", lambda g: None if (g.get("host") or {}).get("file_peak_mib") is None else g["host"]["file_peak_mib"] / 1024),
    ]
    columns = [c for c in columns if any(c[1](run.groups[g]) is not None for g in groups)]
    breakdown = [row for row in memory.get("breakdown_mib") or [] if row.get("self")]
    devices = (memory.get("gpu") or {}).get("devices") or []
    extra_rows = 1 if breakdown or len(devices) > 1 else 0
    wide = style.width >= 5
    top_rows = 1 if wide else len(columns)
    top_height = (0.28 + 0.17 * len(groups)) * top_rows
    device_rows = max(len(breakdown), len(devices))
    # Narrow: the per-GPU panel and the engine's table go one under the other too.
    both = bool(breakdown) and len(devices) > 1
    bottom_height = ((0.34 + 0.17 * device_rows) * (2 if both and not wide else 1)) if extra_rows else 0.0
    fig = plt.figure(figsize=(style.width, style.height or top_height + bottom_height + 0.2))
    grid = fig.add_gridspec(1 + extra_rows, 1, height_ratios=[top_height] + ([bottom_height] if extra_rows else []))
    top = grid[0].subgridspec(top_rows, len(columns) if wide else 1)
    rows = []
    letter = 0
    for index, (title, getter) in enumerate(columns):
        ax = fig.add_subplot(top[0, index] if wide else top[index, 0])
        values = [getter(run.groups[g]) for g in groups]
        _hbars(ax, groups, values, _fmt, title, first=index == 0 or not wide)
        rows += [[title, g, "", v] for g, v in zip(groups, values)]
        _letter(ax, letter, style)
        letter += 1
    if extra_rows:
        parts = (1 if len(devices) > 1 else 0) + (1 if breakdown else 0)
        bottom = grid[1].subgridspec(1, parts) if wide else grid[1].subgridspec(parts, 1)
        slot = 0
        if len(devices) > 1:
            ax = fig.add_subplot(bottom[0, slot] if wide else bottom[slot, 0])
            slot += 1
            labels = [f"GPU {d['index']}" for d in devices]
            values = [d["peak_above_baseline_mib"] / 1024 for d in devices]
            _hbars(ax, labels, values, _fmt, "Peak per GPU, above idle (GiB)")
            rows += [["Peak per GPU, above idle (GiB)", label, "", v] for label, v in zip(labels, values)]
            _letter(ax, letter, style)
            letter += 1
        if breakdown:
            ax = fig.add_subplot(bottom[0, slot] if wide else bottom[slot, 0])
            labels = [f"{r.get('name')} (per GPU)" if str(r.get("name", "")).startswith("Meta") else str(r.get("name")) for r in breakdown]
            left = [0.0] * len(breakdown)
            for k, part in enumerate(("model", "context", "compute")):
                values = [(r.get(part) or 0) / 1024 for r in breakdown]
                ax.barh(range(len(labels)), values, left=left, height=0.62, color=SERIES[k], edgecolor="white", linewidth=0.6, label=part, hatch=HATCHES[k] if style.print_safe else None)
                rows += [["Engine's own table (GiB)", label, part, v] for label, v in zip(labels, values)]
                left = [a + b for a, b in zip(left, values)]
            _categories(ax, labels)
            ax.set_xlim(0, (max(left) or 1.0) * 1.05)
            ax.set_title("Engine's own table (GiB)")
            ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=4))
            ax.grid(axis="x")
            ax.spines["bottom"].set_visible(False)
            ax.legend(loc="upper right", bbox_to_anchor=(1.0, -0.16), ncol=3)
            _letter(ax, letter, style)
    peak = (memory.get("gpu") or {}).get("peak_above_baseline_total_mib")
    caption = f"Memory of {run.label}{_on(run)} by phase"
    if peak:
        caption += f". GPU memory peaks at {peak / 1024:.1f} GiB above idle over the GPUs in use"
    out.save(fig, "memory", ["panel", "row", "part", "gib"], rows, caption + ".")
    plt.close(fig)
    return True


def _movement_rows(run, groups):
    def disk(index):
        def getter(group):
            disks = (group.get("flows") or {}).get("disks") or {}
            return sum(d.get(index, 0.0) for d in disks.values()) if disks else None
        return getter

    def flow(key):
        return lambda group: (group.get("flows") or {}).get(key)

    table = [
        ("Disk to RAM, this process", lambda g: (g.get("host") or {}).get("disk_read_mib")),
        ("Disk to RAM, all processes", disk("read_mib")),
        ("RAM to disk, all processes", disk("write_mib")),
        ("Swap in", flow("swap_in_mib")),
        ("Swap out", flow("swap_out_mib")),
        ("Page cache read again", flow("page_cache_refault_mib")),
        ("RAM to GPU, sampled", flow("pcie_to_gpu_mib_est")),
        ("GPU to RAM, sampled", flow("pcie_from_gpu_mib_est")),
        ("GPU to GPU, NVLink", flow("nvlink_mib")),
    ]
    rows = []
    for label, getter in table:
        values = [getter(run.groups[g]) for g in groups]
        if any((v or 0) >= 1.0 for v in values):
            rows.append((label, values))
    return rows


def _size(mib):
    if mib is None or mib < 1.0:
        return "-"
    if mib >= 1024 * 100:
        return f"{mib / 1024:,.0f} GiB"
    if mib >= 1024:
        return f"{mib / 1024:.1f} GiB"
    return f"{mib:,.0f} MiB"


def movement_figure(plt, np, run, out):
    style = out.style
    groups = [g for g in run.working_groups() if g != "idle"]
    rows = _movement_rows(run, groups)
    if not rows:
        return False
    from matplotlib.colors import ListedColormap
    matrix = np.array([[(v if v and v >= 1.0 else np.nan) for v in values] for _, values in rows], dtype=float)
    scale = np.log10(np.maximum(matrix, 1.0))
    top = np.nanmax(scale) or 1.0
    fig, ax = plt.subplots(figsize=(style.width, style.height or 0.34 + 0.2 * len(rows)))
    cmap = ListedColormap(SEQUENTIAL).with_extremes(bad="#ffffff")
    ax.imshow(np.ma.masked_invalid(scale), aspect="auto", cmap=cmap, vmin=0, vmax=top)
    ax.set_xticks(range(len(groups)), groups)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_yticks(range(len(rows)), [label for label, _ in rows])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(groups)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    csv_rows = []
    for i, (label, values) in enumerate(rows):
        for j, value in enumerate(values):
            csv_rows.append([label, groups[j], value])
            fraction = (math.log10(max(value, 1.0)) / top) if value and value >= 1.0 else 0.0
            ax.text(j, i, _size(value), ha="center", va="center", fontsize=style.font_size - 0.8, color="#ffffff" if fraction > 0.55 else (INK if value and value >= 1.0 else MUTED))
    out.save(fig, "movement", ["flow", "phase", "mib"], csv_rows,
             f"Data moved between tiers per phase for {run.label}{_on(run)}. Color is on a log scale. Disk and swap counters include other processes on the machine; "
             "PCIe traffic is sampled about once a second and so is an estimate.")
    plt.close(fig)
    return True


def _cdf_axes(ax, title, xlabel):
    ax.set_title(title)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(_fixed([0, 0.5, 1.0]))
    ax.grid(axis="both")
    ax.spines["left"].set_visible(False)


def _fixed(values):
    from matplotlib.ticker import FixedLocator
    return FixedLocator(values)


def _plot_cdf(np, ax, values, color, label, linestyle="-"):
    # Gaps can be zero when a server sends tokens in one write, and a log axis has no place for that.
    ordered = np.sort(np.maximum(np.array(values, dtype=float), 1e-3))
    ax.step(ordered, np.arange(1, len(ordered) + 1) / len(ordered), where="post", color=color, label=label, linestyle=linestyle)


def serving_figure(plt, np, run, out):
    style = out.style
    levels = run.serve_levels()
    if not levels:
        return False
    requests = run.requests()
    by_level = {}
    for row in requests:
        if row.get("level") is not None and row.get("ttft_s") is not None:
            by_level.setdefault(row["level"], []).append(row)
    have_raw = bool(by_level)
    columns = 3 if have_raw else 1
    # Without per-request data there is one short bar chart, not three panels to line up with.
    fig, axes, _ = _panel_grid(plt, columns, style, (1.6 if style.width >= 5 else 1.3) if have_raw else 0.34 + 0.2 * len(levels))
    rows = []
    names = [f"{t['concurrency']}" for _, t in levels]
    values = [t["output_tokens_per_s"] for _, t in levels]
    _hbars(axes[0], names, values, _fmt, "Output tokens/s by concurrent clients")
    rows += [["output_tokens_per_s", n, "", v] for n, v in zip(names, values)]
    _letter(axes[0], 0, style)
    if have_raw:
        _cdf_axes(axes[1], "Time to first token, share of requests", "ms")
        _cdf_axes(axes[2], "Time between tokens, share of gaps", "ms")
        for k, (name, t) in enumerate(levels):
            label = f"{t['concurrency']} client" + ("s" if t["concurrency"] > 1 else "")
            linestyle = LINE_STYLES[k % len(LINE_STYLES)] if style.print_safe else "-"
            ttft = [r["ttft_s"] * 1000 for r in by_level.get(t["concurrency"], [])]
            gaps = [g * 1000 for r in by_level.get(t["concurrency"], []) for g in r.get("itl_s", [])]
            if ttft:
                _plot_cdf(np, axes[1], ttft, SERIES[k], label, linestyle)
                rows += [["ttft_ms", t["concurrency"], i, round(v, 3)] for i, v in enumerate(ttft)]
            if gaps:
                _plot_cdf(np, axes[2], gaps, SERIES[k], label, linestyle)
                rows += [["itl_ms", t["concurrency"], i, round(v, 3)] for i, v in enumerate(gaps)]
        for ax in axes[1:]:
            _log_ticks(ax.xaxis, *ax.get_xlim())
        if len(levels) >= 2:
            # Above the panels: where the curves run depends on the run, so no spot inside a panel is always free.
            _figure_legend(fig, *axes[1].get_legend_handles_labels(), "right", len(levels))
        _letter(axes[1], 1, style)
        _letter(axes[2], 2, style)
    first, last = levels[0][1], levels[-1][1]
    caption = f"Serving {run.label}{_on(run)}: output throughput by number of concurrent clients"
    if len(levels) >= 2:
        caption += f" ({first['output_tokens_per_s']:,.0f} tokens/s at {first['concurrency']}, {last['output_tokens_per_s']:,.0f} at {last['concurrency']})"
    if have_raw:
        caption += ", and the distributions of time to first token and time between tokens as a client saw them"
    out.save(fig, "serving", ["series", "clients", "index", "value"], rows, caption + ".")
    plt.close(fig)
    return True


def gpu_time_figure(plt, run, out):
    style = out.style
    groups = (run.report.get("nsys") or {}).get("groups") or {}
    chosen = [g for g in ("prefill", "decode", "serve", "run") if groups.get(g, {}).get("devices")]
    if not chosen:
        return False
    wide = style.width >= 5
    fig = plt.figure(figsize=(style.width, style.height or sum(0.42 + 0.17 * max(len(groups[g]["devices"]), 6) for g in chosen) * (1 if wide else 2)))
    grid = fig.add_gridspec(len(chosen), 2, width_ratios=[1.0, 0.9]) if wide else fig.add_gridspec(2 * len(chosen), 1)
    rows = []
    parts = (("kernels", "kernel_ms", SERIES[0]), ("collectives", "comm_ms", SERIES[1]), ("copies", "copy_ms", SERIES[2]), ("idle", "idle_ms", NEUTRAL_FILL))
    for r, group in enumerate(chosen):
        entry = groups[group]
        ax = fig.add_subplot(grid[r, 0] if wide else grid[2 * r, 0])
        devices = entry["devices"]
        labels = [f"GPU {d['device']}" for d in devices]
        left = [0.0] * len(devices)
        for k, (name, key, color) in enumerate(parts):
            shares = [(100 * d[key] / d["span_ms"]) if d["span_ms"] else 0.0 for d in devices]
            ax.barh(range(len(devices)), shares, left=left, height=0.62, color=color, edgecolor="white", linewidth=0.6, label=name, hatch=HATCHES[k] if style.print_safe else None)
            rows += [[group, label, name, round(v, 2)] for label, v in zip(labels, shares)]
            left = [a + b for a, b in zip(left, shares)]
        _categories(ax, labels)
        ax.set_xlim(0, 100)
        ax.set_title(f"{group}: share of each GPU's time (%)")
        ax.xaxis.set_major_locator(_fixed([0, 25, 50, 75, 100]))
        ax.grid(axis="x")
        ax.spines["bottom"].set_visible(False)
        if r == 0:
            _figure_legend(fig, *ax.get_legend_handles_labels(), "left", 4)
        _letter(ax, 2 * r, style)
        top = entry.get("top_kernels", [])[:8]
        ax2 = fig.add_subplot(grid[r, 1] if wide else grid[2 * r + 1, 0])
        _hbars(ax2, [_short(k["name"], 30) for k in top], [100 * k["share"] for k in top], lambda v: f"{v:.0f}%", f"{group}: top kernels, share of kernel time",
               color=[SERIES[1] if nsys.is_collective(k["name"]) else SERIES[0] for k in top])
        rows += [[group, k["name"], "kernel share", round(100 * k["share"], 2)] for k in top]
        _letter(ax2, 2 * r + 1, style)
    out.save(fig, "gpu_time", ["phase", "row", "part", "percent"], rows,
             f"How GPU time divides during {', '.join(chosen)} for {run.label}{_on(run)}, from a trace. Tracing slows the host, so idle reads higher than in an untraced run; "
             "use this figure for where time goes, not for speed.")
    plt.close(fig)
    return True


def _short(name, limit):
    return name if len(name) <= limit else name[:limit - 1] + "…"


def _function_label(name):
    """Readable in a figure: no argument lists or symbol versions, and libraries without symbols named for what they are. The CSV keeps the
    full names."""
    name = re.sub(r"\(.*", "", name).split("@@")[0].split("@GLIBC")[0]
    if name.endswith("!?"):
        library = name[:-2]
        for prefix, label in (("libcudart", "CUDA runtime"), ("libcuda", "GPU driver"), ("libnvidia", "GPU driver"), ("libnccl", "NCCL"), ("libc.so", "libc"), ("libstdc++", "libstdc++"), ("libgomp", "OpenMP runtime")):
            if library.startswith(prefix):
                return label
        return re.sub(r"\.so.*", "", library)
    if name == "[vdso]":
        return "clock (vdso)"
    return _short(name.lstrip("_"), 24)


def hotspots_figure(plt, run, out):
    style = out.style
    profile = (run.report.get("cpu_profile") or {}).get("groups") or {}
    chosen = [g for g in ("load", "prefill", "decode", "serve", "run") if profile.get(g, {}).get("samples", 0) >= 20]
    if not chosen:
        return False
    fig, axes, _ = _panel_grid(plt, len(chosen), style, 1.75 if style.width >= 5 else 1.35)
    rows = []
    tops = {group: [x for x in profile[group]["self"] if x["share"] >= 0.01][:7] for group in chosen}
    most = max(len(top) for top in tops.values())
    for index, (ax, group) in enumerate(zip(axes, chosen)):
        top = tops[group]
        labels = [_function_label(x["name"]) for x in top]
        values = [100 * x["share"] for x in top]
        ax.barh(range(len(top)), values, height=0.62, color=SERIES[0], linewidth=0)
        _categories(ax, labels, rows=most)
        peak = max(values) if values else 1.0
        for position, x in enumerate(top):
            ax.text(values[position] + peak * 0.04, position, f"{values[position]:.0f}%, {x.get('threads', 1)} thr", va="center", ha="left", fontsize=style.font_size - 0.8)
        ax.set_xlim(0, peak * 2.0)
        ax.set_title(f"{group}: {profile[group]['samples']:,} samples")
        ax.xaxis.set_visible(False)
        ax.spines["bottom"].set_visible(False)
        rows += [[group, x["name"], round(100 * x["share"], 2), x.get("threads", 1), x.get("cpu_s"), x.get("pct_of_wall_on_busiest_thread")] for x in top]
        _letter(ax, index, style)
    out.save(fig, "hotspots", ["phase", "function", "percent_of_samples", "threads", "cpu_s", "percent_of_wall_on_busiest_thread"], rows,
             f"Host CPU time by function for {run.label}{_on(run)}, from per-thread CPU-time sampling, with the number of threads (thr) that ran each function. "
             "Sampling slows the run, so use this for where time goes, not for speed.")
    plt.close(fig)
    return True


def _method(run):
    report = run.report
    parts = [f"Measured with llmperfscope {__version__}"]
    sampler = report.get("sampler") or {}
    if sampler.get("hz_target"):
        parts.append(f"sampling GPU gauges and process memory at {sampler['hz_target']:.0f} Hz and cumulative counters about twice a second")
    hardware = run.hardware()
    if hardware:
        parts.append(f"on {hardware}")
    if report.get("gpu_driver"):
        parts.append(f"driver {report['gpu_driver']}")
    text = ", ".join(parts) + "."
    if report.get("perturbed"):
        text += " This run was traced or sampled for detail; its speed is not representative."
    if report.get("adapter") == "llama.cpp:llama-bench":
        text += " llama-bench feeds random tokens, not real text."
    before = report.get("machine_before") or {}
    if before.get("model_page_cache_fraction") is not None:
        text += f" {before['model_page_cache_fraction'] * 100:.0f}% of the model was in the page cache at the start."
    command = report.get("identity", {})
    if command.get("git_commit"):
        text += f" Engine commit {command['git_commit'][:10]}."
    if command.get("container_image"):
        text += f" Container image {command['container_image']}."
    return text


def make_run_figures(path, out_dir, style, kinds=RUN_KINDS, panels=DEFAULT_PANELS, per_gpu=False, xlim=None):
    plt = _matplotlib()
    import numpy as np
    run = RunData(path)
    out = Output(out_dir, style)
    skipped = []
    makers = {
        "lifecycle": lambda: lifecycle(plt, np, run, out, panels, per_gpu, xlim),
        "phases": lambda: phases_figure(plt, run, out),
        "memory": lambda: memory_figure(plt, run, out),
        "movement": lambda: movement_figure(plt, np, run, out),
        "serving": lambda: serving_figure(plt, np, run, out),
        "gpu-time": lambda: gpu_time_figure(plt, run, out),
        "hotspots": lambda: hotspots_figure(plt, run, out),
    }
    with plt.rc_context(_rc(style)):
        for kind in kinds:
            if not makers[kind]():
                skipped.append(kind)
    out.finish(_method(run))
    return out.written, skipped


def _compare_bars(plt, out, runs, stem, metrics, caption):
    """Small multiples, one per metric, one bar per run. Metrics no run has are dropped."""
    style = out.style
    metrics = [m for m in metrics if any(m[1](r.report) is not None for r in runs)]
    if not metrics:
        return False
    labels = [r.label for r in runs]
    fig, axes, labelled = _panel_grid(plt, len(metrics), style, 0.34 + 0.2 * len(runs))
    rows = []
    for index, (ax, (title, value, error)) in enumerate(zip(axes, metrics)):
        values = [value(r.report) for r in runs]
        errors = [error(r.report) or 0.0 for r in runs] if error else None
        _hbars(ax, labels, values, _fmt, title, errors=errors, first=labelled[index])
        rows += [[title, label, v, (errors[i] if errors else "")] for i, (label, v) in enumerate(zip(labels, values))]
        _letter(ax, index, style)
    out.save(fig, stem, ["metric", "run", "value", "stddev"], rows, caption)
    plt.close(fig)
    return True


def _get(report, *keys):
    node = report
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _div(value, by):
    return None if value is None else value / by


def compare_phases(plt, out, runs):
    style = out.style
    present = [g for g in PHASE_ORDER if g != "idle" and any(r.groups.get(g, {}).get("wall_s", 0.0) >= 0.005 for r in runs)]
    if not present:
        return False
    # A phase keeps its color whichever runs are shown. Past eight, the smallest fold into "other".
    order = [g for g in PHASE_ORDER if g in present][:8]
    fig, ax = plt.subplots(figsize=(style.width, style.height or 0.5 + 0.2 * len(runs)))
    left = [0.0] * len(runs)
    rows = []
    for k, group in enumerate(order):
        values = [r.groups.get(group, {}).get("wall_s", 0.0) for r in runs]
        ax.barh(range(len(runs)), values, left=left, height=0.62, color=SERIES[k], edgecolor="white", linewidth=0.6, label=group, hatch=HATCHES[k] if style.print_safe else None)
        rows += [[r.label, group, v] for r, v in zip(runs, values)]
        left = [a + b for a, b in zip(left, values)]
    _categories(ax, [r.label for r in runs])
    ax.set_xlabel("Wall time (s)")
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=6))
    ax.grid(axis="x")
    ax.spines["bottom"].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=len(order))
    out.save(fig, "compare_phases", ["run", "phase", "wall_s"], rows, f"Wall time by lifecycle phase for {', '.join(r.label for r in runs)}.")
    plt.close(fig)
    return True


def compare_serving(plt, np, out, runs):
    style = out.style
    series = [(r, r.serve_levels()) for r in runs]
    series = [(r, levels) for r, levels in series if levels]
    if not series:
        return False
    fig, axes, _ = _panel_grid(plt, 3, style, 1.9 if style.width >= 5 else 1.35)
    rows = []
    panels = (("Output tokens/s", lambda t: t.get("output_tokens_per_s"), False), ("Time to first token, median (ms)", lambda t: (t.get("ttft_ms") or {}).get("p50"), True),
              ("Time between tokens, median (ms)", lambda t: (t.get("itl_ms") or {}).get("p50"), False))
    for index, (ax, (title, getter, log)) in enumerate(zip(axes, panels)):
        for k, (run, levels) in enumerate(series):
            x = [t["concurrency"] for _, t in levels]
            y = [getter(t) for _, t in levels]
            linestyle = LINE_STYLES[k % len(LINE_STYLES)] if style.print_safe else "-"
            ax.plot(x, y, color=SERIES[k], marker="o", markersize=3, markeredgecolor="white", markeredgewidth=0.5, label=run.label, linestyle=linestyle)
            rows += [[title, run.label, a, b] for a, b in zip(x, y)]
        ax.set_title(title)
        ax.set_xlabel("Concurrent clients")
        ticks = sorted({t["concurrency"] for _, levels in series for _, t in levels})
        ax.xaxis.set_major_locator(_fixed(ticks))
        if log:
            ax.set_yscale("log")
            _log_ticks(ax.yaxis, *ax.get_ylim())
        else:
            ax.set_ylim(bottom=0)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(_compact))
        ax.grid(axis="y")
        ax.spines["left"].set_visible(False)
        _letter(ax, index, style)
    handles, labels = axes[0].get_legend_handles_labels()
    if len(series) >= 2:
        _figure_legend(fig, handles, labels, "left", len(series))
    out.save(fig, "compare_serving", ["metric", "run", "clients", "value"], rows, f"Serving under increasing concurrency for {', '.join(r.label for r, _ in series)}, as a streaming client saw it.")
    plt.close(fig)
    return True


def make_compare_figures(specs, out_dir, style, kinds=COMPARE_KINDS):
    """specs: [(label, run directory)]."""
    plt = _matplotlib()
    import numpy as np
    runs = [RunData(path, label) for label, path in specs]
    out = Output(out_dir, style)
    names = ", ".join(r.label for r in runs)
    makers = {
        "speed": lambda: _compare_bars(plt, out, runs, "compare_speed", [
            ("Prefill (tokens/s)", lambda r: _get(r, "throughput", "prefill", "tokens_per_s"), lambda r: _get(r, "throughput", "prefill", "stddev")),
            ("Decode (tokens/s)", lambda r: _get(r, "throughput", "decode", "tokens_per_s"), lambda r: _get(r, "throughput", "decode", "stddev"))],
            f"Speed of {names}. Error bars show the standard deviation over repetitions."),
        "memory": lambda: _compare_bars(plt, out, runs, "compare_memory", [
            ("GPU memory, peak above idle (GiB)", lambda r: _div(_get(r, "memory", "gpu", "peak_above_baseline_total_mib"), 1024), None),
            ("Host anonymous, peak (GiB)", lambda r: _div(_get(r, "memory", "host", "anon_peak_mib"), 1024), None),
            ("Host file-backed, peak (GiB)", lambda r: _div(_get(r, "memory", "host", "file_peak_mib"), 1024), None)],
            f"Peak memory of {names}."),
        "energy": lambda: _compare_bars(plt, out, runs, "compare_energy", [
            ("Prefill, GPU energy (J/token)", lambda r: _get(r, "energy", "prefill_j_per_token"), None),
            ("Decode, GPU energy (J/token)", lambda r: _get(r, "energy", "decode_j_per_token"), None),
            ("Serving, GPU energy (J/token)", lambda r: _get(r, "energy", "serve_j_per_token"), None)],
            f"GPU energy per token of {names}, from the GPUs' energy counters."),
        "phases": lambda: compare_phases(plt, out, runs),
        "serving": lambda: compare_serving(plt, np, out, runs),
    }
    skipped = []
    with plt.rc_context(_rc(style)):
        for kind in kinds:
            if not makers[kind]():
                skipped.append(kind)
    out.finish(_method(runs[0]) if runs else "")
    return out.written, skipped


MATRIX_METRICS = [("prefill_tps", "Prefill speed"), ("decode_tps", "Decode speed"), ("output_tps", "Output speed"), ("ttft_ms", "Time to first token"), ("gpu_peak_mib", "GPU memory"),
                  ("host_anon_peak_mib", "Host memory"), ("load_s", "Load time"), ("decode_j_per_token", "Decode energy"), ("serve_j_per_token", "Serving energy")]


def make_matrix_figure(path, out_dir, style):
    """Per case and metric: B over A. Marker shape carries the verdict, so color is not the only channel."""
    plt = _matplotlib()
    with open(path) as f:
        table = json.load(f)
    cases = [c for c in table["cases"] if "metrics" in c]
    if not cases:
        raise SystemExit("this matrix has no finished cases")
    metrics = [(key, title) for key, title in MATRIX_METRICS if any(key in c["metrics"] for c in cases)]
    out = Output(out_dir, style)
    with plt.rc_context(_rc(style)):
        columns = len(metrics) if style.width >= 5 else min(2, len(metrics))
        grid_rows = math.ceil(len(metrics) / columns)
        fig, axes = plt.subplots(grid_rows, columns, sharey=True, squeeze=False, figsize=(style.width, style.height or (0.5 + 0.19 * len(cases)) * grid_rows + 0.3))
        axes = list(axes.flatten())
        for ax in axes[len(metrics):]:
            ax.set_visible(False)
        axes = axes[:len(metrics)]
        rows = []
        looks = {"better": ("^", GOOD, GOOD), "worse": ("v", CRITICAL, CRITICAL), "same": ("o", "white", MUTED)}
        for index, (ax, (key, title)) in enumerate(zip(axes, metrics)):
            ratios = []
            for i, case in enumerate(cases):
                value = case["metrics"].get(key)
                if not value or value.get("ratio") is None:
                    continue
                ratio = value["ratio"]
                ratios.append(ratio)
                # A difference resting on too few runs is a hint. A paper figure should not call it a result.
                verdict = "same" if value.get("weak") else value["verdict"]
                marker, face, edge = looks.get(verdict, looks["same"])
                ax.scatter([ratio], [i], marker=marker, s=16, facecolor=face, edgecolor=edge, linewidth=0.7, zorder=3)
                rows.append([case["label"], key, value["a"], value["b"], ratio, verdict])
            ax.axvline(1.0, color=AXIS, linewidth=0.7, zorder=1)
            ax.set_title(title)
            furthest = max([abs(r - 1.0) for r in ratios] + [0.01])
            step = next((c for c in (0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0) if c >= furthest), furthest)
            ax.set_xlim(max(0.0, 1.0 - step * 1.35), 1.0 + step * 1.35)
            ax.xaxis.set_major_locator(_fixed([max(0.0, 1.0 - step), 1.0, 1.0 + step]))
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.grid(axis="y")
            ax.spines["left"].set_visible(False)
            _letter(ax, index, style)
        axes[0].set_yticks(range(len(cases)), [c["label"] for c in cases])
        axes[0].set_ylim(len(cases) - 0.5, -0.5)
        handles = [plt.Line2D([], [], linestyle="", marker=m, markerfacecolor=f, markeredgecolor=e, markersize=4, markeredgewidth=0.7, label=label)
                   for label, (m, f, e) in (("better", looks["better"]), ("no real change", looks["same"]), ("worse", looks["worse"]))]
        _figure_legend(fig, handles, [handle.get_label() for handle in handles], "right", 3)
        fig.supxlabel(f"{table['b']} / {table['a']}", fontsize=style.font_size, color=INK2)
        counts = {"better": 0, "worse": 0}
        for row in rows:
            if row[5] in counts:
                counts[row[5]] += 1
        out.save(fig, "matrix", ["case", "metric", table["a"], table["b"], "ratio", "verdict"], rows,
                 f"{table['b']} against {table['a']} over {len(cases)} cases, in alternating fresh processes ({table.get('order', 'ABBA')}). A result counts as better or worse only when the "
                 f"arms do not overlap and the difference exceeds the noise floor; {counts['better']} results were better and {counts['worse']} worse.")
        plt.close(fig)
    out.finish("")
    return out.written, []
