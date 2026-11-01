"""Command line."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from . import __version__, adapters, capabilities, compare, driver, figures, html, nsys, report, runner, share


def _default_out(label):
    return f"runs/{label or 'run'}-{time.strftime('%Y%m%d-%H%M%S')}"


def _parse_marks(values):
    """regex=phase or regex=phase:group."""
    marks = []
    for value in values or []:
        regex, _, target = value.rpartition("=")
        if not regex:
            raise SystemExit(f"--mark needs regex=phase, got {value!r}")
        phase, _, group = target.partition(":")
        marks.append((regex, phase, group or None))
    return marks


def _parse_env(values):
    out = {}
    for value in values or []:
        key, sep, val = value.partition("=")
        if not sep:
            raise SystemExit(f"--env needs KEY=VALUE, got {value!r}")
        out[key] = val
    return out


def _common(parser):
    parser.add_argument("--out", help="run directory (default runs/<label>-<time>)")
    parser.add_argument("--label", default="")
    parser.add_argument("--hz", type=float, default=50.0, help="fast sampling rate")
    parser.add_argument("--no-pcie", action="store_true", help="do not sample PCIe traffic")
    parser.add_argument("--timeout", type=float, default=0.0, help="stop the target after this many seconds")
    parser.add_argument("--stall", type=float, default=None,
                        help="stop the target after this many seconds without output or resource change "
                             "(default 600 for engines that report progress, otherwise off)")
    parser.add_argument("--precheck", choices=["warn", "wait", "require", "skip"], default="warn",
                        help="what to do when the machine is not clean before the run")
    parser.add_argument("--adapter", default="auto")
    parser.add_argument("--mark", action="append", help="phase from output: 'regex=phase' or 'regex=phase:group'")
    parser.add_argument("--env", action="append", help="KEY=VALUE for the target")
    parser.add_argument("--tee", action="store_true", help="also show the target's output")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--nsys", action="store_true", help="record the GPU timeline (perturbs speed)")
    parser.add_argument("--nsys-arg", action="append", default=[], help="extra argument for nsys profile")
    parser.add_argument("--cpu-profile", action="store_true", help="sample host CPU stacks, no perf needed (perturbs speed)")
    parser.add_argument("--cpu-hz", type=int, default=200)
    parser.add_argument("--cold-start", action="store_true", help="drop the model from the page cache first, to measure a real disk load")
    parser.add_argument("--model-file", help="the model's path, when llmperfscope cannot tell from the command")


def _options(args):
    label = args.label
    return runner.Options(
        cold_start=args.cold_start,
        model=args.model_file,
        pcie=not args.no_pcie,
        out=args.out or _default_out(label),
        label=label,
        hz=args.hz,
        timeout_s=args.timeout,
        stall_s=args.stall,
        precheck=args.precheck,
        adapter=args.adapter,
        marks=_parse_marks(args.mark),
        env=_parse_env(args.env),
        tee=args.tee,
        quiet=args.quiet,
        nsys=args.nsys,
        nsys_args=args.nsys_arg,
        cpu_profile=args.cpu_profile,
        cpu_hz=args.cpu_hz,
    )


def _finish(result, options):
    print(report.render(result))
    page = html.write(options.out)
    print(f"[llmperfscope] charts: {page}\n[llmperfscope] all numbers: {Path(options.out) / 'run.json'}")
    return 0 if result["status"] == "ok" else 1


def _split_command(argv):
    """Everything after `--` is the target."""
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    return argv, []


def cmd_run(args, target):
    if not target:
        raise SystemExit("usage: llmperfscope run [options] -- command ...")
    options = _options(args)
    try:
        return _finish(runner.Run(target, options).execute(), options)
    except runner.Refused as e:
        print(f"[llmperfscope] refused: {e}", file=sys.stderr)
        return 3


def _rows_from(args):
    if args.prompts:
        filters = dict(item.split("=", 1) for item in args.filter or [])
        rows = driver.load_prompts(args.prompts, filters)
        if not rows:
            raise SystemExit("no prompts left after the filter")
        return rows[:args.limit] if args.limit else rows
    return driver.synthetic_prompts(args.synthetic_words, args.limit or 8)


def _serve_workload(args, preset, marks_load):
    """Waits until the server is ready, warms it up, drives each concurrency level, collects the server's own metrics."""

    def workload(run):
        if marks_load:
            run.emit("load", t=run.t_spawn)
        base = args.url.rstrip("/")
        is_alive = (lambda: Path(f"/proc/{run.pid}").exists())
        ready_s = driver.wait_ready(base + args.ready_path if args.ready_path else base + preset["ready_path"], is_alive,
                                    timeout_s=args.ready_timeout)
        run.log(f"server ready after {ready_s:.2f} s")
        extra = dict(preset.get("extra_body") or {})
        if args.extra_body:
            extra.update(json.loads(args.extra_body))
        client = driver.Client(base + preset["base_path"], model=args.model, max_tokens=args.max_tokens, extra_body=extra,
                               ignore_eos=not args.allow_eos, api_key=args.api_key)
        if args.pull:
            run.emit("download")
            driver.pull_ollama_model(base, args.model)
        if not client.model:
            try:
                client.discover_model()
            except (OSError, ValueError, KeyError) as e:
                run.log(f"could not read the model name from /models: {e!r}")
        rows = _rows_from(args)
        result = {"ready_s": round(ready_s, 3), "model": client.model, "prompts": len(rows)}

        run.emit("warmup")
        # A longer reply than strictly needed: a model that thinks first shows no text for its first tokens.
        warm = client.request({"prompt": "Warm up.", "max_tokens": 32} if "messages" not in rows[0]
                              else {"messages": [{"role": "user", "content": "Warm up."}], "max_tokens": 32})
        if not warm["ok"]:
            raise RuntimeError(f"warmup request failed: {warm.get('error')}")
        result["warmup_ttft_ms"] = round(warm["ttft_s"] * 1000, 2)

        metrics_path = args.metrics_path if args.metrics_path is not None else preset.get("metrics_path")
        metrics_before = driver.fetch_metrics(base + metrics_path) if metrics_path else None
        watcher = driver.GaugeWatcher(base + metrics_path) if metrics_before else None
        if watcher:
            watcher.start()
        throughput = {}
        with open(run.dir / "requests.jsonl", "w") as log:
            current = {}

            def record(item):
                row = {k: v for k, v in item.items() if k not in ("t_send", "t_done")}
                row["level"] = current.get("concurrency")
                row["itl_s"] = [round(gap, 6) for gap in item.get("itl_s", [])]
                log.write(json.dumps(row) + "\n")

            for concurrency in [int(x) for x in str(args.concurrency).split(",")]:
                name = f"serve_c{concurrency}"
                current["concurrency"] = concurrency
                run.emit(name, group="serve", concurrency=concurrency)
                results, wall = driver.run_load(client, rows, concurrency=concurrency, repeat=args.repeat, seed=args.seed,
                                                emit_each=record, level=f"c{concurrency}", unique=not args.reuse_prompts)
                summary = driver.summarize_load(results, wall)
                summary["concurrency"] = concurrency
                throughput[name] = summary
                run.log(f"{name}: {summary.get('output_tokens_per_s')} tokens/s, {summary.get('failed')} failed")
                # Tokens go on the phase, so energy per token comes out of the same bookkeeping as everywhere else.
                with run.lock:
                    for phase in run.phases:
                        if phase["name"] == name:
                            phase["tokens"] = summary.get("completion_tokens", 0)
        if watcher:
            watcher.stop()
            result["server_gauges_peak"] = watcher.peaks
        if metrics_path:
            after = driver.fetch_metrics(base + metrics_path)
            if metrics_before is not None and after is not None:
                result["server_metrics_delta"] = driver.metrics_delta(metrics_before, after)
        result["throughput"] = throughput
        return result

    return workload


def _add_energy_per_token(result):
    """Energy per output token for each concurrency level, from the phase it ran in."""
    for phase in result.get("phases", []):
        if phase["group"] != "serve" or not phase.get("tokens"):
            continue
        energy = (phase.get("metrics", {}).get("gpu") or {}).get("energy_j")
        entry = result.get("throughput", {}).get(phase["name"])
        if energy is not None and entry is not None:
            entry["j_per_output_token"] = round(energy / phase["tokens"], 4)


def cmd_serve(args, target):
    if not target:
        raise SystemExit("usage: llmperfscope serve [options] -- server command ...")
    options = _options(args)
    options.stop_timeout_s = args.stop_timeout
    preset_name = args.preset or adapters.preset_for(target)
    preset = adapters.SERVER_PRESETS[preset_name]
    options.stop_signal = preset.get("stop_signal", "TERM")
    if not args.url:
        args.url = adapters.guess_url(target, preset_name)
    for extra in preset.get("extra_args", []):
        if extra not in target:
            target = target + [extra]
    if preset_name == "ollama" and not args.model:
        raise SystemExit("Ollama starts with no model: name one with --model, for example --model qwen3:0.6b --pull")
    print(f"[llmperfscope] server preset {preset_name}, expecting it at {args.url}")
    knows_load_lines = adapters.select(options.adapter, target).name == "llama.cpp:llama-server"
    run = runner.Run(target, options, workload=_serve_workload(args, preset, marks_load=not knows_load_lines))
    try:
        result = run.execute()
    except runner.Refused as e:
        print(f"[llmperfscope] refused: {e}", file=sys.stderr)
        return 3
    _add_energy_per_token(result)
    with open(Path(options.out) / "run.json", "w") as f:
        json.dump(result, f, indent=1)
    return _finish(result, options)


def _docker_pid(name):
    done = subprocess.run(["docker", "inspect", "-f", "{{.State.Pid}}", name], capture_output=True, text=True, check=False)
    if done.returncode != 0 or not done.stdout.strip().isdigit():
        raise SystemExit(f"no running container {name!r}: {done.stderr.strip()}")
    return int(done.stdout.strip())


def cmd_attach(args, _target):
    """Profile something already running, such as a container. Load phases are gone, everything after is measured."""
    pid = args.pid or (_docker_pid(args.docker) if args.docker else None)
    if not pid:
        raise SystemExit("attach needs --pid or --docker")
    options = _options(args)
    options.stall_s = 0
    if args.url:
        preset = adapters.SERVER_PRESETS[args.preset or "openai"]
        workload = _serve_workload(args, preset, marks_load=False)
    else:
        def workload(run):
            run.emit("observe")
            time.sleep(args.duration)
            return {}
    run = runner.Run([f"/proc/{pid}/exe"], options, workload=workload, attach_pid=pid)
    try:
        result = run.execute()
    except runner.Refused as e:
        print(f"[llmperfscope] refused: {e}", file=sys.stderr)
        return 3
    _add_energy_per_token(result)
    with open(Path(options.out) / "run.json", "w") as f:
        json.dump(result, f, indent=1)
    return _finish(result, options)


def _serve_args(parser):
    parser.add_argument("--url", help="server address (default: from --port, a container's published port, or the engine's usual port)")
    parser.add_argument("--pull", action="store_true", help="Ollama: download --model first; this time is its own phase")
    parser.add_argument("--preset", choices=sorted(adapters.SERVER_PRESETS), help="default: guessed from the command")
    parser.add_argument("--model", help="model name for requests (default: first of /v1/models)")
    parser.add_argument("--prompts", help="JSONL with `prompt` or `messages` per row")
    parser.add_argument("--filter", action="append", help="keep rows where column=value")
    parser.add_argument("--limit", type=int, help="at most this many prompts")
    parser.add_argument("--synthetic-words", type=int, default=512, help="length of generated prompts when --prompts is not given")
    parser.add_argument("--concurrency", default="1", help="comma-separated levels, each runs the whole prompt list")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--allow-eos", action="store_true", help="let the model stop early (default: fixed output length)")
    parser.add_argument("--reuse-prompts", action="store_true",
                        help="send prompts unchanged, to measure a server's prefix cache (default: a short nonce in front defeats it)")
    parser.add_argument("--extra-body", help="JSON merged into every request")
    parser.add_argument("--api-key", help="bearer token (never stored in the report)")
    parser.add_argument("--ready-path", help="health path (default from the preset)")
    parser.add_argument("--ready-timeout", type=float, default=3600.0)
    parser.add_argument("--metrics-path", help="Prometheus path (default from the preset, empty to turn off)")
    parser.add_argument("--stop-timeout", type=float, default=120.0, help="seconds between the stop signal and SIGKILL")


def cmd_show(args, _target):
    print(report.render(compare.load_run(args.run)))
    return 0


def cmd_html(args, _target):
    print(html.write(args.run))
    return 0


def _figure_style(args):
    return figures.Style(width=figures.parse_width(args.width), height=args.height or 0.0, formats=tuple(args.format.split(",")), dpi=args.dpi,
                         font={"sans": "sans-serif", "serif": "serif"}.get(args.font, args.font), font_size=args.font_size, print_safe=args.print_safe,
                         title=args.title or "", letters=args.letters)


def cmd_figure(args, _target):
    style = _figure_style(args)
    if args.matrix:
        written, skipped = figures.make_matrix_figure(args.matrix, args.out or str(Path(args.matrix).parent / "figures"), style)
    elif args.compare:
        specs = []
        for item in args.compare:
            label, sep, path = item.partition("=")
            if not sep:
                raise SystemExit(f"--compare needs LABEL=RUN_DIR, got {item!r}")
            specs.append((label, path))
        kinds = figures.COMPARE_KINDS if args.kind == "all" else tuple(args.kind.split(","))
        unknown = set(kinds) - set(figures.COMPARE_KINDS)
        if unknown:
            raise SystemExit(f"unknown --kind {sorted(unknown)}; with --compare choose from {list(figures.COMPARE_KINDS)}")
        written, skipped = figures.make_compare_figures(specs, args.out or "figures-compare", style, kinds)
    else:
        if not args.run:
            raise SystemExit("usage: llmperfscope figure RUN_DIR, or --compare LABEL=RUN_DIR ..., or --matrix matrix.json")
        kinds = figures.RUN_KINDS if args.kind == "all" else tuple(args.kind.split(","))
        unknown = set(kinds) - set(figures.RUN_KINDS)
        if unknown:
            raise SystemExit(f"unknown --kind {sorted(unknown)}; choose from {list(figures.RUN_KINDS)}")
        panels = tuple(args.panels.split(",")) if args.panels else figures.DEFAULT_PANELS
        unknown = set(panels) - set(figures.ALL_PANELS)
        if unknown:
            raise SystemExit(f"unknown --panels {sorted(unknown)}; choose from {list(figures.ALL_PANELS)}")
        xlim = tuple(float(x) for x in args.xlim.split(",")) if args.xlim else None
        written, skipped = figures.make_run_figures(args.run, args.out or str(Path(args.run) / "figures"), style, kinds, panels, args.per_gpu, xlim)
    for path in written:
        print(path)
    if skipped:
        print(f"[llmperfscope] nothing to draw for: {', '.join(skipped)} (gpu-time needs a run with --nsys, hotspots --cpu-profile, serving `llmperfscope serve`)", file=sys.stderr)
    return 0


def cmd_anonymize(args, _target):
    print(share.anonymize(args.run, args.out))
    return 0


def cmd_diff(args, _target):
    runs_a = [compare.load_run(p) for p in args.a]
    runs_b = [compare.load_run(p) for p in args.b]
    print(compare.render_diff(compare.diff(runs_a, runs_b), args.a_name, args.b_name))
    return 0


def cmd_matrix(args, _target):
    refusal = runner.platform_refusal()
    if refusal:
        print(f"[llmperfscope] refused: {refusal}", file=sys.stderr)
        return 3

    def side(value):
        name, sep, path = value.partition("=")
        if not sep:
            raise SystemExit(f"expected NAME=BIN_DIR, got {value!r}")
        return name, path

    noise = None
    if args.noise:
        with open(args.noise) as f:
            noise = json.load(f)
    table = compare.run_matrix(compare.load_cases(args.cases), side(args.a), side(args.b), args.out, order=args.order,
                               timeout_s=args.timeout, a_env=_parse_env(args.a_env), b_env=_parse_env(args.b_env), only=args.only,
                               noise=noise, recheck=args.recheck)
    print(Path(args.out, "table.md").read_text())
    worse = [c["label"] for c in table["cases"] if c.get("verdict") in ("regression", "tradeoff")]
    if worse:
        print("worse in some metric: " + ", ".join(worse))
    return 1 if worse else 0


def cmd_noise(args, _target):
    floors = compare.noise_from_null(args.matrix)
    with open(args.out, "w") as f:
        json.dump(floors, f, indent=1)
    print(f"noise floors for {len(floors)} cases written to {args.out}")
    return 0


def cmd_nsys_report(args, _target):
    print(json.dumps(nsys.analyze(args.sqlite, comm_regex=args.comm_regex), indent=1))
    return 0


def cmd_check(_args, _target):
    print(capabilities.render(capabilities.collect()))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog="llmperfscope", description="Whole-lifecycle profiler for LLM inference engines.")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="profile any command: llmperfscope run -- command ...")
    _common(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("serve", help="start an OpenAI-compatible server, load it, stop it, profile the whole life")
    _common(p)
    _serve_args(p)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("attach", help="profile a running process or container")
    _common(p)
    _serve_args(p)
    p.add_argument("--pid", type=int)
    p.add_argument("--docker", help="container name")
    p.add_argument("--duration", type=float, default=60.0, help="seconds to observe when no --url is given")
    p.set_defaults(func=cmd_attach)

    p = sub.add_parser("show", help="print the report of a run")
    p.add_argument("run")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("html", help="write the charts of a run to report.html")
    p.add_argument("run")
    p.set_defaults(func=cmd_html)

    p = sub.add_parser("figure", help="figures for papers: vector PDF and SVG, high-resolution PNG, the data as CSV, suggested captions")
    p.add_argument("run", nargs="?", help="run directory")
    p.add_argument("--compare", nargs="+", metavar="LABEL=RUN_DIR", help="compare runs against each other")
    p.add_argument("--matrix", help="a matrix.json from `llmperfscope matrix`")
    p.add_argument("--kind", default="all", help=f"comma-separated. For a run: {', '.join(figures.RUN_KINDS)}. With --compare: {', '.join(figures.COMPARE_KINDS)}")
    p.add_argument("--out", help="output directory (default: <run>/figures)")
    p.add_argument("--format", default="pdf,png,svg", help="comma-separated formats")
    p.add_argument("--width", default="double", help="single (3.5 in), double (7.16 in), or inches")
    p.add_argument("--height", type=float, help="inches (default: fits the content)")
    p.add_argument("--dpi", type=int, default=300, help="for PNG")
    p.add_argument("--font", default="sans", help="sans, serif, or a font name")
    p.add_argument("--font-size", type=float, default=7.5, help="points")
    p.add_argument("--print-safe", action="store_true", help="add line styles, for grayscale print")
    p.add_argument("--letters", action="store_true", help="label panels (a), (b), ...")
    p.add_argument("--title", help="figure title (default: none, papers use the caption)")
    p.add_argument("--panels", help=f"lifecycle panels, comma-separated: {', '.join(figures.ALL_PANELS)}")
    p.add_argument("--per-gpu", action="store_true", help="one line per GPU in lifecycle panels")
    p.add_argument("--xlim", help="lifecycle time range in seconds: START,END")
    p.set_defaults(func=cmd_figure)

    p = sub.add_parser("anonymize", help="a copy of a run without host and user names, paths, serial numbers or prompts, to share or publish")
    p.add_argument("run")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_anonymize)

    p = sub.add_parser("diff", help="compare finished runs")
    p.add_argument("--a", nargs="+", required=True)
    p.add_argument("--b", nargs="+", required=True)
    p.add_argument("--a-name", default="A")
    p.add_argument("--b-name", default="B")
    p.set_defaults(func=cmd_diff)

    p = sub.add_parser("matrix", help="A/B across many cases, alternating order, machine re-checked before every arm")
    p.add_argument("--cases", required=True)
    p.add_argument("--a", required=True, help="NAME=BIN_DIR")
    p.add_argument("--b", required=True, help="NAME=BIN_DIR")
    p.add_argument("--a-env", action="append")
    p.add_argument("--b-env", action="append")
    p.add_argument("--order", default=compare.DEFAULT_ORDER, help="arms per case, A and B in turn. Fewer than 3 per side only gives hints")
    p.add_argument("--recheck", type=int, default=1, help="when a case looks worse, run this many more rounds of arms before calling it")
    p.add_argument("--only", help="regex on case labels")
    p.add_argument("--noise", help="floors from `llmperfscope noise`")
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_matrix)

    p = sub.add_parser("noise", help="noise floors from a null matrix (same binary on both sides)")
    p.add_argument("matrix")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_noise)

    p = sub.add_parser("nsys-report", help="analyze an exported nsys sqlite file")
    p.add_argument("sqlite")
    p.add_argument("--comm-regex", default=nsys.DEFAULT_COMM_REGEX)
    p.set_defaults(func=cmd_nsys_report)

    p = sub.add_parser("check", help="what can be measured on this machine")
    p.set_defaults(func=cmd_check)
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    argv, target = _split_command(argv)
    args = build_parser().parse_args(argv)
    return args.func(args, target)


if __name__ == "__main__":
    sys.exit(main())
