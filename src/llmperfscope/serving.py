"""Driving a server: start it, wait until it answers, send prompts at each concurrency level, read its metrics, stop it. Shared by `serve`,
`attach` and server cases of `matrix`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import adapters, driver, procfs, runner


def add_arguments(parser):
    parser.add_argument("--url", help="server address (default: from --port, a container's published port, or the engine's usual port)")
    parser.add_argument("--pull", action="store_true", help="Ollama: download --model first; this time is its own phase")
    parser.add_argument("--preset", choices=sorted(adapters.SERVER_PRESETS), help="default: guessed from the command")
    parser.add_argument("--model", help="model name for requests (default: first of /v1/models)")
    parser.add_argument("--prompts", help="JSONL with `prompt` or `messages` per row, or `mix` for the public llm-speed-prompt-mix set (standard tier unless --filter says)")
    parser.add_argument("--filter", action="append", help="keep rows where column=value")
    parser.add_argument("--limit", type=int, help="at most this many prompts")
    parser.add_argument("--max-prompt-chars", type=int, help="leave out longer prompts, for a server with a small context")
    parser.add_argument("--synthetic-words", type=int, default=512, help="length of generated prompts when --prompts is not given")
    parser.add_argument("--concurrency", default="1", help="comma-separated levels, each runs the whole prompt list")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--allow-eos", action="store_true", help="let the model stop early (default: fixed output length)")
    parser.add_argument("--reuse-prompts", action="store_true",
                        help="send prompts unchanged, to measure a server's prefix cache (default: a short nonce in front defeats it)")
    parser.add_argument("--turns", type=int, default=0,
                        help="conversations of this many turns: each request repeats the questions and replies before it, to measure the prompt cache or context checkpoints")
    parser.add_argument("--extra-body", help="JSON merged into every request")
    parser.add_argument("--api-key", help="bearer token (never stored in the report)")
    parser.add_argument("--ready-path", help="health path (default from the preset)")
    parser.add_argument("--ready-timeout", type=float, default=3600.0)
    parser.add_argument("--metrics-path", help="Prometheus path (default from the preset, empty to turn off)")
    parser.add_argument("--stop-timeout", type=float, default=120.0, help="seconds between the stop signal and SIGKILL")


def defaults(**overrides):
    """The same settings as the command line, for code that starts a server without one. `overrides` use the options' long names."""
    parser = argparse.ArgumentParser(add_help=False)
    add_arguments(parser)
    args = parser.parse_args([])
    unknown = set(overrides) - set(vars(args))
    if unknown:
        raise ValueError(f"unknown server settings: {sorted(unknown)}")
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def _prompt_chars(row):
    if "prompt" in row:
        return len(row["prompt"])
    return sum(len(str(message.get("content", ""))) for message in row["messages"])


def rows_from(args):
    if not args.prompts:
        return driver.synthetic_prompts(args.synthetic_words, args.limit or 8)
    filters = dict(item.split("=", 1) for item in args.filter or [])
    path = args.prompts
    if path == "mix":
        path = driver.prompt_mix_path()
        filters.setdefault("tier", "standard")
    rows = driver.load_prompts(path, filters)
    if args.max_prompt_chars:
        rows = [row for row in rows if _prompt_chars(row) <= args.max_prompt_chars]
    if not rows:
        raise SystemExit("no prompts left after the filter")
    return rows[:args.limit] if args.limit else rows


def _overall_speed(throughput):
    """Prefill and decode as the server timed them, at the lowest concurrency where nothing queues. They read like llama-bench's numbers
    and `diff` and `matrix` judge them the same way, but with real prompts."""
    levels = sorted((t for t in throughput.values() if "concurrency" in t), key=lambda t: t["concurrency"])
    if not levels:
        return {}
    out = {}
    for key, name in (("server_prefill_tokens_per_s", "prefill"), ("server_decode_tokens_per_s", "decode")):
        if levels[0].get(key):
            out[name] = {"tokens_per_s": levels[0][key], "source": "server", "real_prompts": True}
    return out


def workload(args, preset, marks_load):
    """Waits until the server is ready, warms it up, drives each concurrency level, collects the server's own metrics. The prompts are read first: downloading them inside
    the run would count as loading the model, and a mistake in them should not have to wait for a model to load."""
    rows = rows_from(args)

    def run_workload(run):
        if marks_load:
            run.emit("load", t=run.t_spawn)
        base = args.url.rstrip("/")
        is_alive = (lambda: procfs.alive(run.pid))
        ready_s = driver.wait_ready(base + args.ready_path if args.ready_path else base + preset["ready_path"], is_alive,
                                    timeout_s=args.ready_timeout)
        run.log(f"server ready after {ready_s:.2f} s")
        extra = dict(preset.get("extra_body") or {})
        if args.reuse_prompts or args.turns > 1:
            # the point of these options is to reach the server's cache
            extra.pop("cache_prompt", None)
        if args.extra_body:
            extra.update(json.loads(args.extra_body))
        client = driver.Client(base + preset["base_path"], model=args.model, max_tokens=args.max_tokens, extra_body=extra,
                               ignore_eos=not args.allow_eos, api_key=args.api_key)
        if args.pull:
            run.emit("download")
            driver.pull_ollama_model(base, args.model)
        model_name = client.model
        if preset.get("uses_default_model") and not client.model:
            model_name = adapters.model_flag_value(run.cmd)
        elif not client.model:
            try:
                model_name = client.discover_model()
                if getattr(client, "models_listed", 1) > 1:
                    run.notes.append(f"the server lists {client.models_listed} models and these requests named the first, {model_name}. --model names another")
            except (OSError, ValueError, KeyError) as e:
                run.log(f"could not read the model name from /models: {e!r}")
        result = {"ready_s": round(ready_s, 3), "model": model_name, "prompts": len(rows)}

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
                if args.turns > 1:
                    results, wall = driver.run_conversations(client, rows, args.turns, emit_each=record)
                else:
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
        throughput.update(_overall_speed(throughput))
        result["throughput"] = throughput
        return result

    return run_workload


def add_energy_per_token(result):
    """Energy per output token for each concurrency level, from the phase it ran in."""
    for phase in result.get("phases", []):
        if phase["group"] != "serve" or not phase.get("tokens"):
            continue
        energy = (phase.get("metrics", {}).get("gpu") or {}).get("energy_j")
        entry = result.get("throughput", {}).get(phase["name"])
        if energy is not None and entry is not None:
            entry["j_per_output_token"] = round(energy / phase["tokens"], 4)


def serve(target, options, args):
    """One whole life of a server. Raises runner.Refused if it cannot be measured."""
    options.stop_timeout_s = args.stop_timeout
    preset_name = args.preset or adapters.preset_for(target)
    preset = adapters.SERVER_PRESETS[preset_name]
    options.stop_signal = preset.get("stop_signal", "TERM")
    if not args.url:
        args.url = adapters.guess_url(target, preset_name)
    extras = list(preset.get("extra_args", []))
    if args.reuse_prompts or args.turns > 1:
        extras += preset.get("cache_report_args", [])
    for extra in extras:
        if extra not in target:
            target = target + [extra]
    if preset_name == "ollama" and not args.model:
        raise SystemExit("Ollama starts with no model: name one with --model, for example --model qwen3:0.6b --pull")
    if not options.quiet:
        print(f"[llmperfscope] server preset {preset_name}, expecting it at {args.url}")
    knows_load_lines = adapters.select(options.adapter, target).name == "llama.cpp:llama-server"
    run = runner.Run(target, options, workload=workload(args, preset, marks_load=not knows_load_lines))
    result = run.execute()
    add_energy_per_token(result)
    with open(Path(options.out) / "run.json", "w") as f:
        json.dump(result, f, indent=1)
    return result
