"""What llmperfscope knows about specific engines. The core works without any of this; an adapter only adds finer phases,
exact throughput and server defaults. A new engine needs one small class here, or just `--mark 'regex=phase'`.

Canonical phase groups, so reports line up across engines:
    startup, load, setup, warmup, prefill, decode, serve, teardown
"""

from __future__ import annotations

import json
import os
import re
import socket

ANSI =re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
RANDOM_TOKENS_NOTE = ("llama-bench feeds random tokens. Caches, expert routing and table lookups behave differently with real text, so a gain here can be "
                      "much smaller there: confirm it with `llmperfscope serve --prompts mix`, or a server case in `llmperfscope matrix`.")
WRAPPERS = {"env", "taskset", "numactl", "timeout", "nice", "ionice", "chrt", "flock", "stdbuf", "setsid", "nohup", "sudo"}


class Adapter:
    name = "generic"
    # Defaults for `llmperfscope serve`: paths on the server, signal to stop it, extra request fields.
    server = None

    def __init__(self, cmd, marks=()):
        self.cmd = cmd
        self.marks = [(re.compile(regex), phase, group) for regex, phase, group in marks]
        self.report = {}

    def prepare(self, cmd, env):
        """Last chance to change the command and environment. Everything changed is stored in the report."""
        return cmd, env

    def on_line(self, stream, t, text, emit):
        """A line of output arrived at CLOCK_MONOTONIC `t`. emit(phase, group=None, t=None, tokens=None, **meta)."""
        for regex, phase, group in self.marks:
            if regex.search(text):
                emit(phase, group=group, t=t)

    def finalize(self, phases, stdout_path, stderr_path):
        """After exit: parse results, fill in tokens per phase. Returns a dict merged into the report."""
        return {}

    def model_path(self):
        return _find(self.cmd, "--model", "--model-path", "--model-id")

    def engine_config(self):
        """The settings on the command line that change how the structure of a model plays out, under names shared by all engines."""
        out = _settings(self.cmd, preset_for(self.cmd))
        if out.get("tensor_parallel") not in (None, "1"):
            out.setdefault("split_mode", "tensor")
        return out

    def target_index(self):
        """Index of the real engine in a command like `taskset -c 1 numactl ... engine ...`, for tools inserted before it."""
        return _skip_wrappers(self.cmd)

    def has_progress_lines(self):
        """True when silence on stdout/stderr really means nothing is happening (the stall watchdog relies on it)."""
        return False


def _skip_wrappers(cmd):
    """Best effort. Options of wrappers are skipped by looking for the first word that is an executable or a known name."""
    i = 0
    while i < len(cmd):
        base = os.path.basename(cmd[i])
        if base in WRAPPERS or cmd[i].startswith("-") or "=" in cmd[i] or _looks_like_wrapper_value(cmd, i):
            i += 1
            continue
        return i
    return 0


def _looks_like_wrapper_value(cmd, i):
    """The value of a wrapper option such as `taskset -c 0-3` or `timeout 600`."""
    if i == 0:
        return False
    previous = cmd[i - 1]
    if previous in ("-c", "-n", "-p", "-k") and os.path.basename(cmd[i]) not in ("sh", "bash"):
        return True
    return re.fullmatch(r"[0-9][0-9smhd.,-]*", cmd[i]) is not None


def _find(cmd, *flags):
    for i, token in enumerate(cmd):
        if token in flags and i + 1 < len(cmd):
            return cmd[i + 1]
        for flag in flags:
            if flag.startswith("--") and token.startswith(flag + "="):
                return token.split("=", 1)[1]
    return None


def _settings(cmd, engine):
    """The flags of the engine itself, not of a wrapper in front of it like `taskset -c`. A flag given twice counts as the last one."""
    words = cmd[_skip_wrappers(cmd):]
    out = {}
    for name, flags in ENGINE_FLAGS.get(engine, {}).items():
        value = _find(list(reversed_pairs(words)), *flags)
        if value is not None:
            out[name] = value
    for name, flags in ENGINE_SWITCHES.get(engine, {}).items():
        if any(flag in words for flag in flags):
            out[name] = True
    return out


def reversed_pairs(words):
    """The words with flag and value pairs in reverse order, so that a search from the start finds the last time a flag was given."""
    pairs = []
    i = 0
    while i < len(words):
        if words[i].startswith("-") and i + 1 < len(words) and not words[i + 1].startswith("-"):
            pairs.append(words[i:i + 2])
            i += 2
        else:
            pairs.append(words[i:i + 1])
            i += 1
    return [word for pair in reversed(pairs) for word in pair]


class MlxGenerate(Adapter):
    """`mlx_lm.generate` says how fast it was and how much memory it took, after the text."""

    name = "mlx_lm.generate"

    def finalize(self, phases, stdout_path, stderr_path):
        try:
            with open(stdout_path, errors="replace") as f:
                text = f.read()
        except OSError:
            return {}
        throughput = {}
        for name, label in (("prefill", "Prompt"), ("decode", "Generation")):
            found = re.search(rf"^{label}: (\d+) tokens, ([\d.]+) tokens-per-sec", text, re.M)
            if found:
                throughput[name] = {"tokens_per_s": float(found.group(2)), "tokens": int(found.group(1)), "source": "engine"}
        out = {"throughput": throughput} if throughput else {}
        peak = re.search(r"^Peak memory: ([\d.]+) GB", text, re.M)
        if peak:
            out["engine_peak_memory_gb"] = float(peak.group(1))
        return out


class LlamaBench(Adapter):
    """llama-bench: `--progress` gives every run's start on stderr, `-o json` gives exact per-rep times on stdout."""

    name = "llama.cpp:llama-bench"

    def __init__(self, cmd, marks=()):
        super().__init__(cmd, marks)
        self.first = True
        self.n_runs = 0

    def prepare(self, cmd, env):
        cmd = list(cmd)
        if "--progress" not in cmd:
            cmd.append("--progress")
        for flag in ("-o", "--output"):
            if flag in cmd:
                cmd[cmd.index(flag) + 1] = "json"
                break
        else:
            cmd += ["-o", "json"]
        return cmd, env

    def model_path(self):
        return _find(self.cmd, "-m", "--model")

    def has_progress_lines(self):
        return True

    def on_line(self, stream, t, text, emit):
        super().on_line(stream, t, text, emit)
        if stream == "stdout":
            if text.lstrip().startswith('"samples_ts"'):
                emit("post_test", group="teardown", t=t)
            return
        m = re.search(r"benchmark (\d+)/(\d+): (.*)", text)
        if not m:
            return
        test, what = int(m.group(1)), m.group(3).strip()
        if what == "starting":
            emit("load" if self.first else "setup", t=t, test=test)
            self.first = False
        elif what == "warmup prompt run":
            emit("warmup_prefill", group="warmup", t=t, test=test)
        elif what == "warmup generation run":
            emit("warmup_decode", group="warmup", t=t, test=test)
        elif what.startswith("depth run"):
            emit("depth", group="setup", t=t, test=test)
        elif what.startswith("prompt run"):
            emit("prefill", t=t, test=test, rep=int(re.search(r"(\d+)/", what).group(1)))
        elif what.startswith("generation run"):
            emit("decode", t=t, test=test, rep=int(re.search(r"(\d+)/", what).group(1)))

    def finalize(self, phases, stdout_path, stderr_path):
        # Tools wrapping the engine can print around its output, so look for the array instead of assuming the whole file is one.
        try:
            with open(stdout_path, encoding="utf-8", errors="replace") as f:
                text = f.read()
            start = re.search(r"^\[\s*$", text, re.M)
            tests, _ = json.JSONDecoder().raw_decode(text, start.start())
        except (OSError, ValueError, AttributeError):
            return {"notes": ["llama-bench JSON output was not readable (the run may have been cut short)"]}
        summary = []
        prefill, decode = [], []
        for number, test in enumerate(tests, start=1):
            n_prompt, n_gen = test.get("n_prompt", 0), test.get("n_gen", 0)
            kind = "decode" if n_gen and not n_prompt else "prefill" if n_prompt and not n_gen else "mixed"
            entry = {
                "test": number,
                "kind": kind,
                "n_prompt": n_prompt,
                "n_gen": n_gen,
                "n_depth": test.get("n_depth", 0),
                "tokens_per_s": test.get("avg_ts"),
                "stddev": test.get("stddev_ts"),
                "samples_tps": test.get("samples_ts"),
                "samples_ns": test.get("samples_ns"),
            }
            summary.append(entry)
            if kind == "decode":
                decode.append(entry)
            elif kind == "prefill":
                prefill.append(entry)
            for phase in phases:
                if phase.get("meta", {}).get("test") == number:
                    if phase["group"] == "prefill":
                        phase["tokens"] = n_prompt
                    elif phase["group"] == "decode":
                        phase["tokens"] = n_gen
                    elif phase["name"] == "warmup_decode":
                        phase["tokens"] = 0
        info = tests[0] if tests else {}
        out = {
            "tests": summary,
            "engine": {k: info.get(k) for k in ("build_commit", "build_number", "cpu_info", "gpu_info", "backends",
                                                "model_type", "model_size", "model_n_params", "n_batch", "n_ubatch",
                                                "n_threads", "n_gpu_layers", "split_mode") if k in info},
        }
        throughput = {}
        if prefill:
            throughput["prefill"] = _merge_tests(prefill)
        if decode:
            throughput["decode"] = _merge_tests(decode)
        out["throughput"] = throughput
        out["consistency"] = _phase_vs_samples(phases, summary)
        out["notes"] = [RANDOM_TOKENS_NOTE]
        return out


def _merge_tests(tests):
    if len(tests) == 1:
        test = tests[0]
        return {"tokens_per_s": test["tokens_per_s"], "stddev": test["stddev"], "samples": test["samples_tps"],
                "tokens_per_run": test["n_prompt"] or test["n_gen"]}
    return {"tests": [{"tokens_per_run": t["n_prompt"] or t["n_gen"], "tokens_per_s": t["tokens_per_s"],
                       "stddev": t["stddev"]} for t in tests],
            "tokens_per_s": tests[0]["tokens_per_s"]}


def _phase_vs_samples(phases, tests):
    """Self-check: the time llmperfscope saw for each run against what the engine timed itself. Far from 1 means the phase
    boundaries cannot be trusted."""
    wall = sampled = 0.0
    for test in tests:
        if test["kind"] == "mixed" or not test["samples_ns"]:
            continue
        group = "decode" if test["kind"] == "decode" else "prefill"
        runs = [p for p in phases if p["group"] == group and p.get("meta", {}).get("test") == test["test"]]
        wall += sum(p["t_end"] - p["t"] for p in runs if "t_end" in p)
        sampled += sum(test["samples_ns"][:len(runs)])
    return {"phase_wall_over_engine_time": round(wall / sampled, 4)} if sampled and wall else {}


# Sub-phases of model loading and serving, from llama.cpp's own log lines. The log lines are not an API: when one of
# them changes only the sub-phase is lost, because readiness comes from /health.
_SERVER_LINES = [
    (re.compile(r"loading model '"), "load", None),
    (re.compile(r"llama_context: constructing llama_context|constructing llama_context"), "setup", None),
    (re.compile(r"initializing slots"), "setup_slots", "setup"),
    (re.compile(r"cleaning up before exit"), "teardown", None),
]


_GRAPHS_REUSED = re.compile(r"graphs reused\s*=\s*(\d+)")
_GRAPH_REALLOCS = re.compile(r"graph reallocs\s*=\s*(\d+)\s*/\s*(\d+)")


class _LlamaCommon(Adapter):
    def model_path(self):
        return _find(self.cmd, "-m", "--model")

    def engine_config(self):
        return _settings(self.cmd, "llama-server")

    def _count_events(self, message):
        """What the server says about its graphs and context checkpoints: work per request that no timer of a phase shows."""
        counters = self.report.setdefault("engine_counters", {})
        match = _GRAPHS_REUSED.search(message)
        if match:
            counters["graphs_reused"] = int(match.group(1))
        match = _GRAPH_REALLOCS.search(message)
        if match:
            counters["graph_reallocs_same_size"] = int(match.group(1))
            counters["graph_reallocs_changed"] = int(match.group(2))
        if "created context checkpoint" in message:
            counters["checkpoints_created"] = counters.get("checkpoints_created", 0) + 1
        elif "restored context checkpoint" in message:
            counters["checkpoints_restored"] = counters.get("checkpoints_restored", 0) + 1
        if not counters:
            del self.report["engine_counters"]

    def prepare(self, cmd, env):
        # JSONL logs are the only form of the memory breakdown visible without trace verbosity, which would slow decode.
        env = dict(env)
        env.setdefault("LLAMA_ARG_LOG_JSONL", "1")
        return cmd, env

    def on_line(self, stream, t, text, emit):
        super().on_line(stream, t, text, emit)
        message = text
        if text.startswith("{"):
            try:
                record = json.loads(text)
            except ValueError:
                record = None
            if isinstance(record, dict):
                if record.get("type") == "fit_memory_breakdown":
                    self.report["memory_breakdown"] = _breakdown_from_json(record)
                    return
                message = str(record.get("msg", ""))
        message = ANSI.sub("", message)
        table_row = parse_breakdown_line(message)
        if table_row:
            self.report.setdefault("memory_breakdown", []).append(table_row)
            return
        self._count_events(message)
        for regex, phase, group in _SERVER_LINES:
            if regex.search(message):
                emit(phase, group=group, t=t)

    def finalize(self, phases, stdout_path, stderr_path):
        return dict(self.report)


class LlamaServer(_LlamaCommon):
    name = "llama.cpp:llama-server"
    server = {"ready_path": "/health", "metrics_path": "/metrics", "base_path": "/v1", "stop_signal": "TERM",
              "extra_body": {"cache_prompt": False}}

    def prepare(self, cmd, env):
        cmd, env = super().prepare(cmd, env)
        if "--metrics" not in cmd:
            cmd = list(cmd) + ["--metrics"]
        return cmd, env


class LlamaPerplexity(_LlamaCommon):
    name = "llama.cpp:llama-perplexity"


class LlamaGeneric(_LlamaCommon):
    name = "llama.cpp"


def _breakdown_from_json(record):
    data = record.get("data", record)
    rows = data.get("rows", []) if isinstance(data, dict) else []
    return [{k: row.get(k) for k in ("kind", "name", "total", "free", "self", "model", "context", "compute", "unaccounted")
             if k in row} for row in rows]


_BREAKDOWN_GPU = re.compile(
    r"\|\s*-\s*(?P<name>.+?)\s*\|\s*(?P<total>\d+)\s*=\s*(?P<free>\d+)\s*\+\s*\(\s*(?P<self>\d+)\s*=\s*(?P<model>\d+)\s*\+\s*"
    r"(?P<context>\d+)\s*\+\s*(?P<compute>\d+)\s*\)\s*\+\s*(?P<unaccounted>\d+)\s*\|")
_BREAKDOWN_OTHER = re.compile(
    r"\|\s*-\s*(?P<name>.+?)\s*\|\s*(?P<self>\d+)\s*=\s*(?P<model>\d+)\s*\+\s*(?P<context>\d+)\s*\+\s*(?P<compute>\d+)\s*\|")


def parse_breakdown_line(text):
    """One row of common_memory_breakdown_print's table."""
    m = _BREAKDOWN_GPU.search(text)
    if m:
        row = {k: (int(v) if v.isdigit() else v) for k, v in m.groupdict().items()}
        row["kind"] = "device"
        return row
    m = _BREAKDOWN_OTHER.search(text)
    if m:
        row = {k: (int(v) if v.isdigit() else v) for k, v in m.groupdict().items()}
        row["kind"] = "host" if row["name"].lower().startswith("host") else "buffer_type"
        return row
    return None


# Engines recognised from outside only: they get readiness, requests and server metrics, phases from markers.
# Flags per engine, with what they mean. Only those that change where the time and memory of a model's parts go.
ENGINE_FLAGS = {
    "llama-server": {
        "ctx_size": ("-c", "--ctx-size"), "batch_size": ("-b", "--batch-size"), "ubatch_size": ("-ub", "--ubatch-size"),
        "gpu_layers": ("-ngl", "--gpu-layers", "--n-gpu-layers"), "split_mode": ("-sm", "--split-mode"), "tensor_split": ("-ts", "--tensor-split"),
        "flash_attn": ("-fa", "--flash-attn"), "cache_type_k": ("-ctk", "--cache-type-k"), "cache_type_v": ("-ctv", "--cache-type-v"),
        "parallel": ("-np", "--parallel"), "threads": ("-t", "--threads"), "override_tensor": ("-ot", "--override-tensor"),
        "n_cpu_moe": ("-ncmoe", "--n-cpu-moe"), "draft_model": ("-md", "--spec-draft-model", "--model-draft"), "spec_type": ("--spec-type",),
        "spec_draft_n_max": ("--spec-draft-n-max",), "spec_draft_p_min": ("--spec-draft-p-min",),
        "ctx_checkpoints": ("-ctxcp", "--ctx-checkpoints", "--swa-checkpoints"), "checkpoint_min_step": ("-cms", "--checkpoint-min-step"),
        "cache_ram": ("-cram", "--cache-ram"), "cache_reuse": ("--cache-reuse",),
    },
    "vllm": {
        "ctx_size": ("--max-model-len",), "tensor_parallel": ("--tensor-parallel-size", "-tp"), "parallel": ("--max-num-seqs",),
        "dtype": ("--dtype",), "cache_type_k": ("--kv-cache-dtype",), "memory_share": ("--gpu-memory-utilization",),
        "speculative": ("--speculative-config",),
    },
    "sglang": {
        "ctx_size": ("--context-length",), "tensor_parallel": ("--tp-size", "--tp"), "dtype": ("--dtype",), "memory_share": ("--mem-fraction-static",),
        "spec_type": ("--speculative-algorithm",), "draft_model": ("--speculative-draft-model-path",),
    },
    "tgi": {
        "ctx_size": ("--max-total-tokens",), "tensor_parallel": ("--num-shard",), "dtype": ("--dtype",), "quantize": ("--quantize",),
    },
}
ENGINE_SWITCHES = {
    "llama-server": {"cpu_moe": ("-cmoe", "--cpu-moe"), "no_mmap": ("--no-mmap",), "mlock": ("--mlock",), "no_cache_prompt": ("--no-cache-prompt",)},
    "vllm": {"no_prefix_cache": ("--no-enable-prefix-caching",)},
    "sglang": {"no_prefix_cache": ("--disable-radix-cache",)},
}


SERVER_PRESETS = {
    "openai": {"ready_path": "/health", "metrics_path": None, "base_path": "/v1", "stop_signal": "TERM", "extra_body": {},
               "port": 8000},
    # llama.cpp's default since #30159 (2026-10-09), 8080 before. A command without --port gets this one, see give_port.
    "llama-server": {**LlamaServer.server, "port": 9931},
    "vllm": {"ready_path": "/health", "metrics_path": "/metrics", "base_path": "/v1", "stop_signal": "TERM", "extra_body": {},
             "port": 8000, "cache_report_args": ["--enable-prompt-tokens-details"]},
    "sglang": {"ready_path": "/health", "metrics_path": "/metrics", "base_path": "/v1", "stop_signal": "TERM", "extra_body": {},
               "port": 30000, "extra_args": ["--enable-metrics"], "cache_report_args": ["--enable-cache-report"]},
    "tgi": {"ready_path": "/health", "metrics_path": "/metrics", "base_path": "/v1", "stop_signal": "TERM", "extra_body": {},
            "port": 3000},
    "ollama": {"ready_path": "/api/version", "metrics_path": None, "base_path": "/v1", "stop_signal": "TERM", "extra_body": {},
               "port": 11434},
    # It lists every model in the Hugging Face cache and loads whichever a request names, so naming one could change what is measured.
    "mlx": {"ready_path": "/health", "metrics_path": None, "base_path": "/v1", "stop_signal": "TERM", "extra_body": {},
            "port": 8080, "uses_default_model": True},
}


def select(name, cmd, marks=()):
    """`name` is "auto" or an adapter name."""
    words = [os.path.basename(token) for token in cmd]
    if name in ("auto", "mlx_lm.generate") and "mlx_lm.generate" in words:
        return MlxGenerate(cmd, marks)
    if name in ("auto", "llama-bench") and "llama-bench" in words:
        return LlamaBench(cmd, marks)
    if name in ("auto", "llama-server") and "llama-server" in words:
        return LlamaServer(cmd, marks)
    if name in ("auto", "llama-perplexity") and "llama-perplexity" in words:
        return LlamaPerplexity(cmd, marks)
    if name in ("auto", "llama.cpp") and any(w.startswith("llama-") for w in words):
        return LlamaGeneric(cmd, marks)
    return Adapter(cmd, marks)


def preset_for(cmd):
    """Server defaults guessed from the command line, container images included."""
    words = [os.path.basename(token).lower() for token in cmd]
    joined = " ".join(token.lower() for token in cmd)
    if "llama-server" in words or "llama.cpp" in joined:
        return "llama-server"
    if "vllm" in joined:
        return "vllm"
    if "sglang" in joined:
        return "sglang"
    if "text-generation-launcher" in words or "text-generation-inference" in joined:
        return "tgi"
    if "ollama" in joined:
        return "ollama"
    if "mlx_lm" in joined or "mlx-lm" in joined:
        return "mlx"
    return "openai"


def model_flag_value(cmd):
    """What a command line names as its model."""
    return _find(cmd, "--model")


def docker_index(cmd):
    """Index of `docker` in a `docker run ...` command, or None."""
    i = _skip_wrappers(cmd)
    if i + 1 < len(cmd) and os.path.basename(cmd[i]) in ("docker", "podman") and cmd[i + 1] == "run":
        return i
    return None


def port_in_use(host, port):
    """Whether something already answers there."""
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def give_port(cmd, preset_name, environ=None):
    """A llama-server command that says where to listen. Its default moved from 8080 to 9931 in llama.cpp #30159, so without --port the
    port depends on the build. The default is kept when it is free, so the server is where its own docs say."""
    environ = os.environ if environ is None else environ
    has_port = any(token == "--port" or token.startswith("--port=") for token in cmd)
    if preset_name != "llama-server" or docker_index(cmd) is not None or has_port or environ.get("LLAMA_ARG_PORT"):
        return cmd
    port = SERVER_PRESETS["llama-server"]["port"]
    if port_in_use("127.0.0.1", port):
        port = free_port()
    return list(cmd) + ["--port", str(port)]


def guess_url(cmd, preset_name, environ=None):
    """Where a server will listen, so nobody has to say it twice. A container's published port wins over the flags inside."""
    environ = os.environ if environ is None else environ
    port = None
    host = "127.0.0.1"
    index = docker_index(cmd)
    if index is not None:
        for i, token in enumerate(cmd[index:], start=index):
            value = None
            if token in ("-p", "--publish") and i + 1 < len(cmd):
                value = cmd[i + 1]
            elif token.startswith("--publish="):
                value = token.split("=", 1)[1]
            if value:
                parts = value.split("/")[0].split(":")
                if len(parts) >= 2 and parts[-2].isdigit():
                    port = int(parts[-2])
                    break
    if port is None:
        for i, token in enumerate(cmd):
            if token == "--port" and i + 1 < len(cmd) and cmd[i + 1].isdigit():
                port = int(cmd[i + 1])
            elif token.startswith("--port=") and token.split("=", 1)[1].isdigit():
                port = int(token.split("=", 1)[1])
    if port is None and preset_name == "ollama" and ":" in environ.get("OLLAMA_HOST", ""):
        tail = environ["OLLAMA_HOST"].rsplit(":", 1)[1]
        if tail.isdigit():
            port = int(tail)
    if port is None and preset_name == "llama-server" and str(environ.get("LLAMA_ARG_PORT", "")).isdigit():
        port = int(environ["LLAMA_ARG_PORT"])
    if port is None:
        port = SERVER_PRESETS[preset_name]["port"]
    return f"http://{host}:{port}"
