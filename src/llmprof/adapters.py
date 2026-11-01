"""What llmprof knows about specific engines. The core works without any of this; an adapter only adds finer phases,
exact throughput and server defaults. A new engine needs one small class here, or just `--mark 'regex=phase'`.

Canonical phase groups, so reports line up across engines:
    startup, load, setup, warmup, prefill, decode, serve, teardown
"""

from __future__ import annotations

import json
import os
import re

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
WRAPPERS = {"env", "taskset", "numactl", "timeout", "nice", "ionice", "chrt", "flock", "stdbuf", "setsid", "nohup", "sudo"}


class Adapter:
    name = "generic"
    # Defaults for `llmprof serve`: paths on the server, signal to stop it, extra request fields.
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
        return None

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
    """Self-check: the time llmprof saw for each run against what the engine timed itself. Far from 1 means the phase
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


class _LlamaCommon(Adapter):
    def model_path(self):
        return _find(self.cmd, "-m", "--model")

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
SERVER_PRESETS = {
    "openai": {"ready_path": "/health", "metrics_path": None, "base_path": "/v1", "stop_signal": "TERM", "extra_body": {},
               "port": 8000},
    "llama-server": {**LlamaServer.server, "port": 8080},
    "vllm": {"ready_path": "/health", "metrics_path": "/metrics", "base_path": "/v1", "stop_signal": "TERM", "extra_body": {},
             "port": 8000},
    "sglang": {"ready_path": "/health", "metrics_path": "/metrics", "base_path": "/v1", "stop_signal": "TERM", "extra_body": {},
               "port": 30000, "extra_args": ["--enable-metrics"]},
    "tgi": {"ready_path": "/health", "metrics_path": "/metrics", "base_path": "/v1", "stop_signal": "TERM", "extra_body": {},
            "port": 3000},
    "ollama": {"ready_path": "/api/version", "metrics_path": None, "base_path": "/v1", "stop_signal": "TERM", "extra_body": {},
               "port": 11434},
}


def select(name, cmd, marks=()):
    """`name` is "auto" or an adapter name."""
    words = [os.path.basename(token) for token in cmd]
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
    return "openai"


def docker_index(cmd):
    """Index of `docker` in a `docker run ...` command, or None."""
    i = _skip_wrappers(cmd)
    if i + 1 < len(cmd) and os.path.basename(cmd[i]) in ("docker", "podman") and cmd[i + 1] == "run":
        return i
    return None


def guess_url(cmd, preset_name):
    """Where a server will listen, so nobody has to say it twice. A container's published port wins over the flags inside."""
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
    if port is None and preset_name == "ollama" and ":" in os.environ.get("OLLAMA_HOST", ""):
        tail = os.environ["OLLAMA_HOST"].rsplit(":", 1)[1]
        if tail.isdigit():
            port = int(tail)
    if port is None:
        port = SERVER_PRESETS[preset_name]["port"]
    return f"http://{host}:{port}"
