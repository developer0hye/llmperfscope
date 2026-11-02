# Adding an engine or a GPU vendor

The core needs nothing from an engine. First see whether you need code at all.

## Often you need no code

**A server with an OpenAI-compatible API** works now, with `--preset openai`. See [engines.md](engines.md).

**A program with useful output** gets phases from a regular expression:

```sh
llmperfscope run --mark 'Loading weights=load' --mark 'Starting generation=decode' -- my-engine
```

**A program you can change** can append lines to `$LLMPERFSCOPE_MARKERS`, or call `llmperfscope.mark` from Python.

## A server preset

To make `llmperfscope serve -- my-engine` work with no options, add an entry to `SERVER_PRESETS` in `adapters.py`, and teach `preset_for` to recognize the command:

```python
"my-engine": {"ready_path": "/health", "metrics_path": "/metrics", "base_path": "/v1", "stop_signal": "TERM", "extra_body": {}, "port": 8000},
```

| Field | Meaning |
|---|---|
| `ready_path` | answers 200 once the model is loaded |
| `metrics_path` | Prometheus text, or `None`. Gauges are watched during the load, counters compared before and after |
| `base_path` | where `/chat/completions` and `/completions` live |
| `stop_signal` | how to stop it cleanly |
| `extra_body` | fields added to every request |
| `port` | used when neither the command nor a published container port names one |
| `extra_args` | added to the command if missing, such as a flag that turns metrics on |

A gauge whose name contains `kv_cache_usage`, `gpu_cache_usage` or `token_usage` feeds the cache pool findings.

## An adapter

An adapter adds finer phases, exact speed and the engine's own memory table. It is a small class in `adapters.py`, and `select` chooses it from the command:

```python
class MyEngine(Adapter):
    name = "my-engine"

    def prepare(self, cmd, env):
        """Last chance to change the command and environment. Everything changed is stored in the report."""
        return cmd + ["--verbose-timings"], env

    def on_line(self, stream, t, text, emit):
        """A line of output arrived at CLOCK_MONOTONIC t. Start a phase with emit(phase, group=None, t=None, tokens=None, **meta)."""
        super().on_line(stream, t, text, emit)  # keeps --mark working
        if text.startswith("model loaded"):
            emit("warmup", t=t)

    def finalize(self, phases, stdout_path, stderr_path):
        """After exit: parse results, fill in tokens per phase. The returned dict is merged into the report."""
        return {"throughput": {"decode": {"tokens_per_s": 42.0, "stddev": 0.5, "samples": [42.0], "tokens_per_run": 128}}}

    def model_path(self):
        """Where the weights are, for the page cache check, --cold-start and the storage finding."""
        return _find(self.cmd, "--model")

    def has_progress_lines(self):
        """True when silence on stdout and stderr really means nothing is happening. The stall watchdog relies on it."""
        return True
```

Use the canonical group names `startup`, `load`, `setup`, `warmup`, `prefill`, `decode`, `serve`, `teardown`, so reports line up across engines. A memory table goes in `memory_breakdown` as rows with `kind` (`device`, `host` or `buffer_type`), `name`, `self`, `model`, `context` and `compute`, and the findings pick it up from there.

`LlamaBench` and `LlamaServer` in the same file are complete examples.

Test an adapter with a fake engine that prints known lines, and check the report says what happened. `tests/test_parsers.py` keeps real output from each engine.

## A GPU vendor

`gpu.py` has a small interface. NVML and Apple Silicon implement it so far:

```python
class GpuBackend:
    name = "none"
    devices = []           # {"index", "name", "uuid", "memory_total_mib", "power_limit_w", "nvlink"}

    def fast(self):        # per device: the G_* columns. Called about 50 times a second, so it must be cheap
    def energy_mj(self, index):   # cumulative energy, or None. May block; read on its own thread
    def set_target(self, pids):   # the processes being measured, for numbers kept per process
    def slow(self):        # per-process GPU memory and NVLink counters, about twice a second
    def util_samples(self):       # the driver's utilization history, if it keeps one
    def pcie_kbps(self, index, direction):   # may block; read on its own thread
    def driver_version(self):
    def close(self):
```

Return `None` for what a vendor does not have, and the matching numbers and findings drop out. `open_backend()` picks the backend. `LLMPERFSCOPE_NO_GPU=1` forces none, to test the host side. Anything slow must not go in `fast`: reading the energy counter at sampling rate on an NVIDIA GPU cost 3.4% of the engine's speed, and also made the counter under-report by half.
