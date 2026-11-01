# Comparing builds and runs

A single ratio from a few runs is mostly noise. The same binary against itself has shown ratios up to 1.006 at 8 GPUs, so a claimed 0.5% win from one pair of runs means nothing. A speed-up that takes more memory is also not a clean win. llmperfscope's comparisons are built around these two facts.

## `matrix`: two builds over many cases

```sh
llmperfscope matrix --cases cases.json --a old=build-old/bin --b new=build-new/bin --out res
```

For each case, it runs both builds in the order set by `--order` (`ABBA` by default), each arm in a fresh process, and waits for a quiet machine before each arm. The result goes to `res/table.md` and `res/matrix.json`, with every arm's full run directory under `res/`. The exit code is 1 if anything got worse, which suits CI.

### The cases file

```json
{
  "defaults": {
    "binary": "llama-bench",
    "args": ["-ngl", "99", "-p", "512", "-n", "128", "-r", "3"],
    "env": {}
  },
  "cases": [
    {"label": "small model, 1 GPU", "model": "/models/small.gguf", "gpus": "0", "args": ["-sm", "none"]},
    {"label": "small model, tensor split", "model": "/models/small.gguf", "gpus": "0,1", "args": ["-sm", "tensor"]}
  ]
}
```

`binary` is looked up in the `BIN_DIR` of each side. `args` from the defaults and the case are joined, `gpus` sets `CUDA_VISIBLE_DEVICES`, `prefix` goes in front of the command (such as `["taskset", "-c", "16-23"]`), and `env` sets variables. A case whose model is missing is skipped and listed. [examples/cases-llama-bench.json](../examples/cases-llama-bench.json) is a starting point.

Useful options: `--only 'regex'` runs some cases, `--a-env KEY=VALUE` and `--b-env KEY=VALUE` run the same binary with different settings, `--timeout` limits each arm, and `--order ABBABAAB` adds repetitions.

### What it judges

| Metric | Better is | Floor |
|---|---|---|
| `prefill_tps`, `decode_tps` | higher | 0.3% |
| `gpu_peak_mib`, `gpu_peak_max_device_mib` | lower | 0.2%, at least 16 MiB |
| `host_anon_peak_mib` | lower | 2%, at least 64 MiB |
| `load_s` | lower | 5%, at least 0.5 s |
| `prefill_j_per_token` | lower | 5% |
| `decode_j_per_token` | lower | 3% |

A result counts as `better` or `worse` only if **both** of these hold: the arms do not overlap, meaning every run of one side beats every run of the other, and the difference of the means exceeds the floor. Otherwise it is `same`.

The floors are the smallest differences that can mean anything for that metric. Memory rounds to MiB and allocators differ a little from run to run, load time depends on the page cache, and the energy counter has its own resolution. Your own noise floor can be higher, which the next section measures.

Each case then gets one verdict:

| Verdict | Means |
|---|---|
| `improvement` | something is better, nothing is worse |
| `neutral` | nothing changed beyond the floors |
| `regression` | something is worse, nothing is better |
| `tradeoff` | some better, some worse |

Failed runs, and runs with `--nsys` or `--cpu-profile`, do not count. They are reported as dropped.

### Measuring the noise floor

Compare a build with itself first. This tells you what "no change" looks like on your machine:

```sh
llmperfscope matrix --cases cases.json --a same=build/bin --b again=build/bin --out null
llmperfscope noise null/matrix.json --out floors.json
llmperfscope matrix --cases cases.json --a old=build-old/bin --b new=build-new/bin --noise floors.json --out res
```

`floors.json` holds, for each case and metric, the spread seen with nothing changed. A later matrix then raises its floor to at least that.

## `diff`: finished runs

```sh
llmperfscope diff --a runs/old-1 runs/old-2 --b runs/new-1 runs/new-2 --a-name old --b-name new
```

The same judgment on runs you already have. Runs that started on a busy machine are counted and mentioned, because they can make either side look worse. You control the run order here, so alternate the sides and do not change anything else between them.

## Figures

```sh
llmperfscope figure --matrix res/matrix.json --out figures
llmperfscope figure --compare "old=runs/old-1" "new=runs/new-1" --kind speed,memory --out figures
```

See [figures.md](figures.md).

## Getting a clean measurement

- Check for other work first. `--precheck wait` waits until two samples in a row are clean: no build running, no other process above 10% of a core (or 1% of all CPUs on a large machine), other processes together under double that, the GPUs in use idle, and at least 20% of memory available. `matrix` does this before every arm.
- Use fresh processes. A long-lived process carries caches and allocator state from one arm to the next.
- Alternate the order. A slow drift, such as heat or background work, then cannot pass for a speed-up.
- Use the same page cache state. Run a warm-up arm, or `--cold-start` on both sides.
- Run enough repetitions for the arms to separate. With `ABBA` and `-r 3` each side has 2 runs of 3 repetitions.
- Keep the run directories. A number without its machine state cannot be trusted later, and `run.json` has that state.
