# Comparing builds and runs

A single ratio from a few runs is mostly noise. The same binary against itself has shown ratios up to 1.006 at 8 GPUs, so a claimed 0.5% win from one pair of runs means nothing. A speed-up that takes more memory is also not a clean win. llmperfscope's comparisons are built around these two facts.

## `matrix`: two builds over many cases

```sh
llmperfscope matrix --cases cases.json --a old=build-old/bin --b new=build-new/bin --out res
```

For each case, it runs both builds in the order set by `--order` (`ABBABAAB` by default, 4 arms per side), each arm in a fresh process, and waits for a quiet machine before each arm, up to an hour. A desktop with a browser open may never be quiet enough: `--precheck warn` runs each arm at once and flags the ones that started on a busy machine. A case that looks worse gets the same number of arms again before anything is called worse (`--recheck`, 1 round by default, 0 to turn it off). The result goes to `res/table.md` and `res/matrix.json`, with every arm's full run directory under `res/`. The exit code is 1 if anything got worse, which suits CI.

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

Useful options: `--only 'regex'` runs some cases, `--a-env KEY=VALUE` and `--b-env KEY=VALUE` run the same binary with different settings, `--timeout` limits each arm, and `--order` sets the arms.

### With real prompts

`llama-bench` feeds random tokens. Caches, expert routing and table lookups behave differently with real text, so a gain can be much smaller there. One real change sped up prefill by 33% with `llama-bench` and by 2% to 11% with real prompts, depending on their length. Before you trust a gain, add a server case. It takes a `serve` block with the settings of `llmperfscope serve`:

```json
{
  "cases": [
    {
      "label": "real prompts, 2 GPUs",
      "binary": "llama-server",
      "model": "/models/large.gguf",
      "gpus": "0,1",
      "args": ["-ngl", "99", "-sm", "tensor", "-c", "131072", "--port", "8123"],
      "serve": {"prompts": "mix", "max_tokens": 16}
    }
  ]
}
```

These cases are judged on what a streaming client saw, as well as on memory and energy. llama-server's own timers also give prefill and decode speed.

### What it judges

| Metric | Better is | Floor |
|---|---|---|
| `prefill_tps`, `decode_tps` | higher | 0.3% |
| `gpu_peak_mib`, `gpu_peak_max_device_mib` | lower | 0.2%, at least 16 MiB |
| `host_anon_peak_mib` | lower | 2%, at least 64 MiB |
| `load_s` | lower | 5%, at least 0.5 s |
| `prefill_j_per_token` | lower | 5% |
| `decode_j_per_token` | lower | 3% |
| `output_tps`, at the highest concurrency | higher | 0.3% |
| `ttft_ms`, `itl_ms`, median at the lowest concurrency | lower | 1% with at least 1 ms, 0.5% |
| `ttft_fixed_ms`, `ttft_per_token_ms`, the two parts of the first token, see [reading-reports.md](reading-reports.md) | lower | 2% with at least 2 ms, 2% with at least 0.005 ms |
| `serve_j_per_token` | lower | 3% |

A result counts as `better` or `worse` only if **both** of these hold: the arms do not overlap, meaning every run of one side beats every run of the other, and the difference of the means exceeds the floor. Otherwise it is `same`.

The floors are the smallest differences that can mean anything for that metric. Memory rounds to MiB and allocators differ a little from run to run, load time depends on the page cache, and the energy counter has its own resolution. Your own noise floor can be higher, which a later section measures.

With real prompts, `table.md` also has the first token by prompt length for each case, one row per group of lengths. A single number for the first token mostly follows the short prompts, so a change that only helps long ones, or only hurts them, shows up here and not there.

### How many arms it takes

Arms that do not overlap are only evidence if there are enough of them, because arms can miss each other by luck:

| Arms per side | Arms miss each other when nothing changed |
|---:|---|
| 1 | always |
| 2 | 1 time in 3 |
| 3 | 1 time in 10 |
| 4 | 1 time in 35 |
| 8 | 1 time in 6,435 |

With fewer than 4 per side, a difference is marked `better?` or `worse?` and counts only as a hint, unless the gap is more than 5 times what each side varies by. Memory that is the same in every arm, or a speed-up of 35%, needs no more runs.

This comes from a real mistake. A nightly run over 10 changes and 13 cases with 2 arms per side flagged many small losses, some on models that never run the changed code. Run again with 4 arms per side, most of them disappeared. With many cases and metrics a few will still look worse by chance at 4 arms per side, which is why `matrix` runs those cases again.

Each case then gets one verdict:

| Verdict | Means |
|---|---|
| `improvement` | something is better, nothing is worse |
| `neutral` | nothing changed beyond the floors |
| `regression` | something is worse, nothing is better |
| `tradeoff` | some better, some worse |
| `inconclusive` | only hints: run more arms |

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

The same judgment on runs you already have. One run per side can only give hints, so aim for 4. Runs that started on a busy machine are counted and mentioned, because they can make either side look worse. You control the run order here, so alternate the sides and do not change anything else between them.

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
- Use real prompts for anything you will quote. See [With real prompts](#with-real-prompts).
- Use the same page cache state. Run a warm-up arm, or `--cold-start` on both sides.
- Run enough arms for them to separate. With the default order and `-r 3`, each side has 4 processes of 3 repetitions.
- Keep the run directories. A number without its machine state cannot be trusted later, and `run.json` has that state.
