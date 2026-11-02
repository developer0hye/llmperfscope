# Quickstart

## Install

```sh
pip install llmperfscope               # the profiler; it has no dependencies
pip install "llmperfscope[figures]"    # adds matplotlib, for paper figures
llmperfscope check                     # what can be measured on this machine
```

Measuring needs Linux or macOS and Python 3.9 or newer. GPU telemetry needs an NVIDIA driver on Linux, and nothing extra on [Apple Silicon](engines.md#apple-silicon); without a GPU you still get phases, host memory, CPU and disk. Reading finished runs and drawing figures works anywhere.

## Try it without a GPU

The repository has real runs in [examples/sample-runs](../examples/sample-runs). You can read and plot them on any machine:

```sh
git clone https://github.com/developer0hye/llmperfscope && cd llmperfscope
llmperfscope show examples/sample-runs/cpu-ram-limited
llmperfscope figure examples/sample-runs/cpu-ram-limited --kind lifecycle --out figures-try
```

## Profile a command

Put `llmperfscope run --` in front of the command you already use:

```sh
llmperfscope run --label first -- llama-bench -m model.gguf -ngl 99 -p 512 -n 128 -r 3
```

The report goes to the terminal, and the run directory holds everything else:

| File | What it is |
|---|---|
| `report.html` | every counter on one time axis, one self-contained page |
| `run.json` | all the numbers, see [run-json.md](run-json.md) |
| `samples.jsonl.gz` | the raw samples behind them |
| `stdout.log`, `stderr.log` | the program's output |
| `requests.jsonl` | one row per request, with `serve` only |

Runs go to `runs/<label>-<time>` unless you pass `--out`.

## Profile a server

`serve` starts the server, waits until it answers, warms it up, drives streaming requests at each concurrency level, reads its metrics and stops it. The whole life is one run:

```sh
llmperfscope serve --concurrency 1,4,8 --max-tokens 64 -- vllm serve Qwen/Qwen3-0.6B
```

Without `--prompts` it sends 8 generated prompts of about 512 words. For real prompts use `--prompts mix`, which downloads [llm-speed-prompt-mix](https://huggingface.co/datasets/developer0hye/llm-speed-prompt-mix) once: 89 prompts in several languages, domains and lengths, or the other 371 with `--filter tier=extended`. Your own JSONL file with a `prompt` or `messages` column per row works too:

```sh
llmperfscope serve --prompts mix --max-prompt-chars 20000 --concurrency 1,4 -- llama-server -m model.gguf -ngl 99 -np 4
llmperfscope serve --prompts mine.jsonl --filter language=ko -- llama-server -m model.gguf -ngl 99
```

Use real prompts for any speed you will quote. `llama-bench` feeds random tokens, and a gain there can be much smaller with real text.

Every engine has its own notes in [engines.md](engines.md).

## Read the answer

Start at the top of the report. **What limits each phase** gives one verdict per phase with the numbers behind it, and the findings below say what is wasting memory or moving data. [reading-reports.md](reading-reports.md) explains each verdict and what to do about it.

When the verdict is about the GPU or the CPU and you want to know which kernel or function, run again with `--nsys` or `--cpu-profile`. See [deep-dives.md](deep-dives.md). These slow the engine down, so use them to find where time goes, not to measure speed.

## Compare two builds

```sh
llmperfscope matrix --cases cases.json --a old=build-old/bin --b new=build-new/bin --out res
```

Both builds run in alternating order in fresh processes, and speed, memory, load time and energy are judged together. See [comparing.md](comparing.md).

## Make a figure

```sh
llmperfscope figure runs/first-20261101-120000 --width single --out figures
```

This writes PDF, PNG and SVG for each figure the run has data for, the data as CSV, and suggested captions. See [figures.md](figures.md).

## Which command do I want?

| You want to | Use |
|---|---|
| profile a program from start to finish | `llmperfscope run -- ...` |
| start a server, load it and stop it | `llmperfscope serve -- ...` |
| profile something already running | `llmperfscope attach --pid N`, or `--docker NAME` |
| read a finished run again | `llmperfscope show RUN`, `llmperfscope html RUN` |
| draw figures | `llmperfscope figure RUN` |
| compare finished runs | `llmperfscope diff --a RUN ... --b RUN ...` |
| compare two builds over many cases | `llmperfscope matrix` |
| measure how much a comparison varies by chance | `llmperfscope noise` |
| share a run without host names and paths | `llmperfscope anonymize RUN --out DIR` |
| see what this machine can measure | `llmperfscope check` |

`llmperfscope <command> --help` lists every option.
