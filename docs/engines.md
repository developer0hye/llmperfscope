# Using it with your engine

Everything works from outside the engine, so no engine needs changes. What differs is how much an engine tells about itself.

Tested versions and what each engine adds:

| Engine | Tested with | Phases come from | Extra in the report |
|---|---|---|---|
| llama.cpp `llama-bench` | master `0ec4027a2` | its progress output | exact per-repetition speed, load sub-phases, memory table |
| llama.cpp `llama-server` | master `0ec4027a2` | its log | memory table, request and slot metrics |
| vLLM server | 0.32.0 | `serve` | cache pool usage, request queue |
| vLLM as a library | 0.32.0 | markers in your script | |
| SGLang | 0.5.22 | `serve` | cache pool usage, weight and cache memory |
| Text Generation Inference | 3.3.7 | `serve` | queue size |
| Ollama | 0.40.9 | `serve` | model download as its own phase |
| MLX `mlx_lm.generate` | mlx-lm 0.32.2 on an M2 | none | its own speed and peak memory |
| MLX `mlx_lm.server` | mlx-lm 0.32.2 on an M2 | `serve` | |
| transformers, PyTorch | as in the vLLM 0.32.0 image | markers in your script | allocated against reserved GPU memory |
| any OpenAI-compatible server | | `serve` | whatever its `/metrics` exposes |
| any other program | | your markers, or none | |

## llama.cpp

```sh
llmperfscope run -- llama-bench -m model.gguf -ngl 99 -p 512 -n 128 -r 3
llmperfscope serve --concurrency 1,4 -- llama-server -m model.gguf -ngl 99 -np 4
```

What llmperfscope changes, all recorded in the report:

- `llama-bench` gets `--progress -o json`, for phase boundaries and per-repetition speed.
- `llama-server` gets `--metrics` and `LLAMA_ARG_LOG_JSONL=1`, for its metrics and memory table.
- `serve` sends `cache_prompt: false` so each request really runs prefill. `--reuse-prompts` keeps the prompt unchanged, and `--extra-body '{"cache_prompt": true}'` turns the cache back on, to measure it.

Wrappers such as `taskset`, `numactl`, `env` and `timeout` in front of the command are fine. With an unusual wrapper, pass `--model-file model.gguf` so llmperfscope can check the page cache and find the disk.

Other tools named `llama-*` get phases from their log, and `--adapter` picks a specific adapter if the name does not match.

## vLLM

```sh
llmperfscope serve --concurrency 1,4,16 -- vllm serve Qwen/Qwen3-0.6B
llmperfscope serve -- docker run --gpus all -p 8000:8000 vllm/vllm-openai --model Qwen/Qwen3-0.6B
```

The cache pool finding comes from `/metrics`. Reading vLLM's load time needs care: `load` covers everything from process start to the first healthy answer, including Python imports and, in a container, the container start.

As a library, mark the phases yourself. [examples/vllm_offline.py](../examples/vllm_offline.py) is a complete script:

```sh
llmperfscope run --label vllm-offline -- python examples/vllm_offline.py Qwen/Qwen3-0.6B
```

## SGLang

```sh
llmperfscope serve --concurrency 1,4 -- python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B
```

SGLang's metrics are off by default, so llmperfscope adds `--enable-metrics`. The default port is 30000; pass `--port` to the server as usual and llmperfscope reads it from the command.

## Text Generation Inference

```sh
llmperfscope serve -- text-generation-launcher --model-id Qwen/Qwen3-0.6B
llmperfscope serve -- docker run --gpus all -p 8080:80 ghcr.io/huggingface/text-generation-inference --model-id Qwen/Qwen3-0.6B
```

A published port tells llmperfscope where to send requests, so `-p 8080:80` needs no `--url`.

## Ollama

Ollama starts with no model, so name one. `--pull` downloads it first and reports the download as its own phase:

```sh
llmperfscope serve --model qwen2.5:0.5b --pull -- ollama serve
```

Ollama loads a model on the first request. That time lands in `warmup`, and the report says so, because a fresh server's first user waits for it.

## transformers and PyTorch

Wrap the script and mark what matters. `mark`, `phase` and `TokenTimer` do nothing when no profiler is running, so the script keeps working on its own:

```python
import llmperfscope

with llmperfscope.phase("load"):
    model = AutoModelForCausalLM.from_pretrained(name, device_map="auto")

timer = llmperfscope.TokenTimer()
with llmperfscope.phase("decode", tokens=128):
    model.generate(**inputs, max_new_tokens=128, streamer=timer)
print(timer.summary())
```

```sh
llmperfscope run -- python script.py
```

`tokens=` gives the phase a speed and energy per token. When PyTorch is loaded, each marker also records its allocated and reserved memory, which is how the report can tell when the allocator holds far more than the tensors use. [examples/transformers_generate.py](../examples/transformers_generate.py) is a complete script.

With no command line at all, `llmperfscope.Profile` samples from a thread of your own process. It is good enough to find a bottleneck, but `run` is better for speed numbers because its sampler stays out of your process:

```python
with llmperfscope.Profile("runs/inline", label="inline") as profile:
    profile.mark("load")
    ...
```

## Apple Silicon

Macs are measured with no root and nothing extra to install, and they get the same phases, verdicts and figures. These are the sources:

| What | From |
|---|---|
| Process memory by kind, CPU, disk reads, threads | libproc and Mach |
| Machine-wide disks, memory, compressor, network | IOKit, Mach and sysctl |
| GPU memory | IOKit's performance statistics |
| GPU time per process | IOKit's `AppUsage` property, undocumented |
| GPU energy and clock | IOReport, undocumented |
| Thermal pressure | the system's `thermalpressurelevel` notification |

If an undocumented source stops working in a later macOS, those numbers go missing and everything else carries on. `llmperfscope check` says what works on your machine.

What reads differently on a Mac:

- **There is one pool of memory.** GPU memory is what the GPU driver holds of RAM, over the whole machine. What the engine's GPU buffers hold also counts in its host memory, so the two do not add up, and a mapped model can be in both.
- **GPU utilization is the engine's own.** It is the engine's share of GPU time, so the display does not count. Other programs using the GPU show up in the check before the run.
- **`gpu_memory_nearly_full` compares with what macOS recommends**, not with all of RAM. On a 16 GB M2 that is 12,124 MiB. Past it, Metal starts paging and inference slows down sharply.
- **The compressor has its own columns.** macOS compresses memory long before it swaps, and also does it to other programs' idle memory, so this traffic alone does not make RAM the limit. `ram` takes critical memory pressure, or memory pressure while neither the GPU nor the CPU is busy.
- **Memory stalls are an approximation.** macOS has no pressure stall counters, so `memory_some_pct` is the time spent at its warning level and `memory_full_pct` at critical. There is nothing for CPU and I/O.
- **Heat changes the speed.** On a fanless M2, full GPU load held for about 85 s before the clock started to fall, and it settled at half: 1,398 to about 700 MHz, and 56.8 to 37.4 tokens/s. GPU utilization stayed at 90% throughout, so look at the clock. The phase is called `gpu (throttled)`, and `--panels power,util,clock` draws it. For a comparison, keep runs short and let the machine cool between arms.
- **Energy is the GPU's alone.** The CPU and memory are not in it.
- **The first run after a build compiles shaders.** On an M2, llama.cpp spent 8.7 s of `startup` on this in a system process, at 0.01 cores of its own, and milliseconds on the next run because macOS keeps the result. Run once before you measure.
- **After `attach`, memory above idle starts from that moment.** macOS does not say which process holds GPU memory, so the engine's own share cannot be taken out as it is on Linux.

llama.cpp takes the same commands as anywhere, and Metal needs nothing more. For MLX:

```sh
llmperfscope run -- python -m mlx_lm.generate --model mlx-community/Qwen3-0.6B-8bit --prompt "Hello" --max-tokens 256
llmperfscope serve --concurrency 1,2 -- python -m mlx_lm.server --model mlx-community/Qwen3-0.6B-8bit
```

`mlx_lm.server` lists every model in your Hugging Face cache and loads whichever a request names. So llmperfscope names none unless you pass `--model`, and it answers with the one on its command line. `mlx_lm.generate` loads and generates in one go, so its run is a single phase; for separate phases, mark them from Python as under [transformers and PyTorch](#transformers-and-pytorch). `Profile` and `phase` also report PyTorch's own memory figures on MPS.

Not available on a Mac:

- `--nsys`. Instruments' Metal System Trace, which comes with Xcode, shows the GPU's timeline.
- `--cpu-profile`. `sample PID 10` shows what the host threads do, and needs no root.
- Containers. They run in a virtual machine there, out of sight of the host. Run the engine directly.

## Any OpenAI-compatible server

```sh
llmperfscope serve --preset openai --url http://127.0.0.1:9000 --ready-path /v1/models --metrics-path /metrics -- my-server --port 9000
```

`--model` names the model when `/v1/models` does not, `--api-key` adds a bearer token (never stored), and `--extra-body '{"top_k": 1}'` adds fields to every request.

By default replies are held to exactly `--max-tokens` with `ignore_eos`, so runs compare like with like. A server that rejects the field needs `--allow-eos`, and then the token counts come from its usage reports.

## Any other program

It already gets startup, a run phase, memory, CPU, disk and GPU. To split it into phases:

```sh
# A line in its output starts a phase. regex=phase, or regex=phase:group to sum several under one group.
llmperfscope run --mark 'Model loaded=decode' -- my-engine

# Or append to a file from any language. llmperfscope exports the path.
echo decode >> "$LLMPERFSCOPE_MARKERS"
```

A marker line is a phase name, a `CLOCK_MONOTONIC` time in nanoseconds followed by a name, or JSON such as `{"t": 123, "phase": "decode", "group": "decode", "tokens": 128}`. Use the canonical names `load`, `setup`, `warmup`, `prefill`, `decode`, `serve`, `teardown` where they fit, so reports line up across engines. Bare phase names get a time when llmperfscope reads them, within 0.2 s; give a time for anything shorter.

## Containers

`docker run` and `podman run` commands work like any other:

```sh
llmperfscope serve -- docker run --gpus all -p 8000:8000 vllm/vllm-openai --model Qwen/Qwen3-0.6B
llmperfscope run -- docker run --gpus all -v "$PWD:/w" my-image python /w/script.py
```

The docker client only sends instructions, so llmperfscope finds the container's first process and follows that. llmperfscope adds `--cidfile`, a mount of the run directory at `/llmperfscope-run`, and `LLMPERFSCOPE_MARKERS` pointing into it. A script in the container can write markers directly. Using `llmperfscope.phase` there needs the package in the image.

Stopping uses `docker stop`, and a container left running is killed.

`--nsys` and `--cpu-profile` are refused here, because they have to start the engine themselves. For those, run the engine directly, or run llmperfscope inside the container.

## Something already running

```sh
llmperfscope attach --docker my-vllm --url http://127.0.0.1:8000 --concurrency 1,4
llmperfscope attach --pid 1234 --duration 120
```

With `--url` it drives requests like `serve`. Without it, it watches for `--duration` seconds. Load is over by then, so those phases are missing. Memory above idle counts from what the process already held, because idle was never seen.

## Choosing the load

| Option | What it does |
|---|---|
| `--prompts mix` | the public [llm-speed-prompt-mix](https://huggingface.co/datasets/developer0hye/llm-speed-prompt-mix) set, downloaded once and pinned: real prompts in several languages, domains and lengths. Each row states its own license |
| `--prompts FILE` | JSONL with `prompt` or `messages` per row |
| `--filter column=value` | keep matching rows, repeatable |
| `--limit N` | at most N prompts |
| `--max-prompt-chars N` | leave out longer prompts, for a server with a small context. The prompt set goes up to 64K tokens |
| `--synthetic-words N` | length of generated prompts when there is no file |
| `--concurrency 1,4,8` | levels, each runs the whole list |
| `--repeat N` | run the list N times per level |
| `--max-tokens N` | reply length |
| `--seed N` | prompt order |
| `--reuse-prompts` | leave prompts unchanged, to measure a prefix cache |

A short nonce goes in front of each prompt by default, so a server's prefix cache cannot answer from memory and prefill is really measured.
