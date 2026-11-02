# Sample runs

Real runs, to read and plot without a GPU or an engine, and to see what a run directory holds. Each was copied with `llmperfscope anonymize`, which removes host and user names, absolute paths, GPU serial numbers, other processes, logs and prompts, and keeps every number.

```sh
llmperfscope show examples/sample-runs/cpu-ram-limited
llmperfscope figure examples/sample-runs/deepseek-v4.1-flash-8gpu-serve --out figures-try
llmperfscope diff --a examples/sample-runs/cpu-enough-ram --b examples/sample-runs/cpu-ram-limited --a-name enough --b-name limited
```

Note that `llmperfscope html` writes `report.html` into the run directory.

| Run | What it shows | How it ran |
|---|---|---|
| `cpu-enough-ram` | a 0.6 GiB model on 8 CPU threads, loaded cold | `llama-bench` on an AMD EPYC 9554, model on a RAID array that reports as HDD |
| `cpu-ram-limited` | the same with 450 MiB of RAM: decode 2.5 against 73.9 tokens/s, and 18.8 GiB re-read from disk | the same, under `systemd-run --user --scope -p MemoryMax=450M -p MemorySwapMax=0` |
| `deepseek-v4.1-flash-8gpu-serve` | a 748B model on 8 GPUs, from load through serving at 1 and 4 clients | `llama-server -sm tensor` on 8 RTX A6000 with NVLink pairs |
| `deepseek-v4.1-flash-8gpu-nsys` | the same model with `--nsys`: kernels, collectives, copies. Traced, so not for speed | `llama-bench -sm tensor` |
| `deepseek-v4.1-flash-8gpu-cpu-profile` | the same with `--cpu-profile`: host functions per phase. Sampled, so not for speed | `llama-bench -sm tensor` |
| `engines/llama.cpp`, `vllm`, `sglang`, `tgi`, `ollama` | each engine serving at 1 and 4 clients with default settings | one RTX 3070 |
| `engines/transformers`, `vllm-library` | engines used as libraries, with markers in the script | one RTX 3070, in a container |
| `apple-m2-long-decode` | heat on a fanless Mac: the GPU clock falls by half after about 85 s, and decode from 56.8 to 37.4 tokens/s, with utilization unchanged | `llama-bench -p 0 -n 8192 -r 3` on Metal, M2 MacBook Air, 16 GB |
| `apple-m2/llama.cpp`, `mlx` | the same model size and prompts on the two engines, 1 and 2 clients. These do not rank them either | `llama-server` and `mlx_lm.server` on the same machine, Qwen3-0.6B at 8 bits |
| `matrix.json` | a patched llama.cpp against upstream over three cases, for `llmperfscope figure --matrix` | 8 RTX A6000 |

The engine runs show what the tool reports, and **they do not rank the engines**. Nothing was tuned, they are not all the same model or format (llama.cpp ran a Qwen3-0.6B Q8_0 GGUF; vLLM, SGLang, TGI and transformers ran the original Qwen3-0.6B weights; Ollama ran qwen2.5:0.5b), and vLLM, SGLang, TGI and Ollama ran in containers.

The llama.cpp builds are master at commit `0ec4027a2`, with 2 changed files for the 8 GPU runs. Each `run.json` records what ran and on what.

These runs are released under the same license as the rest of this repository.
