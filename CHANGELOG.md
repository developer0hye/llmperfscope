# Changelog

## Unreleased

- Reports know the model's structure: from a GGUF header or a Hugging Face directory they show its parts (attention, experts, shared experts, router, head, tables), what a token reads of each, and the engine settings that change how they play out. See `docs/llm-structure.md`.
- `decode_against_memory_bound` gives the least time a token needs to read its weights at the memory's peak speed, from what the driver says or `--memory-gb-per-s`.
- Serving reports fit the time between tokens against the context behind them, which is what the KV cache costs, and summarize speculative decoding and the prompt cache. llama-server's graph reuse and reservations count too.
- `--turns N` runs conversations, so a prompt cache or context checkpoints get measured: first token and cached share per turn. `--reuse-prompts` now really reaches llama-server's cache, which the preset used to turn off.
- `--cpu-profile` tells waiting from work: time under a call that waits for the GPU is counted apart, `inclusive` rows have a `waiting_share`, and the report and the `host_waits_for_gpu` finding name the places that wait.
- Serving reports split the first token into a fixed part and a part per prompt token, with a table by prompt length. `matrix` judges both parts (`ttft_fixed_ms`, `ttft_per_token_ms`) and lists the table per case, and `first_token_fixed_part` says when the fixed part is most of it.
- Measuring works on macOS, with no root and nothing extra to install: process memory by kind, GPU time per process, GPU memory, energy and clock, thermal pressure, the memory compressor, disks by physical device, and `--cold-start`. `--nsys`, `--cpu-profile` and containers stay Linux only. See [Apple Silicon](docs/engines.md#apple-silicon).
- MLX: a server preset for `mlx_lm.server`, and an adapter that reads the speed and peak memory `mlx_lm.generate` reports. `Profile` and `phase` report PyTorch's memory on MPS.
- `serve` reads the prompts before starting the server. Their first download used to count as loading the model.
- `serve` says in the report when a server lists several models and the requests named the first.
- The report leaves out data movement columns the machine has nothing to report for, and `temp_c_max`, `mem_mhz_avg` and `power_gauge_w` are null when unknown, not 0.
- `matrix` runs 4 arms per side by default, and runs a case that looks worse again (`--recheck`) before calling it. With 2 per side, arms miss each other by chance 1 time in 3.
- `--cpu-profile` counts the time inside long system calls, which used to show as one sample, and every profile is checked against the CPU time measured from outside.
- On a system other than Linux, `run`, `serve`, `attach` and `matrix` say what works there, instead of failing on a missing `/proc`.
- `llmperfscope kernels`: each GPU kernel's time per token, its share of a token and the most it could win, with collectives counted without their waiting. `--vs` compares two traces kernel by kernel.
- `--prompts mix` downloads a public set of real prompts once. Reports from `llama-bench` note that it feeds random tokens, and `matrix` can compare builds with a server and real prompts.
- `diff` and `matrix` mark a difference resting on too few runs as a hint, `better?` or `worse?`, and a comparison with only hints as `inconclusive`. Figures do not show hints as results.

## 0.1.0

First public release.

- `run`, `serve` and `attach`: profile any command, server, container or running process over its whole life, with no code changes.
- Per-phase verdicts on what limits it, findings on memory and on data moving between disk, RAM and GPU, and on how serving scales.
- Support for llama.cpp, vLLM, SGLang, Text Generation Inference, Ollama, transformers and any OpenAI-compatible server, plus phase markers for anything else.
- `--nsys` for GPU kernels, collectives and copies, `--cpu-profile` for host hot spots without `perf`, `--cold-start` for a first load.
- `diff`, `matrix` and `noise`: comparisons that judge speed, memory, load time and energy together.
- `figure`: paper figures as PDF, SVG and PNG, with their data as CSV and suggested captions.
- `anonymize`: a copy of a run that can be published.
- `report.html`: every counter on one time axis.
