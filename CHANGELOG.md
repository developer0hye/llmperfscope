# Changelog

## Unreleased

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
