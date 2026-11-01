# Changelog

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
