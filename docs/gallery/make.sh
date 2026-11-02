#!/bin/bash
# Regenerates every image in docs/gallery from the runs in examples/sample-runs. From the repository root:
#   pip install -e ".[figures]" && docs/gallery/make.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
runs=examples/sample-runs
out=docs/gallery
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

scope() { python3 -m llmperfscope "$@" > /dev/null; }

scope figure "$runs/deepseek-v4.1-flash-8gpu-serve" --kind lifecycle,phases,memory,movement,serving --letters --format png --out "$work/serve"
scope figure "$runs/deepseek-v4.1-flash-8gpu-nsys" --kind gpu-time --format png --out "$work/nsys"
scope figure "$runs/deepseek-v4.1-flash-8gpu-cpu-profile" --kind hotspots --format png --out "$work/cpu"
scope figure "$runs/cpu-ram-limited" --kind lifecycle --panels host_memory,cpu,disk,stall --format png --out "$work/ram"
scope figure --compare "enough RAM=$runs/cpu-enough-ram" "450 MiB of RAM=$runs/cpu-ram-limited" --kind speed --format png --out "$work/ramcompare"
scope figure --compare "llama.cpp=$runs/engines/llama.cpp" "vLLM=$runs/engines/vllm" "SGLang=$runs/engines/sglang" "TGI=$runs/engines/tgi" "Ollama=$runs/engines/ollama" \
    --kind serving,memory,phases --format png --out "$work/engines"
scope figure --matrix "$runs/matrix.json" --format png --out "$work/matrix"
scope figure "$runs/apple-m2-long-decode" --kind lifecycle --panels power,util,clock --format png --out "$work/apple"

cp "$work/serve/lifecycle.png" "$out/lifecycle.png"
cp "$work/serve/phases.png" "$out/phases.png"
cp "$work/serve/memory.png" "$out/memory.png"
cp "$work/serve/movement.png" "$out/movement.png"
cp "$work/serve/serving.png" "$out/serving.png"
cp "$work/nsys/gpu_time.png" "$out/gpu_time.png"
cp "$work/cpu/hotspots.png" "$out/hotspots.png"
cp "$work/ram/lifecycle.png" "$out/ram_limited_lifecycle.png"
cp "$work/ramcompare/compare_speed.png" "$out/ram_limited_speed.png"
cp "$work/apple/lifecycle.png" "$out/apple_heat.png"
cp "$work/engines/compare_serving.png" "$out/engines_serving.png"
cp "$work/engines/compare_memory.png" "$out/engines_memory.png"
cp "$work/engines/compare_phases.png" "$out/engines_phases.png"
cp "$work/matrix/matrix.png" "$out/matrix.png"
ls -la "$out"
