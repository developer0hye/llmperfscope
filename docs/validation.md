# How far the numbers can be trusted

A profiler is an instrument, so it has to be checked like one. Here is what has been measured, and what has not.

## What the tool itself costs

Speed with llmperfscope against without it, alternating fresh processes, 4 arms each of 10 repetitions, on 8 RTX A6000:

| Case | Prefill | Decode |
|---|---:|---:|
| Qwen3-30B-A3B Q8_0, 1 GPU | 1.00 | 1.00 |
| Qwen3-30B-A3B Q8_0, 8 GPUs, tensor split | 0.998 | 1.000 |
| DeepSeek-V4.1-Flash, 8 GPUs, tensor split | 1.001 | 1.001 |

The same binary against itself varies by about 1% at 8 GPUs, so these are within noise. Sampling happens in another process, pinned away from the engine's CPUs when one is free. `--nsys` and `--cpu-profile` do slow the engine, and those runs are marked.

## Known answers

| What was set up | What the report said |
|---|---|
| A 639 MiB memory-mapped model on the CPU, inside a 450 MiB cgroup (`examples/sample-runs/cpu-ram-limited`) | `ram` for decode, with 18.8 GiB read back from disk and 49,217 major faults; decode 2.5 tokens/s against 73.9 with enough RAM |
| The same model with `--cold-start`, with enough RAM (`cpu-enough-ram`) | 0% of the model cached, 0.59 GiB read during load, and no disk reads during inference |
| Partial offload | `cpu (many threads)` |
| GPU memory from NVML, against llama.cpp's own table | equal to within driver overhead |
| GPU kernel times from `--nsys`, against a separately written analysis of the same trace | equal |
| Phase times, against the engine's own timers | ratio 1.000 to 1.002 |
| Energy counter, against the power gauge | ratio 0.96 to 1.04 in the sample runs |

The test suite does this on every commit, with processes of known behavior standing in for an engine: known memory use, known disk reads, known thread counts, scripted GPU power. It checks that the report says what happened:

```sh
cd tests && python3 -m unittest
```

`test_parsers` also keeps real output from each engine, so a format change shows up as a failure.

## Apple Silicon

On an M2 MacBook Air, 16 GB, macOS 27.2. With against without llmperfscope, alternating 4 fresh processes each, prefill 0.9995 and decode 1.000, against a spread of 0.5% between runs of the same thing:

| What was set up | What the report said |
|---|---|
| A Metal compute job of known length, from a process with nothing else to do (`tests/metal_burn.py`) | the GPU credited to it, over 50% busy, with energy and a clock. A process doing only CPU work on the same machine gets no GPU numbers |
| That job next to the display | 92.7% of GPU time for the job and 6.4% for WindowServer, which leaves 0.9% for everything else |
| `llama-bench` | 92.9 tokens/s, phase times against its own timers at 1.0001 |
| The same binary against itself, 3 arms per side, with `llama-bench` and with `llama-server` | every speed and memory ratio within 0.6%, and a 0.3% hint marked as a hint |
| GPU clock under full load, before heat | 1,398 MHz, the M2's top clock |
| Three repetitions of 8,192 tokens, nearly 10 minutes, no fan (`apple-m2-long-decode`) | `gpu (throttled)`, with 56.8, then 37.4 and 38.2 tokens/s, as the clock fell from 1,398 to about 700 MHz and power from 9 to 2.5 W. GPU utilization stayed at 90%, so it alone would not have shown this |
| GPU memory, `llama-server` against its own table (`apple-m2/llama.cpp`) | 2,779 MiB above idle against 2,683 MiB in its table |
| A 639 MiB model with `--cold-start`, from a USB SSD and from the internal one | 0% cached both times, read at 427 and 725 MiB/s, each from its own disk |
| `mlx_lm.generate` | prefill and decode as it reports them, 107.4 tokens/s decode |

Not cross-checked: GPU energy, because the only other source, `powermetrics`, needs root. It reads 9 W at full clock, which is plausible for this GPU. CPU and memory energy are not measured at all.

## Engines tested end to end

llama.cpp natively on one RTX 3070, on 8 RTX A6000 and with Metal on an M2. MLX natively on an M2. vLLM 0.32.0 (server and library), SGLang 0.5.22, TGI 3.3.7, Ollama 0.40.9 and transformers in containers on one RTX 3070, with a small model. Their runs are in [examples/sample-runs/engines](../examples/sample-runs/engines). Other versions may name their metrics differently: a missing gauge drops that finding and nothing else.

## Mistakes the tool has already made

These are real failures that shaped how it works. They show what to watch for in any tool of this kind:

- **Asking a server for its first model.** `mlx_lm.server` lists every model in the Hugging Face cache and loads whichever a request names. Taking the first from `/v1/models` loaded a 9B model instead of the 0.6B one on the command line, and the report said 9.5 tokens/s and 7 GiB where the right answer was 63 tokens/s and 2.4 GiB. The numbers only looked wrong next to what `mlx_lm.generate` printed for itself. Now `serve` names no model on a server that has a default, and a report says so when a server lists several.
- **Getting ready inside the measurement.** The first use of `--prompts mix` downloaded the prompts while the server was starting, so a load of 2 s showed as 26 s. Prompts are now read before the server starts.
- **Reading another system's counters by the same names.** The first macOS version counted the memory compressor as swap. macOS compresses all the time, so every phase of a healthy run came out as `ram`. The compressor now has its own columns, and `ram` takes real memory pressure with nothing else busy.
- **Reading the NVML energy counter too often is expensive and wrong.** It takes 2.5 ms per GPU. At the sampling rate of 50 Hz it cost 3.4% of prefill speed on 8 GPUs, and it under-reported by half (818 W against 1,395 W on the gauge). Read about twice a second it agrees (1,602 W against 1,591 W), so it now has its own thread and every report carries the ratio as a self-check.
- **`setitimer` sampling sends every signal to the main thread when it is running.** That makes a program look single-threaded, so the CPU profiler now uses one CPU-time timer per thread.
- **The software power cap bit is set far below the limit, even on an idle GPU.** Taking it at face value "found" throttling everywhere and made `--precheck wait` wait forever. It now counts only near the power limit.
- **A signal-based profiler barely sees a long system call.** A `munmap` of 12 s showed as 8 samples, 0.06% of the load. Samples now count for the timer periods the kernel merged, and each profile is checked against the CPU time measured from outside.
- **Nearest-symbol lookup in a stripped library names the wrong function.** Names are now trusted only inside the symbol's extent.
- **Two arms per side were not enough for `matrix`.** Arms miss each other by chance 1 time in 3 with 2 per side, and a nightly run flagged losses on models that never run the changed code. The default is now 4 arms per side, a case that looks worse is run again, and anything resting on fewer arms is marked as a hint.
- **A benchmark's input can decide its result.** A change sped up prefill by 33% with `llama-bench`'s random tokens and by 2% to 11% on real prompts. Random tokens read a table all over, and real text does not. Reports from `llama-bench` now say so, `--prompts mix` makes real prompts one option away, and `matrix` can compare with them.
- **Sampling dropped to 16 Hz with 2 s gaps** in 1-GPU arms of long comparison chains, and this was not reproduced alone. The cause is unknown, possibly NVML blocking on GPUs going idle. Reports note a rate far below the target, and cumulative counters stay right. Gauges in those phases can miss short spikes.

## Limits to keep in mind

- PCIe traffic is a sample about once a second, not a count.
- NVML's utilization needs about a second to mean anything, so shorter phases are not judged. It also says only that a kernel was running, not how hard it worked: a collective that spins on a wait reads 100%.
- System-wide counters include other processes. On a shared machine, trust the columns for this process.
- A busy machine skews numbers without any visible sign. Runs record what else was running, and flag it.
- It has been tested on NVIDIA GPUs only. Without NVML you get no GPU numbers, only a note saying so.
- The verdict rules are thresholds chosen on these runs. They come with their evidence so you can disagree with them. They are starting points for looking, not proofs.
