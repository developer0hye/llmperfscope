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

## Engines tested end to end

llama.cpp natively on one RTX 3070 and on 8 RTX A6000. vLLM 0.32.0 (server and library), SGLang 0.5.22, TGI 3.3.7, Ollama 0.40.9 and transformers in containers on one RTX 3070, with a small model. Their runs are in [examples/sample-runs/engines](../examples/sample-runs/engines). Other versions may name their metrics differently: a missing gauge drops that finding and nothing else.

## Mistakes the tool has already made

These are real failures that shaped how it works. They show what to watch for in any tool of this kind:

- **Reading the NVML energy counter too often is expensive and wrong.** It takes 2.5 ms per GPU. At the sampling rate of 50 Hz it cost 3.4% of prefill speed on 8 GPUs, and it under-reported by half (818 W against 1,395 W on the gauge). Read about twice a second it agrees (1,602 W against 1,591 W), so it now has its own thread and every report carries the ratio as a self-check.
- **`setitimer` sampling sends every signal to the main thread when it is running.** That makes a program look single-threaded, so the CPU profiler now uses one CPU-time timer per thread.
- **The software power cap bit is set far below the limit, even on an idle GPU.** Taking it at face value "found" throttling everywhere and made `--precheck wait` wait forever. It now counts only near the power limit.
- **A signal-based profiler barely sees a long system call.** A `munmap` of 12 s showed as 8 samples, 0.06% of the load. Samples now count for the timer periods the kernel merged, and each profile is checked against the CPU time measured from outside.
- **Nearest-symbol lookup in a stripped library names the wrong function.** Names are now trusted only inside the symbol's extent.
- **Two arms per side were not enough for `matrix`.** Arms miss each other by chance 1 time in 3 with 2 per side, and a nightly run flagged losses on models that never run the changed code. The default is now 4 arms per side, a case that looks worse is run again, and anything resting on fewer arms is marked as a hint.
- **Sampling dropped to 16 Hz with 2 s gaps** in 1-GPU arms of long comparison chains, and this was not reproduced alone. The cause is unknown, possibly NVML blocking on GPUs going idle. Reports note a rate far below the target, and cumulative counters stay right. Gauges in those phases can miss short spikes.

## Limits to keep in mind

- PCIe traffic is a sample about once a second, not a count.
- NVML's utilization needs about a second to mean anything, so shorter phases are not judged. It also says only that a kernel was running, not how hard it worked: a collective that spins on a wait reads 100%.
- System-wide counters include other processes. On a shared machine, trust the columns for this process.
- A busy machine skews numbers without any visible sign. Runs record what else was running, and flag it.
- It has been tested on NVIDIA GPUs only. Without NVML you get no GPU numbers, only a note saying so.
- The verdict rules are thresholds chosen on these runs. They come with their evidence so you can disagree with them. They are starting points for looking, not proofs.
