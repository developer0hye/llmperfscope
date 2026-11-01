# Going deeper: which kernel, which function

A normal run says what limits a phase. These options say which part of the code it is. They slow the engine down, so reports mark those runs `perturbed`, and comparisons leave them out. Use them to find where time goes, and measure speed in a separate run.

They have to start the engine themselves, so they are refused with `attach` and with `docker run` commands. Inside a container, run llmperfscope there too.

## `--nsys`: the GPU's time

```sh
llmperfscope run --nsys -- llama-bench -m model.gguf -ngl 99 -sm tensor -p 512 -n 32 -r 1
```

Needs [Nsight Systems](https://developer.nvidia.com/nsight-systems) on `PATH` or under `CUDA_HOME`. Device hardware counters are not needed, so it works without admin rights. `llmperfscope check` says whether it is found.

Per phase and per GPU, the report then has:

- time in kernels, in collectives (NCCL), in copies and idle
- the kernels that take the most time, per token too
- the host's time in CUDA calls
- the exact bytes copied, per token too

```
decode: each GPU spends 21% of its time in kernels, 12% in collectives, 0% in copies and 67% idle (tracing slows the host, so idle reads high).
Top kernels: ncclDevKernel_AllReduce_f32_RING 36%, mul_mat_vec_q 24%, mul_mat_vec_f 15%.
```

Keep traced runs short, because a trace grows quickly. `--nsys-arg` passes extra arguments to `nsys profile`, and `llmperfscope figure RUN --kind gpu-time` draws the result. To analyze a trace you already exported with `nsys export --type sqlite`:

```sh
llmperfscope nsys-report trace.sqlite
```

## `--cpu-profile`: the host's time

```sh
llmperfscope run --cpu-profile -- llama-bench -m model.gguf -ngl 99 -p 512 -n 128 -r 3
```

Samples the stacks of the host threads from inside the process, so it works where `perf` is not allowed. It needs a C compiler once, to build a small shim, which it keeps next to the package or in your cache directory. The shim is loaded with `LD_PRELOAD`, so a static binary cannot be sampled.

Each thread has its own CPU-time timer. That matters: the classic `setitimer` signal goes to the main thread whenever it is running, which makes every multi-threaded program look single-threaded.

A signal only arrives when a system call returns, so time inside one long call would show as a single sample. The kernel reports how many timer periods it merged, and each sample counts for all of them, so a `munmap` of 12 s shows as 12 s. The report checks the profile against the CPU time measured from outside, and says so if it accounts for under 90% of it.

Per phase, the report has the functions with the most self time, the threads that ran them, and the share of the phase each one can account for at most. That last number is the most a speed-up of that function could win back. `--cpu-hz` sets the rate, 200 by default. Draw it with `llmperfscope figure RUN --kind hotspots`.

Function names come from the binary's symbols through `nm`. A stripped library shows as `library!?`, and a name is only trusted inside its symbol's extent, because the nearest symbol in a stripped library is usually wrong. Figures name such libraries for what they are, for example `GPU driver`.

## `--cold-start`: a first load

```sh
llmperfscope run --cold-start -- llama-bench -m model.gguf -ngl 99
```

Drops the model from the page cache first, without root, using `posix_fadvise`. The load then reads from disk as a user's first run would. This one does not slow anything down, and the run stays usable for speed. Pass `--model-file` if llmperfscope cannot find the model in the command. Pages that another process has mapped cannot be dropped, and the report states how much of the model was still cached at the start.

## What it cannot see

- **GPU hardware counters**, such as cache hit rates or memory bandwidth per kernel, need `RmProfilingAdminOnly=0` or root. llmperfscope does not read them yet.
- **`perf`** needs `perf_event_paranoid` of 2 or less. `--cpu-profile` covers most of what it would answer, but not hardware events.
- **Host memory bandwidth** is not visible without privileges.
- **PCIe traffic** is a sample about once a second, not a count. `--nsys` gives the exact bytes of copies, and the difference is a collective going through host memory or a GPU polling it.

`llmperfscope check` lists all of this for your machine, and what it would take to enable the rest.
