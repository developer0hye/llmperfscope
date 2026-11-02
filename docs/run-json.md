# What a run directory holds

Everything in a report and a figure comes from these files, so you can compute your own numbers from them.

| File | Content |
|---|---|
| `run.json` | the report as data, described below |
| `samples.jsonl.gz` | the raw samples, one JSON object per line |
| `requests.jsonl` | one row per request, with `serve` and `attach --url` |
| `stdout.log`, `stderr.log` | the target's output |
| `markers.log` | phase markers the target wrote |
| `report.html` | the charts |
| `figures/` | from `llmperfscope figure` |

## `run.json`

```python
import json

with open("runs/first/run.json") as f:
    run = json.load(f)

print(run["status"], run["groups"]["decode"]["wall_s"])
print(run["throughput"]["decode"]["tokens_per_s"])
print(run["diagnosis"]["bottlenecks"]["decode"]["limit"])
```

The format is identified by `schema`, currently `llmperfscope.run/1`. New keys can appear without a version change. Any key may be missing when a run has no data for it, so read them with care.

### What ran, and on what

| Key | Content |
|---|---|
| `schema`, `tool` | format and tool version |
| `label`, `status`, `stop_reason`, `exit_code` | how it ended. `status` is `ok`, `failed`, `timeout`, `stalled`, `interrupted` or `workload_failed` |
| `perturbed` | true with `--nsys` or `--cpu-profile`. Do not read speed from this run |
| `attached` | true with `attach` |
| `command`, `env` | what ran, and the variables that change engine behavior. Anything that looks like a credential is left out |
| `identity` | hash of the binary, git commit and dirty file count, or the container image |
| `adapter` | which adapter read the engine's output |
| `host`, `platform` | machine, kernel, CPUs, governor, transparent huge pages, Python version |
| `cpu` | totals over the run: user and system CPU seconds, major faults, context switches |
| `devices`, `devices_used`, `gpu_driver` | every GPU, and those the target held memory on or, on a Mac, used. On Apple Silicon `unified_memory` is true, and `memory_total_mib` is what macOS recommends the GPU hold, which `memory_total_is_working_set_limit` marks |
| `disks` | model and kind per disk: `nvme`, `ssd`, `hdd` |
| `machine_before`, `machine_after` | foreign load, available RAM, page cache, how much of the model was cached, where it is stored, `clean` and the `reasons` if not |
| `sampler` | achieved rate, largest gap, its own CPU time and pinned CPU |
| `consistency` | self-checks: `energy_over_power_gauge`, `cpu_profile_coverage`, and engine timers against phase times |
| `notes` | anything else a reader should know |

### Time

`phases` is the list of phases in order. Times are seconds from the start of the baseline. Each phase has `name`, `group`, `start_s`, `end_s`, often `tokens`, and `metrics` with the same content as a group below.

`groups` sums phases of one group, such as all `decode` phases over the repetitions. This is what the report tables show:

```
groups.decode
  wall_s, tokens, phases
  gpu       mem_start_mib, mem_end_mib, mem_peak_mib, mem_peak_max_device_mib,
            energy_j, j_per_token, power_avg_w, power_gauge_w,
            sm_mhz_avg, sm_mhz_min, util_avg, util_min_device, util_max_device, util_mem_avg,
            temp_c_max, throttled, throttle
  host      anon_start_mib, anon_end_mib, anon_peak_mib, file_start_mib, file_end_mib, file_peak_mib,
            shmem_peak_mib, pinned_peak_mib, swap_peak_mib, threads_max, user_s, sys_s, busy_cores,
            major_faults, disk_read_mib, disk_write_mib, threads_cpu_s, busiest_thread
  flows     disks.<name>.{read_mib, write_mib, busy_pct}, swap_in_mib, swap_out_mib,
            compressed_mib, decompressed_mib (macOS),
            page_cache_refault_mib, direct_reclaim_mib, page_cache_change_mib, network_rx_mib, network_tx_mib,
            pcie_to_gpu_mib_est, pcie_from_gpu_mib_est, nvlink_mib, mem_available_min_mib, swap_used_max_mib
  pressure  cpu_some_pct, io_some_pct, io_full_pct, memory_some_pct, memory_full_pct
  system_cpu busy_pct, iowait_pct, steal_pct
```

GPU figures cover the GPUs in `devices_used`, and memory is above the idle baseline. `host` covers the target's process tree. `flows` and `pressure` cover the whole machine, except for `pcie_*` and `nvlink_mib`, which cover used GPUs.

### Results

| Key | Content |
|---|---|
| `throughput.prefill`, `.decode` | `tokens_per_s`, `stddev`, `samples`, `tokens_per_run`. With `serve`, from the server's own timers at the lowest concurrency, with `real_prompts` set |
| `throughput.serve_c4` (one per level) | `concurrency`, `requests`, `failed`, `output_tokens_per_s`, `server_prefill_tokens_per_s`, `server_decode_tokens_per_s`, `ttft_ms` and `itl_ms` with `mean`, `p50`, `p90`, `p99`, `max`, and `j_per_output_token` |
| `memory.gpu` | `devices` with baseline, peak, process peak and what stays after exit, and `peak_above_baseline_total_mib` |
| `memory.host` | anonymous, file-backed, shared, pinned and swap peaks, kernel max RSS |
| `memory.breakdown_mib` | the engine's own table, where it has one |
| `energy` | `total_j`, `prefill_j_per_token`, `decode_j_per_token`, `serve_j_per_token` |
| `workload` | with `serve`: model, prompts, readiness time, server gauges and metrics |
| `torch_memory` | PyTorch's allocated and reserved peaks, from markers |
| `nsys` | with `--nsys`, per group: `devices`, `top_kernels`, `copies`, `host_api`, `per_token_ms` |
| `cpu_profile` | with `--cpu-profile`, per group: `samples`, `self`, `inclusive`, `modules`, `threads` |

### Diagnosis

```
diagnosis
  bottlenecks.<phase>  limit, confidence, why, evidence
  memory[], movement[], serving[], timeline[]  severity (info | warning), what, text, evidence
  next_steps[]
```

`what` is a stable name, such as `disk_reads_during_inference`, listed in [reading-reports.md](reading-reports.md). Match on it rather than on the text. `evidence` holds the numbers the statement rests on.

## `samples.jsonl.gz`

One tick per line:

```json
{"t": 1160007421812320, "g": [[...], [...]], "h": [...], "s": {...}}
```

- `t`: `CLOCK_MONOTONIC` in nanoseconds.
- `g`: one row per GPU, about 50 times a second: memory MiB, power W, energy counter, core clock, memory clock, utilization, memory utilization, throttle bits, temperature. The columns are `G_*` in `gpu.py`. The energy column is empty here, because the counter is slow to read.
- `h`: the target's process tree: anonymous, file-backed, shared, high-water mark, swap, pinned (MiB), threads, processes, user and system ticks, major and minor faults. The columns are `H_*` in `sampler.py`. Counters only grow: a process that exited keeps its last values.
- `s`: on slow ticks, about twice a second: `energy`, `pcie`, `nvlink`, `util` (the driver's utilization history, or on a Mac the target's share of GPU time), `gpu_ns` (on a Mac, the target's cumulative GPU time per GPU), `self_mib`, `foreign_mib`, and `sys` with system-wide `disk`, `vm`, `mem`, `psi`, `cpu` and `net` counters.

```python
from llmperfscope.sampler import load_ticks

ticks = load_ticks("runs/first/samples.jsonl.gz")
```

## `requests.jsonl`

One row per request, as a streaming client saw it: `ok`, `level` (concurrency), `ttft_s`, `itl_s` (time between tokens, each one), `chunks`, `usage` and, when the server gives them, `server_timings`. Failed requests have `error`.
