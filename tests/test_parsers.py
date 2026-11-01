"""Parsers and decision rules, on inputs with known answers. Needs no GPU and starts no process."""

import json
import os
import sqlite3
import unittest

import helpers
from llmprof import adapters, analyze, compare, container, diagnose, driver, machine, markers, nsys, procfs, report, system

BENCH_JSON = """[
  {
    "n_prompt": 512,
    "n_gen": 0,
    "n_depth": 0,
    "avg_ts": 1000.0,
    "stddev_ts": 10.0,
    "samples_ns": [ 512000000, 512000000 ],
    "samples_ts": [ 1000.0, 1000.0 ]
  },
  {
    "n_prompt": 0,
    "n_gen": 128,
    "n_depth": 0,
    "avg_ts": 64.0,
    "stddev_ts": 1.0,
    "samples_ns": [ 2000000000, 2000000000 ],
    "samples_ts": [ 64.0, 64.0 ]
  }
]
"""


class Markers(unittest.TestCase):
    def test_lines(self):
        self.assertEqual(markers.parse_marker_line("load", 5), {"t": 5, "phase": "load"})
        self.assertEqual(markers.parse_marker_line("123 load done", 5), {"t": 123, "phase": "load done"})
        self.assertEqual(markers.parse_marker_line('{"phase": "x", "t": 9}', 5)["t"], 9)
        self.assertIsNone(markers.parse_marker_line("", 5))
        self.assertIsNone(markers.parse_marker_line("{not json", 5))

    def test_old_events_are_not_read_again(self):
        path = os.path.join(helpers.temp_dir(), "m")
        with open(path, "w") as f:
            f.write("old\n")
        reader = markers.MarkerReader(path, skip_existing=True)
        with open(path, "a") as f:
            f.write("new\n")
        self.assertEqual([e["phase"] for e in reader.poll()], ["new"])

    def test_reader_waits_for_whole_lines(self):
        path = os.path.join(helpers.temp_dir(), "m")
        reader = markers.MarkerReader(path)
        self.assertEqual(reader.poll(), [])
        with open(path, "a") as f:
            f.write("one\ntw")
        self.assertEqual([e["phase"] for e in reader.poll()], ["one"])
        with open(path, "a") as f:
            f.write("o\n")
        self.assertEqual([e["phase"] for e in reader.poll()], ["two"])


class Procfs(unittest.TestCase):
    def test_stat_with_spaces_and_parens_in_the_name(self):
        line = b"42 (a b) c)) S 1 42 42 0 -1 4194560 100 0 7 0 11 13 0 0 20 0 3 0 99 1000 10 1 2 3 4 5"
        st = procfs._parse_stat(line)
        self.assertEqual(st["comm"], "a b) c)")
        self.assertEqual((st["ppid"], st["minflt"], st["majflt"], st["utime"], st["stime"], st["threads"]), (1, 100, 7, 11, 13, 3))


class Adapters(unittest.TestCase):
    def test_finds_the_engine_behind_wrappers(self):
        cmd = ["timeout", "-k", "30", "600", "env", "A=B", "taskset", "-c", "16-23,80-87", "numactl", "--interleave=all", "/x/llama-bench", "-m", "y"]
        self.assertEqual(adapters._skip_wrappers(cmd), 11)
        self.assertEqual(adapters._skip_wrappers(["python3", "-c", "1"]), 0)

    def test_urls(self):
        self.assertEqual(adapters.guess_url(["llama-server", "-m", "x", "--port", "9000"], "llama-server"), "http://127.0.0.1:9000")
        self.assertEqual(adapters.guess_url(["llama-server", "-m", "x"], "llama-server"), "http://127.0.0.1:8080")
        self.assertEqual(adapters.guess_url(["vllm", "serve", "m"], "vllm"), "http://127.0.0.1:8000")
        docker = ["docker", "run", "--gpus", "all", "-p", "9001:8000", "vllm/vllm-openai", "--model", "m", "--port", "8000"]
        self.assertEqual(adapters.guess_url(docker, "vllm"), "http://127.0.0.1:9001")
        self.assertEqual(adapters.guess_url(["docker", "run", "-p", "127.0.0.1:9002:80/tcp", "tgi"], "tgi"), "http://127.0.0.1:9002")

    def test_presets(self):
        self.assertEqual(adapters.preset_for(["/a/llama-server", "-m", "x"]), "llama-server")
        self.assertEqual(adapters.preset_for(["vllm", "serve", "m"]), "vllm")
        self.assertEqual(adapters.preset_for(["python", "-m", "sglang.launch_server"]), "sglang")
        self.assertEqual(adapters.preset_for(["docker", "run", "ghcr.io/huggingface/text-generation-inference:3"]), "tgi")
        self.assertEqual(adapters.preset_for(["ollama", "serve"]), "ollama")
        self.assertEqual(adapters.preset_for(["my-server"]), "openai")

    def test_docker_index(self):
        self.assertEqual(adapters.docker_index(["docker", "run", "x"]), 0)
        self.assertIsNone(adapters.docker_index(["docker", "ps"]))
        self.assertIsNone(adapters.docker_index(["llama-bench"]))

    def test_bench_flags_are_added_once(self):
        adapter = adapters.select("auto", ["llama-bench", "-m", "x", "-o", "md"])
        self.assertIsInstance(adapter, adapters.LlamaBench)
        cmd, _ = adapter.prepare(adapter.cmd, {})
        self.assertEqual(cmd.count("--progress"), 1)
        self.assertEqual(cmd[cmd.index("-o") + 1], "json")

    def _bench_phases(self):
        adapter = adapters.select("auto", ["llama-bench", "-m", "x"])
        events = []

        def emit(phase, group=None, t=None, tokens=None, **meta):
            events.append({"name": phase, "group": group or phase, "t": t, "t_end": t + 1000, **({"meta": meta} if meta else {})})

        lines = ["benchmark 1/2: starting", "benchmark 1/2: warmup prompt run", "benchmark 1/2: prompt run 1/2", "benchmark 1/2: prompt run 2/2",
                 "benchmark 2/2: starting", "benchmark 2/2: warmup generation run", "benchmark 2/2: generation run 1/2", "benchmark 2/2: generation run 2/2"]
        for i, line in enumerate(lines):
            adapter.on_line("stderr", i * 1000, "llama-bench: " + line, emit)
        return adapter, events

    def test_bench_progress_lines_become_phases(self):
        _, events = self._bench_phases()
        self.assertEqual([e["name"] for e in events], ["load", "warmup_prefill", "prefill", "prefill", "setup", "warmup_decode", "decode", "decode"])
        self.assertEqual(events[3]["meta"], {"test": 1, "rep": 2})

    def test_bench_results_survive_text_around_them(self):
        adapter, events = self._bench_phases()
        path = os.path.join(helpers.temp_dir(), "out")
        with open(path, "w") as f:
            f.write("[1/8] nsys said something\n" + BENCH_JSON + "Generating report\n")
        extras = adapter.finalize(events, path, path)
        self.assertEqual(extras["throughput"]["prefill"]["tokens_per_s"], 1000.0)
        self.assertEqual(extras["throughput"]["decode"]["tokens_per_s"], 64.0)
        self.assertEqual([e.get("tokens") for e in events if e["group"] == "prefill"], [512, 512])
        self.assertEqual([e.get("tokens") for e in events if e["group"] == "decode"], [128, 128])

    def test_memory_table(self):
        row = adapters.parse_breakdown_line("x: | - CUDA0 (A6000) | 48000 = 10000 + (37000 = 30000 +   1000 +  6000) + 1000 |")
        self.assertEqual((row["name"], row["kind"], row["model"], row["compute"], row["unaccounted"]), ("CUDA0 (A6000)", "device", 30000, 6000, 1000))
        row = adapters.parse_breakdown_line("x: | - Host |    900 =   500 +    300 +    100 |")
        self.assertEqual((row["kind"], row["self"]), ("host", 900))
        self.assertIsNone(adapters.parse_breakdown_line("x: nothing here"))

    def test_memory_table_as_json(self):
        rows = adapters._breakdown_from_json({"type": "fit_memory_breakdown", "data": {"rows": [{"kind": "device", "name": "CUDA0", "self": 5}]}})
        self.assertEqual(rows[0]["self"], 5)
        rows = adapters._breakdown_from_json({"type": "fit_memory_breakdown", "rows": [{"kind": "host", "name": "Host", "self": 6}]})
        self.assertEqual(rows[0]["self"], 6)


class Containers(unittest.TestCase):
    def test_options_go_right_after_run(self):
        directory = helpers.temp_dir()
        target = container.DockerTarget("docker", directory)
        cmd = target.prepare(["taskset", "-c", "1", "docker", "run", "--gpus", "all", "img", "--flag"], 3, "x")
        self.assertEqual(cmd[:5], ["taskset", "-c", "1", "docker", "run"])
        self.assertEqual(cmd[5], "--cidfile")
        self.assertEqual(cmd[-4:], ["--gpus", "all", "img", "--flag"])
        self.assertIn(f"{os.path.realpath(directory)}:/llmprof-run", cmd)

    def test_identity_names_the_image(self):
        target = container.DockerTarget("docker", helpers.temp_dir())
        self.assertEqual(target.identity(), {})
        target.image = "vllm/vllm-openai:v1 sha256:abc"
        self.assertEqual(target.identity(), {"container_image": "vllm/vllm-openai:v1", "container_image_id": "sha256:abc"})


class Driver(unittest.TestCase):
    def test_percentiles(self):
        values = list(range(1, 101))
        self.assertEqual(driver.percentile(values, 50), 50)
        self.assertEqual(driver.percentile(values, 99), 99)
        self.assertEqual(driver.percentile([5], 99), 5)
        self.assertIsNone(driver.percentile([], 50))

    def test_prompt_filters(self):
        path = os.path.join(helpers.temp_dir(), "p.jsonl")
        with open(path, "w") as f:
            f.write(json.dumps({"prompt": "a", "tier": "standard"}) + "\n")
            f.write(json.dumps({"messages": [], "tier": "full"}) + "\n")
        self.assertEqual(len(driver.load_prompts(path)), 2)
        self.assertEqual(len(driver.load_prompts(path, {"tier": "standard"})), 1)

    def test_prometheus(self):
        before = '# HELP x\nreqs_total{a="b"} 5\nother 1\n'
        after = 'reqs_total{a="b"} 9\nother 1\nnew 2\n'
        self.assertEqual(driver.metrics_delta(before, after), {'reqs_total{a="b"}': 4.0, "new": 2.0})

    def test_usage_gauges_are_whole_words(self):
        for name in ("vllm:kv_cache_usage_perc", "llamacpp:requests_processing", 'vllm:num_requests_waiting{engine="0"}'):
            self.assertTrue(driver.GaugeWatcher.is_usage_gauge(name), name)
        for name in ("vllm:generation_tokens_created", "vllm:iteration_tokens_total", "vllm:cache_config_info", "process_cpu_seconds_total"):
            self.assertFalse(driver.GaugeWatcher.is_usage_gauge(name), name)

    def test_nonce_goes_in_front_without_changing_the_original(self):
        rows = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        self.assertEqual(driver._with_nonce(rows, "[1] ")[0]["content"], "[1] s")
        self.assertEqual(rows[0]["content"], "s")
        self.assertIs(driver._with_nonce(rows, ""), rows)

    def test_summary(self):
        results = [{"ok": True, "ttft_s": 0.5, "itl_s": [0.1, 0.1], "chunks": 3, "usage": {"completion_tokens": 3, "prompt_tokens": 100}},
                   {"ok": False, "error": "boom", "itl_s": [], "chunks": 0}]
        out = driver.summarize_load(results, 2.0)
        self.assertEqual((out["requests"], out["failed"], out["completion_tokens"], out["first_error"]), (2, 1, 3, "boom"))
        self.assertAlmostEqual(out["output_tokens_per_s"], 1.5)
        self.assertAlmostEqual(out["prefill_tokens_per_s_mean"], 200.0)
        self.assertAlmostEqual(out["decode_tokens_per_s_mean"], 10.0)
        self.assertTrue(out["tokens_exact"])


class Secrets(unittest.TestCase):
    def test_credentials_stay_out_of_the_report(self):
        self.assertEqual(machine.relevant_env({"CUDA_VISIBLE_DEVICES": "0", "HF_TOKEN": "x", "VLLM_API_KEY": "y", "PATH": "/bin"}), {"CUDA_VISIBLE_DEVICES": "0"})
        self.assertEqual(machine.redact_command(["s", "--api-key", "abc", "--hf-token=def", "--port", "1"]),
                         ["s", "--api-key", "<redacted>", "--hf-token=<redacted>", "--port", "1"])


class Compare(unittest.TestCase):
    def test_overlap_means_no_verdict(self):
        self.assertEqual(compare.judge("decode_tps", [100, 110], [105, 112])["verdict"], "same")

    def test_separated_arms_above_the_floor(self):
        self.assertEqual(compare.judge("decode_tps", [100, 101], [110, 111])["verdict"], "better")
        self.assertEqual(compare.judge("decode_tps", [110, 111], [100, 101])["verdict"], "worse")

    def test_tiny_separated_difference_is_not_a_change(self):
        self.assertEqual(compare.judge("decode_tps", [1000.0, 1000.1], [1000.5, 1000.6])["verdict"], "same")

    def test_noise_floor_is_respected(self):
        self.assertEqual(compare.judge("decode_tps", [100, 100.1], [101, 101.1])["verdict"], "better")
        self.assertEqual(compare.judge("decode_tps", [100, 100.1], [101, 101.1], noise=0.013)["verdict"], "same")

    def test_memory_direction(self):
        self.assertEqual(compare.judge("gpu_peak_mib", [1000, 1000], [2000, 2000])["verdict"], "worse")
        self.assertEqual(compare.judge("gpu_peak_mib", [2000, 2000], [1000, 1000])["verdict"], "better")
        self.assertEqual(compare.judge("gpu_peak_mib", [1000, 1000], [1008, 1008])["verdict"], "same")

    def test_failed_and_traced_runs_do_not_count(self):
        good = {"status": "ok", "throughput": {"decode": {"tokens_per_s": 100.0}}}
        bad = {"status": "failed", "throughput": {"decode": {"tokens_per_s": 999.0}}}
        traced = {"status": "ok", "perturbed": True, "throughput": {"decode": {"tokens_per_s": 999.0}}}
        out = compare.diff([good, bad], [good, traced])
        self.assertEqual(out["dropped_runs"], 2)
        self.assertEqual(out["metrics"]["decode_tps"]["ratio"], 1.0)

    def test_speed_win_with_memory_cost_is_called_a_tradeoff(self):
        a = {"status": "ok", "throughput": {"decode": {"tokens_per_s": 100.0}}, "memory": {"gpu": {"peak_above_baseline_total_mib": 1000}}}
        b = {"status": "ok", "throughput": {"decode": {"tokens_per_s": 120.0}}, "memory": {"gpu": {"peak_above_baseline_total_mib": 2000}}}
        out = compare.diff([a], [b])
        self.assertEqual(out["verdict"], "tradeoff")
        self.assertIn("gpu_peak_mib", out["worse"])
        self.assertIn("decode_tps", out["better"])

    def test_only_a_cost_is_a_regression(self):
        a = {"status": "ok", "throughput": {"decode": {"tokens_per_s": 100.0}}}
        b = {"status": "ok", "throughput": {"decode": {"tokens_per_s": 80.0}}}
        self.assertEqual(compare.diff([a], [b])["verdict"], "regression")

    def test_noise_from_null_matrix(self):
        path = os.path.join(helpers.temp_dir(), "m.json")
        with open(path, "w") as f:
            json.dump({"cases": [{"label": "c", "metrics": {"decode_tps": {"a_runs": [100, 101], "b_runs": [102, 100]}}}]}, f)
        self.assertAlmostEqual(compare.noise_from_null(path)["c"]["decode_tps"], 2 / 100.75, places=4)


def _ticks(duration_s=10, hz=10, watts=100.0):
    """One GPU at constant power, memory stepping up, host counters growing evenly."""
    ticks = []
    for i in range(duration_s * hz + 1):
        t = int(i * 1e9 / hz)
        seconds = i / hz
        mem = 100 if seconds < 2 else 1100 if seconds < 8 else 600
        tick = {"t": t,
                "g": [[mem, watts, int(watts * seconds * 1000), 1800, 7000, 90, 10, 0, 50]],
                "h": [100 * 1024, 200 * 1024, 0, 300 * 1024, 0, 0, 4, 1, int(seconds * 100), int(seconds * 50), i, 0]}
        if i % hz == 0:
            tick["s"] = {"self_mib": [mem - 100], "foreign_mib": [0],
                         "io": [int(seconds) * (1 << 20), 0, 0, 0],
                         "sys": {"disk": {"sda": [int(seconds) * 2048, 0, int(seconds) * 500, 0, 0]},
                                 "vm": {"pswpin": int(seconds) * 256, "pswpout": 0, "workingset_refault_file": 0, "pgscan_direct": 0},
                                 "mem": {"MemAvailable": 1000 - int(seconds), "Cached": 500, "SwapFree": 100, "SwapTotal": 200},
                                 "psi": {"io_some": int(seconds * 0.25 * 1e6), "memory_some": 0, "memory_full": 0, "cpu_some": 0},
                                 "cpu": [int(seconds) * 100, 0, 0, int(seconds) * 300, 0, 0, 0, 0],
                                 "net": [0, 0]},
                         "pcie": [[2048, 1024]],
                         "threads": {"7": ["main", int(seconds * 100)], "8": ["cuda-EvtHandlr", int(seconds * 10)]}}
        ticks.append(tick)
    return ticks


class Analyze(unittest.TestCase):
    def setUp(self):
        self.phases = [{"name": "baseline", "group": "baseline", "t": 0},
                       {"name": "load", "group": "load", "t": int(1e9)},
                       {"name": "decode", "group": "decode", "t": int(3e9), "tokens": 500},
                       {"name": "decode", "group": "decode", "t": int(5e9), "tokens": 500},
                       {"name": "tail", "group": "tail", "t": int(8e9)}]
        self.out = analyze.summarize(_ticks(), self.phases, int(10e9))

    def test_energy_is_power_times_time(self):
        decode = self.out["groups"]["decode"]
        self.assertAlmostEqual(decode["wall_s"], 5.0, places=3)
        self.assertAlmostEqual(decode["gpu"]["energy_j"], 500.0, delta=1.0)
        self.assertAlmostEqual(decode["gpu"]["power_avg_w"], 100.0, delta=0.5)
        self.assertAlmostEqual(self.out["energy"]["decode_j_per_token"], 0.5, places=2)

    def test_energy_counter_is_checked_against_the_gauge(self):
        self.assertAlmostEqual(analyze.energy_self_check(self.out["groups"]), 1.0, delta=0.02)
        # What the counter really did when read at 50 Hz.
        groups = {"decode": {"wall_s": 4.35, "gpu": {"power_avg_w": 818.0, "power_gauge_w": 1395.0}}}
        self.assertAlmostEqual(analyze.energy_self_check(groups), 0.586, places=3)
        self.assertIsNone(analyze.energy_self_check({"decode": {"wall_s": 0.5, "gpu": {"power_avg_w": 1.0, "power_gauge_w": 2.0}}}))

    def test_memory_is_above_the_idle_baseline(self):
        self.assertEqual(self.out["memory"]["gpu"]["devices"][0]["baseline_mib"], 100)
        self.assertEqual(self.out["memory"]["gpu"]["peak_above_baseline_total_mib"], 1000)
        self.assertEqual(self.out["groups"]["decode"]["gpu"]["mem_peak_mib"], 1000)

    def test_a_process_already_running_keeps_its_memory(self):
        # Attached: the GPU holds 1100 MiB from the first sample on, of which the process's own is 1000.
        ticks = _ticks()
        for tick in ticks:
            tick["g"][0][0] = 1100
            if "s" in tick:
                tick["s"]["self_mib"] = [1000 if tick["t"] >= int(1e9) else 0]
        phases = [{"name": "baseline", "group": "baseline", "t": 0}, {"name": "observe", "group": "observe", "t": int(1e9)}]
        out = analyze.summarize(ticks, phases, int(10e9), attached=True)
        self.assertEqual(out["memory"]["gpu"]["devices"][0]["baseline_mib"], 100)
        self.assertEqual(out["memory"]["gpu"]["peak_above_baseline_total_mib"], 1000)

    def test_cpu_and_threads(self):
        decode = self.out["groups"]["decode"]["host"]
        self.assertAlmostEqual(decode["user_s"], 5.0, delta=0.05)
        self.assertAlmostEqual(decode["sys_s"], 2.5, delta=0.05)
        self.assertEqual(decode["busiest_thread"]["name"], "main")
        self.assertGreater(decode["busiest_thread"]["busy"], 0.9)

    def test_data_movement(self):
        decode = self.out["groups"]["decode"]
        flows = decode["flows"]
        self.assertAlmostEqual(decode["host"]["disk_read_mib"], 5.0, delta=0.1)
        self.assertAlmostEqual(flows["disks"]["sda"]["read_mib"], 5.0, delta=0.1)
        self.assertAlmostEqual(flows["disks"]["sda"]["busy_pct"], 50.0, delta=1.0)
        self.assertAlmostEqual(flows["swap_in_mib"], 5.0, delta=0.1)
        self.assertAlmostEqual(flows["pcie_to_gpu_mib_est"], 5.0, delta=0.5)
        self.assertAlmostEqual(flows["pcie_from_gpu_mib_est"], 10.0, delta=1.0)
        self.assertAlmostEqual(decode["pressure"]["io_some_pct"], 25.0, delta=0.5)
        self.assertAlmostEqual(decode["system_cpu"]["busy_pct"], 25.0, delta=0.5)


class Bottlenecks(unittest.TestCase):
    run_info = {"disks": {"sda": {"kind": "hdd"}, "nvme0n1": {"kind": "nvme"}}, "host": {"cpus": 16}}

    def limit(self, **kwargs):
        return diagnose.classify_phase("decode", helpers.group(**kwargs), self.run_info)["limit"]

    def test_swap(self):
        self.assertEqual(self.limit(flows={"swap_in_mib": 100.0}, gpu={"util_avg": 99}), "ram")

    def test_memory_stall(self):
        self.assertEqual(self.limit(pressure={"memory_full_pct": 20.0}), "ram")

    def test_page_cache_thrashing_is_a_ram_limit_and_says_why(self):
        # Measured: a 450 MiB limit on a 639 MiB mmap model, decode fell from 48.7 to 2.3 tokens/s.
        result = diagnose.classify_phase("decode", helpers.group(flows={"page_cache_refault_mib": 10000.0}, pressure={"memory_some_pct": 9.0}), self.run_info)
        self.assertEqual(result["limit"], "ram")
        self.assertIn("read again", result["why"])
        self.assertNotIn("0 MiB/s of swap", result["why"])

    def test_disk_names_the_kind(self):
        result = self.limit(host={"disk_read_mib": 500.0}, flows={"disks": {"sda": {"read_mib": 500.0, "busy_pct": 90.0}}}, gpu={"util_avg": 10})
        self.assertEqual(result, "disk (hdd)")
        result = self.limit(host={"disk_read_mib": 500.0}, flows={"disks": {"nvme0n1": {"read_mib": 500.0, "busy_pct": 10.0}}}, pressure={"io_some_pct": 40.0})
        self.assertEqual(result, "disk (nvme)")

    def test_someone_elses_disk_traffic_is_not_our_bottleneck(self):
        result = self.limit(host={"disk_read_mib": 0.0}, flows={"disks": {"sda": {"read_mib": 500.0, "busy_pct": 90.0}}}, gpu={"util_avg": 95, "util_min_device": 95, "util_max_device": 95})
        self.assertEqual(result, "gpu")

    def test_gpu(self):
        self.assertEqual(self.limit(gpu={"util_avg": 96, "util_min_device": 95, "util_max_device": 97}), "gpu")

    def test_throttled(self):
        self.assertEqual(self.limit(gpu={"util_avg": 96, "throttled": True, "throttle": ["hw_thermal"]}), "gpu (throttled)")

    def test_power_cap_bit_far_below_the_limit_is_not_throttling(self):
        # Seen on real GPUs: the bit was set at 120 W of a 300 W limit.
        info = {**self.run_info, "devices_used": [0], "devices": [{"index": 0, "power_limit_w": 300.0}]}
        gpu = {"util_avg": 96, "util_min_device": 96, "util_max_device": 96, "throttled": True, "throttle": ["sw_power_cap"], "power_avg_w": 120.0}
        self.assertEqual(diagnose.classify_phase("decode", helpers.group(gpu=gpu), info)["limit"], "gpu")
        gpu["power_avg_w"] = 295.0
        self.assertEqual(diagnose.classify_phase("decode", helpers.group(gpu=gpu), info)["limit"], "gpu (throttled)")

    def test_imbalance(self):
        self.assertEqual(self.limit(gpu={"util_avg": 70, "util_min_device": 40, "util_max_device": 99}), "gpu imbalance")

    def test_cpu_many_threads(self):
        self.assertEqual(self.limit(gpu={"util_avg": 7}, host={"busy_cores": 8.0, "busiest_thread": {"name": "x", "busy": 1.0}}), "cpu (many threads)")

    def test_cpu_one_thread(self):
        self.assertEqual(self.limit(gpu={"util_avg": 2}, host={"busy_cores": 1.0, "busiest_thread": {"name": "x", "busy": 0.94}}), "cpu (one thread)")

    def test_spinning_host_with_a_busy_gpu_is_not_called_a_cpu_limit(self):
        self.assertEqual(self.limit(gpu={"util_avg": 46}, host={"busy_cores": 1.0, "busiest_thread": {"name": "x", "busy": 0.89}}), "gpu + host wait")
        self.assertEqual(self.limit(gpu={"util_avg": 75}, host={"busy_cores": 1.0, "busiest_thread": {"name": "x", "busy": 1.0}}), "gpu + host wait")

    def test_host_work_without_a_saturated_thread(self):
        self.assertEqual(self.limit(gpu={"util_avg": 6}, host={"busy_cores": 1.2, "busiest_thread": {"name": "x", "busy": 0.5}}), "cpu")

    def test_nothing_saturated(self):
        self.assertEqual(self.limit(gpu={"util_avg": 5}, host={"busy_cores": 0.1}), "none visible")

    def test_contention(self):
        self.assertEqual(self.limit(gpu={"util_avg": 30}, pressure={"cpu_some_pct": 50.0}, host={"busy_cores": 1.0}), "cpu contention")

    def test_short_phase_is_not_judged(self):
        result = diagnose.classify_phase("prefill", helpers.group(wall_s=0.2, gpu={"util_avg": 3}), self.run_info)
        self.assertEqual(result["confidence"], "none")

    def test_short_phase_still_reports_a_hard_signal(self):
        result = diagnose.classify_phase("load", helpers.group(wall_s=0.5, host={"disk_read_mib": 400.0}, pressure={"io_some_pct": 50.0}), self.run_info)
        self.assertEqual(result["limit"], "disk (disk)")


class MemoryFindings(unittest.TestCase):
    def names(self, run):
        return {f["what"] for f in diagnose.memory_findings(run)}

    def test_nearly_full(self):
        run = {"devices": [{"index": 0, "memory_total_mib": 1000}],
               "memory": {"gpu": {"devices": [{"index": 0, "peak_mib": 990, "peak_above_baseline_mib": 900, "baseline_mib": 90}]}}}
        self.assertIn("gpu_memory_nearly_full", self.names(run))
        run["memory"]["gpu"]["devices"][0]["peak_mib"] = 500
        self.assertNotIn("gpu_memory_nearly_full", self.names(run))

    def test_uneven_gpus(self):
        run = {"memory": {"gpu": {"devices": [{"index": 0, "peak_mib": 2000, "peak_above_baseline_mib": 2000, "baseline_mib": 0},
                                              {"index": 1, "peak_mib": 1000, "peak_above_baseline_mib": 1000, "baseline_mib": 0}]}}}
        self.assertIn("gpu_memory_uneven", self.names(run))

    def test_transient_peak(self):
        run = {"groups": {"warmup": {"gpu": {"mem_peak_mib": 5000}}, "decode": {"gpu": {"mem_peak_mib": 4000, "mem_end_mib": 4000}}}}
        finding = [f for f in diagnose.memory_findings(run) if f["what"] == "gpu_transient_peak"][0]
        self.assertEqual(finding["evidence"]["phase"], "warmup")

    def test_host_growth_looks_like_a_leak(self):
        run = {"groups": {"serve": {"host": {"anon_start_mib": 1000, "anon_end_mib": 2000}}}}
        self.assertIn("host_memory_growing", self.names(run))
        run = {"groups": {"serve": {"host": {"anon_start_mib": 1000, "anon_end_mib": 1010}}}}
        self.assertNotIn("host_memory_growing", self.names(run))

    def test_swap_and_ram(self):
        run = {"memory": {"host": {"swap_peak_mib": 500}}, "machine_before": {"mem_total_mib": 1000},
               "groups": {"decode": {"flows": {"mem_available_min_mib": 10}}}}
        self.assertEqual(self.names(run) & {"process_swapped", "host_ram_nearly_full"}, {"process_swapped", "host_ram_nearly_full"})

    def test_engine_breakdown(self):
        run = {"memory": {"breakdown_mib": [{"kind": "device", "name": "G", "self": 1000, "compute": 500, "unaccounted": 400}]}}
        self.assertEqual(self.names(run) & {"gpu_memory_unaccounted", "compute_buffers_large"}, {"gpu_memory_unaccounted", "compute_buffers_large"})

    def test_tensor_split_table_is_not_read_as_missing_memory(self):
        # Real row from 8 GPUs: the total is the sum over them, the buffers one GPU's share.
        row = {"kind": "device", "name": "Meta()", "total": 363886, "free": 63648, "self": 36718, "model": 36151, "context": 220, "compute": 346, "unaccounted": 263519}
        self.assertNotIn("gpu_memory_unaccounted", self.names({"memory": {"breakdown_mib": [row]}}))

    def test_weights_left_on_the_host(self):
        # Real row: 200 GiB of a 473 GiB model stayed in host buffers.
        run = {"memory": {"gpu": {"devices": [{"index": 0, "peak_mib": 38000, "peak_above_baseline_mib": 37000, "baseline_mib": 583}]},
                          "breakdown_mib": [{"kind": "host", "name": "Host", "self": 200546, "model": 200487}]}}
        self.assertIn("weights_on_host", self.names(run))

    def test_torch_allocator_cache(self):
        run = {"torch_memory": {"max_reserved_mib": 8000, "max_allocated_mib": 3000}}
        self.assertIn("torch_reserved_above_allocated", self.names(run))

    def test_reserved_cache_pool(self):
        run = {"workload": {"server_gauges_peak": {"vllm:kv_cache_usage_perc": 0.03}}}
        self.assertIn("kv_cache_mostly_unused", self.names(run))
        run = {"workload": {"server_gauges_peak": {"vllm:kv_cache_usage_perc": 0.99}}}
        self.assertIn("kv_cache_full", self.names(run))


class MovementFindings(unittest.TestCase):
    def names(self, run):
        return {f["what"] for f in diagnose.movement_findings(run)}

    def test_model_on_slow_storage(self):
        for location, expected in (({"path": "x", "disks": [{"kind": "hdd"}]}, True), ({"path": "x", "network": True}, True), ({"path": "x", "disks": [{"kind": "nvme"}]}, False)):
            run = {"machine_before": {"model_storage": location}}
            self.assertEqual("model_on_slow_storage" in self.names(run), expected)

    def test_fast_array_of_rotating_disks_is_not_a_warning(self):
        # Seen on a 50 TB RAID: rotational according to the kernel, 1.5 GB/s in fact.
        location = {"path": "x", "disks": [{"kind": "hdd"}]}
        run = {"machine_before": {"model_storage": location}, "groups": {"load": {"wall_s": 3.0, "host": {"disk_read_mib": 4700.0}}}}
        finding = [f for f in diagnose.movement_findings(run) if f["what"] == "model_on_slow_storage"][0]
        self.assertEqual(finding["severity"], "info")
        run = {"machine_before": {"model_storage": location}, "groups": {"load": {"wall_s": 30.0, "host": {"disk_read_mib": 4700.0}}}}
        finding = [f for f in diagnose.movement_findings(run) if f["what"] == "model_on_slow_storage"][0]
        self.assertEqual(finding["severity"], "warning")

    def test_disk_reads_during_decode(self):
        run = {"groups": {"decode": {"wall_s": 10, "host": {"disk_read_mib": 500.0, "major_faults": 9000}}}}
        self.assertIn("disk_reads_during_inference", self.names(run))
        run = {"groups": {"decode": {"wall_s": 10, "host": {"disk_read_mib": 0.0, "major_faults": 0}}}}
        self.assertNotIn("disk_reads_during_inference", self.names(run))

    def test_swap_during_decode(self):
        run = {"groups": {"decode": {"wall_s": 10, "flows": {"swap_in_mib": 100.0, "swap_out_mib": 0.0}}}}
        self.assertIn("swapping_during_inference", self.names(run))

    def test_ram_gpu_shuttling(self):
        flows = {"pcie_to_gpu_mib_est": 500.0, "pcie_from_gpu_mib_est": 100.0}
        run = {"groups": {"decode": {"wall_s": 10, "tokens": 10, "flows": flows, "gpu": {"util_avg": 50}}}}
        self.assertIn("heavy_host_gpu_traffic", self.names(run))
        run = {"groups": {"decode": {"wall_s": 10, "tokens": 1000, "flows": flows, "gpu": {"util_avg": 50}}}}
        self.assertNotIn("heavy_host_gpu_traffic", self.names(run))

    def test_a_sampled_rate_says_nothing_about_a_short_or_idle_phase(self):
        flows = {"pcie_to_gpu_mib_est": 500.0, "pcie_from_gpu_mib_est": 100.0}
        run = {"groups": {"decode": {"wall_s": 0.6, "tokens": 10, "flows": flows, "gpu": {"util_avg": 50}}}}
        self.assertNotIn("heavy_host_gpu_traffic", self.names(run))
        run = {"groups": {"decode": {"wall_s": 10, "tokens": 10, "flows": flows, "gpu": {"util_avg": 0}}}}
        self.assertNotIn("heavy_host_gpu_traffic", self.names(run))

    def test_cpu_gpu_split(self):
        run = {"groups": {"decode": {"wall_s": 10, "gpu": {"util_avg": 10}, "host": {"busy_cores": 8.0}}}}
        self.assertIn("cpu_gpu_split", self.names(run))

    def test_cold_and_warm_load(self):
        cold = {"machine_before": {"model_page_cache_fraction": 0.0}, "groups": {"load": {"wall_s": 10, "host": {"disk_read_mib": 4000.0}}}}
        self.assertIn("load_read_from_disk", self.names(cold))
        warm = {"machine_before": {"model_page_cache_fraction": 1.0}, "groups": {"load": {"wall_s": 1, "host": {"disk_read_mib": 0.0}}}}
        self.assertIn("load_from_page_cache", self.names(warm))


class Serving(unittest.TestCase):
    def names(self, gain, failed=0):
        run = {"throughput": {"serve_c1": {"concurrency": 1, "output_tokens_per_s": 100.0, "ttft_ms": {"p50": 20.0}, "requests": 8, "failed": 0},
                              "serve_c4": {"concurrency": 4, "output_tokens_per_s": 100.0 * gain, "ttft_ms": {"p50": 500.0}, "requests": 8, "failed": failed}}}
        return {f["what"] for f in diagnose.serving_findings(run)}

    def test_a_server_that_handles_one_request_at_a_time(self):
        # Ollama with its default setting: 371 tokens/s at 4 clients against 368 at 1.
        self.assertIn("no_parallel_scaling", self.names(1.01))

    def test_a_server_that_scales(self):
        self.assertEqual(self.names(3.1), {"parallel_scaling"})

    def test_failed_requests(self):
        self.assertIn("failed_requests", self.names(3.1, failed=2))

    def test_one_finding_per_family_of_gauges(self):
        run = {"workload": {"server_gauges_peak": {"s:token_usage": 0.01, "s:full_token_usage": 0.01, "s:swa_token_usage": 0.0}}}
        self.assertEqual(len([f for f in diagnose.memory_findings(run) if f["what"] == "kv_cache_mostly_unused"]), 1)


class ServeLevels(unittest.TestCase):
    def test_each_load_level_gets_a_verdict(self):
        busy = helpers.group(gpu={"util_avg": 96, "util_min_device": 96, "util_max_device": 96})
        idle = helpers.group(gpu={"util_avg": 5}, host={"busy_cores": 0.1})
        run = {"groups": {"serve": busy}, "phases": [{"group": "serve", "name": "serve_c1", "metrics": idle}, {"group": "serve", "name": "serve_c8", "metrics": busy}]}
        out = diagnose.diagnose(run)["bottlenecks"]
        self.assertEqual(set(out), {"serve_c1", "serve_c8"})
        self.assertEqual(out["serve_c1"]["limit"], "client or idle")
        self.assertEqual(out["serve_c8"]["limit"], "gpu")


class Storage(unittest.TestCase):
    def test_slowest_member_decides(self):
        self.assertEqual(system.storage_kind({"disks": [{"kind": "nvme"}, {"kind": "hdd"}]}), "hdd")
        self.assertEqual(system.storage_kind({"disks": [{"kind": "ssd"}]}), "ssd")
        self.assertEqual(system.storage_kind({"network": True}), "network")
        self.assertEqual(system.storage_kind({"in_ram": True}), "ram")
        self.assertIsNone(system.storage_kind(None))

    @unittest.skipUnless(os.path.exists("/proc/self/mountinfo"), "Linux only")
    def test_file_location_of_a_real_file(self):
        path = os.path.join(helpers.temp_dir(), "f")
        with open(path, "wb") as f:
            f.write(b"x")
        location = system.file_location(path)
        self.assertEqual(location["path"], os.path.realpath(path))
        self.assertIn("fstype", location)

    @unittest.skipUnless(os.path.exists("/proc/diskstats"), "Linux only")
    def test_system_counters_read_on_this_machine(self):
        sampler = system.SystemSampler()
        row = sampler.read()
        self.assertIn("MemAvailable", row["mem"])
        for values in row["disk"].values():
            self.assertEqual(len(values), 5)


class Nsys(unittest.TestCase):
    """A synthetic trace with exactly known kernels, gaps and copies."""

    def setUp(self):
        self.path = os.path.join(helpers.temp_dir(), "t.sqlite")
        db = sqlite3.connect(self.path)
        db.execute("create table StringIds (id integer, value text)")
        db.executemany("insert into StringIds values (?, ?)", [(1, "mul_mat<float>(int)"), (2, "ncclDevKernel_AllReduce"), (3, "cuLaunchKernel_v2")])
        db.execute("create table CUPTI_ACTIVITY_KIND_KERNEL (deviceId int, start int, end int, shortName int)")
        ms = 1_000_000
        # Device 0, second 1 (decode): 3 kernels of 2 ms, then a 1 ms collective, a 2 ms gap between each.
        rows = []
        t = 1000 * ms
        for _ in range(3):
            rows.append((0, t, t + 2 * ms, 1))
            t += 4 * ms
        rows.append((0, t, t + 1 * ms, 2))
        db.executemany("insert into CUPTI_ACTIVITY_KIND_KERNEL values (?,?,?,?)", rows)
        db.execute("create table CUPTI_ACTIVITY_KIND_MEMCPY (deviceId int, start int, end int, bytes int, copyKind int)")
        db.execute("insert into CUPTI_ACTIVITY_KIND_MEMCPY values (0, ?, ?, ?, 2)", (t + 2 * ms, t + 3 * ms, 2 << 20))
        db.execute("create table ENUM_CUDA_MEMCPY_OPER (id int, name text, label text)")
        db.execute("insert into ENUM_CUDA_MEMCPY_OPER values (2, 'x', 'Device-to-Host')")
        db.execute("create table CUPTI_ACTIVITY_KIND_RUNTIME (globalTid int, start int, end int, nameId int)")
        db.execute("insert into CUPTI_ACTIVITY_KIND_RUNTIME values (5, ?, ?, 3)", (1000 * ms, 1010 * ms))
        db.execute("create table TARGET_INFO_SESSION_START_TIME (utcEpochNs int)")
        db.execute("insert into TARGET_INFO_SESSION_START_TIME values (5000000000)")
        db.commit()
        db.close()

    def test_time_split_and_exact_copies(self):
        # The session started at wall 5 s; t0 is wall 5 s, so decode at start_s 0.9 covers trace time 900 ms onward.
        phases = [{"name": "load", "group": "load", "start_s": 0.0, "end_s": 0.9},
                  {"name": "decode", "group": "decode", "start_s": 0.9, "end_s": 2.0, "tokens": 2}]
        out = nsys.analyze(self.path, phases, 5_000_000_000)
        group = out["groups"]["decode"]
        device = group["devices"][0]
        self.assertEqual(device["kernel_ms"], 6.0)
        self.assertEqual(device["comm_ms"], 1.0)
        self.assertEqual(device["copy_ms"], 1.0)
        self.assertEqual(device["idle_ms"], 7.0)
        self.assertEqual(device["kernels"], 3)
        self.assertEqual(group["per_token_ms"]["kernel_ms"], 3.0)
        self.assertEqual(group["copies"]["Device-to-Host"]["mib"], 2.0)
        self.assertEqual(group["copies"]["Device-to-Host"]["mib_per_token"], 1.0)
        self.assertEqual(group["top_kernels"][0]["name"], "mul_mat")
        self.assertEqual(group["host_api"]["top"][0]["ms"], 10.0)

    def test_without_phases_the_trace_is_one_group(self):
        self.assertIn("all", nsys.analyze(self.path)["groups"])


class Rendering(unittest.TestCase):
    def test_empty_and_partial_reports_do_not_crash(self):
        self.assertIn("ok", report.render({"status": "ok"}))
        out = analyze.summarize(_ticks(), [{"name": "baseline", "group": "baseline", "t": 0}, {"name": "decode", "group": "decode", "t": int(1e9), "tokens": 10}], int(10e9))
        run = {"status": "ok", "label": "x", **out, "disks": {"sda": {"kind": "hdd"}}, "throughput": {"decode": {"tokens_per_s": 5.0, "stddev": 0.1}}}
        run["diagnosis"] = diagnose.diagnose(run)
        text = report.render(run)
        for needle in ("What limits each phase", "Lifecycle", "Data movement", "sda: hdd"):
            self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
