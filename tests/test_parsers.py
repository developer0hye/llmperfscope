"""Parsers and decision rules, on inputs with known answers. Needs no GPU and starts no process."""

import json
import os
import sqlite3
import unittest
from unittest import mock

import helpers
from llmperfscope import adapters, analyze, compare, container, cpuprof, diagnose, driver, kernels, machine, markers, nsys, procfs, report, serving, system

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
    def test_mlx_generate_says_how_fast_it_was(self):
        output = os.path.join(helpers.temp_dir(), "stdout.log")
        with open(output, "w") as f:
            f.write("Some text.\n==========\nPrompt: 30 tokens, 97.041 tokens-per-sec\nGeneration: 256 tokens, 106.873 tokens-per-sec\nPeak memory: 0.887 GB\n")
        adapter = adapters.select("auto", ["python", "-m", "mlx_lm.generate", "--model", "x"])
        self.assertEqual(adapter.name, "mlx_lm.generate")
        result = adapter.finalize([], output, "/nonexistent")
        self.assertEqual(result["throughput"]["prefill"]["tokens_per_s"], 97.041)
        self.assertEqual(result["throughput"]["decode"], {"tokens_per_s": 106.873, "tokens": 256, "source": "engine"})
        self.assertEqual(result["engine_peak_memory_gb"], 0.887)

    def test_mlx_server_gets_its_preset(self):
        self.assertEqual(adapters.preset_for(["python", "-m", "mlx_lm.server", "--model", "x"]), "mlx")
        self.assertEqual(adapters.preset_for(["mlx_lm.server", "--model", "x"]), "mlx")
        self.assertTrue(adapters.SERVER_PRESETS["mlx"]["uses_default_model"])
        self.assertEqual(adapters.model_flag_value(["python", "-m", "mlx_lm.server", "--model", "a/b"]), "a/b")
        self.assertIsNone(adapters.model_flag_value(["python", "-m", "mlx_lm.server"]))

    def test_finds_the_engine_behind_wrappers(self):
        cmd = ["timeout", "-k", "30", "600", "env", "A=B", "taskset", "-c", "16-23,80-87", "numactl", "--interleave=all", "/x/llama-bench", "-m", "y"]
        self.assertEqual(adapters._skip_wrappers(cmd), 11)
        self.assertEqual(adapters._skip_wrappers(["python3", "-c", "1"]), 0)

    def test_urls(self):
        self.assertEqual(adapters.guess_url(["llama-server", "-m", "x", "--port", "9000"], "llama-server"), "http://127.0.0.1:9000")
        self.assertEqual(adapters.guess_url(["llama-server", "-m", "x"], "llama-server", environ={}), "http://127.0.0.1:9931")
        self.assertEqual(adapters.guess_url(["llama-server", "-m", "x"], "llama-server", environ={"LLAMA_ARG_PORT": "9100"}), "http://127.0.0.1:9100")
        self.assertEqual(adapters.guess_url(["vllm", "serve", "m"], "vllm"), "http://127.0.0.1:8000")
        docker = ["docker", "run", "--gpus", "all", "-p", "9001:8000", "vllm/vllm-openai", "--model", "m", "--port", "8000"]
        self.assertEqual(adapters.guess_url(docker, "vllm"), "http://127.0.0.1:9001")
        self.assertEqual(adapters.guess_url(["docker", "run", "-p", "127.0.0.1:9002:80/tcp", "tgi"], "tgi"), "http://127.0.0.1:9002")

    def test_llama_server_is_told_where_to_listen(self):
        # Measured: llama.cpp moved its default port from 8080 to 9931, and `serve -- llama-server -m x` waited an hour at 8080
        with mock.patch("llmperfscope.adapters.port_in_use", return_value=False):
            self.assertEqual(adapters.give_port(["llama-server", "-m", "x"], "llama-server", {}), ["llama-server", "-m", "x", "--port", "9931"])
        with mock.patch("llmperfscope.adapters.port_in_use", return_value=True), mock.patch("llmperfscope.adapters.free_port", return_value=40123):
            self.assertEqual(adapters.give_port(["llama-server", "-m", "x"], "llama-server", {}), ["llama-server", "-m", "x", "--port", "40123"])
        # what the command or the environment says stays, and other engines and containers are left alone
        for cmd, preset, environ in ((["llama-server", "--port", "9000"], "llama-server", {}), (["llama-server", "--port=9000"], "llama-server", {}),
                                     (["llama-server"], "llama-server", {"LLAMA_ARG_PORT": "9100"}), (["vllm", "serve", "m"], "vllm", {}),
                                     (["docker", "run", "-p", "8080:8080", "ghcr.io/ggml-org/llama.cpp:server"], "llama-server", {})):
            self.assertEqual(adapters.give_port(cmd, preset, environ), cmd)

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
        self.assertIn(f"{os.path.realpath(directory)}:/llmperfscope-run", cmd)

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


class IdleMemory(unittest.TestCase):
    def test_a_late_sample_does_not_move_the_baseline(self):
        # Seen in CI: one of 16 samples already saw the target's 2000 MiB, and the memory it used came out 125 MiB short.
        self.assertEqual(analyze.idle_level([100] * 15 + [2100]), 100)

    def test_no_samples(self):
        self.assertEqual(analyze.idle_level([]), 0)


class FirstTokenByLength(unittest.TestCase):
    @staticmethod
    def request(tokens, ms, cached=0):
        return {"ok": True, "ttft_s": ms / 1000.0, "itl_s": [], "chunks": 1, "cached_tokens": cached,
                "usage": {"prompt_tokens": tokens, "completion_tokens": 1}}

    def test_the_line_through_known_costs(self):
        lengths = [20, 40, 60, 80, 100, 150, 200, 300, 400, 800, 1600, 3200, 6400]
        out = driver.first_token_by_length([self.request(n, 150.0 + 0.7 * n) for n in lengths])
        fit = out["first_token_fit"]
        self.assertAlmostEqual(fit["fixed_ms"], 150.0, delta=0.5)
        self.assertAlmostEqual(fit["per_token_ms"], 0.7, delta=0.001)
        self.assertEqual(fit["median_prompt_tokens"], 200)
        self.assertAlmostEqual(fit["fixed_share_at_median"], 150.0 / 290.0, delta=0.002)

    def test_slow_requests_and_long_prompts_do_not_decide_the_line(self):
        lengths = [20, 40, 60, 80, 100, 150, 200, 300, 400]
        results = [self.request(n, 150.0 + 0.7 * n) for n in lengths * 3]
        results += [self.request(60, 900.0), self.request(80, 1200.0), self.request(30000, 90000.0)]
        fit = driver.first_token_by_length(results)["first_token_fit"]
        self.assertAlmostEqual(fit["fixed_ms"], 150.0, delta=8.0)
        self.assertAlmostEqual(fit["per_token_ms"], 0.7, delta=0.05)

    def test_tokens_from_the_cache_do_not_count(self):
        results = [self.request(1000, 100.0 + n, cached=1000 - n) for n in (10, 20, 40, 80, 160, 320, 640, 900)]
        fit = driver.first_token_by_length(results)["first_token_fit"]
        self.assertAlmostEqual(fit["per_token_ms"], 1.0, delta=0.001)
        self.assertAlmostEqual(fit["fixed_ms"], 100.0, delta=0.5)

    def test_groups(self):
        out = driver.first_token_by_length([self.request(50, 10.0), self.request(60, 30.0), self.request(5000, 100.0)])
        self.assertEqual(out["first_token_by_length"], [
            {"tokens": "0-99", "requests": 2, "mean_ms": 20.0, "p50_ms": 10.0},
            {"tokens": "4000+", "requests": 1, "mean_ms": 100.0, "p50_ms": 100.0}])

    def test_no_line_when_lengths_do_not_differ_enough(self):
        self.assertNotIn("first_token_fit", driver.first_token_by_length([self.request(100 + i, 50.0) for i in range(20)]))
        self.assertNotIn("first_token_fit", driver.first_token_by_length([self.request(2 ** i, 50.0) for i in range(4)]))

    def test_summary_carries_them_and_the_report_prints_them(self):
        results = [self.request(n, 150.0 + 0.7 * n) for n in (20, 40, 60, 80, 100, 150, 200, 300)]
        out = driver.summarize_load(results, 2.0)
        out["concurrency"] = 1
        text = report.render({"status": "ok", "throughput": {"serve_c1": out}})
        self.assertIn("first token = 150 ms + 0.70 ms per prompt token", text)
        self.assertIn("0-99:", text)

    def test_a_finding_when_most_of_it_does_not_depend_on_length(self):
        results = [self.request(n, 150.0 + 0.7 * n) for n in (20, 40, 60, 80, 100, 150, 200, 300)] * 3
        summary = driver.summarize_load(results, 2.0)
        summary["concurrency"] = 1
        found = {f["what"] for f in diagnose.serving_findings({"throughput": {"serve_c1": summary}})}
        self.assertIn("first_token_fixed_part", found)
        results = [self.request(n, 1.0 + 2.0 * n) for n in (20, 40, 60, 80, 100, 150, 200, 300)] * 3
        summary = driver.summarize_load(results, 2.0)
        summary["concurrency"] = 1
        found = {f["what"] for f in diagnose.serving_findings({"throughput": {"serve_c1": summary}})}
        self.assertNotIn("first_token_fixed_part", found)


class WaitsForTheGpu(unittest.TestCase):
    @staticmethod
    def tree(stacks):
        tree = {}
        for frames, weight in stacks:
            cpuprof._add_wait(tree, frames, weight)
        return tree

    def test_sites_are_where_the_calls_part(self):
        wait = ["cuStreamSynchronize", "cudaStreamSynchronize", "backend_synchronize"]
        tree = self.tree([
            (wait + ["alloc_graph", "process_ubatch", "decode", "loop", "main"], 30),
            (wait[:2] + ["allreduce", "compute", "process_ubatch", "decode", "loop", "main"], 60),
            (wait + ["read_state", "create_checkpoint", "pre_decode", "loop", "main"], 10),
        ])
        self.assertEqual(tree["count"], 100)
        self.assertEqual(cpuprof.wait_sites(tree), [("compute", 60), ("alloc_graph", 30), ("pre_decode", 10)])

    def test_one_caller_is_named_by_its_innermost_function(self):
        tree = self.tree([(["cudaEventSynchronize", "allreduce", "compute", "main"], 5)])
        self.assertEqual(cpuprof.wait_sites(tree), [("allreduce", 5)])

    def test_a_rare_caller_is_not_a_site(self):
        tree = self.tree([(["cudaEventSynchronize", "a", "main"], 980), (["cudaEventSynchronize", "b", "main"], 10)])
        self.assertEqual(cpuprof.wait_sites(tree), [("a", 980)])

    def test_a_function_that_waits_itself_counts_next_to_what_it_calls(self):
        # The way a real run looked: an AllReduce that waits on its events, and calls a copy that waits too.
        tree = self.tree([
            (["cudaEventSynchronize", "ggml_cuda_ar_allreduce", "meta_compute", "decode", "main"], 500),
            (["cudaEventSynchronize", "copy_impl", "ggml_cuda_ar_allreduce", "meta_compute", "decode", "main"], 370),
            (["cudaStreamSynchronize", "synchronize", "sample", "main"], 130),
        ])
        self.assertEqual(sum(count for _, count in cpuprof.wait_sites(tree)), 1000)
        self.assertEqual(cpuprof.wait_sites(tree), [("ggml_cuda_ar_allreduce", 500), ("copy_impl", 370), ("sample", 130)])

    def test_names_without_their_clutter(self):
        self.assertEqual(cpuprof.function_label("bool ggml_cuda_ar_copy_impl<__nv_bfloat16, float>(ggml_cuda_ar_pipeline*, int)"), "ggml_cuda_ar_copy_impl")
        self.assertEqual(cpuprof.function_label("server_context_impl::pre_decode()"), "server_context_impl::pre_decode")
        self.assertEqual(cpuprof.function_label("ggml_backend_sched_alloc_graph"), "ggml_backend_sched_alloc_graph")
        self.assertEqual(cpuprof.function_label("cudaStreamSynchronize@@libcudart.so.13"), "cudaStreamSynchronize")

    def test_nothing_waited(self):
        self.assertEqual(cpuprof.wait_sites({}), [])

    def test_versioned_symbols_count(self):
        self.assertTrue(cpuprof._is_gpu_wait("cudaStreamSynchronize@@libcudart.so.13"))
        self.assertFalse(cpuprof._is_gpu_wait("cudaLaunchKernel@@libcudart.so.13"))


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
        out = compare.diff([a, a], [b, b])
        self.assertEqual(out["verdict"], "tradeoff")
        self.assertIn("gpu_peak_mib", out["worse"])
        self.assertIn("decode_tps", out["better"])

    def test_only_a_cost_is_a_regression(self):
        a = {"status": "ok", "throughput": {"decode": {"tokens_per_s": 100.0}}}
        b = {"status": "ok", "throughput": {"decode": {"tokens_per_s": 80.0}}}
        self.assertEqual(compare.diff([a, a], [b, b])["verdict"], "regression")

    def test_chance_of_arms_separating_with_nothing_changed(self):
        self.assertEqual(compare.separation_chance(1, 1), 1.0)
        self.assertAlmostEqual(compare.separation_chance(2, 2), 1 / 3)
        self.assertAlmostEqual(compare.separation_chance(3, 3), 1 / 10)
        self.assertAlmostEqual(compare.separation_chance(4, 4), 1 / 35)

    def test_one_run_per_side_proves_nothing(self):
        a = {"status": "ok", "throughput": {"decode": {"tokens_per_s": 100.0}}}
        b = {"status": "ok", "throughput": {"decode": {"tokens_per_s": 80.0}}}
        out = compare.diff([a], [b])
        self.assertEqual(out["verdict"], "inconclusive")
        self.assertEqual(out["worse"], [])
        self.assertEqual(out["maybe_worse"], ["decode_tps"])
        self.assertIn("worse?", compare.render_diff(out))
        self.assertIn("every time", compare.render_diff(out))

    def test_two_close_arms_per_side_are_a_hint(self):
        """Real numbers: a unit flagged 0.990 on a model that never runs its code."""
        verdict = compare.judge("prefill_tps", [8001.5, 8043.7], [7954.2, 7936.4])
        self.assertEqual(verdict["verdict"], "worse")
        self.assertTrue(verdict["weak"])

    def test_four_separated_arms_per_side_are_a_result(self):
        """Real numbers again: the same size of difference, but arms that stay apart."""
        verdict = compare.judge("decode_tps", [163.6, 163.5, 163.6, 163.7], [163.3, 162.9, 162.2, 162.9])
        self.assertEqual(verdict["verdict"], "worse")
        self.assertFalse(verdict["weak"])

    def test_a_clear_gap_needs_no_more_runs(self):
        """Prefill +35% with 2 arms per side, and memory that is the same in every arm."""
        self.assertFalse(compare.judge("prefill_tps", [984.8, 979.3], [1326.8, 1323.6])["weak"])
        self.assertFalse(compare.judge("gpu_peak_mib", [6220, 6220], [6284, 6284])["weak"])

    def _matrix(self, values, **options):
        """A matrix over one case, with each arm's decode speed scripted by build and arm count."""
        arms = []

        class FakeRun:
            def __init__(self, command, opts):
                self.bin = command[0]
                self.opts = opts

            def execute(self):
                side = "new" if "/new/" in self.bin else "old"
                speed = values[side][sum(1 for arm in arms if arm == side) % len(values[side])]
                arms.append(side)
                return {"status": "ok", "throughput": {"decode": {"tokens_per_s": speed}}}

        out = helpers.temp_dir()
        cases = [{"label": "case", "binary": "llama-bench", "prefix": [], "args": [], "env": {}, "model": None, "gpus": None}]
        with mock.patch("llmperfscope.compare.runner.Run", FakeRun):
            table = compare.run_matrix(cases, ("old", "/old/bin"), ("new", "/new/bin"), out, log=lambda text: None, **options)
        return table, arms, out

    def test_a_change_that_looks_worse_gets_more_arms(self):
        table, arms, _ = self._matrix({"old": [101.0, 101.2, 101.1, 101.3, 100.0, 100.2, 100.1, 100.3], "new": [100.0, 100.1, 100.2, 100.3, 101.5, 101.0, 100.9, 101.2]})
        self.assertEqual(len(arms), 16)
        self.assertEqual(arms[:8], ["old", "new", "new", "old", "new", "old", "old", "new"])
        self.assertEqual(arms[8:], ["new", "old", "old", "new", "old", "new", "new", "old"])
        self.assertEqual(table["cases"][0]["arms"], {"a": 8, "b": 8})
        self.assertEqual(table["cases"][0]["verdict"], "neutral")

    def test_a_real_regression_survives_the_extra_arms(self):
        table, arms, out = self._matrix({"old": [100.0, 100.1], "new": [90.0, 90.1]})
        self.assertEqual(len(arms), 16)
        self.assertEqual(table["cases"][0]["verdict"], "regression")
        with open(os.path.join(out, "table.md")) as f:
            self.assertIn("**0.900** worse", f.read())

    def test_no_extra_arms_when_nothing_looks_worse(self):
        _, arms, _ = self._matrix({"old": [100.0, 100.1], "new": [110.0, 110.1]})
        self.assertEqual(len(arms), 8)

    def test_recheck_can_be_turned_off(self):
        _, arms, _ = self._matrix({"old": [100.0, 100.1], "new": [90.0, 90.1]}, recheck=0)
        self.assertEqual(len(arms), 8)

    def test_noise_from_null_matrix(self):
        path = os.path.join(helpers.temp_dir(), "m.json")
        with open(path, "w") as f:
            json.dump({"cases": [{"label": "c", "metrics": {"decode_tps": {"a_runs": [100, 101], "b_runs": [102, 100]}}}]}, f)
        self.assertAlmostEqual(compare.noise_from_null(path)["c"]["decode_tps"], 2 / 100.75, places=4)


class Kernels(unittest.TestCase):
    """Two GPUs, 10 tokens, known times. GPU 0 reaches each collective first and waits; GPU 1 shows what the collectives really cost."""

    def _run(self, mul_mat_ms=120.0):
        return {"nsys": {"groups": {"decode": {
            "tokens": 10,
            "devices": [{"device": 0, "kernel_ms": 100.0, "comm_ms": 30.0, "copy_ms": 2.0, "span_ms": 300.0},
                        {"device": 1, "kernel_ms": 100.0, "comm_ms": 10.0, "copy_ms": 2.0, "span_ms": 300.0}],
            "top_kernels": [{"name": "mul_mat", "ms": mul_mat_ms, "share": 0.5, "n": 40, "avg_us": 3000.0},
                            {"name": "rms_norm", "ms": 80.0, "share": 0.3, "n": 20, "avg_us": 4000.0},
                            {"name": "ncclDevKernel_AllReduce_f32_RING", "ms": 40.0, "share": 0.2, "n": 20, "avg_us": 2000.0}]}}}}

    def test_time_per_token_per_gpu(self):
        data = kernels.per_token(self._run()["nsys"]["groups"]["decode"])
        self.assertAlmostEqual(data["kernel_ms"], 10.0)
        self.assertAlmostEqual(data["collective_ms"], 1.0)
        self.assertAlmostEqual(data["waiting_ms"], 1.0)
        rows = {row["name"]: row for row in data["kernels"]}
        self.assertAlmostEqual(rows["mul_mat"]["ms_per_token"], 6.0)
        self.assertAlmostEqual(rows["mul_mat"]["calls_per_token"], 2.0)

    def test_a_collective_counts_without_its_waiting(self):
        rows = {row["name"]: row for row in kernels.per_token(self._run()["nsys"]["groups"]["decode"])["kernels"]}
        collective = rows["ncclDevKernel_AllReduce_f32_RING"]
        self.assertTrue(collective["collective"])
        self.assertFalse(rows["mul_mat"]["collective"])
        self.assertAlmostEqual(collective["ms_per_token"], 2.0)
        self.assertAlmostEqual(collective["cost_ms_per_token"], 1.0)

    def test_what_a_kernel_could_win(self):
        self.assertAlmostEqual(kernels.gain_if_free(20.0, 5.0), 1 / 3)
        self.assertIsNone(kernels.gain_if_free(20.0, 25.0))

    def test_shares_need_a_run_without_tracing(self):
        speed = {"throughput": {"decode": {"tokens_per_s": 50.0}}}
        text = kernels.render(self._run(), speed)
        self.assertIn("20.00 ms per token without tracing", text)
        self.assertIn("| gaps, host work and waiting between kernels | 8.80 | 44% |", text)
        self.assertIn("| mul_mat | 6.000 | 2.0 | 3000.0 | 30.0% | +42.9% |", text)
        self.assertIn("ncclDevKernel_AllReduce_f32_RING (collective)", text)
        without = kernels.render(self._run())
        self.assertIn("Add --speed", without)
        self.assertNotIn("speed-up if free", without)

    def test_a_run_without_a_trace_says_so(self):
        self.assertIn("--nsys", kernels.render({"throughput": {}}))

    def test_a_server_trace_is_per_output_token(self):
        # Measured: `kernels` on a real `serve --nsys` said it needed a run with --nsys.
        run = {"nsys": {"groups": {"serve": self._run()["nsys"]["groups"]["decode"]}}}
        text = kernels.render(run)
        self.assertIn("## serve: 10 tokens traced", text)
        self.assertIn("per output token", text)
        speed = {"throughput": {"serve_c2": {"concurrency": 2, "output_tokens_per_s": 90.0}, "serve_c1": {"concurrency": 1, "output_tokens_per_s": 50.0}}}
        text = kernels.render(run, speed)
        self.assertIn("20.00 ms per token without tracing", text)
        self.assertNotIn("has no speed", text)
        self.assertIn("## serve", kernels.render_diff(run, run))
        self.assertNotIn("## serve", kernels.render(self._run()))

    def test_diff_shows_which_kernel_changed(self):
        text = kernels.render_diff(self._run(), self._run(mul_mat_ms=100.0), "old", "new")
        self.assertIn("| old ms/token | new ms/token |", text)
        self.assertIn("| mul_mat | 6.000 | 5.000 | -1.000 |", text)
        self.assertIn("| rms_norm | 4.000 | 4.000 | +0.000 |", text)


class RealPrompts(unittest.TestCase):
    def _opener(self, urls):
        import contextlib
        import io

        def opener(url, timeout=None):
            urls.append(url)
            row = {"messages": [{"role": "user", "content": "x" * (10 if "long" not in url else 5000)}], "tier": "standard" if "mbpp" in url or "long_code" in url else "extended", "language": "en"}
            return contextlib.closing(io.BytesIO(json.dumps(row).encode()))
        return opener

    def test_downloads_once_from_a_pinned_revision(self):
        directory = helpers.temp_dir()
        urls = []
        with mock.patch("builtins.print"):
            path = driver.prompt_mix_path(directory, self._opener(urls))
            self.assertEqual(driver.prompt_mix_path(directory, self._opener(urls)), path)
        self.assertEqual(len(urls), len(driver.MIX_FILES))
        self.assertTrue(all(driver.MIX_REVISION in url for url in urls))
        self.assertEqual(len(driver.load_prompts(path)), len(driver.MIX_FILES))

    def test_the_standard_tier_is_the_default_and_filters_still_work(self):
        directory = helpers.temp_dir()
        with mock.patch("builtins.print"):
            path = driver.prompt_mix_path(directory, self._opener([]))
        with mock.patch("llmperfscope.driver.prompt_mix_path", lambda: path):
            self.assertEqual(len(serving.rows_from(serving.defaults(prompts="mix"))), 2)
            self.assertEqual(len(serving.rows_from(serving.defaults(prompts="mix", filter=["tier=extended"]))), len(driver.MIX_FILES) - 2)
            self.assertEqual(len(serving.rows_from(serving.defaults(prompts="mix", max_prompt_chars=100))), 1)

    def test_a_failed_download_says_what_to_do(self):
        def refuse(url, timeout=None):
            raise OSError("no network")
        with mock.patch("builtins.print"), self.assertRaises(SystemExit) as caught:
            driver.prompt_mix_path(helpers.temp_dir(), refuse)
        self.assertIn("--prompts", str(caught.exception))

    def test_unknown_server_settings_are_refused(self):
        with self.assertRaises(ValueError):
            serving.defaults(concurency="1")

    def test_prefill_and_decode_as_the_server_timed_them(self):
        results = [{"ok": True, "chunks": 4, "ttft_s": 0.1, "itl_s": [0.01] * 3, "usage": {"completion_tokens": 4, "prompt_tokens": 100},
                    "server_timings": {"prompt_n": 100, "prompt_ms": 50.0, "predicted_n": 4, "predicted_ms": 40.0}},
                   {"ok": True, "chunks": 4, "ttft_s": 0.1, "itl_s": [0.01] * 3, "usage": {"completion_tokens": 4, "prompt_tokens": 300},
                    "server_timings": {"prompt_n": 300, "prompt_ms": 150.0, "predicted_n": 4, "predicted_ms": 40.0}}]
        summary = driver.summarize_load(results, 1.0)
        self.assertEqual(summary["server_prefill_tokens_per_s"], 2000.0)
        self.assertEqual(summary["server_decode_tokens_per_s"], 100.0)
        speed = serving._overall_speed({"serve_c4": {**summary, "concurrency": 4, "server_prefill_tokens_per_s": 500.0}, "serve_c1": {**summary, "concurrency": 1}})
        self.assertEqual(speed["prefill"]["tokens_per_s"], 2000.0)
        self.assertTrue(speed["prefill"]["real_prompts"])

    def test_serving_metrics_read_the_right_level(self):
        run = {"status": "ok", "throughput": {"serve_c4": {"concurrency": 4, "output_tokens_per_s": 90.0, "ttft_ms": {"p50": 900.0}, "itl_ms": {"p50": 30.0}},
                                              "serve_c1": {"concurrency": 1, "output_tokens_per_s": 40.0, "ttft_ms": {"p50": 100.0}, "itl_ms": {"p50": 20.0}}}}
        self.assertEqual(compare.METRICS["output_tps"][0](run), 90.0)
        self.assertEqual(compare.METRICS["ttft_ms"][0](run), 100.0)
        self.assertEqual(compare.METRICS["itl_ms"][0](run), 20.0)
        self.assertIsNone(compare.METRICS["output_tps"][0]({"throughput": {"decode": {"tokens_per_s": 1.0}}}))


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


class OnAMac(unittest.TestCase):
    run_info = {"disks": {"disk0": {"kind": "ssd"}}, "host": {"cpus": 8, "os": "macOS 27.2"},
                "devices": [{"index": 0, "memory_total_mib": 12124, "memory_total_is_working_set_limit": True, "unified_memory": True}], "devices_used": [0]}

    def phase(self, **kwargs):
        return diagnose.classify_phase("decode", helpers.group(**kwargs), self.run_info)

    def test_the_compressor_alone_does_not_make_ram_the_limit(self):
        # Measured: a machine in normal use, nothing short of memory.
        self.assertEqual(self.phase(flows={"compressed_mib": 5000.0}, gpu={"util_avg": 10}, host={"busy_cores": 0.1})["limit"], "none visible")

    def test_swapping_other_programs_does_not_override_a_busy_gpu(self):
        # Measured on llama-server at warning pressure: 43 MiB/s of swap, decode speed within 3% of an unloaded machine.
        result = self.phase(flows={"swap_out_mib": 430.0, "compressed_mib": 850.0}, pressure={"memory_some_pct": 100.0}, gpu={"util_avg": 89})
        self.assertEqual(result["limit"], "gpu")

    def test_pressure_with_traffic_and_nothing_busy_is_ram_but_not_certain(self):
        result = self.phase(flows={"compressed_mib": 9980.0}, pressure={"memory_some_pct": 83.0}, gpu={"util_avg": 16}, host={"busy_cores": 0.3})
        self.assertEqual((result["limit"], result["confidence"]), ("ram", "medium"))
        self.assertIn("memory compressor", result["why"])
        self.assertIn("memory pressure", result["why"])
        self.assertNotIn("stalled", result["why"])

    def test_critical_pressure_is_ram_whatever_else_is_busy(self):
        result = self.phase(pressure={"memory_full_pct": 20.0, "memory_some_pct": 100.0}, gpu={"util_avg": 99})
        self.assertEqual((result["limit"], result["confidence"]), ("ram", "high"))

    def test_heat_is_named_and_needs_no_clock(self):
        gpu = {"util_avg": 88, "throttled": True, "throttle": ["sw_thermal"]}
        result = self.phase(gpu=gpu)
        self.assertEqual(result["limit"], "gpu (throttled)")
        self.assertIn("thermal pressure", result["why"])
        self.assertNotIn("MHz", result["why"])
        self.assertIn("1243 MHz", self.phase(gpu={**gpu, "sm_mhz_avg": 1243.0})["why"])

    def test_advice_names_what_a_mac_has(self):
        result = self.phase(gpu={"util_avg": 95})
        self.assertIn("Metal System Trace", result["why"])
        self.assertNotIn("--nsys", result["why"])
        steps = diagnose.next_steps(self.run_info, {"decode": {"limit": "gpu + host wait"}})
        self.assertTrue(steps)
        self.assertFalse([step for step in steps if "--nsys" in step or "--cpu-profile" in step])
        linux = diagnose.next_steps({"host": {"cpus": 8}}, {"decode": {"limit": "gpu"}})
        self.assertIn("--nsys", linux[0])

    def _memory(self, gpu_peak=1000, **host):
        group = helpers.group(gpu={"mem_peak_mib": gpu_peak, "mem_start_mib": gpu_peak, "mem_end_mib": gpu_peak}, host={"anon_peak_mib": 8000, "file_peak_mib": 8000, **host})
        run = {**self.run_info, "groups": {"decode": group}, "memory": {"gpu": {"devices": [{"index": 0, "peak_mib": gpu_peak, "peak_above_baseline_mib": gpu_peak, "baseline_mib": 0}]}, "host": host}}
        return {f["what"]: f for f in diagnose.memory_findings(run)}

    def test_gpu_and_host_memory_are_not_called_a_copy(self):
        findings = self._memory()
        self.assertNotIn("host_ram_held_while_on_gpu", findings)
        self.assertNotIn("weights_mapped_in_page_cache", findings)

    def test_full_means_what_macos_recommends(self):
        finding = self._memory(gpu_peak=12000)["gpu_memory_nearly_full"]
        self.assertIn("recommends", finding["text"])
        self.assertIn("iogpu.wired_limit_mb", finding["text"])

    def test_a_compressed_process_is_not_called_swapped_to_disk(self):
        finding = self._memory(swap_peak_mib=431.0)["process_swapped"]
        self.assertIn("compressed", finding["text"])

    def test_the_compressor_finding(self):
        group = helpers.group(flows={"compressed_mib": 2000.0, "decompressed_mib": 264.0}, pressure={"memory_some_pct": 100.0})
        findings = {f["what"]: f for f in diagnose.movement_findings({**self.run_info, "groups": {"serve": group}})}
        self.assertIn("2,264 MiB", findings["memory_compressor_busy"]["text"])

    def test_shaders_compiled_elsewhere_are_explained(self):
        # Measured: the first run of llama.cpp on Metal, 8.7 s of startup at 0.01 cores with the machine 66% busy.
        startup = helpers.group(wall_s=8.7, host={"busy_cores": 0.01}, system_cpu={"busy_pct": 65.5})
        findings = {f["what"]: f for f in diagnose.movement_findings({**self.run_info, "groups": {"startup": startup}})}
        self.assertIn("Metal shader compiler", findings["work_in_another_process"]["text"])
        quiet = helpers.group(wall_s=8.7, host={"busy_cores": 0.01}, system_cpu={"busy_pct": 5.0})
        self.assertFalse(diagnose.movement_findings({**self.run_info, "groups": {"startup": quiet}}))
        elsewhere = diagnose.movement_findings({"host": {"cpus": 8}, "groups": {"startup": startup}})
        self.assertNotIn("Metal", elsewhere[0]["text"])

    def test_the_model_disk_is_named_even_if_another_is_busier(self):
        load = helpers.group(wall_s=1.0, host={"disk_read_mib": 600.0}, flows={"disks": {"disk0": {"read_mib": 260.0, "busy_pct": 90.0}, "disk4": {"read_mib": 590.0, "busy_pct": 50.0}}})
        before = {"model_storage": {"path": "/x", "disks": [{"name": "disk4", "kind": "ssd"}]}, "model_page_cache_fraction": 0.0}
        findings = {f["what"]: f for f in diagnose.movement_findings({**self.run_info, "groups": {"load": load}, "machine_before": before})}
        self.assertIn("disk4", findings["load_read_from_disk"]["text"])


class ReportTables(unittest.TestCase):
    def test_links_a_machine_lacks_get_no_column(self):
        run = {"status": "ok", "groups": {"decode": helpers.group(flows={"disks": {"disk0": {"read_mib": 5.0, "write_mib": 1.0}}, "swap_in_mib": 0.0, "swap_out_mib": 0.0, "compressed_mib": 3.0, "decompressed_mib": 1.0}, host={"disk_read_mib": 2.0, "major_faults": 7})}}
        text = report.render(run)
        self.assertIn("compressed", text)
        self.assertNotIn("RAM to GPU", text)
        self.assertNotIn("GPU to GPU", text)
        row = next(line for line in text.splitlines() if line.startswith("| decode | 2 |"))
        self.assertTrue(row.endswith("| 7 |"), row)

    def test_all_the_links_on_a_machine_that_has_them(self):
        run = {"status": "ok", "groups": {"decode": helpers.group(flows={"pcie_to_gpu_mib_est": 5.0, "pcie_from_gpu_mib_est": 3.0, "nvlink_mib": 9.0, "swap_in_mib": 0.0, "swap_out_mib": 0.0}, host={"disk_read_mib": 2.0})}}
        text = report.render(run)
        self.assertIn("RAM to GPU (est)", text)
        self.assertIn("GPU to GPU", text)
        self.assertNotIn("compressed", text)

    def test_one_pool_of_memory_is_explained(self):
        run = {"status": "ok", "devices": [{"index": 0, "unified_memory": True}], "groups": {"decode": helpers.group(gpu={"mem_peak_mib": 100, "mem_end_mib": 100})}}
        self.assertIn("one pool of memory", report.render(run))
        run["devices"][0].pop("unified_memory")
        self.assertNotIn("one pool of memory", report.render(run))


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

    def test_memory_held_after_the_run_means_nothing_when_attached(self):
        device = {"index": 0, "peak_mib": 2000, "peak_above_baseline_mib": 1500, "baseline_mib": 500, "after_exit_mib": 2000}
        run = {"memory": {"gpu": {"devices": [device]}}, "groups": {"tail": {"wall_s": 1.0}}}
        self.assertIn("gpu_memory_after_exit", self.names(run))
        run["attached"] = True
        self.assertNotIn("gpu_memory_after_exit", self.names(run))

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
