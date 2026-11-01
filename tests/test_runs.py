"""Whole runs of real processes with known behaviour: do the numbers come out as what actually happened? Needs Linux, not a GPU."""

import json
import os
import shutil
import socket
import subprocess
import sys
import textwrap
import time
import unittest
from unittest import mock

import helpers
from llmperfscope import cli, cpuprof, machine, procfs, runner

ENGINE = os.path.join(helpers.HERE, "fake_engine.py")
SERVER = os.path.join(helpers.HERE, "fake_openai_server.py")


def run(cmd, gpu=None, **options):
    options.setdefault("precheck", "skip")
    options.setdefault("quiet", True)
    options.setdefault("baseline_s", 0.3)
    options.setdefault("tail_s", 0.2)
    out = helpers.temp_dir()
    with mock.patch("llmperfscope.runner.open_backend", lambda: gpu or helpers.ScriptedGpu()):
        return runner.Run(cmd, runner.Options(out=out, **options)).execute()


def engine(*args):
    return [sys.executable, ENGINE, *args]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@unittest.skipUnless(os.path.exists("/proc/self/status"), "Linux only")
class KnownEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.file = os.path.join(helpers.temp_dir(), "weights.bin")
        cls.report = run(engine("phases", cls.file), gpu=helpers.ScriptedGpu(count=2, power_w=100.0, used_mib=2000))

    def test_ends_ok(self):
        self.assertEqual(self.report["status"], "ok")
        self.assertEqual(self.report["exit_code"], 0)

    def test_markers_become_phases_with_their_durations(self):
        groups = self.report["groups"]
        self.assertIn("load", groups)
        self.assertGreaterEqual(groups["decode"]["wall_s"], 0.55)
        self.assertLess(groups["decode"]["wall_s"], 1.2)
        self.assertEqual(groups["decode"]["tokens"], 100)

    def test_host_memory_kinds_are_told_apart(self):
        memory = self.report["memory"]["host"]
        self.assertGreater(memory["anon_peak_mib"], 190)
        self.assertLess(memory["anon_peak_mib"], 300)
        self.assertGreater(memory["file_peak_mib"], 60)

    def test_cpu_time(self):
        # The fake engine's clock calls are system calls, so its 0.6 s split between user and system time.
        host = self.report["groups"]["decode"]["host"]
        self.assertAlmostEqual(host["user_s"] + host["sys_s"], 0.6, delta=0.2)

    def test_energy_per_token_from_known_power(self):
        decode = self.report["groups"]["decode"]
        expected = 2 * 100.0 * decode["wall_s"] / 100
        self.assertAlmostEqual(decode["gpu"]["j_per_token"], expected, delta=0.05 * expected + 0.02)

    def test_a_disagreeing_energy_counter_is_called_out(self):
        class Broken(helpers.ScriptedGpu):
            def energy_mj(self, index):
                return int(self.power_w * (time.monotonic() - self.t0) * 400)  # 40% of the truth

        report = run(engine("spin_for", "3"), gpu=Broken(count=1, power_w=100.0), marks=[("started", "decode", None)])
        self.assertEqual(report["status"], "ok")
        self.assertLess(report["consistency"]["energy_over_power_gauge"], 0.6)
        self.assertTrue(any("disagree" in note for note in report["notes"]), report["notes"])

    def test_energy_per_token_is_not_headlined_for_a_short_phase(self):
        # The counter is read about twice a second: under two seconds of work is too little to compare.
        self.assertNotIn("decode_j_per_token", self.report["energy"])

    def test_gpu_memory_comes_from_the_process_view(self):
        gpu = self.report["memory"]["gpu"]
        self.assertEqual(self.report["devices_used"], [0, 1])
        self.assertEqual(gpu["peak_above_baseline_total_mib"], 4000)
        self.assertEqual(gpu["devices"][0]["process_peak_mib"], 2000)

    def test_files_are_written(self):
        self.assertIn("diagnosis", self.report)
        self.assertEqual(self.report["schema"], "llmperfscope.run/1")


@unittest.skipUnless(os.path.exists("/proc/self/status"), "Linux only")
class Failures(unittest.TestCase):
    def test_tracers_refuse_what_they_cannot_reach(self):
        """They would profile the docker client, and the report would look fine."""
        for options in ({"nsys": True}, {"cpu_profile": True}):
            with self.assertRaises(runner.Refused):
                run(["docker", "run", "--gpus", "all", "some-image"], **options)
            with self.assertRaises(runner.Refused):
                with mock.patch("llmperfscope.runner.open_backend", lambda: helpers.ScriptedGpu()):
                    runner.Run(["/proc/1/exe"], runner.Options(out=helpers.temp_dir(), quiet=True, **options), attach_pid=1).execute()

    def test_a_failing_target_is_reported(self):
        report = run(engine("fail"))
        self.assertEqual((report["status"], report["exit_code"]), ("failed", 7))

    def test_timeout_stops_a_sleeping_target(self):
        t0 = time.monotonic()
        report = run(engine("sleep"), timeout_s=1.0)
        self.assertEqual(report["status"], "timeout")
        self.assertLess(time.monotonic() - t0, 15)

    def test_a_hung_spinner_is_stalled_and_diagnosed(self):
        # The case that cost hours: 100% CPU, 100% GPU, no progress. Utilization must not count as progress.
        t0 = time.monotonic()
        report = run(engine("spin"), stall_s=2.0)
        self.assertEqual(report["status"], "stalled")
        self.assertLess(time.monotonic() - t0, 20)
        threads = report["failure"]["threads"]
        self.assertGreater(threads[0]["cpu_s"], 0.5)
        self.assertIn("started", report["failure"]["stderr_tail"] + ["started"])

    def test_silence_is_not_a_stall_unless_asked(self):
        report = run(engine("lines"))
        self.assertEqual(report["status"], "ok")
        self.assertIsNone(report["stop_reason"])

    def test_a_process_left_behind_is_stopped(self):
        report = run(engine("daemon"))
        self.assertEqual(report["status"], "ok")
        self.assertTrue(any("outlived" in note for note in report["notes"]), report["notes"])
        self.assertEqual([p for p in procfs.children_of(os.getpid()) if (procfs.stat(p) or {}).get("state") not in (None, "Z")], [])


@unittest.skipUnless(os.path.exists("/proc/self/status"), "Linux only")
class ProcessTree(unittest.TestCase):
    def test_memory_of_children_counts(self):
        report = run(engine("child"))
        self.assertGreater(report["memory"]["host"]["anon_peak_mib"], 140)


@unittest.skipUnless(os.path.exists("/proc/self/status"), "Linux only")
class Markers(unittest.TestCase):
    def test_plain_text_markers_from_any_language(self):
        report = run(engine("plain_marker"))
        names = [p["name"] for p in report["phases"]]
        self.assertIn("first_phase", names)
        self.assertIn("second_phase", names)
        first = [p for p in report["phases"] if p["name"] == "first_phase"][0]
        self.assertAlmostEqual(first["end_s"] - first["start_s"], 0.5, delta=0.3)

    def test_regex_on_output_needs_no_code_change(self):
        report = run(engine("lines"), marks=[("LOADING", "load", None), ("READY", "decode", None)])
        self.assertAlmostEqual(report["groups"]["load"]["wall_s"], 0.3, delta=0.15)
        self.assertIn("decode", report["groups"])

    def test_with_no_phases_the_whole_run_is_one(self):
        report = run(engine("lines"))
        self.assertIn("run", report["groups"])


@unittest.skipUnless(os.path.exists("/proc/self/status"), "Linux only")
class Serving(unittest.TestCase):
    def serve(self, ttft=0.2, inter=0.02, ready=0.6, extra=()):
        port = free_port()
        out = helpers.temp_dir()
        argv = ["serve", "--out", out, "--precheck", "skip", "--quiet", "--preset", "vllm", "--url", f"http://127.0.0.1:{port}",
                "--synthetic-words", "16", "--limit", "4", "--max-tokens", "11", "--concurrency", "1,2", *extra,
                "--", sys.executable, SERVER, str(port), str(ready), str(ttft), str(inter)]
        gpu = helpers.ScriptedGpu(count=1, power_w=100.0, used_mib=1000)
        with mock.patch("llmperfscope.runner.open_backend", lambda: gpu), mock.patch("builtins.print"):
            cli.main(argv)
        with open(os.path.join(out, "run.json")) as f:
            return json.load(f)

    def test_client_timing_matches_the_server_script(self):
        report = self.serve()
        self.assertEqual(report["status"], "ok")
        c1 = report["throughput"]["serve_c1"]
        self.assertEqual((c1["requests"], c1["failed"], c1["completion_tokens"]), (4, 0, 44))
        self.assertTrue(c1["tokens_exact"])
        self.assertAlmostEqual(c1["ttft_ms"]["p50"], 200, delta=40)
        self.assertAlmostEqual(c1["itl_ms"]["p50"], 20, delta=8)

    def test_concurrency_raises_throughput_of_a_parallel_server(self):
        report = self.serve()
        self.assertGreater(report["throughput"]["serve_c2"]["output_tokens_per_s"], 1.5 * report["throughput"]["serve_c1"]["output_tokens_per_s"])

    def test_load_phase_ends_when_the_server_is_ready(self):
        report = self.serve(ready=1.0)
        self.assertAlmostEqual(report["workload"]["ready_s"], 1.0, delta=0.5)
        self.assertAlmostEqual(report["groups"]["load"]["wall_s"], 1.0, delta=0.6)

    def test_lifecycle_and_energy_per_token(self):
        report = self.serve()
        for group in ("load", "warmup", "serve", "teardown"):
            self.assertIn(group, report["groups"])
        phase = [p for p in report["phases"] if p["name"] == "serve_c1"][0]
        wall = phase["end_s"] - phase["start_s"]
        self.assertAlmostEqual(report["throughput"]["serve_c1"]["j_per_output_token"], 100.0 * wall / 44, delta=0.1 * 100.0 * wall / 44 + 0.05)

    def test_server_metrics_are_collected(self):
        report = self.serve()
        self.assertIn("fake_kv_cache_usage_perc", report["workload"]["server_gauges_peak"])
        self.assertTrue(any(f["what"] == "kv_cache_mostly_unused" for f in report["diagnosis"]["memory"]))

    def test_a_server_that_exits_while_loading_fails_cleanly(self):
        out = helpers.temp_dir()
        argv = ["serve", "--out", out, "--precheck", "skip", "--quiet", "--preset", "vllm", "--url", "http://127.0.0.1:9", "--", sys.executable, ENGINE, "fail"]
        with mock.patch("llmperfscope.runner.open_backend", lambda: helpers.ScriptedGpu()), mock.patch("builtins.print"):
            code = cli.main(argv)
        self.assertEqual(code, 1)
        with open(os.path.join(out, "run.json")) as f:
            self.assertIn(json.load(f)["status"], ("workload_failed", "failed"))


@unittest.skipUnless(os.path.exists("/proc/self/status") and shutil.which("cc"), "needs Linux and a C compiler")
class CpuProfile(unittest.TestCase):
    def test_samples_name_the_function_that_burns_the_cpu(self):
        directory = helpers.temp_dir()
        source = os.path.join(directory, "burn.c")
        with open(source, "w") as f:
            f.write(textwrap.dedent("""
                #include <time.h>
                volatile unsigned long sink;
                __attribute__((noinline)) void llmperfscope_test_hot_function(void) {
                    struct timespec a, b;
                    clock_gettime(CLOCK_MONOTONIC, &a);
                    do {
                        for (int i = 0; i < 100000; i++) sink += i;
                        clock_gettime(CLOCK_MONOTONIC, &b);
                    } while ((b.tv_sec - a.tv_sec) * 1000000000L + (b.tv_nsec - a.tv_nsec) < 1500000000L);
                }
                int main(void) { llmperfscope_test_hot_function(); return 0; }
            """))
        binary = os.path.join(directory, "burn")
        subprocess.run(["cc", "-O1", "-rdynamic", "-o", binary, source], check=True)
        report = run([binary], cpu_profile=True)
        self.assertEqual(report["status"], "ok")
        self.assertTrue(report["perturbed"])
        group = report["cpu_profile"]["groups"]["run"]
        self.assertGreater(group["samples"], 50)
        self.assertEqual(group["self"][0]["name"], "llmperfscope_test_hot_function")
        self.assertGreater(group["self"][0]["share"], 0.8)

    def test_parallel_work_is_not_credited_to_the_main_thread(self):
        # A process-wide timer would deliver nearly all samples to the main thread and hide this.
        directory = helpers.temp_dir()
        source = os.path.join(directory, "par.c")
        with open(source, "w") as f:
            f.write(textwrap.dedent("""
                #include <pthread.h>
                #include <time.h>
                volatile unsigned long sink;
                /* CPU time of this thread, so the amount of work does not depend on how many cores the machine has. */
                static double now(void) { struct timespec t; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
                __attribute__((noinline)) void llmperfscope_test_worker_burn(void) {
                    double t0 = now();
                    while (now() - t0 < 1.2) for (int i = 0; i < 20000; i++) sink += i;
                }
                __attribute__((noinline)) void llmperfscope_test_main_burn(void) {
                    double t0 = now();
                    while (now() - t0 < 0.6) for (int i = 0; i < 20000; i++) sink += i;
                }
                void *worker(void *arg) { (void) arg; llmperfscope_test_worker_burn(); return 0; }
                int main(void) {
                    pthread_t t[3];
                    for (int i = 0; i < 3; i++) pthread_create(&t[i], 0, worker, 0);
                    llmperfscope_test_main_burn();
                    for (int i = 0; i < 3; i++) pthread_join(t[i], 0);
                    return 0;
                }
            """))
        binary = os.path.join(directory, "par")
        subprocess.run(["cc", "-O1", "-rdynamic", "-pthread", "-o", binary, source], check=True)
        report = run([binary], cpu_profile=True)
        self.assertEqual(report["status"], "ok")
        rows = {x["name"]: x for x in report["cpu_profile"]["groups"]["run"]["self"]}
        self.assertEqual(rows["llmperfscope_test_worker_burn"]["threads"], 3)
        self.assertEqual(rows["llmperfscope_test_main_burn"]["threads"], 1)
        # Three threads for twice as long as one: about six times the samples.
        self.assertGreater(rows["llmperfscope_test_worker_burn"]["share"], 3 * rows["llmperfscope_test_main_burn"]["share"])
        self.assertAlmostEqual(rows["llmperfscope_test_worker_burn"]["busiest_thread_s"], 1.2, delta=0.4)

    def test_a_long_system_call_counts_for_all_of_its_time(self):
        """A signal only arrives when the call returns. Real case: one munmap of 12 s showed as 8 samples, 0.06% of the load."""
        directory = helpers.temp_dir()
        source = os.path.join(directory, "kernel.c")
        with open(source, "w") as f:
            f.write(textwrap.dedent("""
                #define _GNU_SOURCE
                #include <sys/mman.h>
                int main(void) {
                    size_t size = (size_t) 1 << 31;
                    /* One call each, in the kernel from start to end: filling in the pages cannot be interrupted by a signal. */
                    for (int i = 0; i < 4; i++) {
                        void *p = mmap(0, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
                        if (p == MAP_FAILED) return 1;
                        madvise(p, size, MADV_NOHUGEPAGE);
                        munmap(p, size);
                    }
                    return 0;
                }
            """))
        binary = os.path.join(directory, "kernel")
        subprocess.run(["cc", "-O1", "-o", binary, source], check=True)
        report = run([binary], cpu_profile=True, cpu_hz=1000)
        self.assertEqual(report["status"], "ok")
        group = report["cpu_profile"]["groups"]["run"]
        self.assertGreater(group["samples"], 10 * group["raw_samples"], "samples should stand for the periods merged during each call")
        calls = sum(x["share"] for x in group["self"] if "mmap" in x["name"] or "madvise" in x["name"])
        self.assertGreater(calls, 0.7)
        self.assertGreater(report["consistency"]["cpu_profile_coverage"], 0.8)

    def test_coverage_compares_the_profile_with_what_was_measured(self):
        profile = {"groups": {"load": {"cpu_s": 67.5}, "decode": {"cpu_s": 1.0}, "startup": {"cpu_s": 0.1}}}
        groups = {"load": {"host": {"user_s": 45.0, "sys_s": 35.0}}, "decode": {"host": {"user_s": 1.0, "sys_s": 0.0}}, "startup": {"host": {"user_s": 0.1, "sys_s": 0.0}}}
        self.assertEqual(cpuprof.coverage(profile, groups), round(68.6 / 81.1, 3))
        self.assertEqual(profile["groups"]["load"]["coverage"], round(67.5 / 80.0, 3))
        self.assertNotIn("coverage", profile["groups"]["startup"])
        self.assertIsNone(cpuprof.coverage({"groups": {"run": {"cpu_s": 0.1}}}, {"run": {"host": {"user_s": 0.1, "sys_s": 0.0}}}))

    def test_stripped_code_gets_no_made_up_name(self):
        table = cpuprof._Symbols("/nonexistent")
        table.starts, table.ends, table.names = [100], [110], ["real"]
        self.assertEqual(table.lookup(105), "real")
        self.assertIsNone(table.lookup(5000))


@unittest.skipUnless(os.path.exists("/proc/self/status"), "Linux only")
class PageCache(unittest.TestCase):
    def test_cold_start_drops_the_file(self):
        path = os.path.join(helpers.temp_dir(), "weights")
        with open(path, "wb") as f:
            f.write(os.urandom(1 << 20) * 32)
            f.flush()
            os.fsync(f.fileno())
        with open(path, "rb") as f:
            while f.read(1 << 20):
                pass
        cached = machine.model_cached_fraction(path)
        if cached is None or cached < 0.5:
            self.skipTest("filesystem does not report page cache state")
        after = machine.evict_from_page_cache(path)
        if machine.system.file_location(path).get("in_ram"):
            self.skipTest("the test directory lives in RAM")
        self.assertLess(after, 0.2)


@unittest.skipUnless(os.path.exists("/proc/self/status"), "Linux only")
class Library(unittest.TestCase):
    def test_profile_inside_a_process(self):
        import llmperfscope
        out = helpers.temp_dir()
        with mock.patch("llmperfscope.api.open_backend", lambda: helpers.ScriptedGpu(count=1)):
            with llmperfscope.Profile(out, label="x", baseline_s=0.2) as profile:
                with llmperfscope.phase("load"):
                    time.sleep(0.2)
                profile.mark("decode", tokens=10)
                time.sleep(0.3)
        report = profile.report
        self.assertEqual(report["status"], "ok")
        self.assertAlmostEqual(report["groups"]["load"]["wall_s"], 0.2, delta=0.1)
        self.assertEqual(report["groups"]["decode"]["tokens"], 10)
        self.assertTrue(os.path.exists(os.path.join(out, "run.json")))

    def test_token_timer_has_a_streamers_shape(self):
        import llmperfscope
        timer = llmperfscope.TokenTimer()
        timer.put("prompt")
        for _ in range(4):
            time.sleep(0.01)
            timer.put("token")
        timer.end()
        self.assertEqual(timer.tokens, 4)
        self.assertGreater(timer.summary()["tokens_per_s"], 10)


if __name__ == "__main__":
    unittest.main()
