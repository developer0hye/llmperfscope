"""The macOS readers against things whose answer is known. These run on a Mac only, and what a machine has no GPU for is skipped."""

import os
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest

import helpers
from llmperfscope import machine, procfs, runner, system
from llmperfscope.gpu import open_backend

if sys.platform == "darwin":
    from llmperfscope import darwin

HERE = os.path.dirname(os.path.abspath(__file__))
MIB = 1 << 20


def child(code):
    """A process doing `code`. It prints a line when it has done the part to look at, then waits to be closed."""
    process = subprocess.Popen([sys.executable, "-I", "-c", textwrap.dedent(code) + "\nprint('ready', flush=True)\nimport sys\nsys.stdin.read()\n"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    process.stdout.readline()
    return process


def finish(process):
    process.stdin.close()
    process.wait(timeout=10)
    process.stdout.close()


@unittest.skipUnless(sys.platform == "darwin", "macOS only")
class Processes(unittest.TestCase):
    def test_cpu_time_is_in_ticks(self):
        process = child("""
            import time
            start = time.process_time()
            while time.process_time() - start < 1.0:
                sum(range(1000))
        """)
        cpu = procfs.stat(process.pid)
        seconds = (cpu["utime"] + cpu["stime"]) / procfs.CLK_TCK
        finish(process)
        self.assertGreater(seconds, 0.9)
        self.assertLess(seconds, 1.5)

    def test_thread_times_add_up_to_the_process(self):
        process = child("""
            import threading, time
            done = threading.Semaphore(0)
            def burn():
                start = time.thread_time()
                while time.thread_time() - start < 0.5:
                    sum(range(1000))
                done.release()
                time.sleep(30)
            for _ in range(2):
                threading.Thread(target=burn, daemon=True).start()
            done.acquire()
            done.acquire()
        """)
        ticks = sum(t[2] for t in procfs.threads(process.pid))
        cpu = procfs.stat(process.pid)
        finish(process)
        self.assertAlmostEqual(ticks, cpu["utime"] + cpu["stime"], delta=0.2 * procfs.CLK_TCK)

    def test_an_unnamed_thread_goes_by_the_process_name(self):
        process = child("x = 1")
        names = {t[1] for t in procfs.threads(process.pid)}
        comm = procfs.stat(process.pid)["comm"]
        finish(process)
        self.assertEqual(names, {comm})

    def test_memory_is_told_apart_by_kind(self):
        with tempfile.NamedTemporaryFile() as mapped:
            mapped.write(b"\1" * (128 * MIB))
            mapped.flush()
            process = child(f"""
                import mmap
                anon = bytearray(256 << 20)
                for i in range(0, len(anon), 16384):
                    anon[i] = 1
                with open({mapped.name!r}, "rb") as f:
                    mapping = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
                total = sum(mapping[i] for i in range(0, len(mapping), 16384))
            """)
            memory = procfs.status(process.pid)
            finish(process)
        self.assertGreater(memory["anon"], 240 * 1024)
        self.assertLess(memory["anon"], 320 * 1024)
        self.assertGreater(memory["file"], 120 * 1024)
        self.assertGreaterEqual(memory["hwm"], memory["rss"])

    def test_disk_reads_are_counted(self):
        with tempfile.NamedTemporaryFile() as data:
            data.write(os.urandom(64 * MIB))
            data.flush()
            os.fsync(data.fileno())
            machine.evict_from_page_cache(data.name)
            process = child(f"""
                with open({data.name!r}, "rb") as f:
                    while f.read(1 << 20):
                        pass
            """)
            counters = procfs.io(process.pid)
            finish(process)
        self.assertGreater(counters["read_bytes"], 48 * MIB)

    def test_a_child_and_its_child_are_in_the_tree(self):
        process = child("""
            import subprocess, sys
            inner = subprocess.Popen([sys.executable, "-I", "-c", "import time; time.sleep(30)"])
            print(inner.pid, flush=True)
        """)
        pids = procfs.tree({process.pid})
        self.assertEqual(len(pids), 2)
        inner = (pids - {process.pid}).pop()
        self.assertTrue(procfs.alive(inner))
        self.assertEqual(procfs.owner(inner), os.getuid())
        self.assertIn(inner, procfs.all_pids())
        self.assertTrue(procfs.exe_path(inner))
        os.kill(inner, 9)
        finish(process)

    def test_found_by_its_environment(self):
        entry = f"LLMPERFSCOPE_TEST_TAG={time.monotonic_ns()}"
        name, _, value = entry.partition("=")
        process = subprocess.Popen([sys.executable, "-I", "-c", "import time; time.sleep(30)"], env={**os.environ, name: value})
        for _ in range(100):
            found = darwin.with_environment(entry)
            if found:
                break
            time.sleep(0.05)
        process.kill()
        process.wait()
        self.assertEqual(found, {process.pid})


@unittest.skipUnless(sys.platform == "darwin", "macOS only")
class System(unittest.TestCase):
    def test_memory(self):
        memory = system.meminfo()
        self.assertEqual(memory["MemTotal"], darwin.sysctl_int("hw.memsize") >> 20)
        self.assertLessEqual(memory["MemAvailable"], memory["MemTotal"])
        self.assertLessEqual(memory["MemFree"], memory["MemAvailable"])

    def test_counters_only_grow(self):
        first, disks = system.cpu_times(), system.disk_counters(set(system.list_disks()))
        memory_before = system.vmstat()
        time.sleep(0.2)
        second, disks_after = system.cpu_times(), system.disk_counters(set(system.list_disks()))
        memory_after = system.vmstat()
        self.assertTrue(all(b >= a for a, b in zip(first, second)))
        self.assertGreater(sum(second), sum(first))
        for name, counters in disks.items():
            self.assertTrue(all(b >= a for a, b in zip(counters, disks_after[name])))
        for key, value in memory_before.items():
            self.assertGreaterEqual(memory_after[key], value)

    def test_vm_counters_are_in_linux_pages(self):
        self.assertEqual(set(system.vmstat()), {"pswpin", "pswpout", "pgmajfault", "compressions", "decompressions"})

    def test_network_counters_survive_wrapping(self):
        before = list(darwin._net_total)
        darwin._add_network(9999, (0xFFFFFF00, 10))
        self.assertEqual(darwin._net_total, before)
        darwin._add_network(9999, (0x100, 15))
        self.assertEqual(darwin._net_total[0] - before[0], 0x200)
        self.assertEqual(darwin._net_total[1] - before[1], 5)
        del darwin._net_last[9999]

    def test_a_volume_leads_to_a_disk_with_counters(self):
        location = system.file_location(__file__)
        self.assertEqual(location["fstype"], "apfs")
        known = system.list_disks()
        for disk in location.get("disks", []):
            self.assertIn(disk["name"], known)

    def test_a_missing_file_has_no_location(self):
        self.assertIsNone(system.file_location("/nonexistent/file"))

    def test_pressure_is_cumulative(self):
        first = system.pressure()
        time.sleep(0.05)
        second = system.pressure()
        self.assertGreaterEqual(second["memory_some"], first["memory_some"])

    def test_host(self):
        host = machine.host_info()
        self.assertTrue(host["os"].startswith("macOS"))
        self.assertTrue(host["cpu"])

    def test_cpu_load_sees_other_processes(self):
        process = subprocess.Popen([sys.executable, "-I", "-c", "\nwhile True:\n    sum(range(1000))\n"])
        time.sleep(0.2)
        load = machine.cpu_load(1.0)
        process.kill()
        process.wait()
        entry = next((r for r in load if r["pid"] == process.pid), None)
        self.assertIsNotNone(entry)
        self.assertGreater(entry["cores"], 0.5)
        self.assertEqual(entry["comm"], entry["comm"].strip())


@unittest.skipUnless(sys.platform == "darwin", "macOS only")
class PageCache(unittest.TestCase):
    def test_eviction_makes_a_cold_file(self):
        with tempfile.NamedTemporaryFile() as data:
            data.write(os.urandom(64 * MIB))
            data.flush()
            os.fsync(data.fileno())
            with open(data.name, "rb") as f:
                while f.read(MIB):
                    pass
            self.assertGreater(machine.model_cached_fraction(data.name), 0.5)
            self.assertLess(machine.evict_from_page_cache(data.name), 0.1)
            with open(data.name, "rb") as f:
                while f.read(MIB):
                    pass
            self.assertGreater(machine.model_cached_fraction(data.name), 0.5)


def can_burn():
    if sys.platform != "darwin":
        return False
    backend = open_backend()
    usable = (backend.name == "apple" and backend.fast()[0][0] is not None and backend.energy_mj(0) is not None and bool(backend.clock.frequencies)
              and any(ns for _pid, ns in backend.accelerator.clients().values()))
    backend.close()
    return usable


@unittest.skipUnless(can_burn(), "needs a GPU that publishes its memory, energy, clock and clients, which a virtual machine's does not")
class Gpu(unittest.TestCase):
    def _run(self, cmd):
        return runner.Run(cmd, runner.Options(out=helpers.temp_dir(), quiet=True, precheck="skip")).execute()

    def test_the_gpu_is_described(self):
        backend = open_backend()
        device = backend.devices[0]
        row = backend.fast()[0]
        backend.close()
        self.assertTrue(device["unified_memory"])
        self.assertGreater(device["memory_total_mib"], 1024)
        self.assertLessEqual(device["memory_total_mib"], system.meminfo()["MemTotal"])
        self.assertGreater(row[0], 0)
        self.assertGreaterEqual(row[5], 0)

    def test_gpu_work_is_credited_to_the_process_that_did_it(self):
        report = self._run([sys.executable, "-I", os.path.join(HERE, "metal_burn.py"), "4"])
        self.assertEqual(report["devices_used"], [0])
        gpu = report["groups"]["run"]["gpu"]
        self.assertGreater(gpu["util_avg"], 50)
        self.assertGreater(gpu["energy_j"], 0.5)
        self.assertGreater(gpu["power_avg_w"], 0.1)
        self.assertGreater(gpu["sm_mhz_avg"], 100)
        self.assertLess(gpu["sm_mhz_avg"], 4000)

    def test_a_process_without_gpu_work_gets_no_gpu_numbers(self):
        report = self._run([sys.executable, "-I", "-c", "import time\nstart = time.time()\nwhile time.time() - start < 2: sum(range(1000))"])
        self.assertEqual(report["devices_used"], [])
        self.assertNotIn("gpu", report["groups"]["run"])

    def test_gpu_time_of_someone_else_is_not_ours(self):
        burner = subprocess.Popen([sys.executable, "-I", os.path.join(HERE, "metal_burn.py"), "8"])
        try:
            time.sleep(1.5)
            backend = open_backend()
            backend.set_target({os.getpid()})
            backend.slow()
            busy = backend.foreign_busy(1.0)
            row = backend.slow()[0]
            samples = backend.util_samples()[0]
            backend.close()
        finally:
            burner.kill()
            burner.wait()
        self.assertGreater(busy[burner.pid], 20)
        self.assertEqual(row["gpu_ns"], {})
        self.assertEqual([value for _, value in samples], [0.0])

    def test_the_counters_keep_growing(self):
        backend = open_backend()
        first = backend.energy_mj(0)
        burner = subprocess.Popen([sys.executable, "-I", os.path.join(HERE, "metal_burn.py"), "2"])
        burner.wait()
        second = backend.energy_mj(0)
        backend.close()
        self.assertGreater(second, first)


if __name__ == "__main__":
    unittest.main()
