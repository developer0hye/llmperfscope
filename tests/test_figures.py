"""Figures: are they what a paper needs, and do they say what the run says? Skipped without matplotlib."""

import csv
import json
import os
import sys
import unittest
import xml.dom.minidom
from unittest import mock

import helpers
from llmperfscope import cli, compare, figures, runner, share

ENGINE = os.path.join(helpers.HERE, "fake_engine.py")
SERVER = os.path.join(helpers.HERE, "fake_openai_server.py")

try:
    import matplotlib  # noqa: F401
    HAVE_MATPLOTLIB = True
except ImportError:
    HAVE_MATPLOTLIB = False


def make_run(tmp=None):
    """A real run of a process with known behaviour, on GPUs with known power and memory."""
    out = tmp or helpers.temp_dir()
    weights = os.path.join(helpers.temp_dir(), "weights.bin")
    gpu = helpers.ScriptedGpu(count=2, power_w=100.0, used_mib=2000)
    with mock.patch("llmperfscope.runner.open_backend", lambda: gpu):
        runner.Run([sys.executable, ENGINE, "phases", weights], runner.Options(out=out, precheck="skip", quiet=True, baseline_s=0.2, tail_s=0.1, label="known <run>")).execute()
    return out


def make_serving_run():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    out = helpers.temp_dir()
    argv = ["serve", "--out", out, "--precheck", "skip", "--quiet", "--preset", "vllm", "--url", f"http://127.0.0.1:{port}", "--synthetic-words", "16", "--limit", "3", "--max-tokens", "6",
            "--concurrency", "1,2", "--", sys.executable, SERVER, str(port), "0.2", "0.05", "0.01"]
    with mock.patch("llmperfscope.runner.open_backend", lambda: helpers.ScriptedGpu(count=1)), mock.patch("builtins.print"):
        cli.main(argv)
    return out


def add_deep_sections(run_dir):
    """What a run with --nsys and --cpu-profile has, with known numbers."""
    path = os.path.join(run_dir, "run.json")
    with open(path) as f:
        report = json.load(f)
    report["nsys"] = {"groups": {"decode": {"tokens": 100, "devices": [{"device": 0, "span_ms": 100.0, "kernel_ms": 60.0, "comm_ms": 20.0, "copy_ms": 5.0, "idle_ms": 15.0, "kernels": 10, "comm_kernels": 2}],
                                            "top_kernels": [{"name": "mul_mat_q", "ms": 60.0, "share": 0.75, "n": 5, "avg_us": 12000.0}, {"name": "rms_norm", "ms": 20.0, "share": 0.25, "n": 5, "avg_us": 4000.0}]}}}
    report["cpu_profile"] = {"groups": {"decode": {"samples": 500, "self": [{"name": "dequantize_row_q8_0", "share": 0.35, "threads": 1, "cpu_s": 1.0},
                                                                            {"name": "libcuda.so.580.1!?", "share": 0.4, "threads": 2, "cpu_s": 1.2}]}}}
    with open(path, "w") as f:
        json.dump(report, f)


@unittest.skipUnless(HAVE_MATPLOTLIB, "needs matplotlib")
@unittest.skipUnless(os.path.exists("/proc/self/status"), "Linux only")
class FromARun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run_dir = make_run()
        add_deep_sections(cls.run_dir)
        cls.out = helpers.temp_dir()
        cls.written, cls.skipped = figures.make_run_figures(cls.run_dir, cls.out, figures.Style())

    def path(self, name):
        return os.path.join(self.out, name)

    def test_every_format_for_every_figure_that_has_data(self):
        for stem in ("lifecycle", "phases", "memory", "gpu_time", "hotspots"):
            for extension in ("pdf", "png", "svg", "csv"):
                self.assertGreater(os.path.getsize(self.path(f"{stem}.{extension}")), 100, f"{stem}.{extension}")

    def test_a_figure_with_no_data_is_skipped_and_said_so(self):
        self.assertIn("serving", self.skipped)
        self.assertFalse(os.path.exists(self.path("serving.pdf")))

    def test_pdf_fonts_are_truetype_not_type3(self):
        # Conference and journal checks reject Type 3 fonts.
        for stem in ("lifecycle", "memory", "gpu_time"):
            with open(self.path(f"{stem}.pdf"), "rb") as f:
                data = f.read()
            self.assertNotIn(b"/Type3", data, stem)
            self.assertTrue(b"/FontFile2" in data or b"/CIDFontType2" in data or b"/TrueType" in data, stem)

    def test_svg_keeps_text_as_text(self):
        with open(self.path("lifecycle.svg")) as f:
            text = f.read()
        xml.dom.minidom.parseString(text)
        self.assertIn("<text", text)
        self.assertIn("GPU memory", text)

    def test_png_has_print_resolution(self):
        with open(self.path("lifecycle.png"), "rb") as f:
            header = f.read(24)
        width = int.from_bytes(header[16:20], "big")
        self.assertGreater(width, 6.5 * 300)

    def test_same_input_gives_the_same_files(self):
        again = helpers.temp_dir()
        figures.make_run_figures(self.run_dir, again, figures.Style(formats=("pdf", "svg")), kinds=("phases", "memory"))
        for name in ("phases.svg", "phases.pdf", "memory.svg", "memory.pdf"):
            with open(self.path(name), "rb") as a, open(os.path.join(again, name), "rb") as b:
                self.assertEqual(a.read(), b.read(), name)

    def test_the_csv_holds_what_the_figure_shows(self):
        with open(self.path("memory.csv")) as f:
            rows = list(csv.DictReader(f))
        with open(os.path.join(self.run_dir, "run.json")) as f:
            report = json.load(f)
        peak = report["groups"]["decode"]["gpu"]["mem_peak_mib"] / 1024
        cell = [r for r in rows if r["panel"].startswith("GPU memory") and r["row"] == "decode"][0]
        self.assertAlmostEqual(float(cell["gib"]), peak, places=6)
        with open(self.path("lifecycle.csv")) as f:
            panels = {row["panel"] for row in csv.DictReader(f)}
        self.assertLessEqual({"gpu_memory", "power", "util", "host_memory", "cpu"}, panels)

    def test_known_power_comes_through_in_the_data(self):
        with open(self.path("lifecycle.csv")) as f:
            power = [float(r["value"]) for r in csv.DictReader(f) if r["panel"] == "power"]
        self.assertAlmostEqual(sum(power) / len(power), 200.0, delta=20.0)  # 2 GPUs at 100 W

    def test_captions_state_what_was_measured(self):
        with open(self.path("captions.md")) as f:
            text = f.read()
        for needle in ("**lifecycle**", "**memory**", "known <run>", "2 x scripted", "Measured with llmperfscope", "50 Hz"):
            self.assertIn(needle, text)

    def test_traced_figures_warn_about_speed(self):
        with open(self.path("captions.md")) as f:
            text = f.read()
        self.assertIn("not for speed", text)

    def test_hotspots_name_stripped_libraries_and_thread_counts(self):
        with open(self.path("hotspots.svg")) as f:
            text = f.read()
        self.assertIn("GPU driver", text)
        self.assertIn("1 thr", text)
        self.assertNotIn("(no symbols)", text)

    def test_options_for_a_paper(self):
        out = helpers.temp_dir()
        written, _ = figures.make_run_figures(self.run_dir, out, figures.Style(width=figures.SINGLE_COLUMN_IN, formats=("svg",), font="serif", print_safe=True, letters=True, title="A title"),
                                              kinds=("lifecycle",), panels=("gpu_memory", "host_memory"), per_gpu=True, xlim=(0.0, 1.0))
        with open(os.path.join(out, "lifecycle.svg")) as f:
            text = f.read()
        self.assertIn("A title", text)
        self.assertIn("(a)", text)
        self.assertIn("GPU 1", text)
        self.assertNotIn("Host CPU", text)
        self.assertIn("stroke-dasharray", text)  # the second line of a panel is dashed for grayscale print

    def test_widths(self):
        self.assertEqual(figures.parse_width("single"), 3.5)
        self.assertEqual(figures.parse_width("double"), 7.16)
        self.assertEqual(figures.parse_width("5"), 5.0)

    def test_command_line(self):
        out = helpers.temp_dir()
        with mock.patch("builtins.print"):
            self.assertEqual(cli.main(["figure", self.run_dir, "--kind", "phases", "--out", out, "--format", "svg", "--width", "single", "--letters", "--font", "serif"]), 0)
        self.assertTrue(os.path.exists(os.path.join(out, "phases.svg")))
        with self.assertRaises(SystemExit):
            cli.main(["figure", self.run_dir, "--kind", "nonsense", "--out", out])
        with self.assertRaises(SystemExit):
            cli.main(["figure", self.run_dir, "--panels", "nonsense", "--out", out])


@unittest.skipUnless(HAVE_MATPLOTLIB, "needs matplotlib")
@unittest.skipUnless(os.path.exists("/proc/self/status"), "Linux only")
class Serving(unittest.TestCase):
    def test_serving_and_compare(self):
        a, b = make_serving_run(), make_serving_run()
        out = helpers.temp_dir()
        written, skipped = figures.make_run_figures(a, out, figures.Style(formats=("svg",)), kinds=("serving",))
        self.assertEqual(skipped, [])
        with open(os.path.join(out, "serving.csv")) as f:
            rows = list(csv.DictReader(f))
        ttft = [float(r["value"]) for r in rows if r["series"] == "ttft_ms"]
        self.assertAlmostEqual(sorted(ttft)[len(ttft) // 2], 50, delta=30)  # the fake server waits 50 ms
        self.assertTrue([r for r in rows if r["series"] == "itl_ms"])
        with open(os.path.join(out, "serving.svg")) as f:
            self.assertIn("2 clients", f.read())

        out = helpers.temp_dir()
        written, skipped = figures.make_compare_figures([("first", a), ("second", b)], out, figures.Style(formats=("svg",)))
        names = {os.path.basename(str(p)) for p in written}
        self.assertLessEqual({"compare_serving.svg", "compare_memory.svg", "compare_phases.svg"}, names)
        self.assertIn("speed", skipped)
        with open(os.path.join(out, "compare_serving.svg")) as f:
            text = f.read()
        self.assertIn("first", text)
        self.assertIn("second", text)


@unittest.skipUnless(HAVE_MATPLOTLIB, "needs matplotlib")
class Matrix(unittest.TestCase):
    def test_matrix_figure(self):
        directory = helpers.temp_dir()
        a = {"status": "ok", "throughput": {"decode": {"tokens_per_s": 100.0}}, "memory": {"gpu": {"peak_above_baseline_total_mib": 1000}}}
        b = {"status": "ok", "throughput": {"decode": {"tokens_per_s": 120.0}}, "memory": {"gpu": {"peak_above_baseline_total_mib": 2000}}}
        case = compare.diff([a, a], [b, b])
        case["label"] = "a case"
        path = os.path.join(directory, "matrix.json")
        with open(path, "w") as f:
            json.dump({"a": "old", "b": "new", "order": "ABBA", "cases": [case, {"label": "missing", "skipped": "model missing"}]}, f)
        out = os.path.join(directory, "figures")
        figures.make_matrix_figure(path, out, figures.Style(formats=("svg",)))
        with open(os.path.join(out, "matrix.csv")) as f:
            rows = {r["metric"]: r for r in csv.DictReader(f)}
        self.assertEqual(rows["decode_tps"]["verdict"], "better")
        self.assertEqual(rows["gpu_peak_mib"]["verdict"], "worse")
        with open(os.path.join(out, "captions.md")) as f:
            self.assertIn("1 results were better and 1 worse", f.read())


class WithoutMatplotlib(unittest.TestCase):
    def test_the_message_says_what_to_install(self):
        with mock.patch.dict(sys.modules, {"matplotlib": None}):
            with self.assertRaises(SystemExit) as caught:
                figures._matplotlib()
        self.assertIn("llmperfscope[figures]", str(caught.exception))


@unittest.skipUnless(os.path.exists("/proc/self/status"), "Linux only")
class Anonymize(unittest.TestCase):
    def setUp(self):
        self.run_dir = make_run()
        self.path = os.path.join(self.run_dir, "run.json")
        with open(self.path) as f:
            report = json.load(f)
        report["host"].update(hostname="secret-box", user="alice")
        report["identity"]["exe"] = "/home/alice/private/engine"
        report["command"] = ["/home/alice/private/engine", "-m", "/data/alice/model.gguf", "--port", "1", "vendor/image:v1"]
        report["devices"][0]["uuid"] = "GPU-1234"
        report["machine_before"] = {"top_processes": [{"comm": "secret-tool"}], "model_storage": {"path": "/data/alice/model.gguf"}, "clean": True}
        report["notes"] = ["seen on secret-box by alice"]
        with open(self.path, "w") as f:
            json.dump(report, f)
        self.out = os.path.join(helpers.temp_dir(), "shared")
        with open(os.path.join(self.run_dir, "requests.jsonl"), "w") as f:
            f.write(json.dumps({"id": "private-prompt", "ttft_s": 0.1, "error": "/home/alice/x", "level": 1}) + "\n")
        share.anonymize(self.run_dir, self.out)
        with open(os.path.join(self.out, "run.json")) as f:
            self.text = f.read()
        self.report = json.loads(self.text)

    def test_names_paths_and_serial_numbers_are_gone(self):
        for secret in ("secret-box", "alice", "GPU-1234", "secret-tool", "/home/", "/data/", "private/engine"):
            self.assertNotIn(secret, self.text, secret)

    def test_what_is_useful_stays(self):
        self.assertIn("vendor/image:v1", self.report["command"])
        self.assertIn("<path>/model.gguf", self.report["command"])
        with open(self.path) as f:
            original = json.load(f)
        self.assertEqual(self.report["groups"]["decode"]["wall_s"], original["groups"]["decode"]["wall_s"])

    def test_logs_and_prompts_are_not_copied(self):
        self.assertEqual(sorted(os.listdir(self.out)), ["requests.jsonl", "run.json", "samples.jsonl.gz"])
        with open(os.path.join(self.out, "requests.jsonl")) as f:
            text = f.read()
        self.assertNotIn("private-prompt", text)
        self.assertNotIn("alice", text)
        self.assertIn("ttft_s", text)

    def test_it_will_not_overwrite_the_original(self):
        with self.assertRaises(SystemExit):
            share.anonymize(self.run_dir, self.run_dir)

    @unittest.skipUnless(HAVE_MATPLOTLIB, "needs matplotlib")
    def test_figures_work_on_the_copy(self):
        out = helpers.temp_dir()
        _, skipped = figures.make_run_figures(self.out, out, figures.Style(formats=("svg",)), kinds=("lifecycle", "phases"))
        self.assertEqual(skipped, [])


if __name__ == "__main__":
    unittest.main()
