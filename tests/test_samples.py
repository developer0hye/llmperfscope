"""The runs in examples/sample-runs: they read and plot on any machine, and they name nobody. Works on any OS. Not in the source archive."""

import contextlib
import gzip
import io
import json
import os
import re
import unittest
from unittest import mock

import helpers
from llmperfscope import capabilities, cli, compare, figures, kernels, report

SAMPLES = os.path.join(os.path.dirname(helpers.HERE), "examples", "sample-runs")
PRIVATE = re.compile(r"/home/|/Users/|/tmp/|\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-|GPU-[0-9a-f]{6}", re.I)

try:
    import matplotlib  # noqa: F401
    HAVE_MATPLOTLIB = True
except ImportError:
    HAVE_MATPLOTLIB = False


def runs():
    return sorted(root for root, _, files in os.walk(SAMPLES) if "run.json" in files)


class OtherSystems(unittest.TestCase):
    """Measuring needs Linux or macOS. Elsewhere it has to say so, and not fall over on what is missing."""

    def _refused(self, argv):
        stderr = io.StringIO()
        with mock.patch("sys.platform", "win32"), contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(argv), 3)
        self.assertIn("needs Linux or macOS", stderr.getvalue())
        self.assertIn("figure", stderr.getvalue())

    def test_run(self):
        self._refused(["run", "--quiet", "--out", helpers.temp_dir(), "--", "true"])

    def test_serve(self):
        self._refused(["serve", "--quiet", "--out", helpers.temp_dir(), "--url", "http://127.0.0.1:1", "--", "llama-server"])

    def test_attach(self):
        self._refused(["attach", "--quiet", "--out", helpers.temp_dir(), "--pid", "1"])

    def test_matrix(self):
        self._refused(["matrix", "--cases", "/nonexistent", "--a", "x=/x", "--b", "y=/y", "--out", helpers.temp_dir()])

    def test_check_tells_what_works(self):
        with mock.patch("sys.platform", "win32"):
            text = capabilities.render(capabilities.collect())
        self.assertIn("needs Linux or macOS", text)
        self.assertIn("reading runs", text)
        self.assertNotIn("nsys", text)


@unittest.skipUnless(os.path.isdir(SAMPLES), "the sample runs are not in the source archive")
class SampleRuns(unittest.TestCase):
    def test_there_are_samples(self):
        self.assertGreaterEqual(len(runs()), 10)

    def test_each_one_reads_and_reports(self):
        for run in runs():
            data = compare.load_run(run)
            self.assertEqual(data["schema"], "llmperfscope.run/1", run)
            self.assertEqual(data["tool"]["name"], "llmperfscope", run)
            self.assertEqual(data["status"], "ok", run)
            self.assertIn("What limits each phase", report.render(data), run)

    def test_nothing_names_a_person_or_a_machine(self):
        for root, _, files in os.walk(SAMPLES):
            for name in files:
                path = os.path.join(root, name)
                opener = gzip.open if name.endswith(".gz") else open
                with opener(path, "rt", errors="replace") as f:
                    text = f.read()
                self.assertIsNone(PRIVATE.search(text), path)
                if name == "run.json":
                    host = json.loads(text).get("host") or {}
                    self.assertEqual((host.get("hostname"), host.get("user")), ("host", "user"), path)

    def test_no_logs_or_prompts(self):
        for run in runs():
            self.assertFalse({"stdout.log", "stderr.log", "report.html", "markers.log"} & set(os.listdir(run)), run)
            requests = os.path.join(run, "requests.jsonl")
            if os.path.exists(requests):
                with open(requests) as f:
                    for line in f:
                        row = json.loads(line)
                        self.assertFalse({"prompt", "messages", "id", "text"} & set(row), requests)

    def test_kernel_table_from_a_real_trace(self):
        traced = compare.load_run(os.path.join(SAMPLES, "deepseek-v4.1-flash-8gpu-nsys"))
        untraced = compare.load_run(os.path.join(SAMPLES, "deepseek-v4.1-flash-8gpu-cpu-profile"))
        text = kernels.render(traced, untraced)
        self.assertIn("ms per token without tracing", text)
        self.assertIn("mul_mat_vec_q", text)
        self.assertIn("(collective)", text)
        data = kernels.per_token(traced["nsys"]["groups"]["decode"])
        token_ms = 1000.0 / untraced["throughput"]["decode"]["tokens_per_s"]
        # What the trace says kernels and collectives cost has to fit inside a token measured without it.
        self.assertLess(data["kernel_ms"] + data["collective_ms"], token_ms)

    def test_ram_limited_run_says_ram(self):
        data = compare.load_run(os.path.join(SAMPLES, "cpu-ram-limited"))
        self.assertEqual(data["diagnosis"]["bottlenecks"]["decode"]["limit"], "ram")
        enough = compare.load_run(os.path.join(SAMPLES, "cpu-enough-ram"))
        self.assertGreater(enough["throughput"]["decode"]["tokens_per_s"] / data["throughput"]["decode"]["tokens_per_s"], 10)

    @unittest.skipUnless(HAVE_MATPLOTLIB, "needs matplotlib")
    def test_figures_for_each_run(self):
        out = helpers.temp_dir()
        style = figures.Style(formats=("svg",), dpi=100)
        for index, run in enumerate(runs()):
            written, _ = figures.make_run_figures(run, os.path.join(out, str(index)), style)
            self.assertTrue(any(str(path).endswith("lifecycle.svg") for path in written), run)

    @unittest.skipUnless(HAVE_MATPLOTLIB, "needs matplotlib")
    def test_figures_show_what_each_run_measured(self):
        out = helpers.temp_dir()
        style = figures.Style(formats=("svg",), dpi=100)
        for sample, kind in (("deepseek-v4.1-flash-8gpu-nsys", "gpu-time"), ("deepseek-v4.1-flash-8gpu-cpu-profile", "hotspots"), ("deepseek-v4.1-flash-8gpu-serve", "serving")):
            _, skipped = figures.make_run_figures(os.path.join(SAMPLES, sample), os.path.join(out, sample), style, (kind,))
            self.assertEqual(skipped, [], sample)

    @unittest.skipUnless(HAVE_MATPLOTLIB, "needs matplotlib")
    def test_nothing_touches_the_edge_of_the_image(self):
        """Matplotlib moves a figure's legend to the corner of the cropped image, where text could be cut off."""
        import matplotlib.image
        out = helpers.temp_dir()
        style = figures.Style(formats=("png",), dpi=200)
        written, _ = figures.make_matrix_figure(os.path.join(SAMPLES, "matrix.json"), os.path.join(out, "matrix"), style)
        written += figures.make_compare_figures([("a", os.path.join(SAMPLES, "engines", "vllm")), ("b", os.path.join(SAMPLES, "engines", "sglang"))], os.path.join(out, "compare"), style)[0]
        written += figures.make_run_figures(os.path.join(SAMPLES, "deepseek-v4.1-flash-8gpu-serve"), os.path.join(out, "serve"), style, ("serving",))[0]
        written += figures.make_run_figures(os.path.join(SAMPLES, "deepseek-v4.1-flash-8gpu-nsys"), os.path.join(out, "nsys"), style, ("gpu-time",))[0]
        images = [str(path) for path in written if str(path).endswith(".png")]
        self.assertGreaterEqual(len(images), 7)
        for path in images:
            ink = matplotlib.image.imread(path)[:, :, :3].min(axis=2) < 0.98
            self.assertFalse(ink[0].any() or ink[-1].any() or ink[:, 0].any() or ink[:, -1].any(), path)

    @unittest.skipUnless(HAVE_MATPLOTLIB, "needs matplotlib")
    def test_matrix_figure(self):
        out = helpers.temp_dir()
        written, _ = figures.make_matrix_figure(os.path.join(SAMPLES, "matrix.json"), out, figures.Style(formats=("svg",), dpi=100))
        self.assertTrue(any(str(path).endswith("matrix.svg") for path in written))


if __name__ == "__main__":
    unittest.main()
