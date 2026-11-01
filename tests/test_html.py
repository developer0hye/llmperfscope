"""The HTML report is a page people open once and trust; these check its content, not its look."""

import json
import os
import unittest
import xml.dom.minidom

import helpers
from llmperfscope import analyze, diagnose, html
from test_parsers import _ticks


class Page(unittest.TestCase):
    def setUp(self):
        self.ticks = _ticks()
        phases = [{"name": "baseline", "group": "baseline", "t": 0}, {"name": "load", "group": "load", "t": int(1e9)},
                  {"name": "decode", "group": "decode", "t": int(3e9), "tokens": 500}, {"name": "tail", "group": "tail", "t": int(8e9)}]
        out = analyze.summarize(self.ticks, phases, int(10e9))
        for phase in phases:
            phase["start_s"] = phase.pop("t") / 1e9
            phase["end_s"] = phase.pop("t_end") / 1e9
        self.run_report = {"status": "ok", "label": "<b>x</b>", "phases": phases, "disks": {"sda": {"kind": "hdd"}},
                           "throughput": {"decode": {"tokens_per_s": 5.0, "stddev": 0.1}}, **out}
        self.run_report["diagnosis"] = diagnose.diagnose(self.run_report)

    def test_the_page_has_the_story(self):
        page = html.render(self.run_report, self.ticks)
        for needle in ("What limits each phase", "GPU memory", "GPU power", "Host memory of the process", "Disk reads", "sda (hdd)", "PCIe traffic", "Phases"):
            self.assertIn(needle, page)

    def test_text_from_a_run_cannot_inject_markup(self):
        page = html.render(self.run_report, self.ticks)
        self.assertNotIn("<b>x</b>", page)
        self.assertIn("&lt;b&gt;x&lt;/b&gt;", page)

    def test_charts_are_well_formed(self):
        page = html.render(self.run_report, self.ticks)
        charts = page.split("<svg")[1:]
        self.assertGreater(len(charts), 5)
        for chart in charts:
            xml.dom.minidom.parseString("<svg" + chart.split("</svg>")[0] + "</svg>")

    def test_no_network_and_no_scripts(self):
        page = html.render(self.run_report, self.ticks)
        self.assertNotIn("<script", page)
        self.assertNotIn("http://", page.replace("http://www.w3.org", ""))
        self.assertNotIn("https://", page)

    def test_written_next_to_the_run(self):
        directory = helpers.temp_dir()
        import gzip
        with gzip.open(os.path.join(directory, "samples.jsonl.gz"), "wt") as f:
            for tick in self.ticks:
                f.write(json.dumps(tick) + "\n")
        with open(os.path.join(directory, "run.json"), "w") as f:
            json.dump(self.run_report, f)
        self.assertTrue(html.write(directory).exists())


if __name__ == "__main__":
    unittest.main()
