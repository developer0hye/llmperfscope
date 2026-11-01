"""The docs have to match the tool: every option in a command exists, every link points at a file, and every finding is explained."""

import glob
import os
import re
import shlex
import unittest

import helpers
from llmperfscope import __version__, cli, figures

ROOT = os.path.dirname(helpers.HERE)
GITHUB = re.compile(r"https://github\.com/developer0hye/llmperfscope/(?:blob|tree)/main/([^)#\s]+)")
RAW = re.compile(r"https://raw\.githubusercontent\.com/developer0hye/llmperfscope/main/([^)#\s]+)")
# Options of the engines and tools that the docs talk about.
NOT_OURS = {"--metrics", "--progress", "--enable-metrics", "--port", "--cidfile", "--verbose-timings", "--model-id", "--model-path", "--type", "--gpus", "--help", "--user", "--scope"}


def documents():
    pages = sorted(os.path.relpath(path, ROOT) for path in glob.glob(os.path.join(ROOT, "docs", "*.md")))
    names = ["README.md", "CHANGELOG.md", "CONTRIBUTING.md"] + pages + ["examples/sample-runs/README.md"]
    found = []
    for name in names:
        path = os.path.join(ROOT, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                found.append((name, f.read()))
    return found


def subparsers():
    parser = cli.build_parser()
    return next(action.choices for action in parser._actions if hasattr(action, "choices") and action.choices and "run" in action.choices)


def commands(text):
    """Command lines in sh blocks, and in inline code. Placeholders such as `<command>` stay out."""
    for block in re.findall(r"```sh\n(.*?)```", text, flags=re.S):
        for line in block.splitlines():
            if line.startswith("llmperfscope "):
                yield line
    for inline in re.findall(r"`(llmperfscope [^`]+)`", text):
        if "<" not in inline:
            yield inline


def words_of(line):
    return shlex.split(line, comments=True)


class Commands(unittest.TestCase):
    def test_every_command_and_option_exists(self):
        parsers = subparsers()
        checked = 0
        for name, text in documents():
            for line in commands(text):
                words = words_of(line)
                if len(words) < 2 or words[1].startswith("--") or words[1] == "...":
                    continue
                self.assertIn(words[1], parsers, f"{name}: {line}")
                known = parsers[words[1]]._option_string_actions
                for word in words[2:]:
                    if word == "--":
                        break
                    if word.startswith("--"):
                        self.assertIn(word.split("=")[0], known, f"{name}: {line}")
                        checked += 1
        self.assertGreater(checked, 40)

    def test_figure_kinds_and_panels_in_the_docs_exist(self):
        for name, text in documents():
            for line in commands(text):
                words = words_of(line)
                if len(words) < 2 or words[1] != "figure":
                    continue
                for flag, allowed in (("--kind", set(figures.RUN_KINDS) | set(figures.COMPARE_KINDS)), ("--panels", set(figures.ALL_PANELS))):
                    if flag in words:
                        for value in words[words.index(flag) + 1].split(","):
                            self.assertIn(value, allowed, f"{name}: {line}")

    def test_options_mentioned_in_prose_exist(self):
        known = set()
        for parser in subparsers().values():
            known |= set(parser._option_string_actions)
        for name, text in documents():
            prose = re.sub(r"```.*?```", "", text, flags=re.S)
            for code in re.findall(r"`([^`]+)`", prose):
                if code.startswith(("llmperfscope ", "--")):
                    command = words_of(code) if "'" not in code else code.split()
                    for word in command:
                        if word == "--":
                            break
                        if word.startswith("--"):
                            option = word.split("=")[0]
                            self.assertTrue(option in known or option in NOT_OURS, f"{name}: {option} in `{code}`")


class Links(unittest.TestCase):
    def test_every_link_points_at_a_file(self):
        checked = 0
        for name, text in documents():
            here = os.path.dirname(os.path.join(ROOT, name))
            for target in re.findall(r"\]\(([^)\s]+)", text):
                match = GITHUB.match(target) or RAW.match(target)
                if match:
                    path = os.path.join(ROOT, match.group(1))
                elif target.startswith(("http://", "https://", "#")):
                    continue
                else:
                    path = os.path.join(here, target.split("#")[0])
                self.assertTrue(os.path.exists(path), f"{name}: {target}")
                checked += 1
        self.assertGreater(checked, 40)

    def test_the_readme_works_on_pypi(self):
        """PyPI shows the README on its own, where a relative link leads nowhere."""
        text = dict(documents())["README.md"]
        for target in re.findall(r"\]\(([^)\s]+)", text):
            self.assertTrue(target.startswith("https://"), target)


class Explanations(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(ROOT, "src", "llmperfscope", "diagnose.py"), encoding="utf-8") as f:
            self.source = f.read()
        self.doc = dict(documents())["docs/reading-reports.md"]

    def test_every_finding_is_explained(self):
        names = set(re.findall(r'add\(\s*"(?:warning|info)"(?: if [^,]*? else "(?:warning|info)")?,\s*"(\w+)"', self.source))
        names |= set(re.findall(r'"what": "(\w+)"', self.source))
        self.assertGreater(len(names), 25)
        for name in names:
            self.assertIn(f"`{name}`", self.doc)

    def test_every_verdict_is_explained(self):
        verdicts = {v for v in re.findall(r'"limit": f?"([^"]+)"', self.source) if "{" not in v}
        self.assertGreater(len(verdicts), 10)
        for verdict in verdicts:
            self.assertIn(f"`{verdict}`", self.doc)


class Versions(unittest.TestCase):
    def test_every_file_names_the_same_version(self):
        with open(os.path.join(ROOT, "CITATION.cff"), encoding="utf-8") as f:
            self.assertIn(f"version: {__version__}\n", f.read())
        self.assertIn(f"## {__version__}\n", dict(documents())["CHANGELOG.md"])


if __name__ == "__main__":
    unittest.main()
