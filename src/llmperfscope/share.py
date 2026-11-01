"""`llmperfscope anonymize`: a copy of a run that can be shared. Host and user names, paths, GPU serial identifiers and foreign process
names are removed; prompts and logs are not copied at all. Numbers stay as they are, so figures and comparisons still work."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

# An absolute path as a whole word, so `vllm/vllm-openai:v1` and `1/2` are left alone.
_PATH = re.compile(r"(?<![\w.:-])/(?:[\w.+@-]+/)+([\w.+@-]*)")
_DROPPED_KEYS = {"uuid", "top_processes", "compute_processes", "libs", "stderr_tail"}


def _clean_text(text, names):
    text = _PATH.sub(lambda m: f"<path>/{m.group(1)}", text)
    for name, replacement in names:
        text = re.sub(rf"\b{re.escape(name)}\b", replacement, text)
    return text


def _clean(node, names):
    if isinstance(node, dict):
        return {_clean_text(str(k), names) if isinstance(k, str) else k: _clean(v, names) for k, v in node.items() if k not in _DROPPED_KEYS}
    if isinstance(node, list):
        return [_clean(v, names) for v in node]
    if isinstance(node, str):
        return _clean_text(node, names)
    return node


def anonymize(run_dir, out_dir):
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    if run_dir.resolve() == out_dir.resolve():
        raise SystemExit("anonymize writes a copy: give a different --out")
    with open(run_dir / "run.json") as f:
        report = json.load(f)
    host = report.get("host") or {}
    names = []
    if host.get("hostname"):
        names.append((host["hostname"], "host"))
    if host.get("user"):
        names.append((host["user"], "user"))
    clean = _clean(report, names)
    if "identity" in clean:
        # What ran, without where it was: hashes and commit stay, they identify code and not a machine.
        clean["identity"].pop("exe", None)
    if clean.get("host"):
        clean["host"]["hostname"] = "host"
        clean["host"]["user"] = "user"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "run.json", "w") as f:
        json.dump(clean, f, indent=1)
    for name in ("samples.jsonl.gz",):
        if (run_dir / name).exists():
            shutil.copyfile(run_dir / name, out_dir / name)
    requests = run_dir / "requests.jsonl"
    if requests.exists():
        # Timing only. Ids can name a private prompt set.
        with open(requests) as source, open(out_dir / "requests.jsonl", "w") as target:
            for line in source:
                row = json.loads(line)
                row.pop("id", None)
                row.pop("error", None)
                target.write(json.dumps(row) + "\n")
    return out_dir
