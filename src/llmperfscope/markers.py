"""Phase markers: the only thing an engine has to do to get finer phases than llmperfscope can see from outside.

`llmperfscope run` exports LLMPERFSCOPE_MARKERS=<file>. Any process, in any language, can append one line per event:

    echo "load_done" >> "$LLMPERFSCOPE_MARKERS"                     # time = when llmperfscope reads it (within 0.2 s)
    echo "$(date +%s%N | ...) ..."                             # or give CLOCK_MONOTONIC ns as the first word
    {"t": <CLOCK_MONOTONIC ns>, "phase": "decode", "group": "decode", "tokens": 128, "meta": {...}}   # or JSON

From Python: `llmperfscope.mark("decode", tokens=128)`. It does nothing when no profiler is running.
"""

from __future__ import annotations

import json
import os
import time

ENV = "LLMPERFSCOPE_MARKERS"

_listeners = []  # in-process Profile objects


def add_listener(listener):
    _listeners.append(listener)


def remove_listener(listener):
    _listeners.remove(listener)


def mark(phase, group=None, tokens=None, **meta):
    """Start phase `phase` now. The previous phase ends here."""
    event = {"t": time.monotonic_ns(), "phase": phase}
    if group:
        event["group"] = group
    if tokens is not None:
        event["tokens"] = tokens
    if meta:
        event["meta"] = meta
    for listener in _listeners:
        listener(event)
    path = os.environ.get(ENV)
    if path:
        # A failed marker loses a phase boundary. It must never take the program being measured down.
        try:
            fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                os.write(fd, (json.dumps(event) + "\n").encode())
            finally:
                os.close(fd)
        except OSError:
            pass


def parse_marker_line(line, now_ns):
    """One line of the marker file -> event dict, or None for a blank or broken line."""
    line = line.strip()
    if not line:
        return None
    if line.startswith("{"):
        try:
            event = json.loads(line)
        except ValueError:
            return None
        if not isinstance(event, dict) or "phase" not in event:
            return None
        event.setdefault("t", now_ns)
        return event
    words = line.split(None, 1)
    if len(words) == 2 and words[0].isdigit():
        return {"t": int(words[0]), "phase": words[1]}
    return {"t": now_ns, "phase": line}


class MarkerReader:
    """Tails the marker file."""

    def __init__(self, path, skip_existing=False):
        self.path = path
        self.pos = 0
        self.tail = b""
        if skip_existing:
            try:
                self.pos = os.path.getsize(path)
            except OSError:
                pass

    def poll(self):
        try:
            with open(self.path, "rb") as f:
                f.seek(self.pos)
                data = f.read()
        except OSError:
            return []
        if not data:
            return []
        self.pos += len(data)
        data = self.tail + data
        lines = data.split(b"\n")
        self.tail = lines.pop()
        now = time.monotonic_ns()
        events = []
        for raw in lines:
            event = parse_marker_line(raw.decode("utf-8", "replace"), now)
            if event:
                events.append(event)
        return events
