"""Cheap process readers on /proc. Deliberately avoids /proc/PID/smaps*: those walk the page tables, take mmap_lock and
would stall an engine that has a several-hundred-GiB model mapped."""

from __future__ import annotations

import os
import sys

CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


def _read(path):
    try:
        with open(path, "rb", buffering=0) as f:
            return f.read(65536)
    except OSError:
        return None


def children_of(pid):
    try:
        tids = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return []
    out = []
    for tid in tids:
        data = _read(f"/proc/{pid}/task/{tid}/children")
        if data:
            out += [int(x) for x in data.split()]
    return out


def _children_by_scan(root_set):
    """Fallback for kernels without CONFIG_PROC_CHILDREN."""
    parents = {}
    for name in os.listdir("/proc"):
        if name.isdigit():
            st = stat(int(name))
            if st:
                parents.setdefault(st["ppid"], []).append(int(name))
    seen = set()
    todo = list(root_set)
    while todo:
        pid = todo.pop()
        if pid in seen:
            continue
        seen.add(pid)
        todo += parents.get(pid, [])
    return seen


_HAS_CHILDREN_FILE = os.path.exists(f"/proc/self/task/{os.getpid()}/children") if os.path.exists("/proc") else False


def tree(roots):
    """All live descendants of `roots`, the roots included. Zombies are left out."""
    if not _HAS_CHILDREN_FILE:
        pids = _children_by_scan(set(roots))
    else:
        pids = set()
        todo = list(roots)
        while todo:
            pid = todo.pop()
            if pid in pids:
                continue
            pids.add(pid)
            todo += children_of(pid)
    return {pid for pid in pids if (st := stat(pid)) and st["state"] not in ("Z", "X")}


_STATUS_KEYS = {
    b"VmHWM:": "hwm",
    b"VmRSS:": "rss",
    b"RssAnon:": "anon",
    b"RssFile:": "file",
    b"RssShmem:": "shmem",
    b"VmSwap:": "swap",
    b"VmPin:": "pin",
    b"VmLck:": "lck",
    b"Threads:": "threads",
}


def status(pid):
    """Memory in kB from /proc/PID/status, or None once the process is gone."""
    data = _read(f"/proc/{pid}/status")
    if data is None:
        return None
    out = {}
    for line in data.split(b"\n"):
        key, _, value = line.partition(b"\t")
        name = _STATUS_KEYS.get(key)
        if name:
            parts = value.split()
            if parts:
                out[name] = int(parts[0])
    return out


def stat(pid):
    data = _read(f"/proc/{pid}/stat")
    if data is None:
        return None
    return _parse_stat(data)


def _parse_stat(data):
    open_paren = data.find(b"(")
    close_paren = data.rfind(b")")
    if open_paren < 0 or close_paren < 0:
        return None
    fields = data[close_paren + 2:].split()
    if len(fields) < 20:
        return None
    return {
        "comm": data[open_paren + 1:close_paren].decode("utf-8", "replace"),
        "state": fields[0].decode(),
        "ppid": int(fields[1]),
        "minflt": int(fields[7]),
        "majflt": int(fields[9]),
        "utime": int(fields[11]),
        "stime": int(fields[12]),
        "threads": int(fields[17]),
    }


def io(pid):
    """Bytes from /proc/PID/io; None when the kernel refuses it (other user, some containers)."""
    data = _read(f"/proc/{pid}/io")
    if data is None:
        return None
    out = {}
    for line in data.split(b"\n"):
        key, _, value = line.partition(b": ")
        if value:
            out[key.decode()] = int(value)
    return out


def threads(pid):
    """(tid, comm, cpu ticks) of every thread of `pid`."""
    try:
        tids = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return []
    out = []
    for tid in tids:
        data = _read(f"/proc/{pid}/task/{tid}/stat")
        if data is None:
            continue
        st = _parse_stat(data)
        if st:
            out.append((int(tid), st["comm"], st["utime"] + st["stime"]))
    return out


def affinity(pid):
    try:
        return os.sched_getaffinity(pid)
    except (OSError, AttributeError):
        return None


def wchan(pid, tid=None):
    data = _read(f"/proc/{pid}/task/{tid}/wchan" if tid else f"/proc/{pid}/wchan")
    return data.decode("utf-8", "replace").strip() if data else None


def alive(pid):
    return os.path.exists(f"/proc/{pid}")


def all_pids():
    return [int(name) for name in os.listdir("/proc") if name.isdigit()]


def owner(pid):
    try:
        return os.stat(f"/proc/{pid}").st_uid
    except OSError:
        return -1


def exe_path(pid):
    return f"/proc/{pid}/exe"


if sys.platform == "darwin":
    from .darwin import CLK_TCK, affinity, all_pids, alive, children_of, exe_path, io, owner, stat, status, threads, tree, wchan  # noqa: F401,F811
