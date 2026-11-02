"""System-wide counters: where data moves between disk, RAM and CPU, and whether the kernel saw anything stall on it.

An LLM engine does not stay on one tier. It pages weights in from an SSD or HDD, keeps them in the page cache, copies them
to a GPU, and under pressure swaps, drops cache and reads the same pages again. These counters are what show that. All of
them are cheap, unprivileged and cumulative, so a phase's value is the difference between its two ends.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

SECTOR = 512
_VMSTAT_KEYS = {
    b"pswpin", b"pswpout", b"pgmajfault", b"pgpgin", b"pgpgout", b"pgscan_direct", b"pgsteal_direct", b"pgscan_kswapd",
    b"workingset_refault_file", b"workingset_refault_anon", b"numa_miss", b"numa_other", b"numa_foreign",
}
_MEMINFO_KEYS = {
    b"MemTotal:", b"MemAvailable:", b"MemFree:", b"Buffers:", b"Cached:", b"Dirty:", b"Writeback:", b"SwapTotal:", b"SwapFree:",
    b"Shmem:", b"AnonPages:", b"Mapped:", b"Mlocked:",
}


def _read(path):
    try:
        with open(path, "rb", buffering=0) as f:
            return f.read(1 << 20)
    except OSError:
        return None


def _text(path):
    data = _read(path)
    return data.decode("utf-8", "replace").strip() if data else None


def classify_disk(name):
    """nvme, ssd or hdd for a whole disk. The kernel's rotational flag is what tells SSD from HDD."""
    if name.startswith("nvme"):
        return "nvme"
    if name.startswith("zram"):
        return "zram"
    rotational = _text(f"/sys/block/{name}/queue/rotational")
    if rotational == "1":
        return "hdd"
    return "ssd" if rotational == "0" else "disk"


def list_disks():
    """Physical disks only. Partitions, LVM and RAID devices would count the same I/O a second time."""
    out = {}
    try:
        names = os.listdir("/sys/block")
    except OSError:
        return out
    for name in sorted(names):
        if name.startswith(("loop", "ram", "dm-", "md", "sr", "fd", "nbd")):
            continue
        model = _text(f"/sys/block/{name}/device/model")
        size = _text(f"/sys/block/{name}/size")
        out[name] = {"kind": classify_disk(name), "model": model, "size_gib": round(int(size) * SECTOR / (1 << 30), 1) if size else None}
    return out


def disk_counters(names):
    """name -> [sectors read, sectors written, ms busy, reads, writes]."""
    data = _read("/proc/diskstats")
    out = {}
    if not data:
        return out
    for line in data.split(b"\n"):
        fields = line.split()
        if len(fields) < 14:
            continue
        name = fields[2].decode()
        if name in names:
            out[name] = [int(fields[5]), int(fields[9]), int(fields[12]), int(fields[3]), int(fields[7])]
    return out


def vmstat():
    data = _read("/proc/vmstat")
    out = {}
    if data:
        for line in data.split(b"\n"):
            key, _, value = line.partition(b" ")
            if key in _VMSTAT_KEYS:
                out[key.decode()] = int(value)
    return out


def meminfo():
    """MiB."""
    data = _read("/proc/meminfo")
    out = {}
    if data:
        for line in data.split(b"\n"):
            parts = line.split()
            if len(parts) >= 2 and parts[0] in _MEMINFO_KEYS:
                out[parts[0][:-1].decode()] = int(parts[1]) // 1024
    return out


def pressure():
    """Kernel pressure stall information: cumulative microseconds that tasks were stalled on CPU, memory or I/O. It answers
    `was anything waiting on this resource` directly, without guessing from utilization."""
    out = {}
    for resource in ("cpu", "memory", "io"):
        data = _read(f"/proc/pressure/{resource}")
        if not data:
            continue
        for line in data.split(b"\n"):
            parts = line.split()
            if len(parts) >= 5 and parts[4].startswith(b"total="):
                out[f"{resource}_{parts[0].decode()}"] = int(parts[4][6:])
    return out


def cpu_times():
    """Machine-wide CPU ticks: user, nice, system, idle, iowait, irq, softirq, steal."""
    data = _read("/proc/stat")
    if not data:
        return None
    fields = data.split(b"\n", 1)[0].split()
    return [int(x) for x in fields[1:9]] if fields and fields[0] == b"cpu" else None


def network_bytes():
    """[rx, tx] over all interfaces but loopback: engines split over machines, or weights on a network filesystem."""
    data = _read("/proc/net/dev")
    if not data:
        return None
    rx = tx = 0
    for line in data.split(b"\n")[2:]:
        name, _, rest = line.partition(b":")
        fields = rest.split()
        if len(fields) >= 9 and name.strip() != b"lo":
            rx += int(fields[0])
            tx += int(fields[8])
    return [rx, tx]


class SystemSampler:
    def __init__(self):
        self.disks = list_disks()
        self.names = set(self.disks)

    def read(self):
        out = {"disk": disk_counters(self.names), "vm": vmstat(), "mem": meminfo(), "psi": pressure()}
        cpu = cpu_times()
        if cpu:
            out["cpu"] = cpu
        net = network_bytes()
        if net:
            out["net"] = net
        return out


def _parent_disk(block_dir):
    """The physical disks under a block device: partition -> disk, LVM or RAID -> its members."""
    path = Path(block_dir).resolve()
    if (path / "partition").exists():
        path = path.parent
    slaves = path / "slaves"
    if slaves.is_dir():
        members = sorted(p.name for p in slaves.iterdir())
        if members:
            found = []
            for member in members:
                found += _parent_disk(f"/sys/class/block/{member}")
            return found
    return [path.name]


def file_location(path):
    """Which tier a file lives on. A model on an HDD or a network filesystem explains a slow load before anything is run."""
    try:
        real = os.path.realpath(path)
        info = os.stat(real)
    except OSError:
        return None
    out = {"path": real, "size_gib": round(info.st_size / (1 << 30), 3)}
    best = ""
    try:
        with open("/proc/self/mountinfo") as f:
            for line in f:
                left, _, right = line.partition(" - ")
                mount_point = left.split()[4]
                if (real == mount_point or real.startswith(mount_point.rstrip("/") + "/")) and len(mount_point) >= len(best):
                    best = mount_point
                    out["fstype"] = right.split()[0]
                    out["mount"] = mount_point
    except OSError:
        pass
    major, minor = os.major(info.st_dev), os.minor(info.st_dev)
    block = f"/sys/dev/block/{major}:{minor}"
    if os.path.exists(block):
        try:
            disks = _parent_disk(block)
        except OSError:
            disks = []
        known = list_disks()
        out["disks"] = [{"name": d, **known.get(d, {"kind": classify_disk(d)})} for d in disks]
    if out.get("fstype") in ("nfs", "nfs4", "cifs", "smb3", "ceph", "fuse.sshfs", "lustre", "gpfs", "glusterfs", "9p", "virtiofs"):
        out["network"] = True
    if out.get("fstype") in ("tmpfs", "ramfs"):
        out["in_ram"] = True
    return out


def storage_kind(location):
    """One word for a file's storage: hdd beats ssd beats nvme, because the slowest member sets the pace."""
    if not location:
        return None
    if location.get("network"):
        return "network"
    if location.get("in_ram"):
        return "ram"
    kinds = {disk.get("kind") for disk in location.get("disks", [])}
    for kind in ("hdd", "disk", "ssd", "nvme"):
        if kind in kinds:
            return kind
    return None


if sys.platform == "darwin":
    from .darwin import cpu_times, disk_counters, file_location, list_disks, meminfo, network_bytes, pressure, vmstat  # noqa: F401,F811
