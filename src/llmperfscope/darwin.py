"""macOS readers, the counterpart of /proc and /sys, none of which need root.

Processes come from libproc and Mach, system counters from Mach and sysctl, and disks and the GPU from IOKit. All of it is cheap enough for the
sampler: a process takes about 10 us, the GPU's gauges about 60 us. Two reads cost more and run less often: the walk over the GPU's clients
(1.3 ms) on slow ticks, and the energy counter (1.6 ms) on its own thread.

Memory on Apple Silicon is one pool, which changes what the numbers mean: GPU memory is the part of RAM that the GPU driver has in use, so it is
not added to the host's. Memory a process's GPU buffers hold is also counted in that process's own memory.

Two sources are undocumented: libIOReport for energy, and the `AppUsage` property for GPU time per process. If either stops working, those
numbers go missing and everything else carries on.
"""

from __future__ import annotations

import ctypes as C
import os
import re
import struct
import subprocess
import sys
import time

_P = C.c_void_p
_UTF8 = 0x08000100
_SECTOR = 512
_LINUX_PAGE = 4096


def _library(path):
    try:
        return C.CDLL(path, use_errno=True)
    except OSError:
        return None


_libc = C.CDLL(None, use_errno=True)
_proc = _library("/usr/lib/libproc.dylib")
_cf = _library("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
_iokit = _library("/System/Library/Frameworks/IOKit.framework/IOKit")
_ioreport = _library("/usr/lib/libIOReport.dylib")


class _Timebase(C.Structure):
    _fields_ = [("numer", C.c_uint32), ("denom", C.c_uint32)]


class _TaskInfo(C.Structure):
    _fields_ = [("virtual", C.c_uint64), ("resident", C.c_uint64), ("user", C.c_uint64), ("system", C.c_uint64), ("threads_user", C.c_uint64), ("threads_system", C.c_uint64),
                ("policy", C.c_int32), ("faults", C.c_int32), ("pageins", C.c_int32), ("cow_faults", C.c_int32), ("sent", C.c_int32), ("received", C.c_int32), ("mach_calls", C.c_int32),
                ("unix_calls", C.c_int32), ("switches", C.c_int32), ("threadnum", C.c_int32), ("running", C.c_int32), ("priority", C.c_int32)]


class _BsdInfo(C.Structure):
    _fields_ = [("flags", C.c_uint32), ("status", C.c_uint32), ("xstatus", C.c_uint32), ("pid", C.c_uint32), ("ppid", C.c_uint32), ("uid", C.c_uint32), ("gid", C.c_uint32),
                ("ruid", C.c_uint32), ("rgid", C.c_uint32), ("svuid", C.c_uint32), ("svgid", C.c_uint32), ("reserved", C.c_uint32), ("comm", C.c_char * 16), ("name", C.c_char * 32),
                ("nfiles", C.c_uint32), ("pgid", C.c_uint32), ("pjobc", C.c_uint32), ("tdev", C.c_uint32), ("tpgid", C.c_uint32), ("nice", C.c_int32), ("start_s", C.c_uint64), ("start_us", C.c_uint64)]


class _ThreadInfo(C.Structure):
    _fields_ = [("user", C.c_uint64), ("system", C.c_uint64), ("usage", C.c_int32), ("policy", C.c_int32), ("run_state", C.c_int32), ("flags", C.c_int32), ("sleep", C.c_int32),
                ("current_priority", C.c_int32), ("priority", C.c_int32), ("max_priority", C.c_int32), ("name", C.c_char * 64)]


class _Rusage(C.Structure):
    _fields_ = [("uuid", C.c_uint8 * 16)] + [(name, C.c_uint64) for name in (
        "user", "system", "pkg_idle", "interrupt", "pageins", "wired", "resident", "footprint", "start", "exit", "child_user", "child_system", "child_idle", "child_interrupt",
        "child_pageins", "child_elapsed", "disk_read", "disk_written", "qos_default", "qos_maintenance", "qos_background", "qos_utility", "qos_legacy", "qos_initiated", "qos_interactive",
        "billed_system", "serviced_system", "logical_writes", "max_footprint", "instructions", "cycles", "billed_energy", "serviced_energy", "interval_max", "runnable")]


class _TaskVmInfo(C.Structure):
    """struct task_vm_info. Older systems fill less of it and leave the rest zero."""

    _fields_ = [("virtual_size", C.c_int64), ("region_count", C.c_int32), ("page_size", C.c_int32), ("resident", C.c_int64), ("resident_peak", C.c_int64)] + [(name, C.c_int64) for name in (
        "device", "device_peak", "internal", "internal_peak", "external", "external_peak", "reusable", "reusable_peak", "purgeable_volatile_pmap", "purgeable_volatile_resident",
        "purgeable_volatile_virtual", "compressed", "compressed_peak", "compressed_lifetime", "footprint", "min_address", "max_address", "footprint_peak", "purgeable_nonvolatile",
        "purgeable_nonvolatile_compressed", "purgeable_volatile", "purgeable_volatile_compressed", "network_nonvolatile", "network_nonvolatile_compressed", "network_volatile",
        "network_volatile_compressed", "media_footprint", "media_footprint_compressed", "media_nofootprint", "media_nofootprint_compressed", "graphics_footprint",
        "graphics_footprint_compressed", "graphics_nofootprint", "graphics_nofootprint_compressed", "neural_footprint", "neural_footprint_compressed", "neural_nofootprint",
        "neural_nofootprint_compressed", "limit_bytes_remaining")] + [("decompressions", C.c_int32)]


class _VmStatistics(C.Structure):
    _fields_ = [("free", C.c_uint32), ("active", C.c_uint32), ("inactive", C.c_uint32), ("wired", C.c_uint32), ("zero_fill", C.c_uint64), ("reactivations", C.c_uint64), ("pageins", C.c_uint64),
                ("pageouts", C.c_uint64), ("faults", C.c_uint64), ("cow_faults", C.c_uint64), ("lookups", C.c_uint64), ("hits", C.c_uint64), ("purges", C.c_uint64), ("purgeable", C.c_uint32),
                ("speculative", C.c_uint32), ("decompressions", C.c_uint64), ("compressions", C.c_uint64), ("swapins", C.c_uint64), ("swapouts", C.c_uint64), ("compressor", C.c_uint32),
                ("throttled", C.c_uint32), ("external", C.c_uint32), ("internal", C.c_uint32), ("uncompressed", C.c_uint64)]


class _SwapUsage(C.Structure):
    _fields_ = [("total", C.c_uint64), ("available", C.c_uint64), ("used", C.c_uint64), ("page_size", C.c_uint32), ("encrypted", C.c_uint32)]


class _StatFs(C.Structure):
    _fields_ = [("bsize", C.c_uint32), ("iosize", C.c_int32), ("blocks", C.c_uint64), ("bfree", C.c_uint64), ("bavail", C.c_uint64), ("files", C.c_uint64), ("ffree", C.c_uint64),
                ("fsid", C.c_int32 * 2), ("owner", C.c_uint32), ("type", C.c_uint32), ("flags", C.c_uint32), ("subtype", C.c_uint32), ("fstype", C.c_char * 16), ("mount", C.c_char * 1024),
                ("source", C.c_char * 1024), ("flags_ext", C.c_uint32), ("reserved", C.c_uint32 * 7)]


_PIDTBSDINFO, _PIDTASKINFO, _PIDTHREADINFO, _PIDLISTTHREADS = 3, 4, 5, 6
_RUSAGE_V4 = 4
_ZOMBIE = 5
_MAX_PIDS = 8192
_TASK_VM_INFO = 22
_HOST_CPU_LOAD_INFO, _HOST_VM_INFO64 = 3, 4
_CTL_NET, _PF_ROUTE, _NET_RT_IFLIST2, _RTM_IFINFO2 = 4, 17, 6, 0x12
_CTL_KERN, _KERN_PROCARGS2 = 1, 49

_libc.mach_host_self.restype = C.c_uint32
_libc.task_name_for_pid.argtypes = [C.c_uint32, C.c_int, C.POINTER(C.c_uint32)]
_libc.task_info.argtypes = [C.c_uint32, C.c_int, C.c_void_p, C.POINTER(C.c_uint32)]
_libc.mach_port_deallocate.argtypes = [C.c_uint32, C.c_uint32]
_libc.host_page_size.argtypes = [C.c_uint32, C.POINTER(C.c_size_t)]
_libc.host_statistics.argtypes = [C.c_uint32, C.c_int, C.c_void_p, C.POINTER(C.c_uint32)]
_libc.host_statistics64.argtypes = [C.c_uint32, C.c_int, C.c_void_p, C.POINTER(C.c_uint32)]
_libc.sysctl.argtypes = [C.POINTER(C.c_int), C.c_uint, C.c_void_p, C.POINTER(C.c_size_t), C.c_void_p, C.c_size_t]
_SELF = C.c_uint32.in_dll(_libc, "mach_task_self_").value
# Asking for this port again would add a reference each time, so it is asked for once.
_HOST = _libc.mach_host_self()

_timebase = _Timebase()
_libc.mach_timebase_info(C.byref(_timebase))
_page = C.c_size_t()
_libc.host_page_size(_HOST, C.byref(_page))
_PAGE_BYTES = _page.value or 16384
if _proc:
    _proc.proc_pidinfo.argtypes = [C.c_int, C.c_int, C.c_uint64, _P, C.c_int]
    _proc.proc_pid_rusage.argtypes = [C.c_int, C.c_int, _P]
    _proc.proc_listchildpids.argtypes = [C.c_int, _P, C.c_int]
    _proc.proc_listallpids.argtypes = [_P, C.c_int]
    _proc.proc_pidpath.argtypes = [C.c_int, _P, C.c_uint32]

CLK_TCK = 100


def _ticks(mach_time):
    """CPU time from libproc is in Mach time units: 1 ns on Intel, 125/3 ns on Apple Silicon."""
    return mach_time * _timebase.numer // _timebase.denom // (1_000_000_000 // CLK_TCK)


def _task(pid):
    info = _TaskInfo()
    return info if _proc.proc_pidinfo(pid, _PIDTASKINFO, 0, C.byref(info), C.sizeof(info)) == C.sizeof(info) else None


def _bsd(pid):
    info = _BsdInfo()
    return info if _proc.proc_pidinfo(pid, _PIDTBSDINFO, 0, C.byref(info), C.sizeof(info)) == C.sizeof(info) else None


def _rusage(pid):
    info = _Rusage()
    return info if _proc.proc_pid_rusage(pid, _RUSAGE_V4, C.byref(info)) == 0 else None


def _vm_info(pid):
    """Memory of a process by kind. A task name port is enough for this, and macOS gives one for the user's own processes."""
    name = C.c_uint32()
    if _libc.task_name_for_pid(_SELF, pid, C.byref(name)) != 0:
        return None
    try:
        info = _TaskVmInfo()
        count = C.c_uint32(C.sizeof(info) // 4)
        return info if _libc.task_info(name.value, _TASK_VM_INFO, C.byref(info), C.byref(count)) == 0 else None
    finally:
        _libc.mach_port_deallocate(_SELF, name.value)


# The same shape as procfs.py ------------------------------------------------------------------------------------------------------------

def children_of(pid):
    buffer = (C.c_int * _MAX_PIDS)()
    count = _proc.proc_listchildpids(pid, buffer, C.sizeof(buffer))
    return [buffer[i] for i in range(max(count, 0)) if buffer[i] > 0]


def all_pids():
    buffer = (C.c_int * _MAX_PIDS)()
    count = _proc.proc_listallpids(buffer, C.sizeof(buffer))
    return [buffer[i] for i in range(max(count, 0)) if buffer[i] > 0]


def alive(pid):
    info = _bsd(pid)
    return info is not None and info.status != _ZOMBIE


def owner(pid):
    info = _bsd(pid)
    return info.uid if info else -1


def exe_path(pid):
    buffer = C.create_string_buffer(4096)
    return buffer.value.decode("utf-8", "replace") if _proc.proc_pidpath(pid, buffer, 4096) > 0 else None


def with_environment(entry):
    """Processes of this user started with `NAME=value` in their environment. macOS cannot adopt orphans, so this is how a process that left the tree gets found:
    what it inherited at its start is still there, whoever its parent is now."""
    wanted = entry.encode()
    me = os.getuid()
    size = 1 << 20
    buffer = C.create_string_buffer(size)
    found = set()
    for pid in all_pids():
        if owner(pid) != me:
            continue
        mib = (C.c_int * 3)(_CTL_KERN, _KERN_PROCARGS2, pid)
        length = C.c_size_t(size)
        if _libc.sysctl(mib, 3, buffer, C.byref(length), None, 0) == 0 and wanted in buffer.raw[:length.value].split(b"\0"):
            found.add(pid)
    return found


def tree(roots):
    pids = set()
    todo = list(roots)
    while todo:
        pid = todo.pop()
        if pid in pids:
            continue
        pids.add(pid)
        todo += children_of(pid)
    return {pid for pid in pids if alive(pid)}


def status(pid):
    """Memory in kB, by the names of /proc/PID/status. Compressed memory stands where swap does on Linux: macOS compresses before it swaps."""
    task = _task(pid)
    usage = _rusage(pid)
    if task is None or usage is None:
        return None
    out = {"rss": task.resident // 1024, "shmem": 0, "pin": usage.wired // 1024, "lck": 0, "threads": task.threadnum}
    vm = _vm_info(pid)
    if vm is not None:
        out.update(hwm=vm.resident_peak // 1024, anon=vm.internal // 1024, file=vm.external // 1024, swap=vm.compressed // 1024)
    else:
        # Should macOS ever refuse the name port, the footprint is the next best thing.
        out.update(hwm=usage.max_footprint // 1024, anon=usage.footprint // 1024, file=max(0, task.resident - usage.footprint) // 1024, swap=0)
    return out


def stat(pid):
    info = _bsd(pid)
    if info is None:
        return None
    out = {"comm": (info.name or info.comm).decode("utf-8", "replace"), "state": "Z" if info.status == _ZOMBIE else "R", "ppid": info.ppid, "minflt": 0, "majflt": 0, "utime": 0, "stime": 0, "threads": 0}
    task = _task(pid)
    if task is not None:
        out.update({"minflt": max(task.faults - task.pageins, 0), "majflt": task.pageins, "utime": _ticks(task.user), "stime": _ticks(task.system), "threads": task.threadnum})
    return out


def io(pid):
    usage = _rusage(pid)
    return None if usage is None else {"read_bytes": usage.disk_read, "write_bytes": usage.disk_written, "rchar": 0, "wchar": 0}


def threads(pid):
    """(thread id, name, cpu ticks). Thread times here are in nanoseconds, unlike the process's. An unnamed thread goes by its process's name, as on Linux."""
    handles = (C.c_uint64 * 1024)()
    size = _proc.proc_pidinfo(pid, _PIDLISTTHREADS, 0, handles, C.sizeof(handles))
    if size <= 0:
        return []
    bsd = _bsd(pid)
    process = (bsd.name or bsd.comm).decode("utf-8", "replace") if bsd else "thread"
    out = []
    for i in range(size // 8):
        info = _ThreadInfo()
        if _proc.proc_pidinfo(pid, _PIDTHREADINFO, handles[i], C.byref(info), C.sizeof(info)) == C.sizeof(info):
            out.append((handles[i], info.name.decode("utf-8", "replace") or process, (info.user + info.system) // (1_000_000_000 // CLK_TCK)))
    return out


def affinity(_pid):
    """macOS has no CPU affinity: it schedules by quality of service."""
    return None


def wchan(_pid, _tid=None):
    return None


# sysctl and Mach ----------------------------------------------------------------------------------------------------------------------

def _sysctl_bytes(name, size=256):
    buffer = C.create_string_buffer(size)
    length = C.c_size_t(size)
    if _libc.sysctlbyname(name.encode(), buffer, C.byref(length), None, C.c_size_t(0)) != 0:
        return None
    return buffer.raw[:length.value]


def sysctl_text(name):
    data = _sysctl_bytes(name)
    return data.split(b"\0", 1)[0].decode("utf-8", "replace") if data else None


def sysctl_int(name):
    data = _sysctl_bytes(name, 8)
    return int.from_bytes(data, "little") if data else None


def _vm():
    info = _VmStatistics()
    count = C.c_uint32(C.sizeof(info) // 4)
    return info if _libc.host_statistics64(_HOST, _HOST_VM_INFO64, C.byref(info), C.byref(count)) == 0 else None


def meminfo():
    """MiB, by the names of /proc/meminfo. Available is the kernel's own figure, the one `memory_pressure` prints as the free percentage and that its pressure levels go by. Speculative
    pages are already counted as free."""
    vm = _vm()
    if vm is None:
        return {}
    to_mib = _PAGE_BYTES / (1 << 20)
    total = (sysctl_int("hw.memsize") or 0) >> 20
    level = sysctl_int("kern.memorystatus_level")
    out = {"MemTotal": total, "MemFree": int((vm.free - vm.speculative) * to_mib), "MemAvailable": total * level // 100 if level else int((vm.free + vm.inactive) * to_mib),
           "Cached": int(vm.external * to_mib)}
    swap = _SwapUsage()
    length = C.c_size_t(C.sizeof(swap))
    if _libc.sysctlbyname(b"vm.swapusage", C.byref(swap), C.byref(length), None, C.c_size_t(0)) == 0:
        out["SwapTotal"], out["SwapFree"] = swap.total >> 20, swap.available >> 20
    return out


def vmstat():
    """By the names of /proc/vmstat where there is one, in units of 4 KiB as the analysis expects. Swap is what goes to the disk. macOS compresses memory long before that and
    on a machine in normal use, so the compressor has counters of its own. It cannot tell a file page read again from one read for the first time, so there are no refaults."""
    vm = _vm()
    if vm is None:
        return {}
    scale = _PAGE_BYTES // _LINUX_PAGE
    return {"pswpin": vm.swapins * scale, "pswpout": vm.swapouts * scale, "pgmajfault": vm.pageins, "compressions": vm.compressions * scale, "decompressions": vm.decompressions * scale}


class Pressure:
    """macOS has no pressure stall counters, but it does publish a memory pressure level: 1 normal, 2 warning, 4 critical. Time spent at warning or
    above counts as tasks stalled on memory, and at critical as fully stalled, so the verdicts read the same on both systems. It is an
    approximation, and CPU and I/O have nothing like it."""

    def __init__(self):
        self.last = time.monotonic()
        self.some = self.full = 0.0

    def read(self):
        now = time.monotonic()
        level = sysctl_int("kern.memorystatus_vm_pressure_level") or 1
        elapsed = (now - self.last) * 1e6
        self.last = now
        if level >= 2:
            self.some += elapsed
        if level >= 4:
            self.full += elapsed
        return {"memory_some": int(self.some), "memory_full": int(self.full)}


_pressure = Pressure()


def pressure():
    return _pressure.read()


def cpu_times():
    """Machine-wide CPU ticks in the order of /proc/stat: user, nice, system, idle, iowait, irq, softirq, steal."""
    ticks = (C.c_uint32 * 4)()
    count = C.c_uint32(4)
    if _libc.host_statistics(_HOST, _HOST_CPU_LOAD_INFO, ticks, C.byref(count)) != 0:
        return None
    user, system, idle, nice = ticks
    return [user, nice, system, idle, 0, 0, 0, 0]


# Offsets in struct if_msghdr2, which holds a struct if_data64 from byte 32 on.
_IF_MSG_TYPE, _IF_FLAGS, _IF_INDEX, _IF_IBYTES = 3, 8, 12, 96
_IFF_LOOPBACK = 0x8


_net_last = {}
_net_total = [0, 0]


def _add_network(index, counters):
    previous = _net_last.get(index)
    if previous is not None:
        for direction in (0, 1):
            _net_total[direction] += (counters[direction] - previous[direction]) & 0xFFFFFFFF
    _net_last[index] = counters


def network_bytes():
    """[rx, tx] over all interfaces but loopback, since this module was loaded. macOS gives ordinary processes these counters rounded to a KiB and cut to 32 bits, even in the
    64-bit fields, so what adds up is the change since the last read."""
    mib = (C.c_int * 6)(_CTL_NET, _PF_ROUTE, 0, 0, _NET_RT_IFLIST2, 0)
    size = C.c_size_t()
    if _libc.sysctl(mib, 6, None, C.byref(size), None, 0) != 0:
        return None
    buffer = C.create_string_buffer(size.value)
    if _libc.sysctl(mib, 6, buffer, C.byref(size), None, 0) != 0:
        return None
    data = buffer.raw[:size.value]
    offset = 0
    while offset + 4 <= len(data):
        length = struct.unpack_from("<H", data, offset)[0]
        if length == 0:
            break
        if data[offset + _IF_MSG_TYPE] == _RTM_IFINFO2 and not struct.unpack_from("<i", data, offset + _IF_FLAGS)[0] & _IFF_LOOPBACK:
            _add_network(struct.unpack_from("<H", data, offset + _IF_INDEX)[0], struct.unpack_from("<QQ", data, offset + _IF_IBYTES))
        offset += length
    return list(_net_total)


class _Thermal:
    """Thermal pressure as the system publishes it: 0 nominal, 1 moderate, 2 heavy, 3 trapping, 4 sleeping. A fanless Mac reaches moderate within minutes of full
    GPU load and slows down, which would otherwise look like the engine slowing down."""

    def __init__(self):
        self.token = C.c_int(-1)
        try:
            _libc.notify_register_check.argtypes = [C.c_char_p, C.POINTER(C.c_int)]
            _libc.notify_get_state.argtypes = [C.c_int, C.POINTER(C.c_uint64)]
            if _libc.notify_register_check(b"com.apple.system.thermalpressurelevel", C.byref(self.token)) != 0:
                self.token = None
        except (AttributeError, ValueError):
            self.token = None

    def level(self):
        if self.token is None:
            return None
        state = C.c_uint64()
        return state.value if _libc.notify_get_state(self.token, C.byref(state)) == 0 else None


_thermal = _Thermal()


def thermal_level():
    return _thermal.level()


def foreign_cpu_scan():
    """pid -> (ticks, name, user id) of every process, from `ps`, which alone may read processes of other users without root. For the check before a run, not for sampling."""
    try:
        text = subprocess.run(["ps", "-axo", "pid=,uid=,cputime=,ucomm="], capture_output=True, text=True, timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    out = {}
    for line in text.splitlines():
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        minutes, _, seconds = parts[2].rpartition(":")
        try:
            out[int(parts[0])] = (int((int(minutes or 0) * 60 + float(seconds)) * CLK_TCK), parts[3].strip(), int(parts[1]))
        except ValueError:
            continue
    return out


def evict_from_page_cache(path):
    """Asks the kernel to drop a file's clean pages. macOS has no posix_fadvise, but invalidating a mapping of the file does the same and needs no root."""
    size = os.path.getsize(path)
    if size == 0:
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        _libc.mmap.restype = C.c_void_p
        _libc.mmap.argtypes = [C.c_void_p, C.c_size_t, C.c_int, C.c_int, C.c_int, C.c_longlong]
        _libc.munmap.argtypes = [C.c_void_p, C.c_size_t]
        _libc.msync.argtypes = [C.c_void_p, C.c_size_t, C.c_int]
        base = _libc.mmap(None, size, 1, 1, fd, 0)  # PROT_READ, MAP_SHARED
        if base is None or base == C.c_void_p(-1).value:
            raise OSError(C.get_errno(), "mmap failed")
        try:
            _libc.msync(base, size, 2)  # MS_INVALIDATE
        finally:
            _libc.munmap(base, size)
    finally:
        os.close(fd)


# Core Foundation and IOKit --------------------------------------------------------------------------------------------------------------

if _cf:
    _cf.CFStringCreateWithCString.restype = _P
    _cf.CFStringCreateWithCString.argtypes = [_P, C.c_char_p, C.c_uint32]
    _cf.CFStringGetCString.argtypes = [_P, C.c_char_p, C.c_long, C.c_uint32]
    _cf.CFDictionaryGetValue.restype = _P
    _cf.CFDictionaryGetValue.argtypes = [_P, _P]
    _cf.CFDictionaryGetCount.restype = C.c_long
    _cf.CFDictionaryGetCount.argtypes = [_P]
    _cf.CFDictionaryCreateMutableCopy.restype = _P
    _cf.CFDictionaryCreateMutableCopy.argtypes = [_P, C.c_long, _P]
    _cf.CFArrayGetCount.restype = C.c_long
    _cf.CFArrayGetCount.argtypes = [_P]
    _cf.CFArrayGetValueAtIndex.restype = _P
    _cf.CFArrayGetValueAtIndex.argtypes = [_P, C.c_long]
    _cf.CFNumberGetValue.argtypes = [_P, C.c_int, _P]
    _cf.CFRelease.argtypes = [_P]
if _iokit:
    _iokit.IOServiceMatching.restype = _P
    _iokit.IOServiceMatching.argtypes = [C.c_char_p]
    _iokit.IOBSDNameMatching.restype = _P
    _iokit.IOBSDNameMatching.argtypes = [C.c_uint32, C.c_uint32, C.c_char_p]
    _iokit.IOServiceGetMatchingService.restype = C.c_uint32
    _iokit.IOServiceGetMatchingService.argtypes = [C.c_uint32, _P]
    _iokit.IOServiceGetMatchingServices.argtypes = [C.c_uint32, _P, C.POINTER(C.c_uint32)]
    _iokit.IOIteratorNext.restype = C.c_uint32
    _iokit.IOIteratorNext.argtypes = [C.c_uint32]
    _iokit.IOObjectRelease.argtypes = [C.c_uint32]
    _iokit.IOObjectRetain.argtypes = [C.c_uint32]
    _iokit.IORegistryEntryGetParentEntry.argtypes = [C.c_uint32, C.c_char_p, C.POINTER(C.c_uint32)]
    _iokit.IORegistryEntryGetChildIterator.argtypes = [C.c_uint32, C.c_char_p, C.POINTER(C.c_uint32)]
    _iokit.IORegistryEntryGetRegistryEntryID.argtypes = [C.c_uint32, C.POINTER(C.c_uint64)]
    _iokit.IORegistryEntryCreateCFProperty.restype = _P
    _iokit.IORegistryEntryCreateCFProperty.argtypes = [C.c_uint32, _P, _P, C.c_uint32]
    _iokit.IORegistryEntrySearchCFProperty.restype = _P
    _iokit.IORegistryEntrySearchCFProperty.argtypes = [C.c_uint32, C.c_char_p, _P, _P, C.c_uint32]

_keys = {}


def _key(text):
    """CF strings are kept for good: this reads the same few keys many times a second."""
    key = _keys.get(text)
    if key is None:
        key = _keys[text] = _cf.CFStringCreateWithCString(None, text.encode(), _UTF8)
    return key


def _text(ref):
    if not ref:
        return None
    buffer = C.create_string_buffer(512)
    return buffer.value.decode("utf-8", "replace") if _cf.CFStringGetCString(ref, buffer, 512, _UTF8) else None


def _number(dictionary, name):
    value = _cf.CFDictionaryGetValue(dictionary, _key(name))
    if not value:
        return None
    out = C.c_int64()
    return out.value if _cf.CFNumberGetValue(value, 4, C.byref(out)) else None


def _services(class_name):
    matching = _iokit.IOServiceMatching(class_name.encode())
    iterator = C.c_uint32()
    if not matching or _iokit.IOServiceGetMatchingServices(0, matching, C.byref(iterator)) != 0:
        return []
    out = []
    while True:
        entry = _iokit.IOIteratorNext(iterator)
        if not entry:
            break
        out.append(entry)
    _iokit.IOObjectRelease(iterator)
    return out


_RECURSIVELY, _PARENTS = 1, 2


def _characteristics(entry):
    """What a disk says about itself, from the device above it in the registry."""
    found = _iokit.IORegistryEntrySearchCFProperty(entry, b"IOService", _key("Device Characteristics"), None, _RECURSIVELY | _PARENTS)
    if not found:
        return {}
    out = {"model": (_text(_cf.CFDictionaryGetValue(found, _key("Product Name"))) or "").strip(), "medium": _text(_cf.CFDictionaryGetValue(found, _key("Medium Type")))}
    _cf.CFRelease(found)
    return out


def _kind(medium):
    return {"Solid State": "ssd", "Rotational": "hdd"}.get(medium, "disk")


_drivers = {}


def _block_drivers():
    """Physical disks only: disk images and APFS containers would count the same I/O again."""
    if not _drivers and _iokit:
        for entry in _services("IOBlockStorageDriver"):
            name = _iokit.IORegistryEntrySearchCFProperty(entry, b"IOService", _key("BSD Name"), None, _RECURSIVELY)
            label = _text(name)
            if name:
                _cf.CFRelease(name)
            info = _characteristics(entry)
            if label and info.get("medium") and "image" not in (info.get("model") or "").lower():
                _drivers[label] = (entry, info)
    return _drivers


def _physical_disk(media):
    """The whole disk under a piece of media: an APFS volume lives on a synthesized disk, which sits on a partition of the real one. The counters are the real one's."""
    found = None
    entry = media
    _iokit.IOObjectRetain(entry)
    while entry:
        name = _iokit.IORegistryEntryCreateCFProperty(entry, _key("BSD Name"), None, 0)
        if name:
            label = _text(name)
            _cf.CFRelease(name)
            if label and re.fullmatch(r"disk\d+", label):
                found = label
        parent = C.c_uint32()
        code = _iokit.IORegistryEntryGetParentEntry(entry, b"IOService", C.byref(parent))
        _iokit.IOObjectRelease(entry)
        entry = parent.value if code == 0 else 0
    return found


def list_disks():
    return {name: {"kind": _kind(info.get("medium")), "model": info.get("model"), "size_gib": None} for name, (_entry, info) in _block_drivers().items()}


def disk_counters(names):
    """name -> [sectors read, sectors written, ms busy, reads, writes], as in /proc/diskstats."""
    out = {}
    for name, (entry, _info) in _block_drivers().items():
        if name not in names:
            continue
        stats = _iokit.IORegistryEntryCreateCFProperty(entry, _key("Statistics"), None, 0)
        if not stats:
            continue
        read_ns, write_ns = _number(stats, "Total Time (Read)") or 0, _number(stats, "Total Time (Write)") or 0
        out[name] = [(_number(stats, "Bytes (Read)") or 0) // _SECTOR, (_number(stats, "Bytes (Write)") or 0) // _SECTOR, (read_ns + write_ns) // 1_000_000,
                     _number(stats, "Operations (Read)") or 0, _number(stats, "Operations (Write)") or 0]
        _cf.CFRelease(stats)
    return out


_NETWORK_FILESYSTEMS = ("nfs", "smbfs", "afpfs", "webdav", "cifs", "macfuse", "osxfuse")


def file_location(path):
    """Which tier a file lives on."""
    try:
        real = os.path.realpath(path)
        size = os.stat(real).st_size
    except OSError:
        return None
    out = {"path": real, "size_gib": round(size / (1 << 30), 3)}
    info = _StatFs()
    # On Intel the plain symbol has the old, smaller structure.
    statfs = getattr(_libc, "statfs$INODE64", None) or _libc.statfs
    if statfs(real.encode(), C.byref(info)) != 0:
        return out
    out["fstype"] = info.fstype.decode()
    out["mount"] = info.mount.decode()
    if out["fstype"] in _NETWORK_FILESYSTEMS:
        out["network"] = True
        return out
    source = info.source.decode()
    match = re.match(r"/dev/(disk\d+)", source)
    if match and _iokit:
        media = _iokit.IOServiceGetMatchingService(0, _iokit.IOBSDNameMatching(0, 0, match.group(1).encode()))
        if media:
            details = _characteristics(media)
            physical = _physical_disk(media)
            _iokit.IOObjectRelease(media)
            if details.get("medium"):
                out["disks"] = [{"name": physical or match.group(1), "kind": _kind(details["medium"]), "model": details.get("model")}]
    return out


# The GPU ----------------------------------------------------------------------------------------------------------------------------------

class Accelerator:
    """The GPU's own statistics: utilization, and what it holds of the one pool of memory."""

    def __init__(self):
        self.entry = _iokit.IOServiceGetMatchingService(0, _iokit.IOServiceMatching(b"IOAccelerator")) if _iokit and _cf else 0
        if not self.entry:
            raise OSError("no GPU found in the IOKit registry")
        self.statistics_key = _key("PerformanceStatistics")
        if not self.read():
            raise OSError("this GPU publishes no statistics")

    def read(self):
        """(memory in use in MiB, utilization %), or None."""
        stats = _iokit.IORegistryEntryCreateCFProperty(self.entry, self.statistics_key, None, 0)
        if not stats:
            return None
        memory, util = _number(stats, "In use system memory"), _number(stats, "Device Utilization %")
        _cf.CFRelease(stats)
        return (None if memory is None else memory >> 20), util

    def clients(self):
        """{GPU client: (pid, GPU time in ns)}. A client is one Metal device a process opened. Its time is the sum over the command queues it still has."""
        iterator = C.c_uint32()
        if _iokit.IORegistryEntryGetChildIterator(self.entry, b"IOService", C.byref(iterator)) != 0:
            return {}
        out = {}
        while True:
            child = _iokit.IOIteratorNext(iterator)
            if not child:
                break
            creator = _iokit.IORegistryEntryCreateCFProperty(child, _key("IOUserClientCreator"), None, 0)
            usage = _iokit.IORegistryEntryCreateCFProperty(child, _key("AppUsage"), None, 0)
            match = re.match(r"pid (\d+)", _text(creator) or "")
            if match:
                total = 0
                for i in range(_cf.CFArrayGetCount(usage) if usage else 0):
                    total += _number(_cf.CFArrayGetValueAtIndex(usage, i), "accumulatedGPUTime") or 0
                identity = C.c_uint64()
                _iokit.IORegistryEntryGetRegistryEntryID(child, C.byref(identity))
                out[identity.value] = (int(match.group(1)), total)
            for ref in (creator, usage):
                if ref:
                    _cf.CFRelease(ref)
            _iokit.IOObjectRelease(child)
        _iokit.IOObjectRelease(iterator)
        return out

    def name(self):
        return sysctl_text("machdep.cpu.brand_string") or "Apple GPU"

    def close(self):
        if self.entry:
            _iokit.IOObjectRelease(self.entry)
            self.entry = 0


if _ioreport and _cf:
    try:
        _ioreport.IOReportCopyChannelsInGroup.restype = _P
        _ioreport.IOReportCopyChannelsInGroup.argtypes = [_P, _P, C.c_uint64, C.c_uint64, C.c_uint64]
        _ioreport.IOReportCreateSubscription.restype = _P
        _ioreport.IOReportCreateSubscription.argtypes = [_P, _P, C.POINTER(_P), C.c_uint64, _P]
        _ioreport.IOReportCreateSamples.restype = _P
        _ioreport.IOReportCreateSamples.argtypes = [_P, _P, _P]
        _ioreport.IOReportCreateSamplesDelta.restype = _P
        _ioreport.IOReportCreateSamplesDelta.argtypes = [_P, _P, _P]
        _ioreport.IOReportChannelGetChannelName.restype = _P
        _ioreport.IOReportChannelGetChannelName.argtypes = [_P]
        _ioreport.IOReportSimpleGetIntegerValue.restype = C.c_int64
        _ioreport.IOReportSimpleGetIntegerValue.argtypes = [_P, C.c_int]
        _ioreport.IOReportStateGetCount.restype = C.c_int
        _ioreport.IOReportStateGetCount.argtypes = [_P]
        _ioreport.IOReportStateGetNameForIndex.restype = _P
        _ioreport.IOReportStateGetNameForIndex.argtypes = [_P, C.c_int]
        _ioreport.IOReportStateGetResidency.restype = C.c_int64
        _ioreport.IOReportStateGetResidency.argtypes = [_P, C.c_int]
    except AttributeError:
        _ioreport = None


def _subscribe(group, subgroup=None):
    """(subscription, channels) for an IOReport group, or None."""
    if not (_ioreport and _cf):
        return None
    found = _ioreport.IOReportCopyChannelsInGroup(_key(group), _key(subgroup) if subgroup else None, 0, 0, 0)
    if not found:
        return None
    mutable = _cf.CFDictionaryCreateMutableCopy(None, _cf.CFDictionaryGetCount(found), found)
    channels = _P()
    subscription = _ioreport.IOReportCreateSubscription(None, mutable, C.byref(channels), 0, None)
    _cf.CFRelease(found)
    return (subscription, channels) if subscription and channels else None


def _channels(samples):
    array = _cf.CFDictionaryGetValue(samples, _key("IOReportChannels"))
    return [_cf.CFArrayGetValueAtIndex(array, i) for i in range(_cf.CFArrayGetCount(array))] if array else []


class EnergyCounter:
    """Cumulative GPU energy from IOReport."""

    def __init__(self):
        self.subscribed = _subscribe("Energy Model")

    def millijoules(self):
        if not self.subscribed:
            return None
        samples = _ioreport.IOReportCreateSamples(self.subscribed[0], self.subscribed[1], None)
        if not samples:
            return None
        try:
            for channel in _channels(samples):
                if _text(_ioreport.IOReportChannelGetChannelName(channel)) == "GPU Energy":
                    return _ioreport.IOReportSimpleGetIntegerValue(channel, 0) // 1_000_000  # nanojoules
        finally:
            _cf.CFRelease(samples)
        return None


def _gpu_frequencies():
    """The GPU's clock in MHz for each of its performance states, lowest first, from the device tree: Hz up to the M2, kHz later."""
    if not _iokit:
        return []
    _iokit.IOServiceNameMatching.restype = _P
    _iokit.IOServiceNameMatching.argtypes = [C.c_char_p]
    _cf.CFDataGetLength.restype = C.c_long
    _cf.CFDataGetLength.argtypes = [_P]
    _cf.CFDataGetBytePtr.restype = _P
    _cf.CFDataGetBytePtr.argtypes = [_P]
    manager = _iokit.IOServiceGetMatchingService(0, _iokit.IOServiceNameMatching(b"pmgr"))
    if not manager:
        return []
    data = _iokit.IORegistryEntryCreateCFProperty(manager, _key("voltage-states9"), None, 0)
    _iokit.IOObjectRelease(manager)
    if not data:
        return []
    raw = C.string_at(_cf.CFDataGetBytePtr(data), _cf.CFDataGetLength(data))
    _cf.CFRelease(data)
    values = [struct.unpack_from("<I", raw, offset)[0] for offset in range(0, len(raw) - 7, 8)]
    return [round(value / (1e6 if value > 1e8 else 1e3)) for value in values if value > 0]


class GpuClock:
    """The GPU's average clock between two reads, from the time it spent in each performance state. It says what no utilization figure does: that the GPU was running slower."""

    def __init__(self):
        self.subscribed = _subscribe("GPU Stats", "GPU Performance States")
        self.frequencies = _gpu_frequencies() if self.subscribed else []
        self.last = None

    def read(self):
        """MHz while the GPU was on since the last read, or None."""
        if not self.frequencies:
            return None
        samples = _ioreport.IOReportCreateSamples(self.subscribed[0], self.subscribed[1], None)
        if not samples:
            return None
        previous, self.last = self.last, samples
        if previous is None:
            return None
        delta = _ioreport.IOReportCreateSamplesDelta(previous, samples, None)
        _cf.CFRelease(previous)
        if not delta:
            return None
        try:
            weighted = total = 0
            for channel in _channels(delta):
                if _text(_ioreport.IOReportChannelGetChannelName(channel)) != "GPUPH":
                    continue
                for i in range(_ioreport.IOReportStateGetCount(channel)):
                    match = re.fullmatch(r"P(\d+)", _text(_ioreport.IOReportStateGetNameForIndex(channel, i)) or "")
                    state = int(match.group(1)) - 1 if match else -1
                    if 0 <= state < len(self.frequencies):
                        residency = _ioreport.IOReportStateGetResidency(channel, i)
                        weighted += residency * self.frequencies[state]
                        total += residency
            return round(weighted / total) if total else None
        finally:
            _cf.CFRelease(delta)


_METAL_LIMIT = """
import ctypes as C
metal = C.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
objc = C.CDLL("/usr/lib/libobjc.dylib")
metal.MTLCreateSystemDefaultDevice.restype = C.c_void_p
objc.sel_registerName.restype = C.c_void_p
objc.sel_registerName.argtypes = [C.c_char_p]
send = C.CFUNCTYPE(C.c_uint64, C.c_void_p, C.c_void_p)(("objc_msgSend", objc))
print(send(metal.MTLCreateSystemDefaultDevice(), objc.sel_registerName(b"recommendedMaxWorkingSetSize")))
"""


def gpu_working_set_mib():
    """What macOS recommends the GPU hold at most, usually two thirds to three quarters of RAM. Past it, Metal starts paging and inference slows down sharply. Asked of a short-lived
    process, so that the profiler itself does not load Metal and become a client of the GPU."""
    try:
        done = subprocess.run([sys.executable, "-I", "-c", _METAL_LIMIT], capture_output=True, text=True, timeout=20, check=True)
        return int(done.stdout.strip()) >> 20 or None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
