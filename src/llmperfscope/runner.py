"""Runs a command under the sampler and writes run.json. Framework-agnostic: an engine is a process tree plus a stream of
phase events, however they are obtained."""

from __future__ import annotations

import ctypes as C
import json
import os
import platform
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import __version__, adapters, analyze, container, cpuprof, diagnose, machine, markers, procfs
from . import nsys as nsys_tool
from .gpu import open_backend
from .sampler import Sampler, load_ticks

SCHEMA = "llmperfscope.run/1"
DEFAULT_STALL_S = 600.0


@dataclass
class Options:
    out: str
    label: str = ""
    hz: float = 50.0
    slow_hz: float = 2.0
    baseline_s: float = 1.0
    tail_s: float = 1.0
    timeout_s: float = 0.0
    stall_s: float = None  # None: DEFAULT_STALL_S when the adapter knows silence means trouble, else off
    tee: bool = False
    quiet: bool = False
    precheck: str = "warn"  # warn | wait | require | skip
    max_wait_s: float = 3600.0
    adapter: str = "auto"
    marks: list = field(default_factory=list)
    env: dict = field(default_factory=dict)
    nsys: bool = False
    nsys_args: list = field(default_factory=list)
    cpu_profile: bool = False
    cpu_hz: int = 200
    stop_signal: str = "TERM"
    stop_timeout_s: float = 120.0
    cold_start: bool = False
    model: str = None
    pcie: bool = True


class Refused(Exception):
    """The machine is not clean and the options say not to measure."""


def _set_subreaper():
    """Daemonized grandchildren stay in our tree instead of reparenting to init, so they are measured and cleaned up."""
    try:
        C.CDLL(None).prctl(36, 1, 0, 0, 0)  # PR_SET_CHILD_SUBREAPER
    except (OSError, AttributeError):
        pass


def _reap_orphans():
    try:
        while os.waitpid(-1, os.WNOHANG)[0] != 0:
            pass
    except ChildProcessError:
        pass


class Run:
    t_spawn = None
    stop_requested_at = None
    killed = False
    workload_thread = None
    wall_baseline_ns = None
    docker = None
    docker_stop_thread = None

    def __init__(self, cmd, options, workload=None, attach_pid=None):
        self.cmd = list(cmd)
        self.opts = options
        self.workload = workload
        self.attach_pid = attach_pid
        self.dir = Path(options.out)
        self.phases = []
        self.lock = threading.Lock()
        self.progress_t = time.monotonic()
        self.proc = None
        self.pid = None
        self.rusage = None
        self.returncode = None
        self.stop_reason = None  # why we ended the target: timeout, stalled, interrupted, workload finished
        self.failure = {}
        self.notes = []
        self.workload_result = {}
        self.workload_error = None
        self.stdout_path = self.dir / "stdout.log"
        self.stderr_path = self.dir / "stderr.log"
        self.adapter = None

    def log(self, text):
        if not self.opts.quiet:
            print(f"[llmperfscope] {text}", flush=True)

    # Phases ---------------------------------------------------------------------------------------------------------

    def emit(self, phase, group=None, t=None, tokens=None, **meta):
        event = {"name": phase, "group": group or phase, "t": t if t is not None else time.monotonic_ns()}
        if tokens is not None:
            event["tokens"] = tokens
        if meta:
            event["meta"] = meta
        with self.lock:
            self.phases.append(event)
        self.progress_t = time.monotonic()
        if phase not in ("decode", "prefill", "post_test"):
            self.log(f"{(event['t'] - (self.t_spawn or event['t'])) / 1e9:9.2f}s  {phase}")

    # Output ---------------------------------------------------------------------------------------------------------

    def _reader(self, stream_name, pipe, path):
        with open(path, "wb") as f:
            while True:
                line = pipe.readline()
                if not line:
                    return
                now = time.monotonic_ns()
                f.write(line)
                f.flush()
                self.progress_t = time.monotonic()
                if self.opts.tee:
                    os.write(2 if stream_name == "stderr" else 1, line)
                try:
                    self.adapter.on_line(stream_name, now, line.decode("utf-8", "replace").rstrip("\r\n"), self.emit)
                except Exception as e:  # noqa: BLE001 - a parser bug must not stop the measurement; the report says so
                    self.notes.append(f"adapter error: {e!r}")

    # Watchdog -------------------------------------------------------------------------------------------------------

    @staticmethod
    def _resource_progress(tick, state):
        """Output is not the only sign of life: a long load shows up as disk reads and growing memory. CPU and GPU
        utilization do not count: a hung collective spins at 100% of both."""
        if not tick:
            return False
        host = tick["h"]
        slow = tick.get("s") or {}
        state["io"] = slow.get("io", state.get("io"))
        sign = (sum(row[0] or 0 for row in tick["g"]), host[0] + host[1], state["io"])
        changed = sign != state.get("sign")
        state["sign"] = sign
        return changed

    def _diagnose(self):
        """What the target is doing when it gets killed. This is what is missing from a hung benchmark's log."""
        out = {"threads": []}
        pids = procfs.tree({self.pid}) if self.pid else set()
        for pid in sorted(pids):
            for tid, comm, ticks in procfs.threads(pid):
                out["threads"].append({"pid": pid, "tid": tid, "comm": comm, "cpu_s": round(ticks / procfs.CLK_TCK, 2),
                                       "wchan": procfs.wchan(pid, tid)})
        out["threads"].sort(key=lambda t: -t["cpu_s"])
        out["threads"] = out["threads"][:16]
        try:
            with open(self.stderr_path, "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 2000))
                out["stderr_tail"] = f.read().decode("utf-8", "replace").splitlines()[-12:]
        except OSError:
            pass
        return out

    def _signal_tree(self, sig):
        if self.attach_pid:
            return
        try:
            os.killpg(self.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass
        # Only processes that left the group. A second signal is not harmless: servers take it as "exit right now, skip cleanup".
        for pid in procfs.tree(set(procfs.children_of(os.getpid()))):
            try:
                if os.getpgid(pid) != self.pid:
                    os.kill(pid, sig)
            except (ProcessLookupError, PermissionError):
                pass

    def request_stop(self, reason):
        self.stop_reason = self.stop_reason or reason
        self.stop_requested_at = time.monotonic()
        # In this order: the tree signal goes to every child of llmperfscope, and `docker stop` must not be one of them yet.
        self._signal_tree(getattr(signal, "SIG" + self.opts.stop_signal))
        if self.docker and self.docker_stop_thread is None:
            self.docker_stop_thread = threading.Thread(target=self.docker.stop, args=(self.opts.stop_timeout_s,), daemon=True,
                                                       name="llmperfscope-docker-stop")
            self.docker_stop_thread.start()

    def _poll_exit(self):
        if self.attach_pid:
            return bool(self.stop_reason) or not os.path.exists(f"/proc/{self.attach_pid}")
        try:
            pid, status, rusage = os.wait4(self.pid, os.WNOHANG)
        except ChildProcessError:
            return True
        if pid == 0:
            return False
        self.rusage = rusage
        self.returncode = os.waitstatus_to_exitcode(status)
        if self.proc is not None:
            self.proc.returncode = self.returncode
        return True

    def _poll_markers(self, reader):
        for event in reader.poll():
            self.emit(event["phase"], group=event.get("group"), t=int(event["t"]), tokens=event.get("tokens"),
                      **event.get("meta", {}))

    def _watch(self, sampler, marker_reader, stall_s):
        t_start = time.monotonic()
        state = {}
        while True:
            if self._poll_exit():
                return
            self._poll_markers(marker_reader)
            now = time.monotonic()
            if self._resource_progress(sampler.latest, state):
                self.progress_t = now
            if self.opts.timeout_s and now - t_start > self.opts.timeout_s and not self.stop_reason:
                self.failure = self._diagnose()
                self.log(f"timeout after {self.opts.timeout_s:.0f} s, stopping the target")
                self.request_stop("timeout")
            workload_alive = self.workload_thread is not None and self.workload_thread.is_alive()
            if stall_s and now - self.progress_t > stall_s and not self.stop_reason and not workload_alive:
                self.failure = self._diagnose()
                self.log(f"no progress for {stall_s:.0f} s, stopping the target")
                self.request_stop("stalled")
            if self.stop_requested_at and now - self.stop_requested_at > self.opts.stop_timeout_s and not self.killed:
                self.log("target ignored the stop signal, sending SIGKILL")
                self.notes.append(f"the target did not exit within {self.opts.stop_timeout_s:.0f} s of the stop signal and was killed")
                self.killed = True
                self._signal_tree(signal.SIGKILL)
            time.sleep(0.2)

    # Pinning --------------------------------------------------------------------------------------------------------

    def _pin_away_from_target(self):
        """All of llmperfscope goes to one CPU the target is not allowed on, when there is one."""
        allowed = None
        for pid in procfs.tree({self.pid}):
            mask = procfs.affinity(pid)
            if mask:
                allowed = mask if allowed is None else allowed | mask
        if allowed is None:
            return None
        free = sorted(os.sched_getaffinity(0) - allowed)
        if not free:
            self.notes.append("target may use every CPU: llmperfscope shares CPUs with it")
            return None
        cpu = free[-1]
        for thread in threading.enumerate():
            native_id = getattr(thread, "native_id", None)
            if native_id:
                try:
                    os.sched_setaffinity(native_id, {cpu})
                except OSError:
                    pass
        return cpu

    # Main -----------------------------------------------------------------------------------------------------------

    def execute(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.update(self.opts.env)
        self.adapter = adapters.select(self.opts.adapter, self.cmd, self.opts.marks)
        cmd, env = self.adapter.prepare(self.cmd, env)
        stall_s = self.opts.stall_s
        if stall_s is None:
            stall_s = DEFAULT_STALL_S if self.adapter.has_progress_lines() else 0

        backend = open_backend()
        if backend.name == "none":
            self.notes.append(f"no GPU telemetry: {backend.reason}")

        gpus = None
        visible = env.get("CUDA_VISIBLE_DEVICES", "")
        if visible and all(part.strip().isdigit() for part in visible.split(",")):
            gpus = {int(part) for part in visible.split(",")}
        ignore = procfs.tree({self.attach_pid}) if self.attach_pid else ()
        model = self.opts.model or self.adapter.model_path()
        if self.opts.cold_start:
            if model:
                left = machine.evict_from_page_cache(model)
                self.log(f"cold start: {(left or 0) * 100:.0f}% of the model still in the page cache")
            else:
                self.notes.append("--cold-start needs the model file: pass --model-file")
        before = None
        if self.opts.precheck == "wait":
            # Printed even with --quiet: silence here would look like a hang.
            before = machine.wait_until_quiet(backend, max_wait_s=self.opts.max_wait_s, model=model, ignore=ignore, gpus=gpus,
                                              log=lambda text: print(f"[llmperfscope] {text}", flush=True))
        elif self.opts.precheck != "skip":
            before = machine.snapshot(backend, model=model, ignore=ignore, gpus=gpus)
        if before and not before["clean"]:
            self.log("machine not clean: " + "; ".join(before["reasons"]))
            if self.opts.precheck == "require":
                backend.close()
                raise Refused("; ".join(before["reasons"]))

        exe_index = self.adapter.target_index()
        try:
            identity = machine.binary_identity(cmd[exe_index]) if cmd else {}
        except OSError as e:  # another user's or a container's binary
            identity = {"error": str(e)}

        if self.opts.nsys:
            cmd = nsys_tool.wrap(cmd, exe_index, self.dir, self.opts.nsys_args)
        if self.opts.cpu_profile:
            env = cpuprof.prepare_env(env, self.dir, self.opts.cpu_hz)

        marker_file = self.dir / "markers.log"
        env[markers.ENV] = str(marker_file)
        # A run directory can be reused. Old events must not become phases of this run.
        try:
            marker_file.unlink()
        except OSError:
            pass
        marker_reader = markers.MarkerReader(marker_file, skip_existing=True)

        docker_at = adapters.docker_index(cmd) if not self.attach_pid else None
        if docker_at is not None:
            self.docker = container.DockerTarget(container.tool_name(cmd, docker_at), self.dir)
            cmd = self.docker.prepare(cmd, docker_at, marker_file)

        _set_subreaper()
        sampler = Sampler(self.dir / "samples.jsonl.gz", backend, hz=self.opts.hz, slow_hz=self.opts.slow_hz, pcie=self.opts.pcie)
        self.t_baseline = time.monotonic_ns()
        self.wall_baseline_ns = time.time_ns()
        sampler.start()
        with self.lock:
            self.phases.append({"name": "baseline", "group": "baseline", "t": self.t_baseline})
        time.sleep(self.opts.baseline_s)

        readers = []
        previous_handlers = {}
        if threading.current_thread() is threading.main_thread():
            def on_signal(_signum, _frame):
                self.request_stop("interrupted")

            for sig in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[sig] = signal.signal(sig, on_signal)
        self.t_spawn = time.monotonic_ns()
        with self.lock:
            self.phases.append({"name": "startup", "group": "startup", "t": self.t_spawn})
        if self.attach_pid:
            self.pid = self.attach_pid
        else:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                         env=env, start_new_session=True)
            self.pid = self.proc.pid
            for name, pipe, path in (("stdout", self.proc.stdout, self.stdout_path),
                                     ("stderr", self.proc.stderr, self.stderr_path)):
                reader = threading.Thread(target=self._reader, args=(name, pipe, path), daemon=True, name=f"llmperfscope-{name}")
                reader.start()
                readers.append(reader)
        sampler.set_roots({self.pid})
        self.log(f"target pid {self.pid}: {' '.join(machine.redact_command(cmd))[:200]}")
        if self.docker:
            self.docker.follow(sampler.add_root)
        time.sleep(0.2)  # lets wrappers such as taskset set the affinity and exec
        spare_cpu = self._pin_away_from_target()

        if self.workload:
            def run_workload():
                try:
                    self.workload_result = self.workload(self) or {}
                except Exception as e:  # noqa: BLE001 - recorded in the report and the target is stopped
                    self.workload_error = repr(e)
                    self.log(f"workload failed: {e!r}")
                finally:
                    if self.attach_pid:
                        self.stop_reason = self.stop_reason or "workload finished"
                    elif not self.stop_reason:
                        self.emit("teardown")
                        self.request_stop("workload finished")

            self.workload_thread = threading.Thread(target=run_workload, daemon=True, name="llmperfscope-workload")
            self.workload_thread.start()

        self._watch(sampler, marker_reader, stall_s)
        t_exit = time.monotonic_ns()
        self._poll_markers(marker_reader)
        for reader in readers:
            reader.join(timeout=5)
        if self.proc is not None:
            self.proc.stdout.close()
            self.proc.stderr.close()
        if self.docker_stop_thread is not None:
            self.docker_stop_thread.join(timeout=self.opts.stop_timeout_s + 30)
        if self.docker:
            if self.docker.remove_if_running():
                self.notes.append("the container outlived its docker client and was killed")
            identity = {**identity, **self.docker.identity()}
        # Leftover children of an exited root are part of this run: ending them is what keeps the GPU free for the next one.
        if not self.attach_pid:
            left = procfs.tree(set(procfs.children_of(os.getpid())))
            if left:
                self.notes.append(f"{len(left)} processes outlived the target and were stopped")
                self._signal_tree(signal.SIGTERM)
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline and procfs.tree(set(procfs.children_of(os.getpid()))):
                    _reap_orphans()
                    time.sleep(0.2)
                self._signal_tree(signal.SIGKILL)
            _reap_orphans()
        with self.lock:
            self.phases.append({"name": "tail", "group": "tail", "t": t_exit})
        time.sleep(self.opts.tail_s)
        sampler.stop()
        sampler.join(timeout=30)
        t_end = time.monotonic_ns()

        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        if self.workload_thread:
            self.workload_thread.join(timeout=5)

        after = machine.snapshot(backend, cpu_seconds=0, ignore=ignore) if self.opts.precheck != "skip" else None
        devices = backend.devices
        backend.close()
        return self._finish(cmd, env, identity, before, after, devices, sampler, t_end, spare_cpu)

    # Report ---------------------------------------------------------------------------------------------------------

    def _status(self):
        if self.stop_reason in ("timeout", "stalled", "interrupted"):
            return self.stop_reason
        if self.workload_error:
            return "workload_failed"
        if self.stop_reason == "workload finished" or self.attach_pid:
            return "ok"
        return "ok" if self.returncode == 0 else "failed"

    def _finish(self, cmd, env, identity, before, after, devices, sampler, t_end, spare_cpu):
        ticks = load_ticks(self.dir / "samples.jsonl.gz")
        with self.lock:
            phases = sorted(self.phases, key=lambda p: p["t"])
        # An engine nothing is known about: the whole life of the process is one phase.
        if not any(p["group"] not in ("baseline", "startup", "tail") for p in phases):
            for phase in phases:
                if phase["name"] == "startup":
                    phase["name"] = phase["group"] = "run"
        # Adapters fill tokens and need phase ends for their self-check.
        for i, phase in enumerate(phases):
            phase["t_end"] = phases[i + 1]["t"] if i + 1 < len(phases) else t_end
        extras = {}
        if not self.attach_pid:
            try:
                extras = self.adapter.finalize(phases, self.stdout_path, self.stderr_path) or {}
            except Exception as e:  # noqa: BLE001 - keep the rest of the report
                self.notes.append(f"adapter finalize failed: {e!r}")
        sections = analyze.summarize(ticks, phases, t_end, attached=bool(self.attach_pid)) if ticks else {"groups": {}, "memory": {}, "energy": {}}

        t0 = phases[0]["t"]
        for phase in phases:
            phase["start_s"] = round((phase.pop("t") - t0) / 1e9, 6)
            phase["end_s"] = round((phase.pop("t_end") - t0) / 1e9, 6)

        report = {
            "schema": SCHEMA,
            "tool": {"name": "llmperfscope", "version": __version__},
            "label": self.opts.label,
            "status": self._status(),
            "exit_code": self.returncode,
            "stop_reason": self.stop_reason,
            # Numbers from a traced or sampled run are for finding where time goes, never for A/B speed comparisons.
            "perturbed": bool(self.opts.nsys or self.opts.cpu_profile),
            "attached": bool(self.attach_pid),
            "wall_time_at_t0_ns": self.wall_baseline_ns,
            "command": machine.redact_command(cmd),
            "env": machine.relevant_env({k: env[k] for k in env if k != markers.ENV}),
            "adapter": self.adapter.name,
            "identity": identity,
            "host": machine.host_info(),
            "devices": devices,
            "machine_before": before,
            "machine_after": after,
            "sampler": {
                "hz_target": self.opts.hz,
                "hz_achieved": round((sampler.ticks - 1) / max((sampler.t_last - sampler.t_first) / 1e9, 1e-9), 1)
                if sampler.ticks > 1 else None,
                "max_gap_ms": round(sampler.max_gap_ns / 1e6, 1),
                "cpu_seconds": round(sampler.cpu_seconds, 3),
                "pinned_cpu": spare_cpu,
                "error": sampler.error,
            },
        }
        report.update(sections)
        report["disks"] = sampler.system.disks
        # The kernel's totals describe the process llmperfscope started. For a container that is only the docker client.
        if self.rusage is not None and not self.docker:
            report.setdefault("memory", {}).setdefault("host", {})["max_rss_kernel_mib"] = round(self.rusage.ru_maxrss / 1024, 1)
            report["cpu"] = {"user_s": round(self.rusage.ru_utime, 3), "sys_s": round(self.rusage.ru_stime, 3),
                             "major_faults": self.rusage.ru_majflt, "voluntary_switches": self.rusage.ru_nvcsw,
                             "involuntary_switches": self.rusage.ru_nivcsw}
        notes = list(self.notes)
        breakdown = extras.pop("memory_breakdown", None)
        if breakdown:
            report.setdefault("memory", {})["breakdown_mib"] = breakdown
        notes += extras.pop("notes", [])
        if "throughput" in extras:
            report["throughput"] = extras.pop("throughput")
        else:
            derived = analyze.throughput_from_phases(sections.get("groups") or {})
            if derived:
                report["throughput"] = derived
        report.update(extras)
        energy_ratio = analyze.energy_self_check(sections.get("groups") or {})
        if energy_ratio is not None:
            report.setdefault("consistency", {})["energy_over_power_gauge"] = energy_ratio
            if not 0.75 <= energy_ratio <= 1.33:
                notes.append(f"energy counter and power gauge disagree by a factor of {energy_ratio}: do not trust the energy numbers of this run")
        if self.workload_result:
            result = dict(self.workload_result)
            if "throughput" in result:
                report.setdefault("throughput", {}).update(result.pop("throughput"))
            report["workload"] = result
        if self.workload_error:
            report["workload_error"] = self.workload_error
        if self.failure:
            report["failure"] = self.failure
        if before and not before["clean"]:
            notes.append("machine was not clean before the run: " + "; ".join(before["reasons"]))
        if sampler.error:
            notes.append(f"sampler stopped early: {sampler.error}")
        achieved = report["sampler"]["hz_achieved"]
        if achieved and achieved < 0.5 * self.opts.hz:
            notes.append(f"the sampler fell behind ({achieved} of {self.opts.hz} Hz, largest gap {report['sampler']['max_gap_ms']} ms): gauges in phases "
                         "shorter than that gap may be missing, cumulative counters are still right")
        if self.opts.nsys:
            report["nsys"] = nsys_tool.finish(self.dir, phases, self.wall_baseline_ns, log=self.log)
        if self.opts.cpu_profile:
            report["cpu_profile"] = cpuprof.summarize(self.dir, phases, self.t_baseline)
        report["notes"] = notes
        report["phases"] = phases
        report["platform"] = {"python": platform.python_version()}
        torch_stats = {}
        for phase in phases:
            for key, value in ((phase.get("meta") or {}).get("torch") or {}).items():
                torch_stats[key] = max(torch_stats.get(key, 0.0), value)
        if torch_stats:
            report["torch_memory"] = torch_stats
        report["diagnosis"] = diagnose.diagnose(report)
        with open(self.dir / "run.json", "w") as f:
            json.dump(report, f, indent=1)
        return report
