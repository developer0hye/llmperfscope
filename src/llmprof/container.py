"""`docker run ...` as a target. The docker client is only a remote control: the engine is a child of the daemon, so the client's
process tree holds nothing worth measuring. llmprof asks docker for the container's first process and follows that instead.
Memory, CPU and per-process GPU memory all stay visible from the host, because a container's processes are host processes."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path


class DockerTarget:
    def __init__(self, tool, run_dir):
        self.tool = tool
        self.run_dir = Path(run_dir).resolve()
        self.cidfile = self.run_dir / "container.id"
        self.container = None
        self.pid = None
        self.image = None
        self._done = threading.Event()

    def prepare(self, cmd, index, marker_file):
        """Adds what llmprof needs right after `docker run`: the id file, and the marker file for scripts inside the container."""
        if self.cidfile.exists():
            self.cidfile.unlink()
        extra = ["--cidfile", str(self.cidfile), "-e", "LLMPROF_MARKERS=/llmprof-run/markers.log", "-v", f"{self.run_dir}:/llmprof-run"]
        return cmd[:index + 2] + extra + cmd[index + 2:]

    def follow(self, on_pid, timeout_s=300.0):
        """Finds the container's first process as soon as docker has one, in the background."""

        def wait():
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline and not self._done.is_set():
                if self.cidfile.exists():
                    self.container = self.cidfile.read_text().strip() or None
                if self.container:
                    done = subprocess.run([self.tool, "inspect", "-f", "{{.State.Pid}}", self.container], capture_output=True, text=True,
                                          check=False)
                    value = done.stdout.strip()
                    if value.isdigit() and int(value) > 0:
                        self.pid = int(value)
                        on_pid(self.pid)
                        image = subprocess.run([self.tool, "inspect", "-f", "{{.Config.Image}} {{.Image}}", self.container],
                                               capture_output=True, text=True, check=False)
                        self.image = image.stdout.strip()
                        return
                time.sleep(0.2)

        threading.Thread(target=wait, daemon=True, name="llmprof-container").start()

    def identity(self):
        """Which image ran, as the report's record of which code this was."""
        if not self.image:
            return {}
        name, _, digest = self.image.partition(" ")
        return {"container_image": name, "container_image_id": digest}

    def stop(self, wait_s=30):
        """Stops the container itself. A signal to the client only reaches it while the client is attached."""
        self._done.set()
        if self.container:
            subprocess.run([self.tool, "stop", "-t", str(int(wait_s)), self.container], capture_output=True, text=True, check=False)

    def remove_if_running(self):
        self._done.set()
        if not self.container:
            return False
        done = subprocess.run([self.tool, "inspect", "-f", "{{.State.Running}}", self.container], capture_output=True, text=True, check=False)
        if done.stdout.strip() == "true":
            subprocess.run([self.tool, "kill", self.container], capture_output=True, text=True, check=False)
            return True
        return False


def tool_name(cmd, index):
    return os.path.basename(cmd[index])
