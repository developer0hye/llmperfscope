"""An engine with known behaviour. usage: fake_engine.py <scenario> [args]"""

import mmap
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import llmperfscope  # noqa: E402

MIB = 1 << 20


def touch_anon(mib):
    block = bytearray(mib * MIB)
    for i in range(0, len(block), 4096):
        block[i] = 1
    return block


def touch_file(path, mib):
    with open(path, "wb") as f:
        f.truncate(mib * MIB)
    f = open(path, "rb")
    mapping = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    total = 0
    for i in range(0, len(mapping), 4096):
        total += mapping[i]
    return f, mapping


def spin(seconds):
    t0 = time.thread_time()
    x = 0
    while time.thread_time() - t0 < seconds:
        x += 1
    return x


def phases():
    """Known durations and allocations, announced with markers."""
    llmperfscope.mark("load")
    block = touch_anon(200)
    f, mapping = touch_file(sys.argv[2], 64)
    time.sleep(0.4)
    llmperfscope.mark("decode", tokens=100)
    spin(0.6)
    llmperfscope.mark("post", group="teardown")
    del block
    mapping.close()
    f.close()
    time.sleep(0.2)


def main():
    scenario = sys.argv[1]
    if scenario == "phases":
        phases()
    elif scenario == "lines":
        print("LOADING", flush=True)
        time.sleep(0.3)
        print("READY", flush=True)
        spin(0.3)
    elif scenario == "fail":
        sys.exit(7)
    elif scenario == "sleep":
        time.sleep(600)
    elif scenario == "spin":
        print("started", flush=True)
        while True:
            pass
    elif scenario == "spin_for":
        print("started", flush=True)
        spin(float(sys.argv[2]))
    elif scenario == "child":
        # The parent stays small: only a view of the whole tree finds the memory.
        child = subprocess.Popen([sys.executable, __file__, "hold", "150"])
        time.sleep(1.2)
        child.terminate()
        child.wait()
    elif scenario == "hold":
        block = touch_anon(int(sys.argv[2]))
        time.sleep(30)
        del block
    elif scenario == "daemon":
        # Leaves a process behind, as a server's helper might.
        subprocess.Popen([sys.executable, __file__, "sleep"], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.2)
    elif scenario == "plain_marker":
        # No library, as a shell script or another language would do it.
        with open(os.environ["LLMPERFSCOPE_MARKERS"], "a") as f:
            f.write("first_phase\n")
        time.sleep(0.5)
        with open(os.environ["LLMPERFSCOPE_MARKERS"], "a") as f:
            f.write(f"{time.monotonic_ns()} second_phase\n")
        time.sleep(0.5)
    else:
        raise SystemExit(f"unknown scenario {scenario}")


if __name__ == "__main__":
    main()
