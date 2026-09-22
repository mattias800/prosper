#!/usr/bin/env python3
"""Self-test for wait_profile.py, against a /proc tree built BY HAND.

The defect this pins (#3400) is an aggregation over the wrong key: samples were grouped by the
thread's `comm` string, so distinct threads sharing a name became one population and its
percentages described none of them. The positive instances below -- two threads with one name,
a TID reused by a new thread, a thread renamed mid-run, a PID reused by a new process -- are
written into a temporary directory here rather than taken from whatever the host is running, so
the case exists whether or not the machine happens to have one.

Run: python3 tools/perf/test_wait_profile.py
"""

import contextlib
import importlib.util
import io
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

TOOL = Path(__file__).resolve().parent / "wait_profile.py"


def load():
    spec = importlib.util.spec_from_file_location("wait_profile", str(TOOL))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stat_line(tid, comm, state, start):
    # proc(5): pid (comm) state, then fields 4..21, then starttime at field 22.
    return f"{tid} ({comm}) {state} " + " ".join(["0"] * 18) + f" {start} 0 0\n"


class FakeProc:
    """A /proc/<pid> tree: process stat plus task/<tid>/{stat,wchan,syscall}."""

    def __init__(self, root, pid=999, start=5000):
        self.root, self.pid = Path(root), pid
        (self.root / str(pid) / "task").mkdir(parents=True)
        self.set_process_start(start)

    def set_process_start(self, start):
        (self.root / str(self.pid) / "stat").write_text(stat_line(self.pid, "prosper-app", "S", start))

    def thread(self, tid, comm, state, start, wchan="futex_do_wait", syscall="202 0xad2048 0x80 0x0"):
        d = self.root / str(self.pid) / "task" / str(tid)
        d.mkdir(parents=True, exist_ok=True)
        (d / "stat").write_text(stat_line(tid, comm, state, start))
        (d / "wchan").write_text(wchan)
        (d / "syscall").write_text("running" if state == "R" else syscall)

    def remove(self, tid):
        d = self.root / str(self.pid) / "task" / str(tid)
        for f in d.iterdir():
            f.unlink()
        d.rmdir()


def run(mod, root, seconds, hz, on_sleep=None, windows=1):
    """Drive main() on a fake clock that advances only when the tool sleeps."""
    clock = {"t": 0.0, "sweeps": 0}

    def fake_time():
        return clock["t"]

    def fake_sleep(dt):
        clock["sweeps"] += 1
        if on_sleep:
            on_sleep(clock["sweeps"])
        clock["t"] += dt

    mod.PROC_ROOT = str(root)
    out = io.StringIO()
    argv = ["wait_profile", "--pid", "999", "--seconds", str(seconds), "--hz", str(hz),
            "--windows", str(windows), "--min-share", "0"]
    with patch.object(sys, "argv", argv), patch.object(mod.time, "time", fake_time), \
            patch.object(mod.time, "sleep", fake_sleep), contextlib.redirect_stdout(out):
        rc = mod.main()
    return out.getvalue(), rc


def main():
    failures = []
    mod = load()

    # 1. THE ISSUE'S CASE: one running thread and one waiting thread, both named "same".
    #    Pre-fix this printed ONE row, `same (2 samples) 50% RUNNING 50% S:futex_do_wait`.
    with tempfile.TemporaryDirectory() as tmp:
        fp = FakeProc(tmp)
        fp.thread(101, "same", "R", 7001)
        fp.thread(102, "same", "S", 7002)
        out, rc = run(mod, tmp, seconds=1, hz=1)
        if rc != 0:
            failures.append(f"case 1: exit {rc}")
        if "50.0%" in out:
            failures.append(f"case 1: two same-named threads were merged into 50/50 shares: {out!r}")
        if "same  tid=101  (1 samples)" not in out or "same  tid=102  (1 samples)" not in out:
            failures.append(f"case 1: expected one row per TID: {out!r}")
        row101 = out.split("tid=101")[1].split("tid=")[0] if "tid=101" in out else ""
        if "100.0%  RUNNING" not in row101:
            failures.append(f"case 1: tid 101 must be 100% RUNNING: {out!r}")
        if "2 distinct threads are named 'same'" not in out:
            failures.append(f"case 1: the name collision is not stated: {out!r}")

    # 2. SCALE, the issue's second figure: 1 running + 15 idle workers of one name read 6.25%.
    with tempfile.TemporaryDirectory() as tmp:
        fp = FakeProc(tmp)
        fp.thread(200, "Job.Worker", "R", 8000)
        for i in range(15):
            fp.thread(201 + i, "Job.Worker", "S", 8001 + i)
        out, _ = run(mod, tmp, seconds=2, hz=1)
        if "6.2%" in out or "6.3%" in out:
            failures.append(f"case 2: population share printed as a thread share: {out!r}")
        if "16 distinct threads are named 'Job.Worker'" not in out:
            failures.append(f"case 2: collision count missing: {out!r}")

    # 3. TID REUSE: tid 300 exits after sweep 1 and a NEW thread (new starttime) takes the TID.
    #    Pre-fix and under a tid-only key both would be one thread's history.
    with tempfile.TemporaryDirectory() as tmp:
        fp = FakeProc(tmp)
        fp.thread(300, "worker", "R", 9000)

        def reuse(sweep):
            if sweep == 1:
                fp.remove(300)
                fp.thread(300, "worker", "S", 9500)
        out, _ = run(mod, tmp, seconds=2, hz=1, on_sleep=reuse)
        if out.count("tid=300  (1 samples)") != 2:
            failures.append(f"case 3: a reused TID must be two identities: {out!r}")

    # 4. RENAME: same identity, comm changes. One row, both names shown.
    with tempfile.TemporaryDirectory() as tmp:
        fp = FakeProc(tmp)
        fp.thread(400, "Thread-1", "S", 9100)

        def rename(sweep):
            if sweep == 1:
                fp.thread(400, "GfxFlipThread", "S", 9100)
        out, _ = run(mod, tmp, seconds=2, hz=1, on_sleep=rename)
        if "Thread-1 -> GfxFlipThread  (renamed)  tid=400  (2 samples)" not in out:
            failures.append(f"case 4: a renamed thread must stay one row naming both: {out!r}")

    # 5. PID REUSE: the process's own starttime changes -> stop, and say why.
    with tempfile.TemporaryDirectory() as tmp:
        fp = FakeProc(tmp)
        fp.thread(500, "main", "S", 9200)

        def respawn(sweep):
            if sweep == 1:
                fp.set_process_start(6000)
        out, _ = run(mod, tmp, seconds=3, hz=1, on_sleep=respawn)
        if "tid=500  (1 samples)" not in out:
            failures.append(f"case 5: samples after the PID changed process were kept: {out!r}")
        if "DIFFERENT process" not in out:
            failures.append(f"case 5: the early stop is not explained: {out!r}")

    # 6. CONTROL: distinct names still produce the ordinary per-thread rows and no NOTE.
    with tempfile.TemporaryDirectory() as tmp:
        fp = FakeProc(tmp)
        fp.thread(601, "a", "R", 1)
        fp.thread(602, "b", "S", 2)
        out, _ = run(mod, tmp, seconds=1, hz=1)
        if "distinct threads are named" in out:
            failures.append(f"case 6: collision note printed without a collision: {out!r}")
        if "100.0%  S:futex_do_wait [futex uaddr=0xad2048]" not in out:
            failures.append(f"case 6: waiting bucket missing its futex address: {out!r}")

    if failures:
        print("FAILURES:")
        for f in failures:
            print(" -", f)
        return 1
    print("wait_profile self-test: all cases pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
