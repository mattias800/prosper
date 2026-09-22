#!/usr/bin/env python3
"""Self-test for lock_holder.py's COVERAGE accounting, against a /proc tree built BY HAND.

The defect this pins (#3401): every OSError reading a thread's syscall file became the same None a
readable non-futex sample returns, so a run that could read nothing printed "No thread was in a
futex wait in ANY sample. That is a real finding" and exited 0. The arms below build each input
class by hand -- readable futex waits, readable non-futex, permission/I-O failures, a thread that
vanishes mid-sweep, an empty or unreadable task list -- rather than relying on what the host
happens to be running or on file modes (a root runner reads a mode-000 file, so the failures are
injected at `open` instead).

Run: python3 tools/perf/test_lock_holder.py
"""

import builtins
import contextlib
import errno
import importlib.util
import io
import struct
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

TOOL = Path(__file__).resolve().parent / "lock_holder.py"
PID = 999


def load():
    spec = importlib.util.spec_from_file_location("lock_holder", str(TOOL))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_tree(root, threads):
    """threads: {tid: (comm, syscall-file-text)}"""
    task = Path(root) / str(PID) / "task"
    task.mkdir(parents=True)
    for tid, (comm, syscall) in threads.items():
        d = task / str(tid)
        d.mkdir()
        (d / "stat").write_text(f"{tid} ({comm}) S 1 0 0\n")
        (d / "syscall").write_text(syscall + "\n")


def run(mod, root, samples=2, fail_open=None, fail_listdir=None, owner=None):
    """fail_open: {tid: errno} raised when that thread's syscall file is opened.
    fail_listdir: errno raised when the task directory is listed.
    owner: tid the fake owner word names, or None to leave read_remote failing."""
    mod.PROC_ROOT = str(root)
    real_open, real_listdir = builtins.open, mod.os.listdir

    def fake_open(path, *a, **k):
        for tid, err in (fail_open or {}).items():
            if str(path).endswith(f"/task/{tid}/syscall"):
                raise OSError(err, errno.errorcode[err], str(path))
        return real_open(path, *a, **k)

    def fake_listdir(path):
        if fail_listdir is not None and str(path).endswith("/task"):
            raise OSError(fail_listdir, errno.errorcode[fail_listdir], str(path))
        return real_listdir(path)

    def fake_remote(pid, addr, size):
        return None if owner is None else struct.pack("<i", owner)

    out = io.StringIO()
    argv = ["lock_holder", "--pid", str(PID), "--samples", str(samples), "--interval", "0"]
    with patch.object(sys, "argv", argv), patch("builtins.open", fake_open), \
            patch.object(mod.os, "listdir", fake_listdir), \
            patch.object(mod, "read_remote", fake_remote), \
            patch.object(mod.time, "sleep"), contextlib.redirect_stdout(out):
        rc = mod.main()
    return out.getvalue(), rc


NEGATIVE = "No futex wait was observed"
OLD_NEGATIVE = "is not blocked on a lock"


def main():
    failures = []
    mod = load()

    def expect_void(label, out, rc):
        if rc != 2:
            failures.append(f"{label}: expected exit 2 (could not evaluate), got {rc}: {out!r}")
        if "UNAVAILABLE" not in out:
            failures.append(f"{label}: expected UNAVAILABLE: {out!r}")
        if NEGATIVE in out or OLD_NEGATIVE in out:
            failures.append(f"{label}: printed a negative finding with no coverage: {out!r}")

    # 1-3. THE ISSUE'S CASE, and its siblings: every syscall read fails. Pre-fix: exit 0 and
    #      "this process is not blocked on a lock".
    for err in (errno.EACCES, errno.EPERM, errno.EIO):
        with tempfile.TemporaryDirectory() as tmp:
            make_tree(tmp, {101: ("main", "202 0x1000 0x80 0x2")})
            out, rc = run(mod, tmp, fail_open={101: err})
            expect_void(f"case 1 ({errno.errorcode[err]})", out, rc)
            if f"{errno.errorcode[err]} x2" not in out:
                failures.append(f"case 1 ({errno.errorcode[err]}): failure reason/count not reported: {out!r}")

    # 4. Task enumeration denied: no sweep saw a thread. Pre-fix: "(target exited early)" + negative.
    with tempfile.TemporaryDirectory() as tmp:
        make_tree(tmp, {101: ("main", "running")})
        out, rc = run(mod, tmp, fail_listdir=errno.EACCES)
        expect_void("case 4", out, rc)
        if "task-list read failures: EACCES x2" not in out:
            failures.append(f"case 4: enumeration failure not named: {out!r}")
        if "(2 sweep(s) could not read the task list)" not in out:
            failures.append(f"case 4: the short sample count does not say why: {out!r}")
        if "target exited" in out:
            failures.append(f"case 4: a permission failure was reported as the target exiting: {out!r}")

    # 5. Empty task list (the directory exists, no threads): nothing observed.
    with tempfile.TemporaryDirectory() as tmp:
        make_tree(tmp, {})
        out, rc = run(mod, tmp)
        expect_void("case 5", out, rc)

    # 6. Every thread vanished between listing and reading: a race, but still zero coverage.
    with tempfile.TemporaryDirectory() as tmp:
        make_tree(tmp, {101: ("main", "running")})
        out, rc = run(mod, tmp, fail_open={101: errno.ENOENT})
        expect_void("case 6", out, rc)
        if "2 vanished mid-sweep" not in out:
            failures.append(f"case 6: vanished threads not counted as such: {out!r}")
        if "0 unreadable" not in out:
            failures.append(f"case 6: a vanished thread was counted as unreadable: {out!r}")

    # 6b. MALFORMED: the file reads, but holds nothing parseable (an empty line, or a futex number
    #     with no address). Not an observation, and not a vanished thread.
    with tempfile.TemporaryDirectory() as tmp:
        make_tree(tmp, {101: ("main", ""), 102: ("worker", "202")})
        out, rc = run(mod, tmp)
        expect_void("case 6b", out, rc)
        if "malformed x4" not in out:
            failures.append(f"case 6b: malformed syscall files not counted as such: {out!r}")

    # 7. CONTROL: fully readable, nobody in a futex wait. A scoped negative, exit 0.
    with tempfile.TemporaryDirectory() as tmp:
        make_tree(tmp, {101: ("main", "running"), 102: ("io", "0 0x5 0x0 0x10")})
        out, rc = run(mod, tmp)
        if rc != 0 or f"{NEGATIVE} in 4 readable thread-sample(s)" not in out:
            failures.append(f"case 7: readable non-futex run must be a scoped negative, rc 0: {out!r}")
        if "PARTIAL" in out or "UNAVAILABLE" in out:
            failures.append(f"case 7: full coverage was reported as partial/unavailable: {out!r}")

    # 8. PARTIAL: one readable non-futex thread, one denied. The negative must carry the gap.
    with tempfile.TemporaryDirectory() as tmp:
        make_tree(tmp, {101: ("main", "running"), 102: ("worker", "202 0x1000 0x80 0x2")})
        out, rc = run(mod, tmp, fail_open={102: errno.EACCES})
        if rc != 0 or f"{NEGATIVE} in 2 readable" not in out:
            failures.append(f"case 8: expected a scoped negative over the readable half: {out!r}")
        if "PARTIAL: 2 thread-sample(s) could not be read" not in out:
            failures.append(f"case 8: missing coverage not stated: {out!r}")

    # 9. POSITIVE, built by hand: two waiters on 0x1000 and a live owner recorded at +8.
    with tempfile.TemporaryDirectory() as tmp:
        make_tree(tmp, {101: ("main", "202 0x1000 0x80 0x2"),
                        102: ("worker", "202 0x1000 0x80 0x2"),
                        103: ("holder", "running")})
        out, rc = run(mod, tmp, owner=103)
        if rc != 0 or "0x1000" not in out or "HELD BY  holder/103" not in out:
            failures.append(f"case 9: a readable contended lock must name its holder: {out!r}")
        if NEGATIVE in out or "UNAVAILABLE" in out:
            failures.append(f"case 9: a positive run printed a negative: {out!r}")
        if "6 thread-sample(s) read, 0 unreadable" not in out:
            failures.append(f"case 9: coverage line wrong: {out!r}")

    if failures:
        print("FAILURES:")
        for f in failures:
            print(" -", f)
        return 1
    print("lock_holder self-test: all cases pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
