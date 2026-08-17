#!/usr/bin/env python3
"""End-to-end regression for `hle_calls --values`: it must capture returns with NO debug info (#2075).

Why this one is end-to-end when its sibling `test_hle_calls.py` is pure
------------------------------------------------------------------------
#2075 was not a logic bug. Every value capture failed on current master because
`gdb.FinishBreakpoint.return_value` is derived from the function's **DWARF return type**, and
prosper's default build type is `Release` (`-O3 -DNDEBUG`, no `-g`) -- so gdb had no type, the
attribute was `None`, and `int(None)` raised once per call. `finish-failures` then equalled the call
count exactly and every row read `(captured 0/N)`.

Nothing short of a real gdb driving a real process over real symbols can reproduce that: the
defect lives in the interaction between gdb, the ELF's debug sections, and this tool's decode. A
recorded-output test would have been green throughout the four days the feature was void.

So this test compiles `test_values_fixture.cpp` **twice** -- once with `-g0` (the exact #2075
condition) and once with `-g` -- and runs the real driver over each. The two arms are what make the
result more than self-agreement: the debug arm's values come from gdb's own ABI-aware decode, the
stripped arm's from this tool's `%rax` read, and the test asserts they produce the SAME histogram.

The price is a dependency on gdb, a C++ compiler, `nm` and `c++filt`. A missing one exits **77**,
which CMake registers as `SKIP_RETURN_CODE`, so ctest reports the run as `(Skipped)` by name rather
than as a pass. That distinction matters here more than usual: this repository already carries a
test (`stack_profile_logic`) written to avoid gdb precisely because a silently skipped test reads
exactly like a passing one. This one cannot avoid gdb, so it says so out loud instead.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER = os.path.join(HERE, "hle_calls.py")
FIXTURE = os.path.join(HERE, "test_values_fixture.cpp")
SKIP = 77

CONTROL = "s_user_getevent"
LOGIN = 0x0
NO_EVENT = 0x80960007
WIDE = "s_fixture_wide"
WIDE_VALUE = 0xDEADBEEF12345678

fails = 0


def check(cond, label):
    global fails
    if cond:
        print("  [ok]   %s" % label)
    else:
        print("  [FAIL] %s" % label)
        fails += 1


def skip(reason):
    print("== test_hle_calls_values: SKIPPED ==")
    print("   %s" % reason)
    print("   This test needs a real gdb (with Python), a C++ compiler, nm and c++filt on Linux.")
    print("   It is the only guard on the #2075 class -- a value capture that silently records")
    print("   nothing -- so a skip here leaves that class UNCHECKED on this machine.")
    sys.exit(SKIP)


def find_compiler():
    for name in (os.environ.get("CXX"), "g++", "c++", "clang++"):
        if name and shutil.which(name):
            return shutil.which(name)
    return None


def gdb_can_launch(gdb):
    """Can this gdb actually start an inferior here? Containers and hardened kernels say no.

    Asked explicitly rather than inferred from a failed run, so an environment that forbids ptrace
    is reported as a skip instead of as a regression in the tool.
    """
    try:
        probe = subprocess.run([gdb, "-batch", "-ex", "python print('py-ok')", "-ex", "run",
                                "--args", "/bin/true"],
                               capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        return "gdb could not be executed: %s" % exc
    if "py-ok" not in probe.stdout:
        return "this gdb has no Python support (the tool is a gdb Python script)"
    blocked = ("ptrace", "Operation not permitted", "not permitted")
    if probe.returncode != 0 and any(b in (probe.stderr + probe.stdout) for b in blocked):
        return "this gdb cannot start an inferior here: %s" % probe.stderr.strip()[:200]
    return None


def compile_fixture(cxx, out, debug):
    cmd = [cxx, "-O0", "-g" if debug else "-g0", "-o", out, FIXTURE]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        print("  [FAIL] could not compile the fixture (%s):\n%s" % (" ".join(cmd), done.stderr))
        return False
    return True


def run_arm(binary, ticks=12):
    """Run the real driver over one fixture build; return (header_fields, rows, raw)."""
    cmd = [sys.executable, DRIVER, "--ticks", str(ticks), "--values", "--order", "8",
           "--filter", "^s_", "--timeout", "180", "--launch", binary]
    done = subprocess.run(cmd, capture_output=True, text=True)
    raw = done.stdout + done.stderr
    fields, rows = {}, {}
    for line in done.stdout.splitlines():
        if line.startswith("clock="):
            for token in line.split():
                if "=" in token:
                    key, _, value = token.partition("=")
                    fields[key] = value
        m = re.match(r"\s*(\d+)\s+(\S+)(.*)$", line)
        if m and not line.startswith("clock="):
            name, tail = m.group(2), m.group(3)
            seen = {}
            for value, count in re.findall(r"(0x[0-9a-f]+) x(\d+)", tail):
                seen[int(value, 16)] = int(count)
            rows[name] = {"calls": int(m.group(1)), "values": seen, "tail": tail}
    return fields, rows, raw


def check_arm(label, fields, rows, raw, expect_source):
    if not fields:
        check(False, "%s: the driver produced a result block\n%s" % (label, raw[-2000:]))
        return
    check(fields.get("window") == "complete" and fields.get("exited") == "0",
          "%s: the window closed on the clock, not on an exit (window=%s exited=%s)"
          % (label, fields.get("window"), fields.get("exited")))
    # THE regression. On master this reads finish-failures=<calls>, and every row (captured 0/N).
    check(fields.get("finish-failures") == "0",
          "%s: no capture failed (finish-failures=%s of calls=%s)"
          % (label, fields.get("finish-failures"), fields.get("calls")))
    check("(captured " not in raw,
          "%s: every call's return was recorded (no `(captured N/M)` row)" % label)
    check(fields.get("positive-control") == "ok",
          "%s: the built-in positive control passed (positive-control=%s)"
          % (label, fields.get("positive-control")))
    check("value-source-MISMATCH" not in raw,
          "%s: the DWARF value and %%rax never disagreed" % label)

    # Which read answered. This is what separates the two arms, and it is the field that would have
    # named #2075 on sight: `dwarf:0,rax:0` with a non-zero call count IS the bug.
    source = fields.get("value-source", "")
    m = re.fullmatch(r"dwarf:(\d+),rax:(\d+)", source)
    check(bool(m), "%s: value-source is reported (%s)" % (label, source))
    if m:
        dwarf, rax = int(m.group(1)), int(m.group(2))
        calls = int(fields.get("calls", "0"))
        check(dwarf + rax == calls and calls > 0,
              "%s: every one of the %d calls yielded a value (dwarf=%d rax=%d)"
              % (label, calls, dwarf, rax))
        if expect_source == "rax":
            check(dwarf == 0 and rax > 0,
                  "%s: with no debug info every value came from %%rax (dwarf=%d rax=%d)"
                  % (label, dwarf, rax))
        else:
            check(rax == 0 and dwarf > 0,
                  "%s: with debug info gdb's own decode answered (dwarf=%d rax=%d)"
                  % (label, dwarf, rax))

    control = rows.get(CONTROL, {"values": {}, "calls": 0})
    check(control["values"].get(LOGIN, 0) == 1,
          "%s: exactly one LOGIN captured for %s (%s)" % (label, CONTROL, control["values"]))
    check(control["values"].get(NO_EVENT, 0) >= 1,
          "%s: the later %s calls answered NO_EVENT (%s)" % (label, CONTROL, control["values"]))
    check(sum(control["values"].values()) == control["calls"],
          "%s: %s's captured values account for all %d of its calls"
          % (label, CONTROL, control["calls"]))

    # A 64-bit value no accident produces: it pins the register read and the mask together. A
    # capture that truncated to 32 bits, or read the wrong register, cannot print this.
    wide = rows.get(WIDE, {"values": {}})
    check(list(wide["values"]) == [WIDE_VALUE],
          "%s: %s's full 64-bit return survived (%s)"
          % (label, WIDE, [hex(v) for v in wide["values"]]))


def main():
    print("== test_hle_calls_values ==")
    if sys.platform != "linux":
        skip("this platform is %s; hle_calls is Linux-only" % sys.platform)
    for tool in ("nm", "c++filt"):
        if not shutil.which(tool):
            skip("%s is not on PATH (the driver enumerates handlers with it)" % tool)
    gdb = os.environ.get("HLE_CALLS_TEST_GDB") or shutil.which("gdb")
    if not gdb:
        skip("gdb is not on PATH")
    reason = gdb_can_launch(gdb)
    if reason:
        skip(reason)
    cxx = find_compiler()
    if not cxx:
        skip("no C++ compiler found (set CXX)")

    work = tempfile.mkdtemp(prefix="hle_calls_values_")
    try:
        # Arm A is the whole point: `-g0` is the shape of prosper's own default Release build, and
        # the configuration on which every value capture failed.
        arms = (("no-debug-info", os.path.join(work, "fixture_nodbg"), False, "rax"),
                ("debug-info", os.path.join(work, "fixture_dbg"), True, "dwarf"))
        for label, path, debug, expect in arms:
            if not compile_fixture(cxx, path, debug):
                global fails
                fails += 1
                continue
            fields, rows, raw = run_arm(path)
            check_arm(label, fields, rows, raw, expect)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print("== FAILURES: %d ==" % fails if fails else "== all checks passed ==")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
