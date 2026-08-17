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

Three further arms guard the load-bias handling (#2605), and they are not interchangeable:

  * `pie` — the same fixture built `-pie` must yield the SAME histogram as its `-no-pie` twin.
  * `mismatched-binary` — a run whose `--binary` names a different image must be REFUSED by name.
    This is the mutation arm for the landing check: without it the run also fails, but as an
    unexplained `VOID` three steps from its cause.
  * `attach-pie` — the only arm that can tell a *derived* bias from a hardcoded one. Under `--launch`
    gdb disables randomization, so `0x555555554000` satisfies every launch-mode assertion above
    (measured: a build with the bias replaced by that constant passes the `pie` arm in full and
    fails only here). This arm reads the real base out of `/proc/<pid>/maps` itself and requires the
    tool to have reported that number. It skips loudly where ptrace forbids attaching to a sibling.

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
import time

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


def attach_is_permitted(gdb):
    """Can this gdb ptrace-attach to a SIBLING process here? `None` if it can, a reason if not.

    Asked rather than inferred from a failed run. `kernel.yama.ptrace_scope=1` -- Ubuntu's default,
    and therefore CI's -- permits attaching only to descendants, and gdb is a child of the driver
    while the fixture is a child of this test, so they are siblings and the attach is refused. That
    is an environment fact, not a regression in the tool, and it must not read as one.
    """
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        probe = subprocess.run([gdb, "-p", str(child.pid), "-batch", "-ex", "quit"],
                               capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        return "gdb could not be run for an attach probe: %s" % exc
    finally:
        child.kill()
        child.wait()
    text = (probe.stderr + probe.stdout).strip()
    if probe.returncode != 0 or "ptrace" in text or "not permitted" in text:
        return ("this gdb cannot attach to a sibling process here (kernel.yama.ptrace_scope?): %s"
                % (text.splitlines()[-1][:160] if text else "gdb said nothing, exit %d"
                   % probe.returncode))
    return None


def executable_base(pid, path):
    """The load address of `path` in `pid`, read straight out of `/proc/<pid>/maps` -- or None.

    Deliberately a SECOND, independent derivation of the number the tool reports: the tool reads
    `AT_ENTRY` out of auxv and subtracts the ELF's link-time entry, while this matches the mapping by
    pathname and takes its lowest start. Both must land on the same address. (The tool does not use
    this method itself because prosper maps the guest image low, so the first line of a live
    `boot_trace`'s maps is a guest mapping, not the executable.)
    """
    lowest = None
    try:
        with open("/proc/%d/maps" % pid) as handle:
            for line in handle:
                parts = line.split()
                if len(parts) >= 6 and os.path.realpath(parts[-1]) == os.path.realpath(path):
                    start = int(parts[0].split("-")[0], 16)
                    lowest = start if lowest is None else min(lowest, start)
    except OSError:
        return None
    return lowest


def compile_fixture(cxx, out, debug, pie=False, optional=False):
    """Build the fixture. The linkage is stated explicitly because BOTH shapes have to be covered.

    hle_calls arms breakpoints at the raw link-time addresses `nm` reports. On an `ET_EXEC` target
    those are the runtime addresses; on a PIE they are offsets from a base the kernel picks per run,
    and the tool has to read that base out of the live process (#2605). Ubuntu's gcc defaults to
    `-pie` while Fedora's does not, so a fixture built without an explicit flag is a DIFFERENT
    program on the two -- which is how the CI arm of this very test once failed
    (`Cannot access memory at address 0x1149`, no result block) while passing locally. Pinning the
    flag makes each arm mean one thing, and the `pie=True` arms below cover the shape this box does
    not produce by default.
    """
    cmd = [cxx, "-O0", "-g" if debug else "-g0",
           "-fPIE" if pie else "-fno-pie", "-pie" if pie else "-no-pie",
           "-o", out, FIXTURE]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        # A toolchain that cannot produce the requested linkage says nothing about the tool, so the
        # PIE arm reports that as a skip. The two main arms still fail: without them there is no
        # test at all.
        prefix = "  [skip] " if optional else "  [FAIL] "
        print("%scould not compile the fixture (%s):\n%s" % (prefix, " ".join(cmd), done.stderr))
        return False
    return True


def run_arm(binary, ticks=12, out_bytes=0, pid=None, symbols=None):
    """Run the real driver over one fixture build; return (header_fields, rows, outs, raw).

    `pid` attaches to an already-running process instead of launching one; `symbols` overrides
    `--binary`, which is how the mismatched-binary arm below builds a wrong address by hand.
    """
    cmd = [sys.executable, DRIVER, "--ticks", str(ticks), "--values", "--order", "8",
           "--filter", "^s_", "--timeout", "180"]
    if out_bytes:
        cmd += ["--out-bytes", str(out_bytes)]
    if symbols:
        cmd += ["--binary", symbols]
    cmd += ["--pid", str(pid)] if pid else ["--launch", binary]
    done = subprocess.run(cmd, capture_output=True, text=True)
    raw = done.stdout + done.stderr
    fields, rows, outs = {}, {}, {}
    current, current_out = None, None
    for line in done.stdout.splitlines():
        if line.startswith("clock="):
            for token in line.split():
                if "=" in token:
                    key, _, value = token.partition("=")
                    fields[key] = value
            continue
        out = re.match(r"\s+out (a\d)\[(\d+)\] (.*)$", line)
        if out and current:
            current_out = (current, out.group(1))
            stats = {"width": int(out.group(2)), "shapes": {}}
            for key, value in re.findall(r"([a-z-]+)=(\d+)", out.group(3)):
                stats[key] = int(value)
            outs[current_out] = stats
            continue
        shape = re.match(r"\s+(@\d+ .*?)\s+x(\d+)$", line)
        if shape and current_out:
            outs[current_out]["shapes"][shape.group(1)] = int(shape.group(2))
            continue
        m = re.match(r"\s*(\d+)\s+(\S+)(.*)$", line)
        if m:
            name, tail = m.group(2), m.group(3)
            seen = {}
            for value, count in re.findall(r"(0x[0-9a-f]+) x(\d+)", tail):
                seen[int(value, 16)] = int(count)
            rows[name] = {"calls": int(m.group(1)), "values": seen, "tail": tail}
            current, current_out = name, None
    return fields, rows, outs, raw


def check_arm(label, fields, rows, raw, expect_source, expect_elf="ET_EXEC", expect_mode="launch"):
    if not fields:
        check(False, "%s: the driver produced a result block\n%s" % (label, raw[-2000:]))
        return
    check(fields.get("window") == "complete" and fields.get("exited") == "0",
          "%s: the window closed on the clock, not on an exit (window=%s exited=%s)"
          % (label, fields.get("window"), fields.get("exited")))
    # WHERE the breakpoints went. On an ET_EXEC target the bias must be exactly zero -- a non-zero
    # one there would mean the tool relocated addresses that were already runtime addresses -- and on
    # a PIE it must not be, because every link-time address is an offset (#2605). The symbol check
    # is the observation behind both: gdb was asked what lives at each sampled armed address.
    check(fields.get("elf") == expect_elf,
          "%s: the header states the target's linkage (elf=%s, expected %s)"
          % (label, fields.get("elf"), expect_elf))
    bias = fields.get("load-bias", "")
    if expect_elf == "ET_DYN":
        check(bias.startswith("0x") and int(bias, 16) != 0,
              "%s: a PIE run reports the load bias it resolved (load-bias=%s)" % (label, bias))
    else:
        check(bias == "0x0", "%s: an ET_EXEC run relocates nothing (load-bias=%s)" % (label, bias))
    checked = fields.get("symbol-check", "")
    match = re.fullmatch(r"(\d+)/(\d+)", checked)
    check(bool(match) and match.group(1) == match.group(2) and int(match.group(2)) > 0,
          "%s: every sampled armed address landed on a function start (symbol-check=%s)"
          % (label, checked))
    check(fields.get("mode") == expect_mode,
          "%s: the run used the mode asked for (mode=%s)" % (label, fields.get("mode")))
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


def check_out_arm(fields, rows, outs, raw):
    """The --out-bytes arm (#2045): what the handler WROTE, which the return register cannot say.

    Every expected value here comes from `test_values_fixture.cpp`, which in turn mirrors the real
    handlers named in the issue — `s_syss_getstatus` writes 12 bytes with `st[6] = 1`, and
    `s_user_getevent` writes `{0, 1}` on the one call that delivers the LOGIN. Both are known from
    source before the run, which is the only kind of check worth having on an instrument.
    """
    label = "out-bytes"
    check(fields.get("out-bytes") == "12", "%s: the header states the sampled width (out-bytes=%s)"
          % (label, fields.get("out-bytes")))

    wrote = outs.get(("s_fixture_writes12", "a0"), {})
    check(wrote.get("read", 0) > 0 and wrote.get("read") == wrote.get("calls"),
          "%s: the writer's out-pointer was sampled on every call (%s)" % (label, wrote))
    check(wrote.get("changed", 0) == wrote.get("read", -1),
          "%s: every sampled call showed a write (changed=%s read=%s)"
          % (label, wrote.get("changed"), wrote.get("read")))
    check(list(wrote.get("shapes", {})) == ["@6 00->01"],
          "%s: the diff is exactly the byte the source writes, st[6]=1 (%s)"
          % (label, list(wrote.get("shapes", {}))))

    # THE distinction the feature exists for. Same width, same call count, both readable — one
    # handler wrote and the other did not, and the counters say which without ambiguity.
    quiet = outs.get(("s_fixture_nowrite", "a0"), {})
    check(quiet.get("read", 0) > 0 and quiet.get("changed", -1) == 0,
          "%s: a handler that writes nothing reads as changed=0, not as a failure (%s)"
          % (label, quiet))
    check(quiet.get("same-zero", 0) == quiet.get("read", -1),
          "%s: its unwritten buffer is same-zero, the unambiguous 'nothing was written' (%s)"
          % (label, quiet))

    # ... and "nothing was READ" must never look like either of those.
    bad = outs.get(("s_fixture_badptr", "a0"), {})
    check(bad.get("unreadable", 0) > 0 and bad.get("read", -1) == 0 and bad.get("changed", -1) == 0,
          "%s: an unmapped out-pointer is unreadable=N read=0 — nothing was READ (%s)" % (label, bad))
    small = outs.get(("s_fixture_handlearg", "a0"), {})
    check(small.get("small", 0) > 0 and small.get("read", -1) == 0,
          "%s: a small handle argument is counted as small, not as unreadable (%s)" % (label, small))
    check(("s_fixture_nullarg", "a0") not in outs
          and "were NULL on every call" in raw,
          "%s: an always-null argument gets the explicit null line, not a silent absence" % label)

    # The out-struct positive control, from the same source contract as the value control: the LOGIN
    # call is the only one that writes, so exactly one sample may differ.
    control = outs.get((CONTROL, "a0"), {})
    check(control.get("changed", -1) == 1,
          "%s: %s wrote its event struct on exactly one call (%s)" % (label, CONTROL, control))
    check(list(control.get("shapes", {})) == ["@4 00->01"],
          "%s: and the write is userId=1 at offset 4 (%s)" % (label, list(control.get("shapes", {}))))
    check(control.get("same-zero", 0) == control.get("read", 0) - 1,
          "%s: every other %s call left the struct untouched (%s)" % (label, CONTROL, control))

    # The ambiguous state, asserted rather than merely documented: a buffer the guest does not clear,
    # rewritten with the same bytes every call. Exactly one call can show a diff; the rest ARE writes
    # and are indistinguishable from silence, which is what `same-nonzero` says and `same-zero` must
    # not. Collapsing the two would make this instrument claim more than it can see.
    rewrite = outs.get(("s_fixture_rewrite", "a0"), {})
    check(rewrite.get("changed", -1) == 1 and rewrite.get("same-nonzero", 0) == rewrite.get("read", 0) - 1,
          "%s: a repeated identical write reads as same-nonzero, not as a write (%s)"
          % (label, rewrite))
    check(rewrite.get("same-zero", -1) == 0,
          "%s: and never as same-zero, which would claim nothing was written (%s)" % (label, rewrite))
    check(rows.get(CONTROL, {}).get("values", {}).get(LOGIN, 0) == 1,
          "%s: the value capture still works alongside the out-byte sampling" % label)
    check(fields.get("finish-failures") == "0" and "(captured " not in raw,
          "%s: sampling out-bytes cost no return-value capture (finish-failures=%s)"
          % (label, fields.get("finish-failures")))


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
        built, measured = {}, {}
        for label, path, debug, expect in arms:
            if not compile_fixture(cxx, path, debug):
                global fails
                fails += 1
                continue
            built[label] = path
            fields, rows, outs, raw = run_arm(path)
            measured[label] = (fields, rows)
            check_arm(label, fields, rows, raw, expect)
        # Third arm: the same no-debug-info binary, now with --out-bytes. Run apart from the two
        # above so the default path (out-bytes off) stays covered rather than only the new one.
        if "no-debug-info" in built:
            fields, rows, outs, raw = run_arm(built["no-debug-info"], out_bytes=12)
            check_out_arm(fields, rows, outs, raw)

        # Fourth arm: a PIE build must produce the SAME measurement as its -no-pie twin (#2605).
        # Every address this tool arms is a link-time address, so on a PIE they are offsets from a
        # base the kernel picks per run; unrelocated, each breakpoint lands in unmapped memory and
        # gdb prints "Cannot insert breakpoint N" per handler with no result block at the end --
        # which is exactly how this test failed on a GitHub ubuntu-24.04 runner while passing on
        # Fedora, the two toolchains disagreeing on the -pie default. #2593 replaced that with a
        # refusal; this arm is the refusal's replacement, and the histogram equality below is what
        # makes it a measurement rather than merely a non-crash.
        pie_path = os.path.join(work, "fixture_pie")
        have_pie = compile_fixture(cxx, pie_path, debug=False, pie=True, optional=True)
        if have_pie:
            fields, rows, _, raw = run_arm(pie_path)
            check_arm("pie", fields, rows, raw, "rax", expect_elf="ET_DYN")
            if "no-debug-info" in measured:
                twin_fields, twin_rows = measured["no-debug-info"]
                # The fixture calls each handler exactly once per loop and the clock closes the
                # window at the end of the 12th, so the histogram is a fixed number -- and the two
                # linkages must agree on all of it, name for name and count for count. A bias that
                # is wrong by any amount cannot produce this: it produces no result block at all.
                check(rows and {n: r["calls"] for n, r in rows.items()}
                      == {n: r["calls"] for n, r in twin_rows.items()},
                      "pie: the PIE histogram equals its -no-pie twin's, handler for handler (%s vs %s)"
                      % (sorted((n, r["calls"]) for n, r in rows.items()),
                         sorted((n, r["calls"]) for n, r in twin_rows.items())))
                check(twin_fields.get("load-bias") == "0x0" and fields.get("load-bias", "0x0") != "0x0",
                      "pie: and it got there through a non-zero bias its twin did not need "
                      "(pie=%s no-pie=%s)" % (fields.get("load-bias"), twin_fields.get("load-bias")))

        # Fifth arm: the check that makes a WRONG address refuse instead of measuring nothing.
        # Constructed by hand rather than by disabling the fix: launch the PIE fixture while naming
        # its -no-pie twin as the symbol source. The addresses are then real, well-formed, and about
        # a different image -- the shape a stale --binary or a rebuilt target produces -- and every
        # one of them lands in unmapped memory. Without this arm, "the PIE arm passed" would say
        # nothing about whether a wrong bias is detectable at all.
        if have_pie and "no-debug-info" in built:
            done = subprocess.run([sys.executable, DRIVER, "--ticks", "4", "--values",
                                   "--filter", "^s_", "--timeout", "60",
                                   "--binary", built["no-debug-info"], "--launch", pie_path],
                                  capture_output=True, text=True)
            message = done.stdout + done.stderr
            # Note which of these three is load-bearing. A run with the landing check REMOVED also
            # exits non-zero and also emits no result block -- gdb fails to insert the breakpoints
            # and the driver reports VOID -- so those two pass either way and prove nothing on their
            # own (measured: with `_symbol_landing` neutered, only the second check below fails).
            # The two that discriminate are the abort line, which only the check can print, and the
            # absence of gdb's insertion error, which only reaching the arming step can produce.
            check(done.returncode != 0,
                  "mismatched-binary: the run is refused (exit %d)" % done.returncode)
            check("HLE_CALLS_ABORT:" in message and "do not land on a function start" in message,
                  "mismatched-binary: and it says which check failed (%s)"
                  % next((l for l in message.splitlines() if "ABORT" in l), "no abort line")[:200])
            check("Cannot insert breakpoint" not in message,
                  "mismatched-binary: it refused BEFORE arming — gdb never tried to insert a "
                  "breakpoint into unmapped memory")
            check("clock=" not in done.stdout,
                  "mismatched-binary: no result block was produced — nothing was measured")

        # Sixth arm: --pid against a PIE with the kernel's own randomization in play. This is the
        # arm that separates a DERIVED bias from a hardcoded one: under gdb the base is
        # 0x555555554000 every time (gdb disables randomization), so a constant would satisfy every
        # launch-mode assertion above. Here the test reads the real base out of /proc itself and
        # requires the tool to have reported that number.
        if have_pie:
            reason = attach_is_permitted(gdb)
            if reason:
                print("  [skip] attach-pie: %s" % reason)
                print("         Nothing else in this test can tell a derived load bias from a "
                      "hardcoded one, so that distinction is UNCHECKED on this machine.")
            else:
                child = subprocess.Popen([pie_path], stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL)
                try:
                    time.sleep(1.0)
                    base = executable_base(child.pid, pie_path)
                    fields, rows, _, raw = run_arm(pie_path, pid=child.pid)
                    check(base is not None and fields.get("load-bias") == "%#x" % base,
                          "attach-pie: the reported bias is the base this process actually got "
                          "(reported=%s /proc says=%s)"
                          % (fields.get("load-bias"), None if base is None else "%#x" % base))
                    check(sum(r["calls"] for r in rows.values()) > 0
                          and fields.get("window") == "complete",
                          "attach-pie: and the window measured real calls (window=%s calls=%s)"
                          % (fields.get("window"), fields.get("calls")))
                    check(fields.get("elf") == "ET_DYN" and fields.get("mode") == "attach",
                          "attach-pie: on a PIE target, in attach mode (elf=%s mode=%s)"
                          % (fields.get("elf"), fields.get("mode")))
                finally:
                    child.kill()
                    child.wait()
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print("== FAILURES: %d ==" % fails if fails else "== all checks passed ==")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
