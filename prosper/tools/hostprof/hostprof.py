#!/usr/bin/env python3
# hostprof — a poor-man's SAMPLING PROFILER for a running prosper process (or any native process),
# built on repeated gdb backtraces. It answers "where is the HOST (C++) side spending CPU?" — the
# render/submit thread, a compute worker, the readback copy — by sampling the live process N times and
# aggregating the innermost (leaf) function across samples into a ranked histogram.
#
# Why this exists (and why not perf):
#   * On locked-down / immutable hosts `perf` is unusable: `kernel.perf_event_paranoid>=2` denies the
#     perf_event syscalls without root (the exact situation on the Bazzite dev box). gdb ptrace-attach,
#     by contrast, works whenever `kernel.yama.ptrace_scope=0` (or from a parent/root), so this keeps
#     working where perf does not.
#   * The existing tools/dbg/*.sh sampling scripts dump RAW all-thread backtraces to a fixed path for a
#     specific title and leave the aggregation to hand-written awk. This generalizes that: any pid/name,
#     leaf/folded/thread aggregation, template+argument-stripped symbol names, and idle-frame filtering.
#   * tools/guest_bt answers the complementary question for GUEST threads (managed C# unwinding through
#     the HLE stub boundary). hostprof is purely the NATIVE (prosper C++ / libc) side.
#
# It is a *statistical* profiler: each sample is one stack per thread, so the histogram approximates the
# fraction of wall-time each leaf sits on top of the stack. More samples -> tighter estimate. It is not
# a substitute for instrumented timing (e.g. PROSPER_RENDER_TIMING) when exact per-phase ms are needed;
# it is the fast "which function is hot right now" first look.
#
# Usage:
#   hostprof.py <pid|process-name> [-n SAMPLES] [-i INTERVAL] [-d DEPTH]
#               [--thread REGEX] [--mode leaf|folded|thread] [--keep-idle] [--raw-names]
#
# Examples:
#   # 30 leaf samples of the busiest thread of a running boot_trace, ranked:
#   hostprof.py boot_trace -n 30 --mode thread
#   # Focus the render/submit thread (matched anywhere in its stack) and rank its leaves:
#   hostprof.py boot_trace -n 40 --thread 'execute_ordered|agc_driver_submit'
#   # Collapsed stacks for a flamegraph (pipe to flamegraph.pl):
#   hostprof.py 12345 -n 200 --thread 'execute_ordered' --mode folded > out.folded
#
# Requires: gdb on PATH. Read-only: it only attaches, samples, and detaches; it never writes to or
# resumes-with-changes the target (each `gdb -batch` attaches, prints, and detaches, leaving the process
# running). Sampling briefly stops the process (a few ms per sample) — do not use on latency-critical runs.

import argparse
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict

# Leaf functions that mean "this thread is parked/idle, not burning CPU". Filtered by default so the
# histogram reflects real work; pass --keep-idle to see the wait breakdown instead. Matched against the
# SIMPLIFIED leaf name (a bare identifier like `pthread_cond_wait`), NOT the raw gdb frame — so `read$`
# actually fires (a raw frame ends in " () at file:line") and the risky short tokens can be word-bounded.
# `select`/`accept` MUST be word-bounded: unanchored they would silently drop real work like
# select_render_target / device_selection / accept_connection (which then looks "not hot" to the user).
IDLE_PATTERNS = re.compile(
    r"(pthread_cond_(?:wait|timedwait)|__futex|futex_(?:wait|abstimed)|nanosleep|clock_nanosleep|"
    r"epoll_wait|__poll|\bpoll\b|ppoll|\bselect\b|cnd_wait|cnd_timedwait|sched_yield|"
    r"__lll_lock_wait|pthread_join|sem_wait|sigwait|do_futex_wait|syscall_cancel|"
    # A bare `syscall`/`__syscall` leaf means the thread is BLOCKED kernel-side (a parked worker in the
    # pool waiting on a futex/io_uring), i.e. not burning user CPU — treat it as idle. Real syscall-heavy
    # work still shows its libc wrapper (e.g. __memmove for a readback, ioctl issuers) as the leaf.
    r"\bsyscall\b|\b__syscall\b|"
    r"read$|recvmsg|recvfrom|\baccept4?\b|waitpid|__wait)"
)

FRAME0_RE = re.compile(r"^#0\s+(?:0x[0-9a-fA-F]+\s+in\s+)?(.*)$")
FRAMEN_RE = re.compile(r"^#(\d+)\s+(?:0x[0-9a-fA-F]+\s+in\s+)?(.*)$")
THREAD_RE = re.compile(r"^Thread\s+\d+")


def resolve_pid(spec):
    if spec.isdigit():
        return int(spec)
    try:
        out = subprocess.run(["pidof", spec], capture_output=True, text=True, timeout=5).stdout.split()
    except FileNotFoundError:
        sys.exit("hostprof: `pidof` not found; pass a numeric pid instead")
    if not out:
        sys.exit(f"hostprof: no running process named '{spec}' (pidof found nothing)")
    if len(out) > 1:
        sys.stderr.write(f"hostprof: {len(out)} processes named '{spec}' ({' '.join(out)}); "
                         f"sampling the first ({out[0]}). Pass a pid to disambiguate.\n")
    return int(out[0])


def busiest_tid(pid):
    """Return the tid of the thread with the most CPU time (utime+stime), or None."""
    import glob, os
    best, best_ticks = None, -1
    for stat in glob.glob(f"/proc/{pid}/task/*/stat"):
        try:
            with open(stat) as f:
                data = f.read()
        except OSError:
            continue
        # comm may contain spaces/parens; split on the LAST ')' so field indexing after it is stable.
        rparen = data.rfind(")")
        if rparen < 0:
            continue
        fields = data[rparen + 2:].split()
        # After comm, fields are: state ppid ... ; utime is the 14th of the whole record => index 11 here
        # (record field 14/15 = utime/stime; here fields[0]=state=field3, so utime=fields[11]).
        try:
            ticks = int(fields[11]) + int(fields[12])
        except (IndexError, ValueError):
            continue
        if ticks > best_ticks:
            best_ticks, best = ticks, os.path.basename(os.path.dirname(stat))
    return best


def simplify(name, raw):
    """Reduce a gdb frame description to a bare qualified function name for aggregation:
    'void ns::f<T>::g<U>(A, B) [clone .isra.0]'  ->  'ns::f::g'. --raw-names keeps gdb's full spelling."""
    if raw:
        return name.strip()
    name = re.sub(r"\s*\[clone[^\]]*\]\s*", "", name)      # drop gdb's [clone ...] annotation
    name = name.replace("(anonymous namespace)::", "")     # anon namespaces add no identity here
    prev = None
    while prev != name:                                    # strip all (possibly nested) template args
        prev = name
        name = re.sub(r"<[^<>]*>", "", name)
    head = name.split("(", 1)[0].strip()                   # everything before the argument list
    if not head:
        return "??"
    # The function name is the LAST whitespace-separated token; any return type is an earlier token.
    return head.split()[-1]


def sample_once(pid, depth):
    """One gdb pass -> (dict{tid_label: [frames]}, gdb_stderr). Frames are (level, rawname)."""
    try:
        p = subprocess.run(
            ["gdb", "-p", str(pid), "-batch", "-nx",
             "-ex", "set pagination off",
             "-ex", "set print frame-arguments none",
             "-ex", f"thread apply all bt {depth}"],
            capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        sys.exit("hostprof: gdb not found on PATH")
    threads = {}
    cur, frames = None, []
    for line in p.stdout.splitlines():
        if THREAD_RE.match(line):
            if cur is not None:
                threads[cur] = frames
            cur, frames = line, []
            continue
        m = FRAMEN_RE.match(line.strip())
        if m and cur is not None:
            frames.append((int(m.group(1)), m.group(2)))
    if cur is not None:
        threads[cur] = frames
    return threads, p.stderr


def self_test():
    """Verify the symbol simplifier against representative gdb frame spellings (no process needed)."""
    cases = [
        ("__memmove_avx512_unaligned_erms ()", "__memmove_avx512_unaligned_erms"),
        ("void std::vector<unsigned char, std::allocator<unsigned char> >::_M_assign_aux"
         "<unsigned char const*>(unsigned char const*, unsigned char const*, std::forward_iterator_tag)"
         " [clone .isra.0] ()", "std::vector::_M_assign_aux"),
        ("prosper::frontend::(anonymous namespace)::execute_item(prosper::frontend::"
         "(anonymous namespace)::VulkanComputeContext&, prosper::gpu::ComputeItem const&)"
         " [clone .constprop.0]", "prosper::frontend::execute_item"),
        ("prosper::gpu::half_to_float(unsigned short)", "prosper::gpu::half_to_float"),
        ("prosper::frontend::register_live_renderer(std::__cxx11::basic_string<char>)::"
         "{lambda(...)#1}::operator()(...) const", "prosper::frontend::register_live_renderer"),
        ("?? ()", "??"),
    ]
    bad = 0
    for raw, want in cases:
        got = simplify(raw, False)
        ok = got == want
        bad += not ok
        print(f"  [{'OK ' if ok else 'BAD'}] {got!r}" + ("" if ok else f"  (want {want!r})"))

    # Idle-filter cases: parked leaves must be filtered, but real functions that merely CONTAIN an idle
    # token as a substring (select_render_target, device_selection, accept_connection) must NOT be —
    # the word-bounded \bselect\b / \baccept4?\b guards. Checked against the simplified name, as the loop
    # does. (idle_want=True means "should be treated as idle/filtered".)
    idle_cases = [
        ("select ()", True), ("pthread_cond_wait ()", True), ("__futex_abstimed_wait64 ()", True),
        ("syscall ()", True), ("__libc_read () at ../sysdeps/unix/sysv/linux/read.c:26", True),
        ("accept4 ()", True),
        ("prosper::gpu::select_render_target(int)", False), ("prosper::vk::device_selection()", False),
        ("prosper::net::accept_connection(int)", False), ("prosper::gpu::half_to_float(unsigned short)", False),
        ("__memmove_avx512_unaligned_erms ()", False),
    ]
    for raw, idle_want in idle_cases:
        got = bool(IDLE_PATTERNS.search(simplify(raw, False)))
        ok = got == idle_want
        bad += not ok
        print(f"  [{'OK ' if ok else 'BAD'}] idle={got} {simplify(raw, False)!r}"
              + ("" if ok else f"  (want idle={idle_want})"))

    print("self-test: PASS" if not bad else f"self-test: {bad} FAILED")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description="Sampling profiler via repeated gdb backtraces.")
    ap.add_argument("target", nargs="?", help="pid or process name (e.g. boot_trace, screenshot)")
    ap.add_argument("--self-test", action="store_true", help="run the symbol-simplifier unit cases and exit")
    ap.add_argument("-n", "--samples", type=int, default=30, help="number of samples (default 30)")
    ap.add_argument("-i", "--interval", type=float, default=0.3, help="seconds between samples (default 0.3)")
    ap.add_argument("-d", "--depth", type=int, default=8, help="backtrace depth per thread (default 8)")
    ap.add_argument("--thread", metavar="REGEX",
                    help="only aggregate the thread whose stack matches REGEX (e.g. a hot function)")
    ap.add_argument("--mode", choices=["leaf", "thread", "folded"], default="leaf",
                    help="leaf: all-thread leaf histogram; thread: busiest thread only; "
                         "folded: collapsed stacks for a flamegraph (default leaf)")
    ap.add_argument("--keep-idle", action="store_true", help="do not filter parked/waiting frames")
    ap.add_argument("--raw-names", action="store_true", help="keep full templated gdb symbol names")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(self_test())
    if not args.target:
        ap.error("target (pid or process name) is required unless --self-test is given")
    args.interval = max(0.0, args.interval)   # a negative interval would crash time.sleep

    pid = resolve_pid(args.target)
    thread_re = re.compile(args.thread) if args.thread else None
    focus_tid = busiest_tid(pid) if args.mode == "thread" and not thread_re else None

    leaves = Counter()
    folded = Counter()
    captured = 0
    sys.stderr.write(f"hostprof: sampling pid {pid} x{args.samples} (depth {args.depth}) ...\n")
    for i in range(args.samples):
        threads, gdb_err = sample_once(pid, args.depth)
        if not threads:
            # First pass produced nothing: almost always gdb couldn't attach (ptrace_scope, wrong pid,
            # process exited). Surface gdb's own reason so the user doesn't misread the empty result as
            # "everything idle" — the whole point of this tool is to work where perf can't.
            if i == 0 and captured == 0:
                hint = next((ln for ln in (gdb_err or "").splitlines()
                             if ln.strip() and "warning:" not in ln.lower()), "")
                sys.stderr.write("hostprof: gdb returned no threads on the first pass"
                                 + (f" — {hint.strip()}" if hint else
                                    " (attach failed? check the pid and kernel.yama.ptrace_scope)")
                                 + "\n")
            time.sleep(args.interval)
            continue
        for label, frames in threads.items():
            if not frames:
                continue
            if thread_re and not any(thread_re.search(f[1]) for f in frames):
                continue
            if focus_tid and f"LWP {focus_tid}" not in label:
                # busiest-thread mode: gdb's thread header carries the LWP tid as "(LWP <tid>)".
                continue
            leaf_raw = frames[0][1]
            # Idle detection runs on the CLEAN function name (not the raw frame): a raw frame ends in
            # " () at file:line", which would defeat anchored patterns like read$, and the clean name is
            # what the word-bounded select/accept guards need to avoid dropping real select_*/accept_*.
            if not args.keep_idle and IDLE_PATTERNS.search(simplify(leaf_raw, False)):
                continue
            leaves[simplify(leaf_raw, args.raw_names)] += 1
            if args.mode == "folded":
                stack = ";".join(simplify(f[1], args.raw_names) for f in reversed(frames))
                folded[stack] += 1
        captured += 1
        if i + 1 < args.samples:
            time.sleep(args.interval)

    if args.mode == "folded":
        for stack, n in folded.most_common():
            print(f"{stack} {n}")
        return

    total = sum(leaves.values())
    if not total:
        sys.exit("hostprof: no non-idle leaves captured (all threads parked, or --thread matched none). "
                 "Try --keep-idle, a different --thread REGEX, or more -n samples.")
    scope = (f"thread ~/{args.thread}/" if thread_re
             else f"busiest thread (tid {focus_tid})" if focus_tid else "all threads")
    print(f"\nhostprof: {total} non-idle leaf samples over {captured} passes, {scope}\n")
    print(f"  {'%':>6}  {'count':>6}  function")
    for name, n in leaves.most_common(25):
        print(f"  {100.0 * n / total:6.1f}  {n:6d}  {name}")


if __name__ == "__main__":
    main()
