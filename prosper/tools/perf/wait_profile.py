#!/usr/bin/env python3
"""wait_profile -- where is every thread's time going, and what is it BLOCKED ON?

LINUX ONLY. It reads /proc/<pid>/task/*/{stat,wchan,syscall}, which do not exist on Windows
or macOS -- there `thread_ids()` simply returns empty. Stated here because the failure is
otherwise a silent one: a Windows reader gets an empty report rather than an explanation,
and an empty report from a wait profiler reads like a process that is not waiting.

The question this answers is the one prosper keeps needing and could not ask: *nothing is
working, so what is everything waiting for?* Blue Prince runs at ~3.2 fps with the GPU 96% idle
(#2215), which is not a "make it faster" problem at all -- it is a wait, and no timer in the tree
names what is being waited on.

WHY /proc SAMPLING RATHER THAN ftrace/perfetto/GPUVis
    Those are better instruments and they are unavailable here: this container exposes no
    tracefs and `perf list` reports zero `sched:`/`syscalls:` tracepoints. /proc needs no
    privileges, no kernel config and no install, so this works on any box that can run the
    emulator -- which is the property that makes it reusable rather than a one-machine trick.

THE ONE DESIGN RULE, and it is the reason to trust the output
    **Every sample lands in exactly one bucket, by construction.** A thread is either running or
    blocked on exactly one thing, so each thread's percentages sum to 100 with no residual and
    nothing derived by subtraction. Six published shares from two lanes were wrong in one day
    (2026-08-08) because a total and its components came from different accountings; a partition
    that cannot express that error is worth more than a more precise one that can.

WINDOWS, NOT AVERAGES
    Blue Prince has performance regimes differing ~50x and its phase boundary is sharp. An
    average across it is a fiction, so this reports per-window and refuses to print a grand mean
    unless --windows 1 is asked for explicitly. See docs/BLUE_PRINCE_STATUS.md.

USAGE
    wait_profile.py --pid <pid> [--seconds 30] [--hz 200] [--windows 6] [--top 12]
    wait_profile.py --name prosper-app --seconds 60
"""
import argparse, collections, os, sys, time

# Linux x86-64 syscall numbers worth naming. Anything else prints as a number, which is still a
# stable identifier -- a wrong name would be worse than a number.
SYSCALLS = {
    0: "read", 1: "write", 7: "poll", 16: "ioctl", 23: "select", 35: "nanosleep",
    56: "clone", 61: "wait4", 202: "futex", 230: "clock_nanosleep", 232: "epoll_wait",
    271: "ppoll", 281: "epoll_pwait", 202 | 0: "futex",
}

def thread_ids(pid):
    try: return sorted(int(t) for t in os.listdir(f"/proc/{pid}/task"))
    except OSError: return []

def read(path):
    try:
        with open(path, "rb") as f: return f.read().decode("utf-8", "replace").strip()
    except OSError: return ""

def thread_name_and_state(pid, tid):
    # `comm` can contain spaces and parentheses, so split on the LAST ')' rather than on spaces.
    raw = read(f"/proc/{pid}/task/{tid}/stat")
    if not raw: return None, None
    close = raw.rfind(")")
    if close < 0: return None, None
    name = raw[raw.find("(") + 1:close]
    rest = raw[close + 2:].split()
    return name, (rest[0] if rest else None)

def bucket(pid, tid):
    """The single bucket this sample belongs to. Never returns None for a live thread."""
    name, state = thread_name_and_state(pid, tid)
    if name is None: return None, None
    if state == "R": return name, "RUNNING"
    wchan = read(f"/proc/{pid}/task/{tid}/wchan") or "?"
    sc = read(f"/proc/{pid}/task/{tid}/syscall").split()
    call, detail = "", ""
    if sc and sc[0] not in ("running", "-1", ""):
        try: n = int(sc[0]); call = SYSCALLS.get(n, f"syscall{n}")
        except ValueError: n, call = -1, ""
        # The syscall's FIRST ARGUMENT is what turns "blocked on a futex" into "blocked on THIS
        # futex". Without it every contended lock in the process collapses into one bucket named
        # futex_do_wait, which tells you that threads wait and never which thing they wait on --
        # and "which" is the whole question when 16 workers and the main thread are all idle.
        # Two threads sharing an address are contending; two threads on different addresses are
        # not, and no aggregate share can distinguish those.
        if call == "futex" and len(sc) > 1:
            detail = f" uaddr={sc[1]}"
        elif call in ("read", "write", "ioctl", "poll", "epoll_wait", "ppoll") and len(sc) > 1:
            detail = f" fd={int(sc[1], 16)}" if sc[1].startswith("0x") else f" fd={sc[1]}"
    label = f"{state}:{wchan}" + (f" [{call}{detail}]" if call else "")
    return name, label

def main():
    ap = argparse.ArgumentParser(description="per-thread wait attribution from /proc")
    ap.add_argument("--pid", type=int)
    ap.add_argument("--name", help="match a process by comm instead of pid")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--hz", type=float, default=200.0)
    ap.add_argument("--windows", type=int, default=6, help="report N equal windows (regimes differ)")
    ap.add_argument("--top", type=int, default=10, help="buckets shown per thread")
    ap.add_argument("--min-share", type=float, default=1.0, help="hide buckets below this %%")
    a = ap.parse_args()

    pid = a.pid
    if pid is None and a.name:
        for d in os.listdir("/proc"):
            if d.isdigit() and read(f"/proc/{d}/comm") == a.name: pid = int(d); break
    if not pid or not os.path.isdir(f"/proc/{pid}"):
        print("wait_profile: no such process (use --pid or --name)", file=sys.stderr); return 2

    nwin = max(1, a.windows)
    per_window = a.seconds / nwin
    interval = 1.0 / a.hz
    # samples[w][thread][bucket] -> count. Counts, not times: every sample is one tick of the same
    # length, so a share is a count ratio and there is no weighting to get wrong.
    samples = [collections.defaultdict(collections.Counter) for _ in range(nwin)]
    totals = [collections.Counter() for _ in range(nwin)]
    missed = 0
    t0 = time.time()
    while True:
        now = time.time()
        elapsed = now - t0
        if elapsed >= a.seconds: break
        w = min(nwin - 1, int(elapsed / per_window))
        if not os.path.isdir(f"/proc/{pid}"): break
        for tid in thread_ids(pid):
            name, label = bucket(pid, tid)
            if name is None: continue          # thread exited mid-sweep; not a bucket
            samples[w][name][label] += 1
            totals[w][name] += 1
        slept = time.time() - now
        if slept < interval: time.sleep(interval - slept)
        else: missed += 1

    print(f"wait_profile pid={pid} seconds={a.seconds:g} hz={a.hz:g} windows={nwin}")
    if missed:
        # An overrun means the sampler could not keep up, which biases toward long-lived states.
        # Printed always, including zero, so the reader never has to wonder whether it was checked.
        print(f"  NOTE: {missed} sweeps overran the sample interval — lower --hz for an unbiased run")
    else:
        print("  sweeps overran: 0")
    for w in range(nwin):
        lo, hi = w * per_window, (w + 1) * per_window
        print(f"\n=== window {w}  t={lo:6.1f}..{hi:6.1f}s ===")
        if not totals[w]:
            print("  (no samples)"); continue
        for name, n in totals[w].most_common():
            print(f"  {name}  ({n} samples)")
            shown = 0.0
            for label, c in samples[w][name].most_common(a.top):
                share = 100.0 * c / n
                if share < a.min_share: continue
                shown += share
                print(f"      {share:5.1f}%  {label}")
            # The residual is printed rather than absorbed: buckets below the threshold or beyond
            # --top are stated, so the column is always visibly complete.
            if shown < 99.95:
                print(f"      {100.0 - shown:5.1f}%  (other buckets, below --min-share or --top)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
