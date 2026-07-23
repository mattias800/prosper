#!/usr/bin/env python3
"""hang_probe - estimate how often a title's main thread ends up stuck, and show what it's stuck on.

Intermittent boot/loading hangs (e.g. the undelivered-GPU-EOP deadlock family #1113/#1195/#987, where
UnityEOPThread parks in k_eq_wait and the main/loader threads park in k_sema_wait) are timing races: a
single run may or may not hit them, so you need a *rate* and a *backtrace of the stuck run*. This tool
automates the run -> wait -> gdb/guest_bt classify -> repeat loop that is otherwise hand-orchestrated.

For each run it launches the title headless through boot_trace, waits, then attaches guest_bt to the
main thread (Thread 1) and classifies it:
  * BLOCKED (hung) - the top HLE frame is a kernel wait (k_sema_wait / k_eq_wait / k_cond_wait / k_ef_wait)
  * RUNNING        - an active render/submit frame (execute_ordered / present / agc_driver_submit / run_command)
Plain gdb cannot unwind the guest main thread through the HLE stub boundary, so guest_bt is required (it
re-sections the flattened module ELFs). On a BLOCKED run the tool prints Thread 1's stack as evidence.

Linux only (needs gdb + guest_bt; ptrace attach must be permitted - kernel.yama.ptrace_scope=0).

Example:
  python3 tools/hang_probe/hang_probe.py --dump /path/PPSA01885-app0 --runs 10 --wait 35
"""
import argparse, os, subprocess, sys, time, tempfile, shutil, signal, re

HERE = os.path.dirname(os.path.abspath(__file__))

WAIT_FRAMES = ("k_sema_wait", "k_eq_wait", "k_cond_wait", "k_ef_wait", "k_usleep")
RUN_FRAMES  = ("execute_ordered", "execute_submit", "agc_driver_submit", "run_command_buffer",
               "present", "SubmitDcb", "run_command")


def classify(gbt_text: str):
    """Return ('BLOCKED'|'RUNNING'|'UNKNOWN', top_frame_str) for the main thread's stack."""
    if any(f in gbt_text for f in RUN_FRAMES):
        m = next((l for l in gbt_text.splitlines() if any(f in l for f in RUN_FRAMES)), "")
        return "RUNNING", m.strip()
    if any(f in gbt_text for f in WAIT_FRAMES):
        m = next((l for l in gbt_text.splitlines() if any(f in l for f in WAIT_FRAMES)), "")
        return "BLOCKED", m.strip()
    return "UNKNOWN", ""


def one_run(args, i):
    save = tempfile.mkdtemp(prefix=f"hangprobe_save{i}_")
    log = tempfile.NamedTemporaryFile(prefix=f"hangprobe_run{i}_", suffix=".log", delete=False).name
    env = dict(os.environ)
    env.update({
        "PROSPER_INITLOG": "1", "PROSPER_GUEST_FS": "1",
        "PROSPER_GUEST_ARGS": "-force-gfx-direct", "PROSPER_RENDER": "1",
        "PROSPER_NO_FRAME_DUMPS": "1", "PROSPER_SAVE0": save,
    })
    for kv in args.extra_env:
        if "=" in kv:
            k, v = kv.split("=", 1); env[k] = v
    proc = None
    try:
        with open(log, "wb") as lf:
            proc = subprocess.Popen([args.boot_trace, args.dump], stdout=lf, stderr=lf, env=env)
        time.sleep(args.wait)
        if proc.poll() is not None:
            return "DEAD", "(process exited before the sample)", log
        cmd = [sys.executable, args.guest_bt, "--pid", str(proc.pid),
               "--initlog", log, "--thread", str(args.thread)]
        if args.dump:
            cmd += ["--dump", args.dump]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=args.gbt_timeout).stdout
        except subprocess.TimeoutExpired:
            return "UNKNOWN", "(guest_bt timed out)", log
        state, frame = classify(out)
        if state == "BLOCKED" and args.show_stack:
            sys.stdout.write("    --- Thread 1 stack ---\n")
            for l in out.splitlines():
                if re.match(r"\s*#\d", l) or "thread 1" in l.lower():
                    sys.stdout.write("    " + l.strip() + "\n")
        return state, frame, log
    finally:
        if proc and proc.poll() is None:
            try:
                proc.send_signal(signal.SIGKILL); proc.wait(timeout=5)
            except Exception:
                pass
        shutil.rmtree(save, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description="Estimate a title's intermittent-hang rate via guest_bt.")
    ap.add_argument("--dump", required=True, help="<TITLE>-app0 directory")
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--wait", type=float, default=35.0, help="seconds to let the title reach the hang window")
    ap.add_argument("--thread", default="1", help="thread to classify (default 1 = main)")
    ap.add_argument("--boot-trace", dest="boot_trace",
                    default=os.path.join(HERE, "..", "..", "build-linux", "boot_trace"))
    ap.add_argument("--guest-bt", dest="guest_bt",
                    default=os.path.join(HERE, "..", "guest_bt", "guest_bt.py"))
    ap.add_argument("--gbt-timeout", type=float, default=120.0)
    ap.add_argument("--extra-env", action="append", default=[],
                    help="K=V to add to the run env (repeatable), e.g. PROSPER_EOP_SYNC=1")
    ap.add_argument("--no-stack", dest="show_stack", action="store_false",
                    help="don't print the stuck main-thread stack on a BLOCKED run")
    ap.add_argument("--keep-logs", action="store_true")
    args = ap.parse_args()

    for p, name in ((args.boot_trace, "boot_trace"), (args.guest_bt, "guest_bt.py")):
        if not os.path.exists(p):
            sys.exit(f"error: {name} not found at {p} (build boot_trace / check --{name.split('.')[0]})")
    if not os.path.isdir(args.dump):
        sys.exit(f"error: dump dir not found: {args.dump}")

    counts = {"BLOCKED": 0, "RUNNING": 0, "DEAD": 0, "UNKNOWN": 0}
    print(f"== hang_probe: {os.path.basename(args.dump.rstrip('/'))}, {args.runs} runs, "
          f"{args.wait:.0f}s wait, thread {args.thread} ==")
    for i in range(1, args.runs + 1):
        state, frame, log = one_run(args, i)
        counts[state] += 1
        print(f"  run {i:>2}: {state:<8} {frame}")
        if args.keep_logs:
            print(f"           log: {log}")
        else:
            try: os.unlink(log)
            except OSError: pass
    total_classified = counts["BLOCKED"] + counts["RUNNING"]
    rate = (100.0 * counts["BLOCKED"] / total_classified) if total_classified else 0.0
    print(f"== BLOCKED={counts['BLOCKED']} RUNNING={counts['RUNNING']} "
          f"DEAD={counts['DEAD']} UNKNOWN={counts['UNKNOWN']} (of {args.runs}) "
          f"=> hang rate ~{rate:.0f}% of classified ==")
    # exit 1 if any hang was observed, so it's usable as a gate/bisect predicate
    sys.exit(1 if counts["BLOCKED"] else 0)


if __name__ == "__main__":
    main()
