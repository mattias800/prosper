#!/usr/bin/env python3
"""hang_probe - estimate how often a title's main thread ends up stuck, and show what it's stuck on.

Intermittent boot/loading hangs (e.g. the undelivered-GPU-EOP deadlock family #1113/#1195/#987, where
UnityEOPThread parks in k_eq_wait and the main/loader threads park in k_sema_wait) are timing races: a
single run may or may not hit them, so you need a *rate* and a *backtrace of the stuck run*. This tool
automates the run -> wait -> gdb/guest_bt classify -> repeat loop that is otherwise hand-orchestrated.

For each run it launches the title headless through boot_trace, waits, then attaches guest_bt to the
main thread (Thread 1) and classifies it:
  * BLOCKED (hung) - the top HLE frame is a kernel wait (k_sema_wait / k_eq_wait / k_cond_wait / k_ef_wait)
  * RUNNING        - an active render/submit frame or a non-blocking selected native frame
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
NATIVE_STOP_FRAME = re.compile(r"^guest-bt-native: (0x[0-9a-f]+ in .+)$", re.MULTILINE)
NATIVE_BLOCK_FUNCTION = re.compile(
    r"^(?:syscall_cancel_arch|futex_(?:(?:abstimed_)?wait(?:_(?:common|cancelable))?(?:64)?|wait_simple)|"
    r"pthread_cond_(?:wait|timedwait|clockwait)|pthread_mutex_(?:lock|timedlock|clocklock)(?:_full)?|"
    r"pthread_rwlock_(?:rdlock|wrlock|timedrdlock|timedwrlock|clockrdlock|clockwrlock)|"
    r"sem_(?:wait|timedwait|clockwait)|clock_nanosleep(?:_time64)?|nanosleep|poll|ppoll|"
    r"select(?:64)?|epoll_(?:wait|pwait|pwait2)|kevent(?:64)?)$")


def native_function(frame: str):
    """Return a normalized function name from a guest-bt-native evidence line."""
    name = frame.partition(" in ")[2].split("(", 1)[0].split("@", 1)[0].strip()
    if name.startswith("__GI_"):
        name = name[len("__GI_"):]
    return name.lstrip("_")


def classify(gbt_text: str):
    """Return ('BLOCKED'|'RUNNING'|'UNKNOWN', top_frame_str) for the main thread's stack."""
    native_match = NATIVE_STOP_FRAME.search(gbt_text)
    # gdb prints the thread it happened to stop on before the guest-bt command switches threads.
    # Once the selected-thread marker exists, no frame before it is classification evidence.
    selected_text = gbt_text[native_match.start():] if native_match else gbt_text
    # A guest kernel wait is decisive even when a lower frame happens to contain a render symbol.
    # The EOP deadlock this probe targets has exactly that shape: the main thread is in k_sema_wait.
    if any(f in selected_text for f in WAIT_FRAMES):
        m = next((l for l in selected_text.splitlines() if any(f in l for f in WAIT_FRAMES)), "")
        return "BLOCKED", m.strip()
    frame = native_match.group(1).strip() if native_match else ""
    # A selected thread stopped in a known host wait is inconclusive even if a lower caller happens
    # to carry an active-render name. Accepting that lower frame would turn a parked thread green.
    if frame and NATIVE_BLOCK_FUNCTION.fullmatch(native_function(frame)):
        return "UNKNOWN", frame
    if any(f in selected_text for f in RUN_FRAMES):
        m = next((l for l in selected_text.splitlines() if any(f in l for f in RUN_FRAMES)), "")
        return "RUNNING", m.strip()
    # Re-sectioning guest modules leaves native boot_trace frames unnamed in the backtrace, but gdb
    # prints the selected thread's current frame immediately before it. A non-blocking current frame
    # means Thread 1 is executing: observed healthy Evergate samples include Prosper render helpers,
    # libc allocation/string work, the Vulkan driver, and unsymbolicated guest PCs. Known host waits
    # remain inconclusive unless guest_bt also found the decisive guest kernel-wait frame above.
    if frame:
        return "RUNNING", frame
    return "UNKNOWN", frame


def run_guest_bt(cmd, timeout):
    """Return (state, evidence, stdout), preserving debugger/tool failures as UNKNOWN evidence."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "UNKNOWN", "(guest_bt timed out)", ""
    if result.returncode != 0:
        detail = next((line.strip() for line in reversed(result.stderr.splitlines()) if line.strip()), "")
        suffix = f": {detail}" if detail else ""
        return "UNKNOWN", f"(guest_bt failed, exit {result.returncode}{suffix})", result.stdout
    state, frame = classify(result.stdout)
    return state, frame, result.stdout


def result_exit_code(counts):
    """1 means a reproduced hang; 2 means the experiment was inconclusive; 0 is a clean result."""
    if counts["BLOCKED"]:
        return 1
    if counts["UNKNOWN"] or counts["DEAD"]:
        return 2
    return 0


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
        state, frame, out = run_guest_bt(cmd, args.gbt_timeout)
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
    # A gate must not report a clean result when no usable backtrace was obtained. Keep reproduced
    # hangs as exit 1 for bisect predicates and use exit 2 for inconclusive/tool-failure runs.
    sys.exit(result_exit_code(counts))


if __name__ == "__main__":
    main()
