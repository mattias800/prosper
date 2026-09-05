#!/usr/bin/env python3
"""Inventory debugging tools and prove selected capabilities on bounded child processes.

Usage: doctor.py [--probe perf|scheduler|debugger ... --output NEW_DIRECTORY]
       doctor.py --json

No probes by default. Exit 0: inventory completed or every requested probe passed.
Exit 1: at least one requested capability was unavailable/inconclusive. Exit 2: usage.
"""
import argparse
import collections
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time

TOOLS = (
    "perf", "gdb", "hotspot", "heaptrack", "valgrind", "bpftrace", "strace",
    "ltrace", "sysprof-cli", "radeontop", "renderdoccmd", "qrenderdoc",
    "vulkaninfo", "spirv-val", "spirv-as", "spirv-dis", "llvm-mc",
    "stackcollapse-perf.pl", "flamegraph.pl",
)
CPU_CONTROL = (
    "import os,time; end=time.monotonic()+1.5; fd=os.open('/dev/zero',os.O_RDONLY); "
    "\nwhile time.monotonic()<end: os.read(fd,4096)"
)
SCHED_CONTROL = """import threading,time
ready=threading.Event()
ack=threading.Event()
def worker():
    for _ in range(100):
        ready.wait()
        ready.clear()
        ack.set()
t=threading.Thread(target=worker)
t.start()
for _ in range(100):
    ready.set()
    ack.wait()
    ack.clear()
    time.sleep(0.005)
t.join()
"""


def read_setting(path):
    try:
        return Path(path).read_text().strip()
    except OSError as exc:
        return f"unavailable: {exc.strerror}"


def inventory():
    modules = {}
    for name in ("renderdoc", "elftools"):
        try:
            modules[name] = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            modules[name] = False
    return {
        "platform": platform.platform(),
        "python": sys.executable,
        "tools": {name: shutil.which(name) for name in TOOLS},
        "python_modules": modules,
        "settings": {
            name: read_setting("/proc/sys/" + name.replace(".", "/"))
            for name in ("kernel.perf_event_paranoid", "kernel.kptr_restrict",
                         "kernel.yama.ptrace_scope")
        },
        "tracefs": {
            path: os.access(path, os.R_OK)
            for path in ("/sys/kernel/tracing/available_events",
                         "/sys/kernel/debug/tracing/available_events")
        },
        "note": "Executable/module presence is inventory, not proof of capability.",
    }


def run(command, directory, name, timeout=15):
    start = time.monotonic()
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                              env={**os.environ, "LC_ALL": "C", "DEBUGINFOD_URLS": ""})
        result = {"command": command, "returncode": proc.returncode,
                  "stdout": proc.stdout, "stderr": proc.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        result = {"command": command, "returncode": None, "stdout": "", "stderr": str(exc)}
    result["elapsed_seconds"] = round(time.monotonic() - start, 4)
    (directory / (name + ".json")).write_text(json.dumps(result, indent=2) + "\n")
    return result


def events_in(text):
    # perf script -F event emits one event name per sample, not arbitrary diagnostics.
    return collections.Counter(line.strip().rstrip(":") for line in text.splitlines()
                               if re.fullmatch(r"[\w:-]+:?", line.strip()))


def sample_verdict(record, script, required):
    counts = events_in(script["stdout"])
    ready = (record["returncode"] == 0 and script["returncode"] == 0
             and all(counts[event] > 0 for event in required))
    return {"status": "READY" if ready else "UNAVAILABLE",
            "samples": {event: counts[event] for event in required},
            "reason": "Required samples recorded and decoded." if ready else
                      "Recording/decoding failed or required samples were absent; inspect logs."}


def perf_probe(directory, scheduler=False):
    if not shutil.which("perf"):
        return {"status": "UNAVAILABLE", "reason": "perf is not on this environment's PATH."}
    cases = [("scheduler", ["sched:sched_switch", "sched:sched_wakeup"], SCHED_CONTROL)] if scheduler else [
        ("cpu", ["cpu-clock:u"], CPU_CONTROL),
        ("kernel", ["cpu-clock:k"], CPU_CONTROL),
    ]
    results = {}
    for name, events, control in cases:
        data = directory / (name + ".data")
        command = ["perf", "record", "-q", "-o", str(data)]
        if not scheduler:
            command += ["-F", "99", "-g"]
        for event in events:
            command += ["-e", event]
        command += ["--", sys.executable, "-c", control]
        rec = run(command, directory, name + "-record")
        decoded = run(["perf", "script", "-i", str(data), "-F", "event"],
                      directory, name + "-decode")
        results[name] = sample_verdict(rec, decoded, events)
    return {"status": "READY" if all(r["status"] == "READY" for r in results.values())
            else "UNAVAILABLE", "controls": results,
            "scope": "Child-process recording; not system-wide scheduler attribution." if scheduler
            else "User and kernel samples; guest symbolication is a separate capability."}


def debugger_probe(directory):
    if not shutil.which("gdb"):
        return {"status": "UNAVAILABLE", "reason": "gdb is not on PATH."}
    # Only our child is attached. No process-name matching, no calls executed in the inferior.
    with subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"]) as child:
        try:
            result = run(["gdb", "-nx", "-nh", "-batch", "-p", str(child.pid),
                          "-ex", 'python print("DOCTOR_ATTACHED_PID=%d" % gdb.selected_inferior().pid)',
                          "-ex", "info threads", "-ex", "detach"], directory, "debugger")
            ready = (result["returncode"] == 0
                     and f"DOCTOR_ATTACHED_PID={child.pid}" in result["stdout"])
            return {"status": "READY" if ready else "UNAVAILABLE",
                    "reason": "Attached to this probe's child and detached." if ready else
                    "Attach not established; inspect debugger.json.",
                    "scope": "Same-environment child only; test host/container crossing separately."}
        finally:
            child.terminate()
            child.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", action="append", choices=("perf", "scheduler", "debugger"),
                        default=[])
    parser.add_argument("--output", type=Path, help="new directory on real disk; retained logs")
    parser.add_argument("--json", action="store_true", help="print machine-readable inventory/report")
    args = parser.parse_args()
    if args.probe and not args.output:
        parser.error("--probe requires --output NEW_DIRECTORY (use disk, not /tmp)")
    report = {"schema_version": 1, "inventory": inventory(), "probes": {}}
    if args.output:
        try:
            args.output = args.output.resolve()
            args.output.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            parser.error(f"cannot create a new output directory: {exc}")
    for probe in dict.fromkeys(args.probe):
        report["probes"][probe] = (debugger_probe(args.output) if probe == "debugger"
                                   else perf_probe(args.output, probe == "scheduler"))
    if args.output:
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for name, path in report["inventory"]["tools"].items():
            print(f"{'INSTALLED' if path else 'MISSING':11} {name}: {path or '-'}")
        print(report["inventory"]["note"])
        for name, result in report["probes"].items():
            print(f"{result['status']:11} {name}: {json.dumps(result)}")
        if args.output:
            print(f"Evidence: {args.output}")
    return int(any(p["status"] != "READY" for p in report["probes"].values()))


if __name__ == "__main__":
    sys.exit(main())
