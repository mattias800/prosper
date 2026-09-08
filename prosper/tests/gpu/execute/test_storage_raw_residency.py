#!/usr/bin/env python3
"""Pair real raw-residency admission witnesses with the executable's full pixel oracle."""
import os
import re
import subprocess
import sys

env = {key: value for key, value in os.environ.items() if not key.startswith("PROSPER_")}
env["PROSPER_COMPUTELOG"] = "1"
result = subprocess.run(sys.argv[1:], env=env, text=True, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, timeout=50)
print(result.stdout, end="")
if result.returncode:
    raise SystemExit(result.returncode)
records = {}
phase = None
pattern = re.compile(r"persistent storage image binding=5 addr=0x[0-9a-f]+ "
                     r"guest=4096 upload-skipped=([01])")
for line in result.stdout.splitlines():
    if line.startswith("raw residency phase: "):
        phase = line.removeprefix("raw residency phase: ")
        if phase in records:
            raise SystemExit(f"duplicate phase: {phase}")
        records[phase] = []
    elif match := pattern.search(line):
        if phase is None:
            raise SystemExit("cache record precedes the first phase")
        records[phase].append(int(match.group(1)))
expected = {
    "full": [],
    "warm full": [1],
    "partial unchanged guest": [1],
    "partial changed guest": [0],
    "warm partial": [1],
    "readable alias transition": [],
}
if records != expected:
    raise SystemExit(f"raw residency admissions differ: {records!r}; expected {expected!r}")
