#!/usr/bin/env python3
"""Require validated warm image reuse and a forced-upload control with the same pixel oracle."""
import os
import subprocess
import sys

env = {key: value for key, value in os.environ.items() if not key.startswith("PROSPER_")}
env["PROSPER_COMPUTELOG"] = "1"
minimum_reuses = int(sys.argv[1])
for forced in (False, True):
    arm = dict(env)
    if forced:
        arm["PROSPER_NO_SKIP_SEED"] = "1"
    result = subprocess.run(sys.argv[2:], env=arm, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=50)
    print(result.stdout, end="")
    if result.returncode:
        raise SystemExit(result.returncode)
    lines = result.stdout.splitlines()
    assert not any(line.startswith("[seed-skip-verify] ") for line in lines), \
        "destructive historical coverage proving must not run"
    reuses = [line for line in lines if "persistent storage image binding=5 " in line
              and "upload-skipped=1" in line]
    if forced:
        assert not reuses, "the forced-seed control must really upload current storage input"
    else:
        assert len(reuses) >= minimum_reuses, \
            f"expected at least {minimum_reuses} validated warm reuses, got {len(reuses)}"
