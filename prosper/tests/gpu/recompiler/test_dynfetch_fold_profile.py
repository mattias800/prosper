"""Prove the opt-in fold profile sees a known selected PC-relative fixture."""

import os
from pathlib import Path
import re
import subprocess
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: test_dynfetch_fold_profile.py TEST_DYNFETCH_FOLD", file=sys.stderr)
        return 2
    executable = Path(sys.argv[1])
    env = os.environ.copy()
    env["PROSPER_STAGE_FOLD_PROFILE"] = "1"
    env["PROSPER_STAGE_FOLD_PROFILE_CALLS"] = "1"
    run = subprocess.run([str(executable)], env=env, text=True,
                         capture_output=True, timeout=120, check=False)
    output = run.stdout + "\n" + run.stderr
    summary = any(
        line.startswith("[stage-fold-profile] ") and
        re.search(r"\bpcrel_calls=[1-9][0-9]*\b", line)
        for line in output.splitlines()
    )
    selected = any(
        line.startswith("[stage-fold-pcrel] ") and
        re.search(r"\bcalls=[1-9][0-9]*\b", line) and
        "failed=0" in line
        for line in output.splitlines()
    )
    if run.returncode != 0 or not summary or not selected:
        print(f"exit={run.returncode} nonzero-summary={summary} "
              f"successful-selected-row={selected}", file=sys.stderr)
        print("\n".join(output.splitlines()[-40:]), file=sys.stderr)
        return 1
    print("selected PC-relative fold reached both aggregate and bucket reports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
