"""Calibrate publication/replacement diagnostics through actual HLE calls and FLOW output."""
import os
import re
import subprocess
import sys

result = subprocess.run([sys.argv[1], "--flow-publication"],
                        env=os.environ | {"PROSPER_AUDIO_FLOW": "1"},
                        capture_output=True, text=True, timeout=30, check=True)
rows = [(int(a), int(b)) for a, b in re.findall(
    r"\[audio-flow\]   port1 .*?published=(\d+) replaced-pending=(\d+)", result.stderr)]
assert [row for row in rows if row[0]] == [(2, 1), (1, 0)], result.stderr
print("FLOW calibrated: same-port replacement detected; consumed grain is not a replacement")
