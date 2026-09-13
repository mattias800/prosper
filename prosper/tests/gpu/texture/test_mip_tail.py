"""Exact process comparison: a cached environment switch needs a fresh process per arm."""
import os
import re
import subprocess
import sys

env = dict(os.environ)
env.pop("PROSPER_NO_SEPARABLE_MIP_TAIL", None)
fast = subprocess.run([sys.argv[1]], env=env, capture_output=True, check=True, timeout=45)
scalar = subprocess.run([sys.argv[1]], env={**env, "PROSPER_NO_SEPARABLE_MIP_TAIL": "1"},
                        capture_output=True, check=True, timeout=45)
assert fast.stdout and fast.stdout == scalar.stdout, "fast/scalar mip-tail bytes differ"
negative = subprocess.run([sys.argv[1], "--corrupt-oracle"], env=env,
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=45)
marker = re.search(rb"mip-tail oracle failures=([1-9][0-9]*)\r?\n", negative.stderr)
assert negative.returncode == 1 and marker, \
    "placement guard failed to reject a deliberately corrupted expected byte"
print(f"Compared {len(fast.stdout)} exact bytes; independent placement and negative controls passed")
