#!/usr/bin/env python3
"""Require real warm seed skipping as well as the fixture's independent pixel oracle."""
import os
import subprocess
import sys

env = {key: value for key, value in os.environ.items() if not key.startswith("PROSPER_")}
env.update(PROSPER_COMPUTELOG="1", PROSPER_SEED_REPROVE="256")
expected_proofs, expected_skips = map(int, sys.argv[1:3])
result = subprocess.run(sys.argv[3:], env=env, text=True, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, timeout=50)
print(result.stdout, end="")
if result.returncode:
    raise SystemExit(result.returncode)
lines = result.stdout.splitlines()
proofs = [line for line in lines if line.startswith("[seed-skip-verify] ")]
skips = [line for line in lines if "seed-skip write-only storage " in line]
# Full cold proof, identical Full warm skip, changed program/launch partial proof,
# then identical Partial warm seed. Reject an implementation that simply re-proves
# every dispatch: it can preserve all pixels while silently removing the optimization.
assert len(proofs) == expected_proofs, f"expected {expected_proofs} proofs, got {len(proofs)}"
assert "poison_survived=0 " in proofs[0], "the initial writer must prove Full"
assert "PARTIAL-COVERAGE" in proofs[1], "the changed writer must prove Partial"
assert len(skips) == expected_skips, f"expected {expected_skips} Full skips, got {len(skips)}"
