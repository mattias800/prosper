"""Require the capture fixture's empty-shader control to fail normally, without a crash."""

import os
import subprocess
import sys
from pathlib import Path


def test_empty_cf9200_shader_fails_without_crashing():
    binary = Path(os.environ["GPU_CAPTURE_SETUP_TEST_BINARY"])
    result = subprocess.run(
        [str(binary), "--empty-cf9200-shader"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 1, f"unexpected child exit {result.returncode}\n{output}"
    assert "[control] CF9200 shader setup returned an empty module" in output, output
    failures = [line.strip() for line in result.stdout.splitlines() if "[FAIL]" in line]
    assert failures == [
        "[FAIL] capture retains the exact root-record program, launch, sites, and shared witness"
    ], output


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_failed_capture_setup.py <test_gpu_capture>")
    os.environ["GPU_CAPTURE_SETUP_TEST_BINARY"] = sys.argv[1]
    test_empty_cf9200_shader_fails_without_crashing()
