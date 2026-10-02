"""Run a pipeline-cache test binary against a fresh temporary directory (argv[1] is the binary)."""

import subprocess
import sys
import tempfile

with tempfile.TemporaryDirectory(prefix="pipeline-cache-file-") as directory:
    subprocess.run([sys.argv[1], directory], check=True, timeout=30)
