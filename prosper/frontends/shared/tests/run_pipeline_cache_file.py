import subprocess
import sys
import tempfile

with tempfile.TemporaryDirectory(prefix="pipeline-cache-file-") as directory:
    subprocess.run([sys.argv[1], directory], check=True, timeout=30)
