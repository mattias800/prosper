"""Separate processes prove driver data survives _Exit and actually reaches Vulkan at startup."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

with tempfile.TemporaryDirectory(prefix="graphics-cache-") as directory:
    path = Path(directory) / "cache"
    env = os.environ.copy()
    env.pop("PROSPER_NO_DISK_PIPELINE_CACHE", None)
    env["PROSPER_GRAPHICS_PIPELINE_CACHE_PATH"] = str(path)

    def run(mode, extra=None):
        subprocess.run([sys.argv[1], mode], env=env | (extra or {}), check=True, timeout=60)

    run("save")
    original = path.read_bytes()
    assert len(original) > 32
    run("load")
    run("reject")
    run("empty", {"PROSPER_NO_DISK_PIPELINE_CACHE": "1"})
    assert path.read_bytes() == original
    damaged = bytearray(original)
    damaged[-1] ^= 1
    path.write_bytes(damaged)
    run("empty")
    path.write_bytes(original[:-1])
    run("empty")
    path.write_bytes(original)
    run("load")
