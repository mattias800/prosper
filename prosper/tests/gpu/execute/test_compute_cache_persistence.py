"""#3425: the compute pipeline cache must survive a process that leaves through std::_Exit.

Separate processes are the whole point: a single process could pass by holding the blob in memory.
Each fixture run ends in _Exit, so only an explicit shutdown snapshot can produce the file.

The last check is the one that would have caught the original defect, which was not a broken save
but an UNCALLED one: the frontend that exits through _Exit must actually name the flush.
"""
import concurrent.futures
import os
from pathlib import Path
import subprocess
import sys
import tempfile

fixture = sys.argv[1]
app_main = Path(sys.argv[2])

env = os.environ.copy()
env.pop("PROSPER_NO_DISK_PIPELINE_CACHE", None)
env.pop("PROSPER_DISK_PIPELINE_CACHE", None)


def run(mode, path, extra=None):
    run_env = env | {"PROSPER_COMPUTE_PIPELINE_CACHE_PATH": str(path)} | (extra or {})
    return subprocess.run([fixture, mode], env=run_env, check=True, timeout=180)


with tempfile.TemporaryDirectory(prefix="compute-cache-") as directory:
    path = Path(directory) / "cache"

    # A cold launch saves; a following launch loads what it wrote.
    run("save", path)
    original = path.read_bytes()
    assert len(original) > 32, f"the envelope must carry a real blob, got {len(original)} bytes"
    run("load", path)

    # The opt-out wins over an explicit path, and must not write or report a save.
    run("disabled", path, {"PROSPER_NO_DISK_PIPELINE_CACHE": "1"})
    assert path.read_bytes() == original, "a disabled run must not touch the existing cache"

    # A damaged body is rejected by the checksum even though its Vulkan header still matches, and a
    # truncated file is rejected by the recorded length. Neither may reach vkCreatePipelineCache.
    damaged = bytearray(original)
    damaged[-1] ^= 0xFF
    path.write_bytes(damaged)
    run("cold", path)
    path.write_bytes(original[:-1])
    run("cold", path)
    path.write_bytes(b"")
    run("cold", path)

    # A good file still loads after all of that, so rejection is about the bytes, not the path.
    path.write_bytes(original)
    run("load", path)

    # Several prosper processes run on one box routinely. Concurrent savers must leave a file that
    # still loads, never a half-written one.
    concurrent_path = Path(directory) / "concurrent"
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for future in [pool.submit(run, "resave", concurrent_path) for _ in range(3)]:
            future.result()
    assert concurrent_path.is_file(), "concurrent savers must leave exactly one cache file"
    leftovers = [p.name for p in Path(directory).iterdir() if ".tmp-" in p.name]
    assert not leftovers, f"concurrent savers must clean up their temporaries: {leftovers}"
    run("load", concurrent_path)

# The defect in #3425 was that flush_live_compute_pipeline_cache() had no caller at all, while its
# own header comment claimed prosper-app called it. Nothing above can see that, because the fixture
# calls the function directly. Assert the wiring separately.
source = app_main.read_text(encoding="utf-8", errors="replace")
assert "flush_live_compute_pipeline_cache" in source, (
    f"{app_main.name} exits through std::_Exit and must flush the compute pipeline cache there; "
    "without that call the cache loads on every launch and is never written"
)
print("compute pipeline cache persists across processes and prosper-app calls the flush")
