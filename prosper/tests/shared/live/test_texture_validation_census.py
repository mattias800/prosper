"""Verify the census hook through real mapped textures, rendering and capture timing."""
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    executable = str(Path(sys.argv[1]).resolve())
    env = {k: v for k, v in os.environ.items() if not k.startswith("PROSPER_")}
    env.update(PROSPER_TEXTURE_VALIDATION_CENSUS="1",
               PROSPER_TEXTURE_WRITE_WATCH_MIN_KB="1024",
               PROSPER_NO_LIVE_PERSISTENT_COLOR_TARGETS="1",
               PROSPER_BACKEND_BUFFER_RESIDENCY_MB="256",
               PROSPER_BACKEND_BUFFER_RESIDENCY_OWNERS="4096")
    if "--negative-control" in sys.argv[2:]:
        env.pop("PROSPER_TEXTURE_VALIDATION_CENSUS")
    with tempfile.TemporaryDirectory(prefix="texture-validation-census-") as directory:
        result = subprocess.run([executable, "--source-snapshot"], env=env, cwd=directory,
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=120)
    print(result.stdout, end="")
    assert result.returncode == 0, f"Renderer fixture failed: {result.returncode}"
    headers = [line for line in result.stdout.splitlines()
               if line.startswith("[texture-validation-census]")]
    assert len(headers) == 1, headers
    # Two independent cold textures do not validate. The next submit changes one source;
    # the last repeats that refreshed source. Both actual rendered colors are checked by C++.
    assert re.search(r"scope=thread-cumulative thread=\d+ reason=thread-exit calls=2$", headers[0]), headers
    rows = [dict(re.findall(r"([\w_]+)=([^ ]+)", line)) for line in result.stdout.splitlines()
            if line.startswith("[texture-validation-bucket]")]
    assert len(rows) == 2, rows
    by_outcome = {r["outcome"]: r for r in rows}
    assert set(by_outcome) == {"match", "exact-failure"}, rows
    for row in rows:
        assert row["cumulative_calls"] == "2" and row["calls"] == "1", row
        assert row["watch"] == "below-minimum" and row["size"] == "below-1MiB", row
        assert row["source_bytes"] == "16", row
    assert by_outcome["match"]["reported_validated_bytes"] == "16", rows
    # Failed comparisons have platform-dependent reported extents; do not infer read traffic.
    assert int(by_outcome["exact-failure"]["reported_validated_bytes"]) in (0, 16), rows
    return 0


if __name__ == "__main__":
    sys.exit(main())
