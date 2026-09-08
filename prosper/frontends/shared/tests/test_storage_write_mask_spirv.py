#!/usr/bin/env python3
"""Validate every module emitted by the CPU store-mask transformation fixture."""

import pathlib
import subprocess
import sys
import tempfile


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: test_storage_write_mask_spirv.py FIXTURE SPIRV_VAL")
    with tempfile.TemporaryDirectory(prefix="prosper-storage-mask-") as directory:
        subprocess.run([sys.argv[1], directory], check=True)
        modules = sorted(pathlib.Path(directory).glob("*.spv"))
        # An empty/truncated fixture export cannot silently pass validation.
        if len(modules) != 96:
            raise SystemExit(f"expected 96 complete mask fixtures, found {len(modules)}")
        for module in modules:
            subprocess.run(
                [sys.argv[2], "--target-env", "vulkan1.3", str(module)], check=True
            )
        print(f"Validated {len(modules)} instrumented storage-image modules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
