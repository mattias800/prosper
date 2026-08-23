#!/usr/bin/env python3
"""Assemble every hand-written control in this folder and run spirv-val over the result.

WHY THIS IS A TEST AND NOT A README LINE. This folder's whole value is that its modules are
*known-good by construction*: they are the arm that says "the SPIR-V is not the problem", and that
sentence is worth nothing if the module is malformed. `tools/vkprobe`'s recorded history is exactly
this failure -- a control that reported confident numbers from an invalid pipeline for weeks,
because nothing checked it. A control nobody validates is a control that will eventually lie, so
the validation is a ctest case rather than an instruction somebody may skip.

It also pins the assembler contract: `--target-env vulkan1.1` so the header says SPIR-V 1.3, which
is the version prosper's own recompiler emits. A module that stops assembling, stops validating, or
silently changes SPIR-V version fails here.

Usage: check_controls.py <spirv-as> <spirv-val> <shader-dir> [<scratch-dir>]
"""
import pathlib
import subprocess
import sys
import tempfile

EXPECTED_VERSION_WORD = 0x00010300  # SPIR-V 1.3


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    spirv_as, spirv_val, shader_dir = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
    sources = sorted(shader_dir.glob("*.spvasm"))
    if not sources:
        # An empty glob passing silently is the same class of defect as an absent validator
        # reporting PASS. Nothing to check means the check is broken, not satisfied.
        print(f"FAIL: no .spvasm sources under {shader_dir}")
        return 1

    failures = 0
    with tempfile.TemporaryDirectory(dir=sys.argv[4] if len(sys.argv) > 4 else None) as scratch:
        for source in sources:
            binary = pathlib.Path(scratch) / (source.stem + ".spv")
            assemble = subprocess.run(
                [spirv_as, "--target-env", "vulkan1.1", str(source), "-o", str(binary)],
                capture_output=True, text=True)
            if assemble.returncode != 0:
                print(f"FAIL {source.name}: spirv-as: {assemble.stderr.strip()}")
                failures += 1
                continue
            words = binary.read_bytes()
            version = int.from_bytes(words[4:8], "little")
            if version != EXPECTED_VERSION_WORD:
                print(f"FAIL {source.name}: SPIR-V version 0x{version:08x}, "
                      f"expected 0x{EXPECTED_VERSION_WORD:08x}")
                failures += 1
                continue
            validate = subprocess.run(
                [spirv_val, "--target-env", "vulkan1.2", str(binary)],
                capture_output=True, text=True)
            if validate.returncode != 0:
                print(f"FAIL {source.name}: spirv-val: {validate.stdout.strip()} "
                      f"{validate.stderr.strip()}")
                failures += 1
                continue
            print(f"ok   {source.name}: assembles, SPIR-V 1.3, spirv-val clean")

    print(f"{len(sources) - failures} of {len(sources)} controls valid")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
