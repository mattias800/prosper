#!/usr/bin/env python3
"""Self-test for fragment_wave_width_relaxable (rdna2_to_spirv.hpp).

The predicate is the codified ruling of #2404/#2410/#2414: NO wave-width reason set is
relaxable to a narrower subgroup, because the vote result is consumed as guest scalar data
(SCC) and a guarded block's emulated scalar state goes stale in the non-entering half-wave.
The test pins that ruling so a future relaxation cannot land silently: relaxing the
predicate to return true for any set must fail this test and force the #2410 criterion
(shape-level dataflow evidence) to be applied first.

Run: python3 tools/perf/test_fragment_wave_width_relaxable.py
"""

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "src" / "gpu" / "recompiler" / "rdna2_to_spirv.hpp"
TOOL = Path(__file__).resolve().parent / "test_fragment_wave_width_relaxable.py"

PROBE = r"""
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <cstdio>
using namespace prosper::gpu;
int main() {
    const uint32_t single[] = {
        kFragmentWaveReasonLaneId, kFragmentWaveReasonWaveAny,
        kFragmentWaveReasonDppRow16, kFragmentWaveReasonPermLane32,
        kFragmentWaveReasonReadLane64, kFragmentWaveReasonShuffle,
        kFragmentWaveReasonWaveBallot,
    };
    for (uint32_t reasons : single)
        if (fragment_wave_width_relaxable(reasons)) {
            printf("FAIL: single reason set 0x%x is relaxable\n", reasons);
            return 1;
        }
    // Every pair, and the full set: the ruling covers combinations too.
    for (uint32_t a = 0; a < 7; a++)
        for (uint32_t b = a + 1; b < 7; b++) {
            const uint32_t combo = (1u << a) | (1u << b);
            if (fragment_wave_width_relaxable(combo)) {
                printf("FAIL: combination 0x%x is relaxable\n", combo);
                return 1;
            }
        }
    if (fragment_wave_width_relaxable(0x7f)) {
        printf("FAIL: the full reason set is relaxable\n");
        return 1;
    }
    printf("fragment_wave_width_relaxable: no reason set is relaxable (ruling intact)\n");
    return 0;
}
"""


def main():
    failures = []
    text = HEADER.read_text(encoding="utf-8")
    if "fragment_wave_width_relaxable" not in text:
        print("FAIL: the predicate is missing from rdna2_to_spirv.hpp")
        return 1
    if "return false" not in text.split("fragment_wave_width_relaxable")[1].split("}")[0]:
        failures.append("the predicate does not return false unconditionally")

    with tempfile.TemporaryDirectory() as tmp:
        cpp = Path(tmp) / "probe.cpp"
        cpp.write_text(PROBE, encoding="utf-8")
        exe = Path(tmp) / "probe.exe"
        compiler = r"C:\Users\matti\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin\g++.exe"
        build = subprocess.run(
            [compiler, "-O1", "-std=gnu++20",
             f"-I{ROOT / 'src'}", str(cpp), "-o", str(exe)],
            capture_output=True, text=True)
        if build.returncode != 0:
            failures.append(f"probe build failed: {build.stderr[:400]}")
        else:
            run = subprocess.run([str(exe)], capture_output=True, text=True)
            if run.returncode != 0:
                failures.append(f"probe failed: {run.stdout} {run.stderr}")
            else:
                print(run.stdout.strip())

    if failures:
        print("FAILURES:")
        for failure in failures:
            print(" -", failure)
        return 1
    print("test_fragment_wave_width_relaxable: all cases pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
