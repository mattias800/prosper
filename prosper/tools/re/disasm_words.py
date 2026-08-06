#!/usr/bin/env python3
"""Disassemble RDNA2 instruction words with llvm-mc (gfx1030).

The recompiler's `[recompile-reject]` diagnostic prints the two decoded dwords of the
instruction it could not translate, as `words=AABBCCDD,EEFFGGHH`. Naming that instruction from an
opcode table is how a wrong lowering gets written (see SONIC_CROSSWORLDS_STATUS.md § Ruled out:
`0x305` is `v_mul_lo_u16`, not `v_lshlrev_b16`). Round-tripping the exact bytes through llvm-mc is
the check that actually names it.

Usage:
    disasm_words.py d7074001,00020081 [more...]      # explicit word pairs
    grep recompile-reject run.log | disasm_words.py  # read `words=..,..` out of a log on stdin

Requires `llvm-mc` on PATH (any LLVM with the AMDGPU target built in).
"""
import re
import subprocess
import sys

WORDS_RE = re.compile(r"words=([0-9a-fA-F]{8}),([0-9a-fA-F]{8})")


def to_bytes(words):
    """Little-endian byte list for a sequence of 32-bit words."""
    out = []
    for w in words:
        v = int(w, 16)
        out += [(v >> s) & 0xFF for s in (0, 8, 16, 24)]
    return out


def disassemble(byte_list):
    text = ",".join("0x%02x" % b for b in byte_list) + "\n"
    try:
        proc = subprocess.run(
            ["llvm-mc", "-arch=amdgcn", "-mcpu=gfx1030", "-disassemble"],
            input=text, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        sys.exit("llvm-mc not found on PATH")
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()
             and not ln.strip().startswith(".")]
    if not lines:
        err = proc.stderr.strip().splitlines()
        return "<no disassembly>" + (" (" + err[-1] + ")" if err else "")
    return lines[0]


def pairs_from_args(argv):
    for arg in argv:
        parts = [p for p in re.split(r"[,\s]+", arg) if p]
        yield parts


def pairs_from_stdin():
    for line in sys.stdin:
        m = WORDS_RE.search(line)
        if m:
            yield [m.group(1), m.group(2)]


def main():
    src = pairs_from_args(sys.argv[1:]) if len(sys.argv) > 1 else pairs_from_stdin()
    seen = set()
    for words in src:
        key = tuple(w.lower() for w in words)
        if key in seen:
            continue
        seen.add(key)
        print("%-20s %s" % (",".join(words), disassemble(to_bytes(words))))


if __name__ == "__main__":
    main()
