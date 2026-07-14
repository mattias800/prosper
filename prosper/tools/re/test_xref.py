#!/usr/bin/env python3
"""Synthetic decoder regression for xref.py's supported x86-64 reference forms."""

import importlib.util
import os
import struct
import sys
import tempfile


sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = importlib.util.spec_from_file_location("prosper_xref", os.path.join(HERE, "xref.py"))
XREF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(XREF)


def rel32(site, insn_size, target):
    return struct.pack("<i", target - (site + insn_size))


def main():
    raw = bytearray(0x200)
    raw[:4] = b"\x7fELF"
    struct.pack_into("<Q", raw, 0x20, 0x40)       # e_phoff
    struct.pack_into("<HH", raw, 0x36, 56, 1)    # e_phentsize, e_phnum
    struct.pack_into("<IIQQQQQQ", raw, 0x40,
                     1, 5, 0x100, 0x1000, 0, 0x100, 0x100, 0x1000)

    slot = 0x1200
    direct_target = 0x1080
    code = bytearray()
    site = 0x1000
    code += b"\xe8" + rel32(site, 5, direct_target)
    site += 5
    code += b"\x48\x8d\x05" + rel32(site, 7, slot)
    site += 7
    code += b"\x48\x8b\x05" + rel32(site, 7, slot)
    site += 7
    code += b"\x48\x89\x05" + rel32(site, 7, slot)
    site += 7
    code += b"\xff\x15" + rel32(site, 6, slot)
    site += 6
    code += b"\xff\x25" + rel32(site, 6, slot)
    raw[0x100:0x100 + len(code)] = code

    path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name
            f.write(raw)
        module = XREF.Module(path)
        expected = {
            (0x1005, "lea"),
            (0x100c, "load"),
            (0x1013, "store"),
            (0x101a, "call*"),
            (0x1020, "jmp*"),
        }
        actual = set(module.code_xref[slot])
        assert actual == expected, (actual, expected)
        assert module.code_xref[direct_target] == [(0x1000, "call")]
    finally:
        if path:
            os.unlink(path)

    print("xref decoder: PASS")


if __name__ == "__main__":
    main()
