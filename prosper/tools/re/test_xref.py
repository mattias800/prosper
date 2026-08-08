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
    site += 6
    code += b"\xc6\x05" + rel32(site, 7, slot) + b"\x01"                  # mov BYTE [rip+d],1  storeb
    site += 7
    code += b"\xc7\x05" + rel32(site, 10, slot) + b"\x00\x00\x00\x00"    # mov DWORD [rip+d],0 stored
    site += 10
    code += b"\x80\x3d" + rel32(site, 7, slot) + b"\x01"                  # cmp BYTE [rip+d],1  cmpb
    raw[0x100:0x100 + len(code)] = code

    # #1314: a seven-byte reference ending exactly at the executable PT_LOAD boundary must be
    # inspected. The old range(n - 7) stopped one byte before this instruction's start.
    tail_site = 0x10f9
    tail_slot = 0x1300
    raw[0x1f9:0x200] = b"\xc6\x05" + rel32(tail_site, 7, tail_slot) + b"\x01"

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
            (0x1026, "storeb"),
            (0x102d, "stored"),
            (0x1037, "cmpb"),
        }
        actual = set(module.code_xref[slot])
        assert actual == expected, (actual, expected)
        assert module.code_xref[direct_target] == [(0x1000, "call")]
        assert module.code_xref[tail_slot] == [(tail_site, "storeb")]
    finally:
        if path:
            os.unlink(path)

    print("xref decoder: PASS")


def imm():
    """`imm` finds a string materialised from instruction immediates, and only that string.

    Two executable segments' worth of arguments matter here. The positive arm is the exact shape
    clang emits for a short std::string — two overlapping 8-byte `movabs` immediates — which leaves
    the .rodata literal with no reference of any kind, so `to <literal VA>` answers 0 for a string
    the program demonstrably uses. The negative arms are what keep the answer meaningful: a string
    present only as a referenced literal must NOT be reported as immediate-built, and neither must a
    stretch of unrelated code that happens to contain scattered single bytes of the query.
    """
    raw = bytearray(0x400)
    raw[:4] = b"\x7fELF"
    struct.pack_into("<Q", raw, 0x20, 0x40)
    struct.pack_into("<HH", raw, 0x36, 56, 2)
    struct.pack_into("<IIQQQQQQ", raw, 0x40,
                     1, 5, 0x100, 0x1000, 0, 0x100, 0x100, 0x1000)     # exec  0x1000..0x1100
    struct.pack_into("<IIQQQQQQ", raw, 0x40 + 56,
                     1, 4, 0x200, 0x2000, 0, 0x100, 0x100, 0x1000)     # rodata 0x2000..0x2100

    built = b"ui/ui_startup"                    # 13 bytes: movabs [0:8] then overlapping [5:13]
    code = bytearray(b"\x90" * 0x100)
    code[0x10:0x12] = b"\x48\xba"               # movabs rdx, imm64
    code[0x12:0x1a] = built[0:8]
    code[0x1a:0x1c] = b"\x48\xbe"               # movabs rsi, imm64
    code[0x1c:0x24] = built[5:13]
    # Decoy: every byte of this string is present in a 28-byte window, but never as a contiguous
    # run of four, which is exactly the shape a loose cover rule would mistake for a construction.
    scattered = b"cfg/boot_flags"
    decoy = bytearray()
    for ch in scattered:
        decoy += bytes((ch, 0))
    code[0x80:0x80 + len(decoy)] = decoy

    # 8 + 4 + 1 — the other shape a 13-byte string arrives in, and the only arm that reaches the
    # tail-completion branch.
    tailed = b"rfl/param_hud"
    code[0x30:0x38] = tailed[0:8]
    code[0x3a:0x3e] = tailed[8:12]
    code[0x40:0x41] = tailed[12:13]

    # Two separate constructions of one string, 0xa0 apart, so they must not chain into one cluster.
    repeated = b"text/text_menu"
    code[0x50:0x58] = repeated[0:8]
    code[0x58:0x60] = repeated[6:14]
    code[0xf0:0xf8] = repeated[0:8]
    code[0xf8:0x100] = repeated[6:14]
    raw[0x100:0x200] = code

    referenced = b"bmpfont/segafont"
    raw[0x200:0x200 + len(built) + 1] = built + b"\0"
    raw[0x220:0x220 + len(referenced) + 1] = referenced + b"\0"

    path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name
            f.write(raw)
        module = XREF.Module(path)

        hits = XREF.find_immediate_builds(module, built)
        assert len(hits) == 1, hits
        first, last, parts = hits[0]
        assert (first, last) == (0x1012, 0x101c), hits
        assert [(lo, hi) for _, lo, hi in parts] == [(0, 8), (5, 13)], parts

        # The fixture emits no reference to the literal copy, which is the situation `imm` exists
        # for; this only records that the fixture models it, and is not a test of the tool.
        assert module.code_xref.get(0x2000, []) == [], module.code_xref.get(0x2000)
        assert module.data_xref.get(0x2000, []) == [], module.data_xref.get(0x2000)

        # A string that only exists as a literal must not be reported as immediate-built. This is a
        # real negative: it fails if the search ever reaches non-executable segments.
        assert XREF.find_immediate_builds(module, referenced) == []
        # ...nor may the byte-scattered decoy satisfy the cover. This is the arm with the lever:
        # relaxing the >= 4-byte window floor makes it fail.
        assert XREF.find_immediate_builds(module, scattered) == []

        # The <= 3-byte tail path: 8-byte store, 4-byte store, then a single trailing byte. Nothing
        # else in this function reaches that branch, because a decoy with no contiguous 4-byte run
        # leaves `strong` empty and returns early.
        tail_hits = XREF.find_immediate_builds(module, tailed)
        assert len(tail_hits) == 1, tail_hits
        sizes = sorted(hi - lo for _, lo, hi in tail_hits[0][2])
        assert sizes == [1, 4, 8], tail_hits

        # Two constructions of one string, further apart than the cluster span, must stay two.
        twice = XREF.find_immediate_builds(module, repeated)
        assert len(twice) == 2, twice
        assert twice[1][0] - twice[0][0] > 0x60, twice

        # Under 4 bytes the window floor cannot exist, so the query is refused rather than answered
        # with every plain occurrence of those bytes in the instruction stream.
        #
        # Every sub-4-byte length goes through the one guard, INCLUDING the empty needle (#2099).
        # b"" used to return [] and print a clean "0 constructions" -- the misleading zero this mode
        # exists to eliminate, reproduced inside the mode. The 3-byte case alone would not have
        # caught it: they were two separate code paths, and only one was guarded.
        for bad in (b"cfg", b"ab", b"a", b""):
            try:
                XREF.find_immediate_builds(module, bad)
            except ValueError:
                pass
            else:
                raise AssertionError(
                    "a %d-byte needle (%r) must be refused, not answered" % (len(bad), bad))
    finally:
        if path:
            os.unlink(path)

    print("xref imm: PASS")


if __name__ == "__main__":
    main()
    imm()
