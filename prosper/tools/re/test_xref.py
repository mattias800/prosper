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
    site += 7
    # #2025: four forms that were not decoded, and every one of them occurs around a std::vector's
    # _M_finish. The `add [rip+d],imm8` is push_back/pop_back advancing the pointer; while it was
    # missed, "who touches this vector" answered with reads only and looked complete.
    code += b"\x48\x2b\x05" + rel32(site, 7, slot)                       # sub rax,[rip+d]     r
    site += 7
    code += b"\x48\x3b\x05" + rel32(site, 7, slot)                       # cmp rax,[rip+d]     r
    site += 7
    code += b"\x4c\x3b\x35" + rel32(site, 7, slot)                       # cmp r14,[rip+d]     r
    site += 7
    code += b"\x48\x83\x05" + rel32(site, 8, slot) + b"\x08"             # add [rip+d],8      rw
    site += 8
    code += b"\x48\x83\x2d" + rel32(site, 8, slot) + b"\x08"             # sub [rip+d],8      rw
    site += 8
    code += b"\x48\x83\x3d" + rel32(site, 8, slot) + b"\x00"             # cmp [rip+d],0       r
    site += 8
    code += b"\xc5\xfc\x11\x05" + rel32(site, 8, slot)                   # vmovups [rip+d],ymm w
    site += 8
    code += b"\xc5\xf8\x10\x05" + rel32(site, 8, slot)                   # vmovups xmm,[rip+d] r
    site += 8
    # High registers (ymm8-15 / xmm8-15) invert VEX.R, giving c5 7c / c5 78. Pinning R=1 missed
    # them: 876 of 14,722 such instructions on a real eboot, 6.0% (#2025 review). Encodings that
    # differ only in R must decode to the SAME kind -- the register number is not the reference.
    code += b"\xc5\x7c\x11\x05" + rel32(site, 8, slot)   # vmovups [rip+d],ymm8+  w
    site += 8
    code += b"\xc5\x78\x10\x05" + rel32(site, 8, slot)   # vmovups xmm8+,[rip+d]  r
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
            # #2025
            (0x103e, "sub"),
            (0x1045, "cmp"),
            (0x104c, "cmp"),
            (0x1053, "addi"),
            (0x105b, "subi"),
            (0x1063, "cmpi"),
            (0x106b, "vstorey"),
            (0x1073, "vloadx"),
            (0x107b, "vstorey"),   # c5 7c: ymm8-15, same kind as c5 fc
            (0x1083, "vloadx"),    # c5 78: xmm8-15, same kind as c5 f8
        }
        actual = set(module.code_xref[slot])
        assert actual == expected, (actual, expected)

        # The classification is the other half of #2025: the tool's usual question is "who WRITES
        # this", and the four missed forms included the only writers. Assert the classes rather than
        # merely that the forms decode -- a form decoded into the wrong class is still a wrong answer
        # to that question.
        by_kind = {k: XREF.access(k) for _, k in actual}
        assert by_kind["addi"] == "rw" and by_kind["subi"] == "rw", by_kind
        assert by_kind["vstorey"] == "w" and by_kind["vloadx"] == "r", by_kind
        assert by_kind["sub"] == "r" and by_kind["cmp"] == "r" and by_kind["cmpi"] == "r", by_kind
        # Nothing may fall through to the unknown class: an unclassified kind would print '?' and
        # silently drop out of the write/read counts the summary line reports.
        unknown = sorted(k for _, k in actual if XREF.access(k) == "?")
        assert not unknown, unknown
        # The summary this drives: writers must be visible. Asserted as the exact SET rather than a
        # count -- a count is satisfied by the wrong six kinds, and the point of #2025 is precisely
        # which forms land in this bucket. `addi`/`subi`/`vstorey` are the ones that were missing.
        writers = {k for _, k in actual if XREF.access(k) in ("w", "rw")}
        assert writers == {"store", "storeb", "stored", "addi", "subi", "vstorey"}, sorted(writers)
        # A VEX form differing only in the R bit must not decode to a different kind or width:
        # the referenced address does not depend on which register the instruction uses.
        vex = sorted(k for _, k in actual if k.startswith("v"))
        assert vex == ["vloadx", "vloadx", "vstorey", "vstorey"], vex
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


def refusal():
    """#2346: inputs this tool cannot answer must be REFUSED, never answered with zeros.

    Every arm here asserts an exception. That is the point: before the fix, each of these inputs
    constructed a Module without complaint and then reported `0 references` for every address --
    output that is byte-identical to a true negative, which is the answer people run this tool to
    establish. The arms are written so that a regression re-opens as a FAILURE to raise, rather than
    as a wrong number nobody notices.
    """
    def build(prefix, phentsize=56, phnum=1, seg_flags=5, body=b"\x90" * 0x40):
        raw = bytearray(0x200)
        raw[:4] = prefix
        struct.pack_into("<Q", raw, 0x20, 0x40)
        struct.pack_into("<HH", raw, 0x36, phentsize, phnum)
        struct.pack_into("<IIQQQQQQ", raw, 0x40,
                         1, seg_flags, 0x100, 0x1000, 0, len(body), len(body), 0x1000)
        raw[0x100:0x100 + len(body)] = body
        fd, path = tempfile.mkstemp(suffix=".bin")
        os.write(fd, bytes(raw))
        os.close(fd)
        return path

    def refuses(path, why, must_mention=None):
        try:
            XREF.Module(path)
        except XREF.BadModule as exc:
            if must_mention and must_mention not in str(exc):
                raise AssertionError("%s: message must mention %r, got: %s" % (why, must_mention, exc))
        else:
            raise AssertionError("%s: constructed a Module instead of refusing" % why)

    paths = []
    try:
        # The reported case: a real SELF starts with this container magic, and its bytes at 0x20/0x36
        # decode as e_phoff=0x110004, phentsize=0, phnum=45792 -- which is why the old code built an
        # empty segment table rather than crashing. The message must name the flattening step,
        # because "refused" without the remedy just moves the dead end.
        p = build(b"\x4f\x15\x3d\x1d"); paths.append(p)
        refuses(p, "a SELF container", must_mention="prx_to_elf.py")

        # Not an ELF and not a SELF either -- must still refuse rather than parse whatever is at 0x20.
        p = build(b"MZ\x90\x00"); paths.append(p)
        refuses(p, "a non-ELF file")

        # phentsize=0 is what a SELF actually yields, and it makes the header loop re-read one offset
        # phnum times. Covered on its own so the guard survives someone relaxing the magic check.
        p = build(b"\x7fELF", phentsize=0); paths.append(p)
        refuses(p, "e_phentsize of 0")

        # A well-formed ELF whose only PT_LOAD is non-executable. Every form this tool decodes lives
        # in an executable segment, so it could only ever answer 0.
        p = build(b"\x7fELF", seg_flags=6); paths.append(p)
        refuses(p, "no executable segment")

        # The backstop, and the arm that matters most: this file passes every structural check above
        # -- real ELF magic, sane phentsize, an executable PT_LOAD -- and still yields no references,
        # because its "code" is all single-byte nops. A module that decodes to nothing has failed
        # whatever its header claimed, so the tool must detect that about itself rather than trust
        # that the checks above were exhaustive.
        #
        # It must be MODULE-SIZED to be refused, and that is not incidental. The backstop's premise
        # ("no references at all means the decoder failed") is false for a small ELF, and this test
        # is how that was found: an earlier version had no threshold and refused the `imm` fixture
        # above, which is 256 bytes of legitimately reference-free executable segment.
        big = 0x10000 + 0x40
        p = build(b"\x7fELF", body=b"\x90" * big); paths.append(p)
        refuses(p, "a module-sized ELF with zero decodable references", must_mention="ZERO references")

        # The negative half of that threshold: the SAME reference-free body, below module scale, must
        # be ACCEPTED. Otherwise the guard has just moved the failure from a silent wrong answer to a
        # loud wrong refusal.
        p = build(b"\x7fELF", body=b"\x90" * 0x40); paths.append(p)
        XREF.Module(p)

        # ...and the control for the backstop itself, built by hand rather than drawn from the same
        # generator as the nulls (charter: a same-source control tests the discriminator, not the
        # domain). ONE direct call in an otherwise module-sized nop field is enough to make it
        # plausible, and it must be accepted. Without this arm the backstop could refuse every
        # large module and still pass every assertion above.
        body = bytearray(b"\x90" * big)
        body[0:5] = b"\xe8" + rel32(0x1000, 5, 0x1020)
        p = build(b"\x7fELF", body=bytes(body)); paths.append(p)
        module = XREF.Module(p)
        assert module.code_xref[0x1020] == [(0x1000, "call")], dict(module.code_xref)
    finally:
        for p in paths:
            os.unlink(p)

    print("xref refusal: PASS")


if __name__ == "__main__":
    main()
    imm()
    refusal()
