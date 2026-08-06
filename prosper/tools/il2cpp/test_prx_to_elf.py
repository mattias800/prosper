#!/usr/bin/env python3
"""Header regression for prx_to_elf.py — the flattened ELF's own bytes, at their ELF64 offsets.

Every assertion below reads the produced FILE at a fixed offset. Nothing here trusts a value the
converter returns or prints, because the whole class of bug this guards (#2016: the program-header
count written into the section-header-count slot) is invisible to anything that does not look at
where the bytes landed.

The synthetic SELF is built so each expected value is distinct from every wrong value a plausible
mutation would leave behind. Each check names the mutation it kills.
"""

import os
import struct
import subprocess
import sys
import tempfile


HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "prx_to_elf.py")

# ELF64 header field offsets (System V gABI). The two that #2016 confused are 0x38 and 0x3c.
E_IDENT, E_TYPE, E_PHOFF, E_SHOFF = 0x00, 0x10, 0x20, 0x28
E_EHSIZE, E_PHENTSIZE, E_PHNUM = 0x34, 0x36, 0x38
E_SHENTSIZE, E_SHNUM, E_SHSTRNDX = 0x3a, 0x3c, 0x3e

PT_LOAD, PT_NOTE = 1, 4

# Deliberately distinct so no two can be confused for one another in a failure message:
SRC_PHNUM = 4          # program headers in the synthetic module
SRC_SHNUM = 0x2a       # stale section count in the source header
SRC_SHSTRNDX = 0x2b    # stale .shstrtab index (real modules carry 0x29-0x2e; 22 checked for #2016)
SRC_SHOFF = 0x1232D528  # stale, past EOF — exactly what PPSA24651's IL2CPP PRX carries
SRC_ETYPE = 0xFE18     # Sony's OS-specific module type

SEG0_VADDR, SEG0_FILESZ = 0x0000, 0x100
SEG2_VADDR, SEG2_FILESZ = 0x2000, 0x080
EXPECT_SIZE = SEG2_VADDR + SEG2_FILESZ


def build_selfish_module():
    """A minimal PS5-shaped SELF: header, segment table, embedded ELF, then the segment payloads.

    Only the parts prx_to_elf.py actually reads are modelled — nseg at 0x18, a 32-byte segment
    table at 0x20, and an embedded ELF image located by its magic.
    """
    nseg = 2
    seg_table_end = 0x20 + nseg * 32
    elf_base = 0x200                      # after the segment table, magic-searchable
    payload0_off = 0x1000
    payload2_off = 0x2000

    buf = bytearray(payload2_off + SEG2_FILESZ)
    struct.pack_into("<H", buf, 0x18, nseg)
    assert seg_table_end <= elf_base

    # SELF segment entries: (flags, file_offset, file_size, mem_size). Bit 0x800 marks a data
    # segment; the target program header index is flags >> 20. Map phdr 0 and phdr 2 — phdr 1 is
    # not PT_LOAD and phdr 3 is bss-only; both must be carried through untouched.
    struct.pack_into("<QQQQ", buf, 0x20 + 0 * 32,
                     (0 << 20) | 0x800, payload0_off, SEG0_FILESZ, SEG0_FILESZ)
    struct.pack_into("<QQQQ", buf, 0x20 + 1 * 32,
                     (2 << 20) | 0x800, payload2_off, SEG2_FILESZ, SEG2_FILESZ)

    # Embedded ELF header, with the stale section-table fields a stripped PS5 module really carries.
    buf[elf_base:elf_base + 16] = bytes([0x7F, ord('E'), ord('L'), ord('F'), 2, 1, 1, 9, 2,
                                         0, 0, 0, 0, 0, 0, 0])
    struct.pack_into("<H", buf, elf_base + E_TYPE, SRC_ETYPE)
    struct.pack_into("<H", buf, elf_base + 0x12, 62)          # e_machine = EM_X86_64
    struct.pack_into("<I", buf, elf_base + 0x14, 1)           # e_version
    struct.pack_into("<Q", buf, elf_base + E_PHOFF, 0x40)
    struct.pack_into("<Q", buf, elf_base + E_SHOFF, SRC_SHOFF)
    struct.pack_into("<H", buf, elf_base + E_EHSIZE, 64)
    struct.pack_into("<H", buf, elf_base + E_PHENTSIZE, 56)
    struct.pack_into("<H", buf, elf_base + E_PHNUM, SRC_PHNUM)
    struct.pack_into("<H", buf, elf_base + E_SHENTSIZE, 64)
    struct.pack_into("<H", buf, elf_base + E_SHNUM, SRC_SHNUM)
    struct.pack_into("<H", buf, elf_base + E_SHSTRNDX, SRC_SHSTRNDX)

    # Program headers: p_type, p_flags, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz, p_align.
    # p_offset is deliberately unequal to p_vaddr — making them equal is the converter's job.
    # phdr 1 is a non-PT_LOAD header (PT_NOTE rather than PT_DYNAMIC only so binutils will parse
    # the synthetic image at all — the converter treats every non-PT_LOAD type identically).
    # phdr 3 is a bss-only PT_LOAD: carried through, but it must not extend the written image.
    phdrs = [
        (PT_LOAD, 5, 0x0400, SEG0_VADDR, 0x11, SEG0_FILESZ, SEG0_FILESZ, 0x1000),
        (PT_NOTE, 6, 0x0500, 0x1500,     0x22, 0x40,        0x40,        8),
        (PT_LOAD, 6, 0x0600, SEG2_VADDR, 0x33, SEG2_FILESZ, SEG2_FILESZ, 0x1000),
        (PT_LOAD, 6, 0x0700, 0x3000,     0x44, 0,           0x100,       0x1000),
    ]
    assert len(phdrs) == SRC_PHNUM
    for i, p in enumerate(phdrs):
        struct.pack_into("<IIQQQQQQ", buf, elf_base + 0x40 + i * 56, *p)

    # Payload bytes, distinguishable per segment.
    buf[payload0_off:payload0_off + SEG0_FILESZ] = bytes([0xA5]) * SEG0_FILESZ
    buf[payload2_off:payload2_off + SEG2_FILESZ] = bytes([0x5A]) * SEG2_FILESZ
    return bytes(buf), phdrs


def u16(b, off):
    return struct.unpack_from("<H", b, off)[0]


def u64(b, off):
    return struct.unpack_from("<Q", b, off)[0]


def check(ok, label, kills):
    if not ok:
        raise AssertionError("%s  [regression for: %s]" % (label, kills))


def main():
    src_bytes, phdrs = build_selfish_module()
    # ctest hands every case a build-tree scratch root (#1613) precisely so tests stay off this
    # machine's tmpfs; fall back to the system temp dir when run standalone.
    scratch = os.environ.get("PROSPER_TEST_SCRATCH_DIR")
    if scratch:
        os.makedirs(scratch, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="prx_to_elf_test_", dir=scratch or None)
    src = os.path.join(tmp, "synthetic.prx")
    dst = os.path.join(tmp, "synthetic.elf")
    with open(src, "wb") as fh:
        fh.write(src_bytes)

    proc = subprocess.run([sys.executable, TOOL, src, dst],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise AssertionError("prx_to_elf.py failed (%d): %s"
                             % (proc.returncode, proc.stdout.decode("utf-8", "replace")))

    with open(dst, "rb") as fh:
        out = fh.read()

    check(len(out) == EXPECT_SIZE,
          "image size %#x, want %#x" % (len(out), EXPECT_SIZE),
          "image sized from something other than max(p_vaddr + p_filesz) over PT_LOAD")

    check(out[0:4] == b"\x7fELF", "no ELF magic at offset 0",
          "the verbatim 64-byte header copy going missing")

    # --- the #2016 pair. These two offsets are four bytes apart and mean unrelated things. ---
    check(u16(out, E_PHNUM) == SRC_PHNUM,
          "e_phnum@0x38 = %d, want %d" % (u16(out, E_PHNUM), SRC_PHNUM),
          "e_phnum zeroed or clobbered — including a 'fix' that moves the zero write to 0x38")
    check(u16(out, E_SHNUM) == 0,
          "e_shnum@0x3c = %d, want 0" % u16(out, E_SHNUM),
          "#2016: e_phnum written into e_shnum (would read %d), or the write dropped entirely "
          "(would read %d)" % (SRC_PHNUM, SRC_SHNUM))
    check(u16(out, E_SHSTRNDX) == 0,
          "e_shstrndx@0x3e = %d, want 0" % u16(out, E_SHSTRNDX),
          "the source module's stale .shstrtab index (%d) surviving the copy — binutils reports "
          "'<corrupt: out of range>' and refuses the file" % SRC_SHSTRNDX)
    check(u64(out, E_SHOFF) == 0,
          "e_shoff@0x28 = %#x, want 0" % u64(out, E_SHOFF),
          "the source module's stale section-table offset (%#x, past EOF) surviving" % SRC_SHOFF)

    # --- fields the three above have to stay consistent with ---
    check(u64(out, E_PHOFF) == 0x40,
          "e_phoff@0x20 = %#x, want 0x40" % u64(out, E_PHOFF),
          "phdr table relocated in the header but not in the file, or vice versa")
    check(u16(out, E_TYPE) == 3,
          "e_type@0x10 = %#x, want 3 (ET_DYN)" % u16(out, E_TYPE),
          "Sony's OS-specific e_type (%#x) reaching a consumer that only knows ET_DYN" % SRC_ETYPE)
    check(u16(out, E_EHSIZE) == 64 and u16(out, E_PHENTSIZE) == 56,
          "e_ehsize/e_phentsize = %d/%d, want 64/56" % (u16(out, E_EHSIZE), u16(out, E_PHENTSIZE)),
          "a header/phdr stride that disagrees with the 0x40 + i*56 layout actually written")
    # e_shentsize is intentionally NOT rewritten: with e_shoff == 0 and e_shnum == 0 it describes
    # nothing, and binutils accepts the file either way (checked with 0 and 64 on the real
    # PPSA24651 module). Pinned so a future change to it is a deliberate one.
    check(u16(out, E_SHENTSIZE) == 64,
          "e_shentsize@0x3a = %d, want 64 (carried through)" % u16(out, E_SHENTSIZE),
          "an undiscussed change to e_shentsize")

    # --- program headers: p_offset must equal p_vaddr, everything else carried through ---
    for i, p in enumerate(phdrs):
        got = struct.unpack_from("<IIQQQQQQ", out, 0x40 + i * 56)
        want = (p[0], p[1], p[3], p[3], p[4], p[5], p[6], p[7])
        check(got == want,
              "phdr[%d] = %r, want %r" % (i, got, want),
              "p_offset not forced to p_vaddr, or a non-PT_LOAD phdr dropped/reordered")

    # --- payload placement: each mapped SELF data segment must land at its p_vaddr ---
    # The rewritten ELF header + phdr table deliberately overwrite the head of the first segment
    # (they occupy the same addresses in the source module), so check from just past the table.
    hdr_end = 0x40 + SRC_PHNUM * 56
    check(out[hdr_end:SEG0_VADDR + SEG0_FILESZ] == bytes([0xA5]) * (SEG0_FILESZ - hdr_end),
          "phdr-0 payload missing at its p_vaddr",
          "the SELF flags>>20 -> phdr index mapping breaking")
    check(out[SEG2_VADDR:SEG2_VADDR + SEG2_FILESZ] == bytes([0x5A]) * SEG2_FILESZ,
          "phdr-2 payload missing at its p_vaddr",
          "a non-zero phdr index in the SELF segment table being mis-decoded")

    # --- differential arm: binutils is the consumer #2016 was actually reported against. ---
    # This arm is only meaningful if the installed binutils can tell the two headers apart, so it
    # carries its own control: the same image with the #2016 header written back in. A run where
    # binutils accepts or rejects BOTH proves nothing about our fix and is reported VOID rather
    # than passed or failed — an arm that cannot show its lever moved is not evidence.
    control = bytearray(out)
    struct.pack_into("<H", control, E_SHNUM, SRC_PHNUM)        # what master wrote here
    struct.pack_into("<H", control, E_SHSTRNDX, SRC_SHSTRNDX)  # left over from the source module
    ctl = os.path.join(tmp, "control-2016.elf")
    with open(ctl, "wb") as fh:
        fh.write(control)

    def objdump_ok(path):
        try:
            r = subprocess.run(["objdump", "-f", path],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except OSError:
            return None, ""
        return (r.returncode == 0 and
                b"file format elf64-x86-64" in r.stdout), r.stdout.decode("utf-8", "replace")

    fixed_ok, fixed_text = objdump_ok(dst)
    control_ok, _ = objdump_ok(ctl)
    if fixed_ok is None:
        print("objdump arm: VOID — binutils not on PATH; the byte checks above are the gate")
    else:
        check(fixed_ok,
              "objdump rejected the flattened image: %s" % fixed_text.strip(),
              "any header inconsistency binutils refuses outright")
        if control_ok:
            print("objdump arm: VOID — this binutils accepts the #2016 header too, so acceptance "
                  "of ours discriminates nothing; the byte checks above are the gate")
        else:
            print("objdump arm: RAN and DISCRIMINATED — accepts the fixed header, "
                  "rejects the #2016 header")

    for f in (src, dst, ctl):
        os.unlink(f)
    os.rmdir(tmp)
    print("prx_to_elf header regression: PASS")


if __name__ == "__main__":
    main()
