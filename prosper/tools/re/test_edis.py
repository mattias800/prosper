#!/usr/bin/env python3
"""Regression for edis.py's ELF program-header parsing + VADDR→file-offset mapping.

The pure mapping logic (load_segments / vaddr_to_file_off) is tested against a synthetic ELF with
no objdump dependency, so it runs on every platform. The end-to-end objdump path is exercised only
where objdump is on PATH (Linux dev/CI); elsewhere it is skipped, never failed."""

import importlib.util
import os
import shutil
import struct
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = importlib.util.spec_from_file_location("prosper_edis", os.path.join(HERE, "edis.py"))
EDIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EDIS)

fails = 0


def check(cond, msg):
    global fails
    if not cond:
        print(f"FAIL: {msg}")
        fails += 1
    else:
        print(f"ok: {msg}")


def gnu_objdump():
    """Path to GNU binutils objdump, or None. edis's -b binary / -M intel / --adjust-vma flags are
    GNU-specific, so a different implementation on PATH (e.g. LLVM objdump on macOS) must be treated
    as 'unavailable' — skipped, not failed."""
    exe = shutil.which("objdump")
    if not exe:
        return None
    try:
        ver = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None
    return exe if "GNU objdump" in ver else None


def build_elf(code, p_offset, p_vaddr):
    """A minimal ELF64 with one PT_LOAD, `code` placed at file offset p_offset."""
    size = max(0x200, p_offset + len(code) + 0x10)
    raw = bytearray(size)
    raw[:4] = b"\x7fELF"
    raw[4] = 2                                        # ELFCLASS64
    struct.pack_into("<Q", raw, 0x20, 0x40)          # e_phoff
    struct.pack_into("<HH", raw, 0x36, 56, 1)        # e_phentsize, e_phnum
    struct.pack_into("<IIQQQQQQ", raw, 0x40,
                     1,          # p_type = PT_LOAD
                     5,          # p_flags = R+X
                     p_offset, p_vaddr, 0,
                     len(code),  # p_filesz
                     len(code),  # p_memsz
                     0x1000)     # p_align
    raw[p_offset:p_offset + len(code)] = code
    return bytes(raw)


def main():
    # --- Case A: FLATTENED image (p_offset == p_vaddr), as prx_to_elf.py emits. ---
    flat = build_elf(b"\x90\x90\xc3", p_offset=0x2000, p_vaddr=0x2000)
    segs = EDIS.load_segments(flat)
    check(segs == [(0x2000, 0x2000, 3)], "load_segments reads the single PT_LOAD (flattened)")
    check(EDIS.vaddr_to_file_off(segs, 0x2000) == 0x2000, "flattened: vaddr maps to equal file offset")
    check(EDIS.vaddr_to_file_off(segs, 0x2002) == 0x2002, "flattened: mid-segment vaddr maps correctly")

    # --- Case B: NORMAL mapping (p_offset != p_vaddr) — the general formula must still hold. ---
    norm = build_elf(b"\x90\x90\xc3", p_offset=0x100, p_vaddr=0x400000)
    nsegs = EDIS.load_segments(norm)
    check(EDIS.vaddr_to_file_off(nsegs, 0x400000) == 0x100,
          "non-flattened: vaddr resolves through p_offset+(vaddr-p_vaddr)")
    check(EDIS.vaddr_to_file_off(nsegs, 0x400002) == 0x102, "non-flattened: mid-segment offset correct")

    # --- Bounds: an address outside every segment is unmapped (None), not a wild offset. ---
    check(EDIS.vaddr_to_file_off(segs, 0x1FFF) is None, "vaddr just below the segment is unmapped")
    check(EDIS.vaddr_to_file_off(segs, 0x2003) is None, "vaddr just past the segment end is unmapped")

    # --- Rejects a non-ELF blob rather than mis-parsing it. ---
    try:
        EDIS.load_segments(b"not an elf" + b"\x00" * 0x100)
        check(False, "load_segments rejects a non-ELF blob")
    except ValueError:
        check(True, "load_segments rejects a non-ELF blob")

    # --- End-to-end objdump path. Runs only with GNU binutils objdump (edis's -b binary / -M intel /
    # --adjust-vma flags are GNU-specific); skipped — never failed — where objdump is absent or is a
    # different implementation (e.g. LLVM objdump on macOS CI). ---
    if gnu_objdump():
        # `c6 05 <disp32> 01` = mov BYTE PTR [rip+disp],0x1 — the shape of Bendy's SAFE-flag setter.
        setter = build_elf(b"\xc6\x05\x00\x00\x00\x00\x01\xc3", p_offset=0x3000, p_vaddr=0x3000)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".elf", delete=False) as tf:
            tf.write(setter)
            elf_path = tf.name
        try:
            out = subprocess.run(
                [sys.executable, os.path.join(HERE, "edis.py"), elf_path, "0x3000", "0x8"],
                capture_output=True, text=True)
            body = out.stdout
            check("mov" in body and "BYTE PTR" in body,
                  "end-to-end: decodes the mov-byte instruction")
            check("3000:" in body, "end-to-end: prints the real guest VADDR, not a 0-based offset")
        finally:
            os.unlink(elf_path)
    else:
        print("skip: GNU objdump not available — end-to-end disassembly path not exercised")

    if fails:
        print(f"\n{fails} FAILED")
        return 1
    print("\n== PASS ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
