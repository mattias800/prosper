#!/usr/bin/env python3
"""edis — disassemble a FLATTENED PS5 module ELF at an eboot-relative offset.

The flattened module ELFs that `tools/il2cpp/prx_to_elf.py` produces (for Il2CppDumper) have
NO section header table — they target readers that only parse program headers. `objdump -d`
and `readelf` reject them outright ("file format not recognized" / "too large section header
offset"), so the usual way to look at guest code is a fiddly
`dd skip=<off> | objdump -D -b binary -m i386:x86-64 --adjust-vma=<off>` incantation that also
has to know the segment's file-offset↔vaddr mapping. This wraps that into a one-liner.

Unlike `tools/dbg/dis.sh` (which disassembles a pre-extracted `/tmp/text.bin` of ONE title), this
takes any flattened module ELF directly and resolves the byte at an eboot/guest VADDR through the
module's own PT_LOAD program headers, so the same command works for every title.

Usage:
    tools/re/edis.py <flattened.elf> <vaddr_or_offset> [length_bytes]

    <vaddr_or_offset>  eboot-relative address to disassemble (hex 0x… or decimal). For a flattened
                       ELF the guest VADDR equals the file offset, but this also handles a normal
                       PT_LOAD mapping by resolving through the containing segment.
    [length_bytes]     how many bytes to decode (default 176; hex or decimal).

Example (locate the safe-flag setter in Bendy's flattened eboot):
    tools/re/edis.py bendy_eboot.elf 0x109a680 0x70

Output is AT&T-free Intel syntax with the real guest addresses, e.g.:
     109a6e4:  c6 05 ba f1 fc 00 01   mov  BYTE PTR [rip+0xfcf1ba],0x1   # 0x20698a5
"""
import struct
import subprocess
import sys
import tempfile
import os


def load_segments(raw):
    """Return the PT_LOAD segments of an ELF64 as (p_offset, p_vaddr, p_filesz) tuples."""
    if raw[:4] != b"\x7fELF":
        raise ValueError("not an ELF (bad magic) — pass a flattened module ELF from prx_to_elf.py")
    if raw[4] != 2:
        raise ValueError("not ELF64")
    e_phoff = struct.unpack_from("<Q", raw, 0x20)[0]
    e_phentsize = struct.unpack_from("<H", raw, 0x36)[0]
    e_phnum = struct.unpack_from("<H", raw, 0x38)[0]
    segs = []
    for i in range(e_phnum):
        base = e_phoff + i * e_phentsize
        if base + 56 > len(raw):
            break
        p_type = struct.unpack_from("<I", raw, base)[0]
        if p_type != 1:  # PT_LOAD
            continue
        p_offset = struct.unpack_from("<Q", raw, base + 0x08)[0]
        p_vaddr = struct.unpack_from("<Q", raw, base + 0x10)[0]
        p_filesz = struct.unpack_from("<Q", raw, base + 0x20)[0]
        segs.append((p_offset, p_vaddr, p_filesz))
    return segs


def vaddr_to_file_off(segs, vaddr):
    """Map a guest VADDR to a file offset via the containing PT_LOAD segment (None if unmapped)."""
    for p_offset, p_vaddr, p_filesz in segs:
        if p_vaddr <= vaddr < p_vaddr + p_filesz:
            return p_offset + (vaddr - p_vaddr)
    return None


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 1
    path = argv[1]
    vaddr = int(argv[2], 0)
    length = int(argv[3], 0) if len(argv) > 3 else 176
    if length <= 0:
        print("edis: length must be positive", file=sys.stderr)
        return 2

    with open(path, "rb") as f:
        raw = f.read()

    segs = load_segments(raw)
    if not segs:
        print("edis: no PT_LOAD segments — not a mappable module image", file=sys.stderr)
        return 2
    file_off = vaddr_to_file_off(segs, vaddr)
    if file_off is None:
        lo = min(v for _, v, _ in segs)
        hi = max(v + s for _, v, s in segs)
        print(f"edis: 0x{vaddr:x} is not in any PT_LOAD segment "
              f"(mapped range 0x{lo:x}..0x{hi:x})", file=sys.stderr)
        return 2

    avail = len(raw) - file_off
    if avail <= 0:
        print(f"edis: file offset 0x{file_off:x} past end of file (0x{len(raw):x})", file=sys.stderr)
        return 2
    if length > avail:
        length = avail  # clamp to what the file actually holds; still disassemble the rest

    chunk = raw[file_off:file_off + length]
    tmp = tempfile.NamedTemporaryFile(prefix="edis-", suffix=".bin", delete=False)
    try:
        tmp.write(chunk)
        tmp.close()
        try:
            out = subprocess.run(
                ["objdump", "-D", "-b", "binary", "-m", "i386:x86-64", "-M", "intel",
                 f"--adjust-vma=0x{vaddr:x}", tmp.name],
                capture_output=True, text=True, check=True).stdout
        except FileNotFoundError:
            print("edis: objdump not found on PATH", file=sys.stderr)
            return 3
        except subprocess.CalledProcessError as e:
            sys.stderr.write(e.stderr)
            return e.returncode or 3
    finally:
        os.unlink(tmp.name)

    # Keep only the instruction lines (objdump prints a file-format banner + a ".data" section header).
    for line in out.splitlines():
        stripped = line.lstrip()
        if stripped[:1].isalnum() and ":\t" in line:
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
