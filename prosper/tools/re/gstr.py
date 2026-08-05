#!/usr/bin/env python3
"""gstr — print the string that lives at a guest VADDR in a flattened module image.

`xref.py from <addr>` answers "what does this function reference", and most of what it prints is
`lea -> 0x…` / `load -> 0x…` into the module's read-only data. On a stripped PS5 title those data
addresses are the *only* way to name a function: a UE4 build has no symbols, but the function that
holds `"GameThread timed out waiting for RenderThread after %.02f secs"` can only be one function.
Turning those addresses into text by hand is the slow step, and `objdump`/`readelf`/`strings` cannot
do it here — `tools/il2cpp/prx_to_elf.py` writes no section header table, so the usual tools reject
the image outright, and `strings` has no way to map a VADDR to a file offset.

gstr resolves a VADDR through the image's PT_LOAD program headers and prints what is there,
auto-detecting UTF-16LE (UE4's TCHAR) as well as ASCII.

    python3 tools/re/gstr.py <flat.elf> 0x718c6ca
    0x718c6ca    W"GameThread timed out waiting for RenderThread after %.02f secs"

The intended pipeline is to feed it every data reference of a frame's function at once:

    python3 tools/re/xref.py <flat.elf> from 0x2121d9e \
      | awk '/^   (lea|load) /{print $3}' \
      | xargs python3 tools/re/gstr.py <flat.elf> \
      | grep '"'

Flatten a module first with `python3 tools/il2cpp/prx_to_elf.py <eboot.bin> <flat.elf>`, or reuse
the images `tools/guest_bt/guest_bt.py` already caches under `tools/guest_bt/.cache/`.

Addresses are **image-relative**, matching what `xref.py` and `edis.py` print: subtract the module
load base before querying an address taken from a live backtrace.

A non-string address prints `bytes <hex>` rather than nothing, so a wrong address is visibly wrong
instead of silently absent, and an address outside every PT_LOAD prints `<unmapped>`.
"""
import argparse
import struct
import sys

MIN_LEN = 3
PT_LOAD = 1


def load_segments(path):
    """Return [(vaddr, filesz, offset)] for every PT_LOAD, plus the raw image."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"\x7fELF":
        raise SystemExit("%s: not an ELF image" % path)
    if data[4] != 2:
        raise SystemExit("%s: only ELF64 is supported" % path)
    (phoff,) = struct.unpack_from("<Q", data, 0x20)
    phentsize, phnum = struct.unpack_from("<HH", data, 0x36)
    segs = []
    for i in range(phnum):
        o = phoff + i * phentsize
        (p_type,) = struct.unpack_from("<I", data, o)
        p_offset, p_vaddr, _p_paddr, p_filesz, _p_memsz = struct.unpack_from("<QQQQQ", data, o + 8)
        if p_type == PT_LOAD:
            segs.append((p_vaddr, p_filesz, p_offset))
    if not segs:
        raise SystemExit("%s: no PT_LOAD segments" % path)
    return segs, data


def read_at(segs, data, va, n):
    for vaddr, filesz, offset in segs:
        if vaddr <= va < vaddr + filesz:
            avail = filesz - (va - vaddr)
            start = offset + (va - vaddr)
            return data[start:start + min(n, avail)]
    return b""


def as_utf16(buf):
    """UE4 TCHAR strings are UTF-16LE. Require a printable, NUL-terminated run."""
    out = []
    for i in range(0, len(buf) - 1, 2):
        unit = buf[i] | (buf[i + 1] << 8)
        if unit == 0:
            break
        if unit < 0x20 or unit > 0x7E:
            return None
        out.append(chr(unit))
    else:
        return None                      # ran off the window with no terminator
    return "".join(out) if len(out) >= MIN_LEN else None


def as_ascii(buf):
    out = []
    for byte in buf:
        if byte == 0:
            break
        if byte < 0x20 or byte > 0x7E:
            return None
        out.append(chr(byte))
    else:
        return None
    return "".join(out) if len(out) >= MIN_LEN else None


def describe(segs, data, va, window):
    buf = read_at(segs, data, va, window)
    if not buf:
        return "<unmapped>"
    wide = as_utf16(buf)
    if wide is not None:
        return 'W"%s"' % wide
    narrow = as_ascii(buf)
    if narrow is not None:
        return 'A"%s"' % narrow
    return "bytes " + buf[:24].hex()


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="print the string at an image-relative guest VADDR in a flattened module image")
    ap.add_argument("image", help="flattened module ELF (prx_to_elf.py output, or a guest_bt cache entry)")
    ap.add_argument("addr", nargs="*", help="image-relative addresses (0x… or decimal); '-' reads stdin")
    ap.add_argument("--window", type=lambda s: int(s, 0), default=512,
                    help="maximum bytes to consider per address (default 512)")
    ap.add_argument("--strings-only", action="store_true",
                    help="print only addresses that decode to a string")
    args = ap.parse_args(argv)

    words = args.addr
    if not words or words == ["-"]:
        words = sys.stdin.read().split()
    try:
        addrs = [int(w, 0) for w in words]
    except ValueError as exc:
        raise SystemExit("bad address: %s" % exc)

    segs, data = load_segments(args.image)
    for va in addrs:
        text = describe(segs, data, va, args.window)
        if args.strings_only and not text.startswith(('W"', 'A"')):
            continue
        print("0x%-12x %s" % (va, text))
    return 0


if __name__ == "__main__":
    sys.exit(main())
