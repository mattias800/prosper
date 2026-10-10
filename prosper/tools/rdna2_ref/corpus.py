"""Readers for the local-only shader corpora (never committed, never reported by name or hash).

Each reader yields ``(index, code_bytes)`` pairs. Reports built from them carry only indexes
and counts. Corpora:

* ``fw``     - firmware ``OrbShdr`` blobs (a prebuilt store of ``*.bin``, or carved straight
               from library images in a dump directory).
* ``eboot``  - raw ``*.bin`` code written by ``tools/re/eboot_shader_elfs.py``.
* ``census`` - raw ``*.bin`` compute-shader dumps (a directory; ``--refused`` selects refused ones).
"""

from __future__ import annotations

import struct
from pathlib import Path

from . import decode as D

BRANCH_OPS = {
    "S_BRANCH",
    "S_CBRANCH_SCC0",
    "S_CBRANCH_SCC1",
    "S_CBRANCH_VCCZ",
    "S_CBRANCH_VCCNZ",
    "S_CBRANCH_EXECZ",
    "S_CBRANCH_EXECNZ",
}
MARKER = b"OrbShdr"


def carve_program(data: bytes, start: int, limit: int = 1 << 20) -> bytes | None:
    """Carve one program starting at ``start``: decode linearly to an ``s_endpgm`` that no
    earlier branch jumps past. Returns None if decoding fails before that point."""
    off = start
    max_target = start
    end = min(len(data), start + limit)
    while off < end:
        try:
            insn = D.decode_one(data, off)
        except D.DecodeError:
            return None
        if insn.name in BRANCH_OPS:
            simm = insn.fields["simm16"]
            simm -= 0x10000 if simm & 0x8000 else 0
            max_target = max(max_target, off + 4 + simm * 4)
        off += insn.size
        if insn.name == "S_ENDPGM" and off >= max_target:
            return data[start:off]
    return None


def carve_orbshdr(blob: bytes, header: int = 0x40) -> list[bytes]:
    """Carve every program that follows an ``OrbShdr`` marker, ``header`` bytes later."""
    out = []
    pos = blob.find(MARKER)
    while pos != -1:
        prog = carve_program(blob, pos + header)
        if prog is None:
            # Other libraries have longer headers: find the start by decode validation.
            for h in range(0x20, 0x200, 4):
                prog = carve_program(blob, pos + h)
                if prog is not None and len(prog) >= 8:
                    break
                prog = None
        if prog:
            out.append(prog)
        pos = blob.find(MARKER, pos + 1)
    return out


def firmware_programs(root: Path, store: Path | None = None) -> list[bytes]:
    """Firmware programs: from a prebuilt ``*.bin`` store if given, else carved from ``root``."""
    progs: list[bytes] = []
    if store and store.is_dir():
        return [p.read_bytes() for p in sorted(store.glob("*.bin"))]
    seen: set[bytes] = set()
    for lib in sorted(root.rglob("libSceGnmDriver*.sprx")):
        for prog in carve_orbshdr(lib.read_bytes()):
            if prog not in seen:
                seen.add(prog)
                progs.append(prog)
    return progs


def eboot_programs(root: Path) -> list[bytes]:
    """Raw code written by ``eboot_shader_elfs.py`` (``<root>/<title>/*.bin``)."""
    return [p.read_bytes() for p in sorted(root.rglob("*.bin"))]


def census_programs(root: Path, refused: bool = False) -> list[bytes]:
    """Dumped compute shaders: ``shaders-ok/*.bin`` or ``shaders-refused/*.bin``."""
    sub = root / ("shaders-refused" if refused else "shaders-ok")
    return [p.read_bytes() for p in sorted(sub.glob("*.bin"))]


def words(data: bytes) -> list[int]:
    """Little-endian dwords of ``data`` (helper for tests and reports)."""
    n = len(data) // 4
    return list(struct.unpack_from(f"<{n}I", data))
