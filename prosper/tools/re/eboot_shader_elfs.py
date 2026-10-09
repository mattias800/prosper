#!/usr/bin/env python3
"""eboot_shader_elfs — find and extract nested AMDGPU shader ELFs from a PS5 eboot.

Some PS5 games embed small shader ELF images in their eboot's read-only data: an ELF
with e_machine == 0xE0 (EM_AMDGPU) carrying `.shader_header` and `.shader_text`
sections, passed to sceAgcCreateShader(header, code). This tool scans a raw eboot for
those images, pulls out both sections, and writes a local-only store plus a
hashes-only manifest. It is the extractor; validation/consumption of the raw code
bytes is a separate step.

Usage: python eboot_shader_elfs.py --eboot <file> --title-id <PPSA...> [--out <dir>]
       [--manifest <file>] [--no-llvm]

The optional llvm-mc decode runs `llvm-mc` from PATH when present, else through WSL Ubuntu
(Windows hosts); with neither, records are simply left unvalidated.

Output (all outside the repo, never committed; default store is
~/prosper-work/eboot_shaders/<title_id>): each nested ELF at
<out>/<sha256-of-elf>.elf (identical ELFs share one file) and each raw
`.shader_text` at <out>/<text_sha256>.bin. The manifest is JSON lines, one record
per candidate hit (UTF-8, no host paths or drive letters).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

ELF_MAGIC = b"\x7fELF"
EI_CLASS_64 = 2
EI_DATA_LSB = 1
EM_AMDGPU = 0xE0
S_ENDPGM = bytes((0x00, 0x00, 0x81, 0xBF))
SHDR_SIZE_64 = 64
SHT_NOBITS = 8
SHADER_HEADER_NAME = ".shader_header"
SHADER_TEXT_NAME = ".shader_text"

_EHDR = struct.Struct("<16sHHIQQQIHHHHHH")
_SHDR = struct.Struct("<IIQQQQIIQQ")
_TITLE_ID_RE = re.compile(r"^PPSA\d{5}$")
_LLVM_BAD_MARKER = "invalid instruction encoding"


class MalformedElf(Exception):
    """A candidate hit that does not parse; counted as rejected, never raised out."""


def find_candidates(data: bytes):
    """Yield file offsets of every ELF64-LE header with e_machine == EM_AMDGPU."""
    pos = 0
    size = len(data)
    while True:
        hit = data.find(ELF_MAGIC, pos)
        if hit < 0 or hit + 0x14 > size:
            return
        if data[hit + 4] == EI_CLASS_64 and data[hit + 5] == EI_DATA_LSB:
            (machine,) = struct.unpack_from("<H", data, hit + 0x12)
            if machine == EM_AMDGPU:
                yield hit
        pos = hit + 1


def _cstr(table: bytes, at: int) -> str:
    end = table.find(b"\x00", at)
    if end < 0:
        raise MalformedElf(f"section name at {at} is unterminated")
    return table[at:end].decode("ascii", errors="strict")


def parse_nested_elf(data: bytes, off: int):
    """Parse the nested ELF at off.

    Returns (sections, elf_size) where sections maps section name to the raw
    section bytes. Raises MalformedElf on any out-of-bounds or incoherent field.
    """
    if off + _EHDR.size > len(data):
        raise MalformedElf("truncated ELF header")
    (
        _ident,
        _type,
        _machine,
        _ver,
        _entry,
        _phoff,
        shoff,
        _flags,
        _ehsize,
        _phentsize,
        _phnum,
        shentsize,
        shnum,
        shstrndx,
    ) = _EHDR.unpack_from(data, off)
    if shnum == 0:
        return {}, _EHDR.size
    if shentsize < SHDR_SIZE_64:
        raise MalformedElf(f"e_shentsize {shentsize} < {SHDR_SIZE_64}")
    table_end = shoff + shnum * shentsize
    if off + table_end > len(data):
        raise MalformedElf("section header table runs past end of file")
    if shstrndx >= shnum:
        raise MalformedElf(f"e_shstrndx {shstrndx} >= e_shnum {shnum}")
    raw_shdrs = []
    for i in range(shnum):
        entry = off + shoff + i * shentsize
        (name_off, _shtype, _flags, _addr, sh_offset, sh_size, _link, _info, _align, _entsize) = (
            _SHDR.unpack_from(data, entry)
        )
        raw_shdrs.append((name_off, _shtype, sh_offset, sh_size))
    (_, str_type, str_data_off, str_size) = raw_shdrs[shstrndx]
    if str_type == SHT_NOBITS or str_size == 0:
        raise MalformedElf("section-name string table is empty or NOBITS")
    if off + str_data_off + str_size > len(data):
        raise MalformedElf("section-name string table runs past end of file")
    strtab = data[off + str_data_off : off + str_data_off + str_size]
    sections: dict[str, bytes] = {}
    data_end = max(_EHDR.size, table_end)
    for name_off, shtype, sh_offset, sh_size in raw_shdrs:
        if name_off >= len(strtab):
            raise MalformedElf("section name offset outside string table")
        try:
            name = _cstr(strtab, name_off)
        except UnicodeDecodeError as exc:
            raise MalformedElf(f"section name is not ASCII: {exc}") from exc
        if sh_size == 0:
            if name and name not in sections:
                sections[name] = b""
            continue
        if shtype != SHT_NOBITS:
            if off + sh_offset + sh_size > len(data):
                raise MalformedElf(f"section {name!r} data runs past end of file")
            data_end = max(data_end, sh_offset + sh_size)
            if name and name not in sections:
                sections[name] = data[off + sh_offset : off + sh_offset + sh_size]
        elif name and name not in sections:
            sections[name] = b""
    return sections, data_end


def _llvm_command():
    """argv prefix that runs llvm-mc, or None when it is unreachable on this host.

    A native `llvm-mc` on PATH wins (Linux/macOS); otherwise a Windows host reaches it
    through WSL Ubuntu. Anything else degrades to "unvalidated", never to an error.
    """
    native = shutil.which("llvm-mc")
    if native is not None:
        return [native]
    if shutil.which("wsl") is not None:
        return ["wsl", "-d", "Ubuntu", "--", "llvm-mc"]
    return None


def llvm_available() -> bool:
    """True when llvm-mc is reachable (native or through WSL Ubuntu); probed per call."""
    cmd = _llvm_command()
    if cmd is None:
        return False
    try:
        proc = subprocess.run([*cmd, "--version"], capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def llvm_invalid_count(code: bytes):
    """Disassemble code as gfx1030; return the invalid-instruction count.

    Returns None when llvm-mc is unreachable or the decode itself fails, so a
    broken instrument reads as unvalidated rather than as clean or dirty.
    """
    cmd = _llvm_command()
    if cmd is None:
        return None
    tokens = " ".join(f"0x{b:02x}" for b in code)
    try:
        proc = subprocess.run(
            [*cmd, "--disassemble", "-triple=amdgcn-amd-amdhsa", "-mcpu=gfx1030"],
            input=tokens,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return (proc.stdout + proc.stderr).count(_LLVM_BAD_MARKER)


def extract(eboot: Path, title_id: str, out_dir: Path, manifest_path: Path, use_llvm: bool) -> dict:
    """Scan eboot, write the store + manifest, return the status counts."""
    data = eboot.read_bytes()
    out_dir.mkdir(parents=True, exist_ok=True)
    want_llvm = use_llvm and llvm_available()
    if use_llvm and not want_llvm:
        print("note: llvm-mc unreachable, all records unvalidated", file=sys.stderr)
    counts = {"ok": 0, "unvalidated": 0, "missing-section": 0, "rejected": 0}
    records = []
    for off in find_candidates(data):
        try:
            sections, elf_size = parse_nested_elf(data, off)
        except MalformedElf as exc:
            print(f"rejected candidate at {off}: {exc}", file=sys.stderr)
            counts["rejected"] += 1
            records.append(
                {
                    "source": "eboot",
                    "title_id": title_id,
                    "file_offset": off,
                    "elf_size": 0,
                    "sections": [],
                    "header_size": 0,
                    "text_size": 0,
                    "sha256": None,
                    "text_sha256": None,
                    "ends_with_s_endpgm": False,
                    "llvm_invalid": None,
                    "status": "rejected",
                }
            )
            continue
        elf = data[off : off + elf_size]
        digest = hashlib.sha256(elf).hexdigest()
        elf_path = out_dir / f"{digest}.elf"
        if not elf_path.exists():
            elf_path.write_bytes(elf)
        header = sections.get(SHADER_HEADER_NAME)
        text = sections.get(SHADER_TEXT_NAME)
        present = [n for n in (SHADER_HEADER_NAME, SHADER_TEXT_NAME) if n in sections]
        text_digest = None
        if text is not None:
            text_digest = hashlib.sha256(text).hexdigest()
            text_path = out_dir / f"{text_digest}.bin"
            if not text_path.exists():
                text_path.write_bytes(text)
        ends = text is not None and len(text) >= 4 and text[-4:] == S_ENDPGM
        if header is None or text is None:
            status = "missing-section"
            invalid = None
        elif not want_llvm:
            status = "unvalidated"
            invalid = None
        else:
            invalid = llvm_invalid_count(text if text is not None else b"")
            status = "unvalidated" if invalid is None else "ok"
        counts[status] += 1
        records.append(
            {
                "source": "eboot",
                "title_id": title_id,
                "file_offset": off,
                "elf_size": elf_size,
                "sections": present,
                "header_size": len(header) if header is not None else 0,
                "text_size": len(text) if text is not None else 0,
                "sha256": digest,
                "text_sha256": text_digest,
                "ends_with_s_endpgm": ends,
                "llvm_invalid": invalid,
                "status": status,
            }
        )
    with manifest_path.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    return {"candidates": len(records), "counts": counts, "manifest": str(manifest_path)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--eboot", required=True, help="path to the game eboot.bin")
    parser.add_argument("--title-id", required=True, help="title id, e.g. PPSA28183")
    parser.add_argument(
        "--out",
        default=None,
        help="store directory (default ~/prosper-work/eboot_shaders/<title_id>; keep it "
        "outside the repo, extracted game content is never committed)",
    )
    parser.add_argument(
        "--manifest", default=None, help="manifest path (default <out>/manifest.jsonl)"
    )
    parser.add_argument(
        "--no-llvm",
        action="store_true",
        help="skip the llvm-mc decode (records become unvalidated)",
    )
    args = parser.parse_args(argv)
    if not _TITLE_ID_RE.match(args.title_id):
        parser.error("--title-id must look like PPSA28183")
    eboot = Path(args.eboot)
    if not eboot.is_file():
        parser.error(f"--eboot not found: {args.eboot}")
    out_dir = (
        Path(args.out)
        if args.out
        else Path.home() / "prosper-work" / "eboot_shaders" / args.title_id
    )
    manifest_path = Path(args.manifest) if args.manifest else out_dir / "manifest.jsonl"
    result = extract(eboot, args.title_id, out_dir, manifest_path, use_llvm=not args.no_llvm)
    counts = result["counts"]
    print(
        f"eboot {eboot.stat().st_size} bytes: "
        f"{result['candidates']} candidates "
        f"(ok={counts['ok']} unvalidated={counts['unvalidated']} "
        f"missing-section={counts['missing-section']} rejected={counts['rejected']})"
    )
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
