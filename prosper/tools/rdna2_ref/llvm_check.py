"""Decode cross-check of the independent decoder against LLVM's AMDGPU disassembler.

LLVM runs only inside WSL. The bytes are assembled as ``.byte`` data by ``llvm-mc`` and
disassembled by ``llvm-objdump -d`` from the same LLVM build, whose output lists each
instruction with its byte offset and its *original* dwords (``llvm-mc --disassemble
--show-encoding`` re-encodes, so a non-canonical literal reports the wrong length and cannot
be used for boundaries). Comparison is per instruction on offset,
length and mnemonic (with the ``_e32/_e64/_dpp/_sdwa`` encoding suffixes stripped). A
disagreement is reported with the bytes; it is settled by the ISA document, never by editing
the expected side.
"""

from __future__ import annotations

import re
import subprocess
import uuid
from dataclasses import dataclass, field

from . import decode as D

LLVM_PIPE = (
    "llvm-mc -triple=amdgcn-amd-amdhsa -mcpu={cpu} -filetype=obj -o {obj} && "
    "llvm-objdump -d --mcpu={cpu} {obj}; rm -f {obj}"
)
_LINE = re.compile(r"^\s*(.*?)\s*//\s*([0-9A-Fa-f]+):((?:\s+[0-9A-Fa-f]{8})+)\s*$")
_SUFFIX = re.compile(r"_(e32|e64|dpp|sdwa|dpp8)$")
_INVALID = ("<unknown>", ".long", ".short", ".byte")


@dataclass
class LlvmInsn:
    """One instruction as LLVM disassembled it."""

    offset: int
    size: int
    mnemonic: str


@dataclass
class LlvmResult:
    """Parsed LLVM disassembly."""

    insns: list = field(default_factory=list)
    invalid: int = 0
    raw: str = ""


def run_llvm(data: bytes, mcpu: str = "gfx1030", timeout: int = 180) -> LlvmResult:
    """Disassemble ``data`` with LLVM for ``-mcpu=<mcpu>`` (inside WSL)."""
    asm = ".text\n" + "".join(
        ".byte " + ",".join(str(b) for b in data[i : i + 64]) + "\n"
        for i in range(0, len(data), 64)
    )
    obj = f"/tmp/h03_{uuid.uuid4().hex}.o"
    cmd = ["wsl", "-d", "Ubuntu", "--", "bash", "-c", LLVM_PIPE.format(cpu=mcpu, obj=obj)]
    proc = subprocess.run(
        cmd, input=asm, capture_output=True, text=True, timeout=timeout, check=False
    )
    return parse_llvm_output(proc.stdout)


def parse_llvm_output(text: str) -> LlvmResult:
    """Parse ``llvm-objdump -d`` text into instructions with offsets and byte lengths."""
    res = LlvmResult(raw=text)
    for line in text.splitlines():
        m = _LINE.match(line)
        if not m:
            continue
        body, off, words = m.group(1).strip(), int(m.group(2), 16), m.group(3).split()
        if not body:
            continue
        mnem = _SUFFIX.sub("", body.split()[0].lower())
        if mnem in _INVALID:
            res.invalid += 1
        res.insns.append(LlvmInsn(off, 4 * len(words), mnem))
    return res


def compare(data: bytes, llvm: LlvmResult) -> dict:
    """Compare boundaries and mnemonics of our decoder against LLVM, keyed by offset.

    Instructions whose opcode this tool has no table for (``name is None``) are compared
    on offset and length only.
    """
    ours, err = D.decode_program(data)
    theirs = {i.offset: i for i in llvm.insns}
    ours_by_off = {i.pc: i for i in ours}
    disagreements = []
    checked = 0
    for a in ours:
        b = theirs.get(a.pc)
        if b is None:
            kind = "boundary: LLVM has no instruction at this offset"
        elif b.size != a.size:
            kind = "length"
        elif a.name is not None and a.mnemonic != b.mnemonic:
            kind = "mnemonic"
            checked += 1
        else:
            checked += a.name is not None
            continue
        disagreements.append(
            {
                "kind": kind,
                "offset": a.pc,
                "ours": (a.mnemonic, a.size),
                "llvm": (b.mnemonic, b.size) if b else None,
                "bytes": data[a.pc : a.pc + max(a.size, b.size if b else 0)].hex(" "),
            }
        )
    for b in llvm.insns:
        if b.offset not in ours_by_off:
            disagreements.append(
                {
                    "kind": "boundary: LLVM instruction not at one of our boundaries",
                    "offset": b.offset,
                    "ours": None,
                    "llvm": (b.mnemonic, b.size),
                    "bytes": data[b.offset : b.offset + b.size].hex(" "),
                }
            )
    return {
        "ours_count": len(ours),
        "llvm_count": len(llvm.insns),
        "llvm_invalid": llvm.invalid,
        "decode_error": err,
        "mnemonic_checked": checked,
        "disagreements": disagreements,
    }
