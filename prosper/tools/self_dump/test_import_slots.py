#!/usr/bin/env python3
"""Regression test for `self_dump --import-slots` (#1625).

The flag answers "which GOT slot does this Sony import land in", which is step 2 of the
"who calls this Sony function" workflow and used to be hand-parsed out of `DT_JMPREL`.

Every fixture here is a PS5-shaped ELF **built byte by byte in this file**, not extracted from a
dump and not produced by any part of self_dump. That matters: a positive control drawn from the
same machinery as the thing it validates only ever tests the discriminator. These bytes are an
independent statement of what the relocation tables mean, so a wrong reading of them fails here.

The cases are chosen so that each one fails for a different reason:

  1. a JUMP_SLOT relocation against an imported symbol IS a slot   (the flag's whole purpose)
  2. a GLOB_DAT relocation against an imported symbol IS a slot     (RELA, not just JMPREL)
  3. a JUMP_SLOT against a DEFINED symbol is NOT an import slot     (index-aligned symbol lookup)
  4. an import with no relocation is listed, never silently dropped (a grep miss must mean "absent")
  5. a non-address relocation type against an import is counted, not reported as a slot
  6. zero rows always names its own cause and exits 3               (#2399's class of defect)
  7. an unmapped relocation table says UNMAPPED rather than "none"  (ditto)
  7b. a table declared at address 0 is absent, not resolved against the vaddr-0 PT_LOAD
  8. the name column is filled from the firmware stub dump, and its provenance is printed

Usage: test_import_slots.py <path-to-self_dump>
"""

import struct
import subprocess
import sys
import tempfile
from pathlib import Path

# --- ELF / Sony dynamic constants used by the fixtures ---------------------------------------
PT_LOAD, PT_DYNAMIC = 1, 2
ET_SCE_DYNEXEC, EM_X86_64 = 0xFE10, 0x3E

DT_NULL, DT_PLTRELSZ, DT_STRTAB, DT_SYMTAB, DT_RELA, DT_RELASZ = 0, 2, 5, 6, 7, 8
DT_RELAENT, DT_STRSZ, DT_SYMENT, DT_PLTREL, DT_JMPREL = 9, 0xA, 0xB, 0x14, 0x17
DT_SCE_NEEDED_MODULE, DT_SCE_IMPORT_LIB, DT_SCE_SYMTABSZ = 0x61000045, 0x61000049, 0x6100003F

R_X86_64_GLOB_DAT, R_X86_64_JUMP_SLOT = 6, 7
R_X86_64_RELATIVE, R_X86_64_DTPMOD64 = 8, 16

# Where each table lives in the fixture's read-write segment (vaddr == file offset there).
V_DYNAMIC, V_STRTAB, V_SYMTAB, V_JMPREL, V_RELA, V_GOT = 0x1000, 0x1200, 0x1300, 0x1400, 0x1500, 0x1F00
SEG_RW, SEG_SIZE, FILE_SIZE = 0x1000, 0x1000, 0x2000

B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-"


class StrTab:
    """Sony's dynamic string table: NUL at 0, then NUL-terminated names."""

    def __init__(self):
        self.blob = bytearray(b"\0")
        self.offs = {}

    def add(self, s: str) -> int:
        if s not in self.offs:
            self.offs[s] = len(self.blob)
            self.blob += s.encode() + b"\0"
        return self.offs[s]


def sym(name_off: int, shndx: int, value: int) -> bytes:
    """Elf64_Sym. st_info = GLOBAL|FUNC, which self_dump does not read but a real one carries."""
    return struct.pack("<IBBHQQ", name_off, 0x12, 0, shndx, value, 0)


def rela(offset: int, symidx: int, rtype: int, addend: int = 0) -> bytes:
    return struct.pack("<QQq", offset, (symidx << 32) | rtype, addend)


def phdr(ptype: int, flags: int, off: int, vaddr: int, filesz: int) -> bytes:
    return struct.pack("<IIQQQQQQ", ptype, flags, off, vaddr, vaddr, filesz, filesz, 0x4000)


def build_fixture(*, jmprel=True, rela_table=True, unmapped_jmprel=False,
                  zero_jmprel_va=False) -> bytes:
    """A minimal PS5-shaped dynamic executable with three imports and one local definition.

    `jmprel`/`rela_table` drop a whole relocation table; `unmapped_jmprel` keeps DT_JMPREL but
    points it at a vaddr no segment covers; `zero_jmprel_va` declares a size for a table whose
    address is 0 — the distinct ways an honest tool must report a zero differently from "I did not
    parse anything", and differently from each other.
    """
    st = StrTab()
    lib_one, lib_two = st.add("libSceTestOne"), st.add("libSceTestTwo")
    module = st.add("testmodule")
    # Symbol names are Sony's NID#libId#modId, the ids in Sony base64 ("A" == 0, "B" == 1).
    n_import1 = st.add("TESTNIDONE1#%s#%s" % (B64[0], B64[1]))
    n_local = st.add("LOCALSYMBOL#%s#%s" % (B64[0], B64[1]))
    n_import2 = st.add("TESTNIDTWO2#%s#%s" % (B64[1], B64[1]))
    n_noslot = st.add("NOSLOTNID12#%s#%s" % (B64[0], B64[1]))

    symtab = b"".join([
        sym(0, 0, 0),                    # 0: the null symbol — no NID, must not shift the indices
        sym(n_import1, 0, 0),            # 1: import, libSceTestOne
        sym(n_local, 1, 0x500),          # 2: DEFINED here — never an import slot
        sym(n_import2, 0, 0),            # 3: import, libSceTestTwo
        sym(n_noslot, 0, 0),             # 4: import with no relocation at all
    ])
    jmprel_tab = b"".join([
        rela(V_GOT + 0x00, 1, R_X86_64_JUMP_SLOT),
        rela(V_GOT + 0x08, 3, R_X86_64_JUMP_SLOT),
        rela(V_GOT + 0x10, 2, R_X86_64_JUMP_SLOT),   # against the DEFINED symbol
    ])
    rela_tab = b"".join([
        rela(V_GOT + 0x20, 1, R_X86_64_GLOB_DAT),
        rela(V_GOT + 0x28, 0, R_X86_64_RELATIVE, 0x1234),      # no symbol -> skipped
        rela(V_GOT + 0x30, 1, R_X86_64_DTPMOD64),              # TLS against an import -> counted
    ])

    dyn = [
        (DT_SCE_IMPORT_LIB, (0 << 48) | lib_one),
        (DT_SCE_IMPORT_LIB, (1 << 48) | lib_two),
        (DT_SCE_NEEDED_MODULE, (1 << 48) | module),
        (DT_STRTAB, V_STRTAB), (DT_STRSZ, len(st.blob)),
        (DT_SYMTAB, V_SYMTAB), (DT_SYMENT, 24), (DT_SCE_SYMTABSZ, len(symtab)),
        (DT_PLTREL, DT_RELA), (DT_RELAENT, 24),
    ]
    if jmprel:
        jmprel_va = V_JMPREL
        if unmapped_jmprel:
            jmprel_va = 0xDEAD0000
        elif zero_jmprel_va:
            jmprel_va = 0
        dyn += [(DT_JMPREL, jmprel_va), (DT_PLTRELSZ, len(jmprel_tab))]
    if rela_table:
        dyn += [(DT_RELA, V_RELA), (DT_RELASZ, len(rela_tab))]
    dyn += [(DT_NULL, 0)]
    dyn_blob = b"".join(struct.pack("<qQ", t, v) for t, v in dyn)

    phdrs = b"".join([
        phdr(PT_LOAD, 5, 0, 0, SEG_RW),                              # r-x
        phdr(PT_LOAD, 6, SEG_RW, SEG_RW, SEG_SIZE),                  # rw-, holds every table
        phdr(PT_DYNAMIC, 6, V_DYNAMIC, V_DYNAMIC, len(dyn_blob)),
    ])
    ehdr = struct.pack(
        "<16sHHIQQQIHHHHHH",
        b"\x7fELF\x02\x01\x01\x09" + b"\0" * 8, ET_SCE_DYNEXEC, EM_X86_64, 1,
        0x100, 64, 0, 0, 64, 56, 3, 64, 0, 0)

    buf = bytearray(b"\0" * FILE_SIZE)
    buf[0:len(ehdr)] = ehdr
    buf[64:64 + len(phdrs)] = phdrs
    buf[V_DYNAMIC:V_DYNAMIC + len(dyn_blob)] = dyn_blob
    buf[V_STRTAB:V_STRTAB + len(st.blob)] = st.blob
    buf[V_SYMTAB:V_SYMTAB + len(symtab)] = symtab
    buf[V_JMPREL:V_JMPREL + len(jmprel_tab)] = jmprel_tab
    buf[V_RELA:V_RELA + len(rela_tab)] = rela_tab
    return bytes(buf)


ALL_OUTPUT = []   # every stdout this test has seen, for the ASCII invariant below


def run(binary: str, blob: bytes, extra=(), names_dir=None):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fixture.elf"
        path.write_bytes(blob)          # binary mode: the fixture must reach the tool byte-exact
        cmd = [binary, str(path), "--import-slots"] + list(extra)
        if names_dir:
            cmd += ["--names", names_dir]
        # Decode explicitly, NOT with text=True. text=True uses
        # locale.getpreferredencoding(), which is UTF-8 on Linux and cp1252 on Windows — so the
        # same bytes would compare differently per platform and a Linux-green run could not see
        # it. errors="replace" keeps a mis-decode visible (U+FFFD is non-ASCII, so the invariant
        # check below fails) rather than raising something that reads as a crash.
        done = subprocess.run(cmd, capture_output=True)
        out = done.stdout.decode("utf-8", errors="replace")
        ALL_OUTPUT.append((" ".join(cmd[1:]), out))
        return done.returncode, out


def slot_rows(out: str):
    """The `<slot> <NID> <library> <name> <reloc>` rows, keyed by slot."""
    rows = {}
    for line in out.splitlines():
        if not line.startswith("0x"):
            continue
        f = line.split()
        if len(f) == 5:
            rows[f[0]] = f[1:]
    return rows


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: test_import_slots.py <path-to-self_dump>", file=sys.stderr)
        return 2
    binary = sys.argv[1]
    failures = []

    def check(name, condition, detail=""):
        if condition:
            print("ok   - %s" % name)
        else:
            print("FAIL - %s %s" % (name, detail))
            failures.append(name)

    # --- Cases 1-5: the full fixture ----------------------------------------------------------
    code, out = run(binary, build_fixture())
    rows = slot_rows(out)

    check("a JUMP_SLOT relocation reports the slot from r_offset",
          rows.get("0x00000001f00", [None])[0] == "TESTNIDONE1", "\n" + out)
    check("the slot row names the import's library",
          rows.get("0x00000001f00", [None, None])[1:2] == ["libSceTestOne"], "\n" + out)
    check("the second import resolves through its own library id",
          rows.get("0x00000001f08", [None, None])[0:2] == ["TESTNIDTWO2", "libSceTestTwo"],
          "\n" + out)
    check("a GLOB_DAT relocation in DT_RELA is a slot too",
          rows.get("0x00000001f20", [None] * 5)[3] == "GLOB_DAT", "\n" + out)
    check("the JUMP_SLOT row is labelled JUMP_SLOT",
          rows.get("0x00000001f00", [None] * 5)[3] == "JUMP_SLOT", "\n" + out)

    # A relocation against a symbol this module DEFINES is not an import slot. This is the arm the
    # index alignment protects: the null symbol carries no NID, so a parser that skipped it would
    # shift every index by one and report symbol 2 (local) under symbol 1's identity.
    check("a relocation against a defined symbol is not an import slot",
          "0x00000001f10" not in rows, "\n" + out)
    check("no row claims the local symbol's name",
          "LOCALSYMBOL" not in out, "\n" + out)

    check("the run reports three slots over two of three imports",
          "3 slots over 2 of 3 imports" in out, "\n" + out)
    check("an import with no relocation is listed rather than dropped",
          "[IMPORTS WITHOUT A SLOT] (1" in out and "NOSLOTNID12" in out, "\n" + out)
    check("a non-address relocation against an import is counted, not reported as a slot",
          "type16 x1" in out, "\n" + out)
    check("both relocation tables are named with their entry counts",
          "entries=3 (DT_JMPREL/DT_PLTRELSZ" in out and "entries=3 (DT_RELA/DT_RELASZ" in out,
          "\n" + out)
    check("a run that found slots exits 0", code == 0, "got exit %d" % code)

    # --- Case 6: no relocation tables at all --------------------------------------------------
    code, out = run(binary, build_fixture(jmprel=False, rela_table=False))
    # Asserted as two ASCII substrings rather than one quoted sentence: the zero, and the reason
    # for it. Matching the whole sentence made this the only check in the file that spanned a
    # non-ASCII character, and it was the only one that failed on Windows.
    check("a module with no relocation tables reports zero slots",
          "0 import slots" in out, "\n" + out)
    check("a module with no relocation tables says WHY it is zero",
          "the dynamic table declares neither DT_JMPREL nor DT_RELA" in out, "\n" + out)
    check("a zero result exits 3 rather than 0", code == 3, "got exit %d" % code)
    check("a zero result still reports what was parsed",
          "imports=3" in out and "jmprel absent" in out, "\n" + out)

    # --- Case 7: a declared table that does not map -------------------------------------------
    # The defect class #2399 is about: an unreadable table must never look like an empty one.
    code, out = run(binary, build_fixture(rela_table=False, unmapped_jmprel=True))
    check("an unmapped relocation table is reported as UNMAPPED", "UNMAPPED" in out, "\n" + out)
    check("an unmapped table is not reported as 'no relocation references an import'",
          "every declared relocation table is unmapped" in out, "\n" + out)
    check("an unmapped table exits 3", code == 3, "got exit %d" % code)

    # --- Case 7b: a table address of 0 is "absent", never "at vaddr 0" ------------------------
    # A PS5 executable's first PT_LOAD IS mapped at vaddr 0, so resolving a zero table address
    # succeeds and hands back the ELF header — which would be parsed as relocations and reported
    # as slots. The wrong answer would look exactly like a right one, which is what makes it worth
    # a case of its own rather than folding into the "no tables" arm.
    code, out = run(binary, build_fixture(rela_table=False, zero_jmprel_va=True))
    check("a zero table address is reported as absent, not resolved to vaddr 0",
          "jmprel absent" in out and "malformed dynamic table" in out, "\n" + out)
    check("a zero table address produces no rows", not slot_rows(out), "\n" + out)
    check("a zero table address exits 3", code == 3, "got exit %d" % code)

    # --- Case 8: the firmware stub dump fills the name column ---------------------------------
    with tempfile.TemporaryDirectory() as names:
        Path(names, "libSceTestOne.c").write_text(
            'void __stub(void) {\n'
            '  if(sprx_dlsym(__handle, "TESTNIDONE1", &__ptr_sceTestFunctionOne)) return;\n'
            '}\n')
        code, out = run(binary, build_fixture(), names_dir=names)
        rows = slot_rows(out)
        check("a named NID resolves to its function name",
              rows.get("0x00000001f00", [None] * 5)[2] == "sceTestFunctionOne", "\n" + out)
        check("an unnamed NID still prints its slot and NID",
              rows.get("0x00000001f08", [None] * 5)[0:3] == ["TESTNIDTWO2", "libSceTestTwo", "-"],
              "\n" + out)
        check("the name table's provenance is printed",
              "1 files, 1 pairs, 1 distinct NIDs" in out, "\n" + out)

    code, out = run(binary, build_fixture(), names_dir="/nonexistent-name-table")
    check("an unreadable name table says so instead of silently emptying the name column",
          "is not a readable directory" in out, "\n" + out)
    check("an unreadable name table does not suppress the slots", code == 0, "got exit %d" % code)

    # --- the flag is opt-in: without it, the tool's existing output is unchanged ---------------
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fixture.elf"
        path.write_bytes(build_fixture())
        done = subprocess.run([binary, str(path)], capture_output=True, text=True)
        check("without --import-slots nothing is printed and the exit code stays 0",
              done.returncode == 0 and "[IMPORT SLOTS]" not in done.stdout,
              "exit %d\n%s" % (done.returncode, done.stdout))

    # --- the output is ASCII on every message path -------------------------------------------
    # The property, not a re-spelling of one message: stdout carries BYTES, and whoever reads them
    # decodes by platform (UTF-8 on Linux, the cp1252/cp437 console code page on Windows). Any
    # non-ASCII byte therefore renders as mojibake in cmd.exe and breaks a `grep` for the very
    # line the tool printed. Checked over every run above — including the zero-result and
    # name-table paths, which is where the five em dashes actually were.
    offenders = []
    for cmd, out in ALL_OUTPUT:
        for lineno, line in enumerate(out.splitlines(), 1):
            bad = {c for c in line if ord(c) > 127}
            if bad:
                offenders.append("%s: line %d: %s in %r"
                                 % (cmd, lineno, sorted(hex(ord(c)) for c in bad), line[:90]))
    check("every byte the tool prints is ASCII, on all %d runs" % len(ALL_OUTPUT),
          not offenders, "\n" + "\n".join(offenders[:10]))

    print("\n%d checks failed" % len(failures) if failures else "\nall checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
