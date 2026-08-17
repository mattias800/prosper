#!/usr/bin/env python3
"""nid_gate_scan — does a guest module *branch on* an imported function's return value?

Importing a NID is not the same as depending on what it returns. When an HLE handler's return
value changes, the question that actually bounds the blast radius is "which titles inspect the
result, and how?" — a title that calls the import and drops eax cannot be affected by any answer,
while a title that compares the result against a specific errno changes control flow.

This resolves that statically, with no boot and no GPU:

  NID -> dynsym index -> JMPREL jump-slot VA -> the `jmp *[rip+d]` stub that goes through it
      -> every `call rel32` that reaches the stub (plus any `call *[rip+d]` straight to the slot)
      -> objdump the bytes right after each call and classify what happens to eax.

THE ERROR ARM IS READ. This used to carry a caveat saying it was not: `classify_window` stopped at
the first branch, so a site that tests the result generically and THEN const-compares it *inside its
error arm* was bucketed as a plain non-zero test, and a `nonzero` verdict meant only "no const
compare before the first branch". Measured then: PPSA08804 compares against 0x809F000F at 0x4e41a32,
past the branch, reported as `nonzero` (#2023 review). That under-reporting is now fixed —
`classify_site_following` forks the walk at every branch the value survives and explores both
successors — and PPSA08804's site is reported as `const`.

What remains, and it is a different claim: a site whose arms leave this scan's reach is reported
`gate-open` or `forward`, NEVER `nonzero`. So the bucket that says "this site cannot tell one error
code from another" now earns that meaning, and the buckets that cannot say it are loud. Pass
`--no-follow-arms` to reproduce the pre-2026-08 numbers.

Buckets (see `classify_site_following` for the default walk, `classify_window` for the legacy one):
    const        some reachable path compares eax against --const (the idiom this was written for)
    nonzero      gated on zero/non-zero, and the value is then dead on EVERY reachable path — the
                 one bucket that means "this site cannot be affected by WHICH non-zero answer"
    gate-open    gated, but at least one arm left the scan's reach (returned, spilled, indirect
                 branch, unmapped, or the block budget ran out) — NOT cleared, read by hand
    other-cmp    compares eax against some *other* immediate
    alu-gate     flags derive from an ALU transform of the result (`and eax,mask; jne`)
    forward      eax is moved/returned/stored and never gated here — the gate, if any, is in a
                 caller or behind a spilled value; needs a look by hand
    ignored      eax is dead before it is read — this call site cannot be affected
    undecodable  objdump could not decode the window: a VOID sample, not a negative one

Usage:
    nid_gate_scan.py <module|app0-dir> --nid <NID> [--const 0x805a1001] [--window 48] [-v]
    nid_gate_scan.py <module|app0-dir> --all-nids [--names <PS5-libs-dir>] [--min-gated N] [-v]

`<module>` is a SELF/PRX/eboot.bin (flattened in memory — no temp files) or an already-flat ELF.
Given a directory, every `eboot.bin` and `*.prx`/`*.sprx` under it is scanned.

Example — the sysmodule gate (#2002); `0x805A1001` is SCE_SYSMODULE_ERROR_UNLOADED:
    nid_gate_scan.py testdata/PPSA04263-app0 --nid fMP5NHUOaMk --const 0x805a1001

`--all-nids` inverts the question: instead of "who branches on THIS value?", it answers **"which
Sony answers does this title depend on at all?"** — every import, classified, in one pass. That is
the bound a *registered-but-mismodelled return value* investigation needs and cannot get any other
way. An absence check (the unimplemented-NID table) cannot see a handler that is registered and
answers wrongly; a runtime return-value histogram says what prosper returned but not whether the
guest looked. This says which call sites can be affected by an answer at all, so the ones that
cannot are removed from the search before a single boot:

    nid_gate_scan.py <DUMP_ROOT>/PPSA05325-app0/eboot.bin --all-nids --names ../PS5-3.20_Libs

Read the buckets the same way as in single-NID mode, and read `ignored` as the load-bearing one:
it is the only bucket that says an answer *cannot* matter at that site. `--names` is symbolication
only — an unknown NID is still scanned and still reported, so a missing library dump costs labels,
never coverage.
"""
import argparse
import os
import re
import struct
import subprocess
import sys

JMP_SLOT = 7  # R_X86_64_JUMP_SLOT
MODRM_RIP = {0x05, 0x0d, 0x15, 0x1d, 0x25, 0x2d, 0x35, 0x3d}


# ---------------------------------------------------------------- image loading

def flatten(path):
    """Return a flat image (p_offset == p_vaddr) of a SELF/PRX, or the file itself if already ELF.

    Same transform as tools/il2cpp/prx_to_elf.py, in memory. PS5 modules keep their dynamic
    string/symbol/relocation tables in a PT_LOAD with no permission bits (flags=0), so the flat
    image carries everything this tool needs.
    """
    f = open(path, "rb").read()
    if f[:4] == b"\x7fELF":
        return f
    nseg = struct.unpack_from("<H", f, 0x18)[0]
    segs = [struct.unpack_from("<QQQQ", f, 0x20 + i * 32) for i in range(nseg)]
    eb = f.find(b"\x7fELF", 0x20 + nseg * 32)
    if eb < 0:
        raise ValueError("no embedded ELF")
    e_phoff = struct.unpack_from("<Q", f, eb + 0x20)[0]
    e_phnum = struct.unpack_from("<H", f, eb + 0x38)[0]
    data_seg = {(fl >> 20): (foff, fsize) for (fl, foff, fsize, _ms) in segs if fl & 0x800}
    phdrs = [list(struct.unpack_from("<IIQQQQQQ", f, eb + e_phoff + i * 56)) for i in range(e_phnum)]
    loads = [p for p in phdrs if p[0] == 1 and p[5] > 0]
    if not loads:
        raise ValueError("no PT_LOAD")
    buf = bytearray(max(p[3] + p[5] for p in loads))
    for i, p in enumerate(phdrs):
        if p[0] == 1 and p[5] > 0 and i in data_seg:
            off, sz = data_seg[i]
            sz = min(sz, p[5])
            buf[p[3]:p[3] + sz] = f[off:off + sz]
    buf[0:64] = f[eb:eb + 64]
    struct.pack_into("<H", buf, 0x10, 3)
    struct.pack_into("<Q", buf, 0x20, 0x40)
    struct.pack_into("<Q", buf, 0x28, 0)
    struct.pack_into("<H", buf, 0x3c, e_phnum)
    for i, p in enumerate(phdrs):
        struct.pack_into("<IIQQQQQQ", buf, 0x40 + i * 56, p[0], p[1], p[3], p[3], p[4], p[5], p[6], p[7])
    return bytes(buf)


class Image:
    def __init__(self, raw):
        self.raw = raw
        e_phoff = struct.unpack_from("<Q", raw, 0x20)[0]
        phentsize, phnum = struct.unpack_from("<HH", raw, 0x36)
        self.segs = []       # (va, foff, filesz, flags)
        self.dyn_va = 0
        for i in range(phnum):
            t, fl, off, va, _pa, fs, _ms, _al = struct.unpack_from("<IIQQQQQQ", raw, e_phoff + i * phentsize)
            if t == 1:
                self.segs.append((va, off, fs, fl))
            elif t == 2:
                self.dyn_va = va
        self.tags = self._dyn_tags()

    def foff(self, va):
        for v, o, fs, _fl in self.segs:
            if v <= va < v + fs:
                return o + (va - v)
        return None

    def _dyn_tags(self):
        tags = {}
        o = self.foff(self.dyn_va) if self.dyn_va else None
        while o is not None and o + 16 <= len(self.raw):
            tag, val = struct.unpack_from("<qQ", self.raw, o)
            o += 16
            if tag == 0:
                break
            tags.setdefault(tag, val)
        return tags

    def tag(self, *names):
        for n in names:
            if n in self.tags:
                return self.tags[n]
        return None

    def dynsym(self):
        """(symtab file offset, strtab file offset, symbol count, entry size), or None.

        Shared by the single-NID lookup and the whole-table sweep so both agree on how the table
        is bounded — the bounding rule below is the one thing that can silently turn a present
        import into a reported absence.
        """
        symtab = self.tag(6, 0x61000039)
        strtab = self.tag(5, 0x61000035)
        if symtab is None or strtab is None:
            return None
        so, to = self.foff(symtab), self.foff(strtab)
        if so is None or to is None:
            return None
        syment = self.tag(0xB) or 24
        count = self.tag(0x6100003F)
        if count:
            n = count // syment
        elif to > so:
            n = (to - so) // syment          # symtab runs up to strtab, the usual PS5 layout
        else:
            # No DT_SCE_SYMTABSZ and strtab does NOT follow symtab: the symbol count is unknown.
            # Falling through with n=0 would report "no-import" for a module that may well import
            # the NID — a false negative that reads exactly like a real absence. Fail loudly.
            raise ValueError("cannot bound symtab: no DT_SCE_SYMTABSZ and strtab precedes symtab")
        return so, to, n, syment

    def symbol_index(self, nid):
        """Index of the dynsym entry whose NID is `nid` ("<NID>#<lib>#<mod>"), or None."""
        d = self.dynsym()
        if d is None:
            return None
        so, to, n, syment = d
        want = nid.encode() + b"#"
        for i in range(n):
            base = so + i * syment
            if base + syment > len(self.raw):
                break
            st_name = struct.unpack_from("<I", self.raw, base)[0]
            if self.raw[to + st_name:to + st_name + len(want)] == want:
                return i
        return None

    def imported_nids(self):
        """Every `<NID>#<lib>#<mod>` dynsym entry, as [(nid, symbol index)] in table order.

        A PS5 dynsym name is the 11-character base64 NID, `#`, the import-library id, `#`, the
        module id. Requiring exactly 11 characters before the first `#` is what keeps ordinary
        C++ mangled names out — and the rule has to be *exactly* 11, not "contains a `#`", because
        both looser and stricter versions fail silently: a mangled name admitted as a phantom
        import reports zero call sites and reads as a clean negative, while a dropped real import
        shrinks a census whose whole point is exhaustiveness.

        The `#` search is clamped to the name's own NUL. Without that clamp a name with no `#` at
        all can borrow the *next* strtab string's, and if that one happened to land at distance 11
        the entry would be admitted — an over-report with nothing about it looking wrong.

        This does NOT distinguish an import from an export: a module's own exports carry the same
        shape. What removes them downstream is the JMPREL/call-site filter, since an export has no
        jump slot bound to it. Duplicates are possible in principle — one NID can appear under two
        library ids — so the caller gets every row rather than a dict that would silently keep one.
        """
        d = self.dynsym()
        if d is None:
            return []
        so, to, n, syment = d
        out = []
        for i in range(n):
            base = so + i * syment
            if base + syment > len(self.raw):
                break
            st_name = struct.unpack_from("<I", self.raw, base)[0]
            start = to + st_name
            stop = self.raw.find(b"\0", start)
            end = self.raw.find(b"#", start, stop if stop >= 0 else None)
            if end < 0 or end - start != 11:
                continue
            out.append((self.raw[start:end].decode("latin1"), i))
        return out

    def jump_slots(self, sym_index):
        """JMPREL jump-slot VAs bound to `sym_index` (usually exactly one)."""
        out = []
        for tv, sv in [(0x17, 0x2), (0x61000029, 0x6100002D)]:
            va, sz = self.tags.get(tv), self.tags.get(sv)
            if va is None or sz is None:
                continue
            base = self.foff(va)
            if base is None:
                continue
            for i in range(sz // 24):
                r_off, r_info, _r_add = struct.unpack_from("<QQq", self.raw, base + i * 24)
                if (r_info & 0xFFFFFFFF) == JMP_SLOT and (r_info >> 32) == sym_index:
                    out.append(r_off)
        return out


# ---------------------------------------------------------------- code scanning

def scan_code(img):
    """Reverse index over executable segments: target VA -> [(site VA, kind)]."""
    xref = {}
    for v, o, fs, fl in img.segs:
        if not (fl & 1):
            continue
        d = img.raw[o:o + fs]
        n = len(d)
        for i in range(n):
            b = d[i]
            if b == 0xE8 and i + 5 <= n:
                disp = struct.unpack_from("<i", d, i + 1)[0]
                xref.setdefault(v + i + 5 + disp, []).append((v + i, "call"))
            elif b == 0xFF and i + 6 <= n and (d[i + 1] & 0xC7) == 0x05:
                reg = (d[i + 1] >> 3) & 7
                if reg in (2, 4):
                    disp = struct.unpack_from("<i", d, i + 2)[0]
                    xref.setdefault(v + i + 6 + disp, []).append((v + i, "call*" if reg == 2 else "jmp*"))
    return xref


_SCRATCH = None


def _scratch_path():
    """One reusable scratch file for objdump input, deliberately NOT under /tmp by default.

    /tmp on the project's Linux box is a RAM-backed tmpfs with a per-user quota shared by every
    concurrent agent, so a tool that creates a file per call is a bad citizen there even when each
    file is tiny and short-lived. Honour TMPDIR when the caller sets one; otherwise use the user
    cache directory, which is real disk.
    """
    global _SCRATCH
    if _SCRATCH is None:
        d = os.environ.get("TMPDIR") or os.path.join(os.path.expanduser("~"), ".cache")
        os.makedirs(d, exist_ok=True)
        _SCRATCH = os.path.join(d, "nid_gate_scan-%d.bin" % os.getpid())
    return _SCRATCH


_OBJDUMP = False   # False = not probed yet; None = no capable objdump found


def objdump_binary():
    """Path to an objdump that can disassemble a RAW BINARY blob, or None.

    Existence is not the test — capability is. macOS ships LLVM's objdump as plain `objdump`, and it
    answers `--version` happily while rejecting `-b binary` outright, so a probe that only checks the
    binary is present reports a working toolchain on a machine where every decode fails. Probe with
    an actual one-byte decode, and accept Homebrew's `gobjdump` as the GNU binutils build.
    """
    global _OBJDUMP
    if _OBJDUMP is not False:
        return _OBJDUMP
    _OBJDUMP = None
    cands = [os.environ["OBJDUMP"]] if os.environ.get("OBJDUMP") else ["objdump", "gobjdump"]
    probe = _scratch_path()
    try:
        with open(probe, "wb") as f:
            f.write(b"\x90")                                   # nop
        for c in cands:
            try:
                r = subprocess.run([c, "-D", "-b", "binary", "-m", "i386:x86-64", "-M", "intel", probe],
                                   capture_output=True, text=True)
            except (OSError, FileNotFoundError):
                continue
            if r.returncode == 0 and "nop" in r.stdout:
                _OBJDUMP = c
                break
    except OSError:
        pass
    return _OBJDUMP


def disasm(blob, base):
    """objdump a raw byte blob as x86-64 Intel; returns [(va, mnemonic, operands)].

    Raises on an objdump failure rather than returning an empty list: a silent empty decode would
    be classified as an ordinary window and drain into a bucket, turning a broken toolchain into a
    plausible-looking measurement.

    `--no-show-raw-insn` is load-bearing, not cosmetic. objdump wraps the raw-byte column after
    seven bytes, and the continuation line carries the SAME `<addr>:\\t<hex bytes>` shape as a real
    instruction — so the parse below read ` 310d73a:\\t03 00 00 00` as an instruction `00` at a VA
    that is *inside* the 11-byte `mov DWORD PTR [rdi+rax*4+0x6e0],0x3` starting at 0x310d733
    (PPSA04263). A phantom mnemonic is inert in the taint walk, but a phantom VA is not: the walk
    uses instruction addresses to continue a block past the end of a window, and continuing from
    the middle of an instruction decodes garbage. Suppressing the byte column removes the wrap and
    with it the whole class.
    """
    od = objdump_binary()
    if od is None:
        raise RuntimeError("no GNU objdump that can disassemble a raw binary was found "
                           "(tried objdump, gobjdump; set OBJDUMP=/path). LLVM's objdump — the "
                           "default on macOS — does not support -b binary.")
    tmp = _scratch_path()
    with open(tmp, "wb") as f:
        f.write(blob)
    r = subprocess.run(
        [od, "-D", "-b", "binary", "-m", "i386:x86-64", "-M", "intel", "--no-show-raw-insn",
         "--adjust-vma=%#x" % base, tmp],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("objdump failed (%d): %s" % (r.returncode, r.stderr.strip()[:200]))
    out = r.stdout
    insns = []
    for line in out.splitlines():
        m = re.match(r"\s*([0-9a-f]+):\s+(\S+)\s*(.*)", line)
        if m:
            insns.append((int(m.group(1), 16), m.group(2), m.group(3).split("#")[0].strip()))
    return insns


def arg0_before(raw, foff, back=24):
    """Best-effort first argument of the call at file offset `foff`: the last `mov edi,imm32`
    (0xbf) or `xor edi,edi` in the `back` bytes ahead of it. Returns (value, exact) where `exact`
    means the setter sits immediately before the call, so nothing can have clobbered edi between.
    A backwards decode cannot be exact in general — treat a non-exact hit as a hint."""
    lo = max(0, foff - back)
    best = None
    for p in range(lo, foff):
        if raw[p] == 0xBF and p + 5 <= foff:
            best = (struct.unpack_from("<I", raw, p + 1)[0], p + 5 == foff)
        elif raw[p] == 0x31 and p + 2 <= foff and raw[p + 1] == 0xFF:
            best = (0, p + 2 == foff)
    return best


_SUBREG = {}
for _wide, _key in (("ax", "a"), ("bx", "b"), ("cx", "c"), ("dx", "d")):
    for _f in ("r%s" % _wide, "e%s" % _wide, _wide, _key + "l", _key + "h"):
        _SUBREG[_f] = _key
for _wide, _key in (("si", "si"), ("di", "di"), ("bp", "bp"), ("sp", "sp")):
    for _f in ("r" + _wide, "e" + _wide, _wide, _wide + "l"):
        _SUBREG[_f] = _key

# A `call` clobbers every SysV caller-saved register, so a result parked in one of these does not
# survive one — but rbx/rbp/r12-r15 do.
CALLER_SAVED = {"a", "c", "d", "si", "di", "r8", "r9", "r10", "r11"}


def canon(operand):
    """Canonical register name for an operand, or None if it is not a bare register."""
    op = operand.strip()
    m = re.fullmatch(r"r(\d{1,2})[dwb]?", op)
    if m:
        return "r" + m.group(1)
    return _SUBREG.get(op)


# Read-modify-write ALU ops: they consume the tainted register, set flags from the result, and
# write it back. `and eax,0x80000000; jne` is a gate on a BIT of the result, so treating these as
# "the value was overwritten, therefore ignored" hides a real dependency in the one bucket whose
# meaning is "cannot be affected". Self-`xor`/`sub` are the zeroing idiom and still kill.
ALU_RMW = {"and", "or", "xor", "add", "sub", "shl", "shr", "sar", "not", "neg", "inc", "dec", "imul"}


def _direct_target(ops):
    """The VA a direct branch goes to, or None when the operand is not a bare absolute address.

    objdump's raw-binary Intel output prints a direct jump's destination as a plain `0x...`
    (already shifted by --adjust-vma, so it is a VA in the flat image). An indirect branch prints
    a register or a memory operand instead, and there is nothing static to follow.
    """
    op = ops.strip()
    return int(op, 16) if re.fullmatch(r"0x[0-9a-f]+", op) else None


def _is_cond_jump(mn):
    """Any conditional jump. Every x86 mnemonic starting with `j` is a jump, and `jmp` is the only
    unconditional one, so this is exact rather than an enumeration that can miss `jp`/`jrcxz`."""
    return mn.startswith("j") and mn != "jmp"


def _walk_block(insns, const, live, spilled, follow=False, fall_va=None):
    """One block of the taint walk. Returns (bucket, evidence, state).

    Tracks the result as a small taint set of registers rather than looking only at eax: the
    common compiler output moves eax into another register and *then* zeroes eax as the enclosing
    function's own return value, so an eax-only reader calls a live gate "ignored" (measured on
    GTA V eboot+0x2d850cc: `mov ecx,eax; xor eax,eax; test ecx,ecx`).

    `follow=False` is the legacy single-window walk: it STOPS at the first branch, which is the
    under-reporting this file's header caveat is about. `follow=True` instead reports the block's
    successors in `state["edges"]` — a zero-gate no longer terminates the walk, it forks it — so
    `classify_site_following` can read the ERROR ARM. `fall_va` is where the block continues if the
    walk runs off the end of the decoded window; None means "the window is the end of what I have".
    """
    const_forms = {"0x%x" % const, str(const)}
    live = set(live)
    gated = None                                          # the gate's own (va, mn, ops), once seen

    def st(edges=()):
        return {"live": live, "spilled": spilled, "edges": list(edges), "gated": gated}

    for i, (va, mn, ops) in enumerate(insns):
        ev = (va, mn, ops)
        if mn.startswith("(bad)") or mn.startswith(".byte") or mn in ("data16", "rex.W", "rex.RB"):
            # The decode ran into data or a prefix objdump could not attach. Say so instead of
            # letting a broken window fall into `forward`, where it is indistinguishable from a
            # real forward and quietly weakens every count in the table.
            return "undecodable", ev, st()
        parts = [p.strip() for p in ops.split(",")] if ops else []
        dst = canon(parts[0]) if parts else None
        src = canon(parts[-1]) if len(parts) > 1 else None
        if mn == "cmp" and dst in live:
            imm = parts[-1]
            if imm in const_forms:
                return "const", ev, st()
            if imm in ("0x0", "0"):
                if not follow:
                    return "nonzero", ev, st()
                gated = gated or ev                       # the gate is real; keep reading past it
                continue
            return "other-cmp", ev, st()
        if mn == "test" and dst in live and src in live:
            if not follow:
                return "nonzero", ev, st()
            gated = gated or ev
            continue
        if mn in ("mov", "movsxd", "movzx", "movsx") and src in live:
            if dst:
                live.add(dst)
            else:
                spilled = True                            # stored to memory; the gate may be later
            continue
        if mn in ("cdqe", "cdq", "cwde"):
            continue
        if mn in ALU_RMW and dst in live:
            if src == dst and mn in ("xor", "sub"):
                live.discard(dst)                         # self-xor / self-sub: the zeroing idiom
                if not live:
                    return ("forward" if spilled else "ignored"), ev, st()
                continue
            return "alu-gate", ev, st()                   # flags now derive from the result
        if mn == "ret":
            return ("forward" if ("a" in live or spilled) else "ignored"), ev, st()
        if mn == "jmp":
            # An unconditional jump ends this basic block; anything after it belongs to unrelated
            # code. Without this the scan walked straight on and could pick up a `cmp` from the
            # NEXT block, over-reporting a gate that the result never reaches.
            if not (live or spilled):
                return "ignored", ev, st()
            tgt = _direct_target(ops) if follow else None
            if tgt is None:
                return "forward", ev, st()                # legacy mode, or an indirect tail-call
            return "edges", ev, st([tgt])
        if _is_cond_jump(mn):
            if not follow:
                continue                                  # legacy: walked straight through
            # Both successors are reachable with the value in the same state, so BOTH are queued —
            # including when the branch tests flags this walk did not set. Exploring only the
            # fall-through is what let a const compare hide in the taken arm.
            tgt = _direct_target(ops)
            nxt = insns[i + 1][0] if i + 1 < len(insns) else fall_va
            if tgt is None or nxt is None:
                return "forward", ev, st()                # indirect, or the window ran out
            return "edges", ev, st([tgt, nxt])
        if mn == "call":
            live -= CALLER_SAVED
            if not live:
                return ("forward" if spilled else "ignored"), ev, st()
            continue
        if dst in live and mn not in ("cmp", "test", "push"):
            live.discard(dst)                             # overwritten from a non-result source
            if not live:
                return ("forward" if spilled else "ignored"), ev, st()
    last = insns[-1] if insns else (0, "", "")
    if not (live or spilled):
        return "ignored", (insns[0] if insns else last), st()
    if follow and fall_va is not None:
        return "edges", last, st([fall_va])               # the block simply continues past the window
    return "forward", (insns[0] if insns else last), st()


def classify_window(insns, const):
    """Classify what the code right after the call does with the returned eax — ONE window, no
    branch following. This is the legacy classifier, kept so `--no-follow-arms` reproduces every
    number this tool published before arm following existed.

    Buckets: const / nonzero / other-cmp / alu-gate all mean "branches on the result"; `forward`
    means the result left the window still live (returned, spilled, or tail-jumped) and needs a
    look by hand; `ignored` means it was dead before any read; `undecodable` means objdump could
    not decode the window, which is a VOID sample, not a negative one.
    """
    bucket, ev, _state = _walk_block(insns, const, {"a"}, False)
    return bucket, ev


# Decoded blocks per call site. A site that needs more than this is reported UNRESOLVED
# (`gate-open` / `forward`), never quietly cleared — running out of budget must not look like a
# clean negative, which is the whole failure mode arm following exists to remove.
ARM_BLOCK_LIMIT = 192

# Worst-to-best. The site's verdict is the strongest thing found on ANY reachable path, so one arm
# that const-compares makes the whole site `const` no matter how many arms drop the value.
_SITE_RANK = ("ignored", "nonzero", "gate-open", "forward", "undecodable",
              "alu-gate", "other-cmp", "const")


MAX_X86_INSN = 16          # an x86-64 instruction is at most 15 bytes; 16 is the safe round number


def block_fetcher(img, window):
    """`va -> (instructions strictly inside [va, va+window), next_va)`, memoised.

    The over-read by one maximum instruction length is what makes both halves exact, and neither is
    optional once the walk follows branches:
      * every returned instruction is COMPLETE. objdump decodes whatever bytes it is handed, so the
        last instruction in a blob cut at a fixed size decodes as `(bad)`/`.byte` junk — which is
        the `undecodable` bucket, i.e. a truncation artifact wearing the name of a real finding.
        Reading 16 bytes past the boundary and then discarding everything that starts at or after
        it leaves only instructions whose bytes were all present.
      * `next_va` is EXACT, with no instruction-length table anywhere: it is simply the first
        instruction that starts at or after the boundary, which is where the block continues if the
        walk runs off the end of the window.
    Junk in the MIDDLE of a window is still reported `undecodable`, because there it is data rather
    than truncation — which is the distinction the bucket is supposed to carry.
    """
    cache = {}

    def fetch(va):
        if va not in cache:
            fo = img.foff(va)
            if fo is None:
                cache[va] = (None, None)                 # unmapped: honestly unresolvable
            else:
                end = va + window
                ins = disasm(img.raw[fo:fo + window + MAX_X86_INSN], va)
                body = [i for i in ins if i[0] < end]
                tail = [i for i in ins if i[0] >= end]
                cache[va] = (body, tail[0][0] if tail else None) if body else (None, None)
        return cache[va]

    return fetch


def classify_site_following(fetch, start, const, limit=ARM_BLOCK_LIMIT):
    """Classify a call site by exploring EVERY reachable path the result stays live on.

    This is the fix for the caveat at the top of this file. The legacy walk stops at the first
    branch, so `test eax,eax; je .ok; cmp eax,<errno>; ...` — a site that gates generically and
    discriminates inside its error arm — was reported as a plain `nonzero`, i.e. as if it could not
    tell two error codes apart. Here the gate forks the walk and both arms are read.

    Buckets are the legacy ones plus **`gate-open`**: the site does gate on the value, and at least
    one reachable arm left this scan's reach (returned it, spilled it, jumped indirectly, or ran the
    block budget out). `nonzero` is correspondingly STRONGER than in the legacy walk — it now means
    the value is gated and then provably dead on every reachable path, so that site cannot tell one
    non-zero answer from another. `gate-open` and `forward` are the not-cleared buckets.
    """
    outcomes, gate_ev, other_ev = set(), None, None
    work = [(start, frozenset({"a"}), False)]
    seen, blocks = set(), 0
    while work:
        key = work.pop()
        if key in seen:
            continue
        seen.add(key)
        va, live, spilled = key
        blocks += 1
        if blocks > limit:
            outcomes.add("forward")                      # budget: unresolved, not cleared
            break
        insns, fall = fetch(va)
        if not insns:
            outcomes.add("forward")                      # unmapped, or nothing decodable: not cleared
            continue
        bucket, ev, state = _walk_block(insns, const, set(live), spilled, True, fall)
        gate_ev = gate_ev or state["gated"]
        if bucket == "edges":
            nlive, nspill = frozenset(state["live"]), state["spilled"]
            work += [(tgt, nlive, nspill) for tgt in state["edges"]]
            continue
        if bucket == "const":
            return "const", ev                           # one positive instance settles the site
        outcomes.add(bucket)
        if bucket in ("other-cmp", "alu-gate", "undecodable"):
            other_ev = other_ev or ev
    if gate_ev:
        if "forward" in outcomes:
            outcomes = (outcomes - {"forward"}) | {"gate-open"}
        outcomes = (outcomes - {"ignored"}) | {"nonzero"}
    verdict = max(outcomes or {"ignored"}, key=_SITE_RANK.index)
    return verdict, (other_ev or gate_ev or (start, "", ""))


ENDBR64 = b"\xf3\x0f\x1e\xfa"


def stub_entry_points(img, site):
    """Addresses a caller may target for the PLT-ish stub whose `jmp *[rip+d]` sits at `site`.

    A caller targets the stub's FIRST byte, which is not always the `jmp`: a CET-enabled stub opens
    with `endbr64`, so calls land 4 bytes earlier. Looking only at the `jmp` address would report
    zero call sites for an entire module with nothing about the output looking wrong — a whole
    title silently classified "imports it but never inspects the result".

    Split out from scan_module so it can be tested without a module fixture: no dump in the local
    corpus uses a CET prologue, so this path is exercised by nothing but its unit test.
    """
    entries = [site]
    fo = img.foff(site - 4)
    if fo is not None and img.raw[fo:fo + 4] == ENDBR64:
        entries.append(site - 4)
    return entries


def call_sites(img, xref, sym_index):
    """Every (site VA, kind) that reaches the import bound to `sym_index`."""
    sites = []
    for slot in img.jump_slots(sym_index):
        for site, kind in xref.get(slot, []):
            if kind == "jmp*":
                for e in stub_entry_points(img, site):
                    sites += [(s, "call") for s, k in xref.get(e, []) if k == "call"]
            elif kind == "call*":                            # direct indirect call through the slot
                sites.append((site, "call*"))
    return sorted(set(sites))


def classify_sites(img, sites, const, window, follow=True):
    """(census, details) for a set of call sites. `details` rows are (site, bucket, ev, arg0)."""
    census, details = {}, []
    fetch = block_fetcher(img, window)
    for site, kind in sites:
        after = site + (6 if kind == "call*" else 5)
        fo = img.foff(after)
        if fo is None:
            continue
        if follow:
            bucket, ev = classify_site_following(fetch, after, const)
        else:
            bucket, ev = classify_window(disasm(img.raw[fo:fo + window], after), const)
        census[bucket] = census.get(bucket, 0) + 1
        details.append((site, bucket, ev, arg0_before(img.raw, img.foff(site))))
    return census, details


def scan_module(path, nid, const, window, verbose=False, follow=True):
    """Returns (status, detail) where status is 'no-import', 'no-slot', or a bucket census dict."""
    try:
        img = Image(flatten(path))
    except Exception as e:                                   # noqa: BLE001 — report, keep going
        return "unreadable: %s" % e, {}
    idx = img.symbol_index(nid)
    if idx is None:
        return "no-import", {}
    if not img.jump_slots(idx):
        return "import-no-slot", {}
    census, details = classify_sites(img, call_sites(img, scan_code(img), idx), const, window, follow)
    if verbose:
        for site, bucket, ev, arg in details:
            argtxt = ("id=%#x%s" % (arg[0], "" if arg[1] else "?")) if arg else "id=?"
            print("    call at %#x %-10s -> %-9s %s %s" % (site, argtxt, bucket, ev[1], ev[2]))
    return "ok", census


GATED = ("const", "nonzero", "other-cmp", "alu-gate", "gate-open")
# Buckets that say "this site was not resolved", as opposed to "resolved, and it does not care".
UNRESOLVED = ("gate-open", "forward", "undecodable")


def load_nid_names(libs_dir):
    """{NID: (function name, library)} from a PS5 firmware `genstub.py` library dump.

    Each generated `libSceXxx.c` binds its imports with
    `sprx_dlsym(__handle, "<NID>", &__ptr_<funcName>)`, so the file is a NID<->name table. This
    is symbolication only: a NID that is absent stays a NID and is still scanned and reported, so
    a missing or partial dump degrades the *labels* and never the census.
    """
    names = {}
    pat = re.compile(r'sprx_dlsym\(__handle,\s*"([^"]+)",\s*&__ptr_([A-Za-z0-9_]+)\)')
    try:
        entries = sorted(os.listdir(libs_dir))
    except OSError as e:
        raise SystemExit("--names %s: %s" % (libs_dir, e))
    for fn in entries:
        if not fn.endswith(".c"):
            continue
        try:
            with open(os.path.join(libs_dir, fn), errors="ignore") as f:
                txt = f.read()
        except OSError:
            continue
        for nid, name in pat.findall(txt):
            names.setdefault(nid, (name, fn[:-2]))
    if not names:
        raise SystemExit("--names %s: no `sprx_dlsym(...)` lines — not a genstub library dump" % libs_dir)
    return names


def sweep_module(path, const, window, names, min_gated, verbose, follow=True):
    """Classify EVERY imported NID of one module. Prints a row per import that has call sites."""
    img = Image(flatten(path))
    if img.dynsym() is None:
        # Returning an empty sweep here would print "0 imported NIDs are called", which reads as a
        # module that imports nothing rather than as a table this tool could not locate. Same
        # fail-loudly rule as the symtab bound in dynsym().
        raise ValueError("no dynamic symbol/string table (DT_SYMTAB/DT_STRTAB): cannot enumerate imports")
    xref = scan_code(img)                                    # once for the module, not once per NID
    rows = []
    for nid, idx in img.imported_nids():
        sites = call_sites(img, xref, idx)
        if not sites:
            continue                                         # imported, never called: no gate here
        census, details = classify_sites(img, sites, const, window, follow)
        gated = sum(census.get(b, 0) for b in GATED)
        rows.append((gated, len(sites), nid, census, details))
    rows.sort(key=lambda r: (-r[0], -r[1], r[2]))
    shown = 0
    for gated, nsites, nid, census, details in rows:
        if gated < min_gated:
            continue
        shown += 1
        name, lib = names.get(nid, ("", ""))
        print("%-12s %-52s %-28s sites=%-5d gated=%-5d %s" % (
            nid, name or "?", lib or "?", nsites, gated,
            " ".join("%s=%d" % kv for kv in sorted(census.items()))))
        if verbose:
            for site, bucket, ev, arg in details:
                if bucket not in GATED:
                    continue
                argtxt = ("id=%#x%s" % (arg[0], "" if arg[1] else "?")) if arg else "id=?"
                print("    call at %#x %-10s -> %-9s %s %s" % (site, argtxt, bucket, ev[1], ev[2]))
    return rows, shown


def sweep_summary(label, rows, shown, min_gated):
    """The trailing `#` block: what was hidden, and why the table is not a clean partition.

    A row that is not gated is NOT automatically a row that cannot matter. `forward` means the
    result left the scan's reach still live and needs a look by hand, `gate-open` means the same of
    an arm, and `undecodable` is a void sample. Reporting only "N called, M shown" invites exactly
    the wrong reading — that everything below the cut is cleared — which is the expensive direction
    in a document whose whole purpose is to stop the next reader re-deriving a dead answer. So state
    the split explicitly, count the rows that are not resolved SEPARATELY from the ones that are not
    gated (a `gate-open` row is both gated and unresolved), and print the site-level bucket totals.
    """
    gated_rows = sum(1 for g, _n, _i, _c, _d in rows if g)
    ignored_only = sum(1 for g, _n, _i, c, _d in rows if not g and set(c) <= {"ignored"})
    unresolved = len(rows) - gated_rows - ignored_only
    open_rows = sum(1 for _g, _n, _i, c, _d in rows if set(c) & set(UNRESOLVED))
    totals = {}
    for _g, _n, _i, c, _d in rows:
        for k, v in c.items():
            totals[k] = totals.get(k, 0) + v
    print("# %s: %d imported NIDs are called; %d shown at --min-gated=%d"
          % (label, len(rows), shown, min_gated))
    print("#   %d gated, %d ignored-only (cannot matter), %d neither "
          "(no gate, and >=1 site not cleared)" % (gated_rows, ignored_only, unresolved))
    # A `gate-open` row IS gated — it branches on the result — so it is counted above and is still
    # not cleared. Reporting only the "neither" column would hide exactly those rows, which is the
    # expensive direction: they read as answered.
    print("#   %d rows carry >=1 site the scan could not resolve (%s) — read those by hand"
          % (open_rows, "/".join(UNRESOLVED)))
    print("#   site buckets: %s" % " ".join("%s=%d" % kv for kv in sorted(totals.items())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="SELF/PRX/ELF module, or an app0 directory")
    ap.add_argument("--nid", help="one NID to scan; omit and pass --all-nids to sweep every import")
    ap.add_argument("--all-nids", action="store_true",
                    help="classify EVERY imported NID of the module in one pass")
    ap.add_argument("--names", metavar="DIR",
                    help="PS5 firmware genstub library dump, to label NIDs with function names")
    ap.add_argument("--min-gated", type=int, default=1,
                    help="with --all-nids, hide imports with fewer than N gated call sites (default 1)")
    ap.add_argument("--const", default="0x805a1001")
    ap.add_argument("--window", type=lambda s: int(s, 0), default=48)
    ap.add_argument("--no-follow-arms", action="store_true",
                    help="legacy single-window walk: stop at the first branch and never read the "
                         "error arm. UNDER-REPORTS const-sensitivity (see this file's header); it "
                         "exists to reproduce numbers published before arm following")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    if bool(a.nid) == bool(a.all_nids):
        ap.error("pass exactly one of --nid <NID> or --all-nids")
    const = int(a.const, 0)

    targets = []
    if os.path.isdir(a.target):
        for root, _dirs, files in os.walk(a.target):
            for f in files:
                if f == "eboot.bin" or f.lower().endswith((".prx", ".sprx")):
                    targets.append(os.path.join(root, f))
    else:
        targets = [a.target]

    names = load_nid_names(a.names) if a.names else {}
    follow = not a.no_follow_arms

    if a.all_nids:
        for t in sorted(targets):
            label = os.path.relpath(t, a.target) if os.path.isdir(a.target) else t
            try:
                rows, shown = sweep_module(t, const, a.window, names, a.min_gated, a.verbose, follow)
            except Exception as e:                           # noqa: BLE001 — report, keep going
                # Loud and per module: a directory sweep must not lose one module's failure in the
                # noise of the others', and this is where dynsym()'s deliberate raises surface.
                print("# %s: SCAN FAILED (no census for this module): %s" % (label, e))
                continue
            sweep_summary(label, rows, shown, a.min_gated)
        return 0

    total = {}
    for t in sorted(targets):
        status, census = scan_module(t, a.nid, const, a.window, a.verbose, follow)
        if status == "no-import" and not a.verbose:
            continue
        label = os.path.relpath(t, a.target) if os.path.isdir(a.target) else t
        print("%-56s %s %s" % (label, status,
                               " ".join("%s=%d" % kv for kv in sorted(census.items())) or ""))
        for k, v in census.items():
            total[k] = total.get(k, 0) + v
    if total:
        print("TOTAL " + " ".join("%s=%d" % kv for kv in sorted(total.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
