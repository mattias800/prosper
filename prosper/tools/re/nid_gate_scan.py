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

Buckets (see `classify_window`):
    const        the window compares eax against --const (the gate idiom this was written for)
    nonzero      `test eax,eax` / `cmp eax,0` / sign test -> conditional branch (any error gates)
    other-cmp    compares eax against some *other* immediate
    forward      eax is moved/returned/stored without a local branch — the gate, if any, is in a
                 caller or behind a spilled value; needs a look by hand
    ignored      eax is dead before it is read — this call site cannot be affected

Usage:
    nid_gate_scan.py <module|app0-dir> --nid <NID> [--const 0x805a1001] [--window 48] [-v]

`<module>` is a SELF/PRX/eboot.bin (flattened in memory — no temp files) or an already-flat ELF.
Given a directory, every `eboot.bin` and `*.prx`/`*.sprx` under it is scanned.

Example — the sysmodule gate (#2002); `0x805A1001` is SCE_SYSMODULE_ERROR_UNLOADED:
    nid_gate_scan.py testdata/PPSA04263-app0 --nid fMP5NHUOaMk --const 0x805a1001
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

    def symbol_index(self, nid):
        """Index of the dynsym entry whose NID is `nid` ("<NID>#<lib>#<mod>"), or None."""
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
        want = nid.encode() + b"#"
        for i in range(n):
            base = so + i * syment
            if base + syment > len(self.raw):
                break
            st_name = struct.unpack_from("<I", self.raw, base)[0]
            if self.raw[to + st_name:to + st_name + len(want)] == want:
                return i
        return None

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
        [od, "-D", "-b", "binary", "-m", "i386:x86-64", "-M", "intel",
         "--adjust-vma=%#x" % base, tmp],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("objdump failed (%d): %s" % (r.returncode, r.stderr.strip()[:200]))
    out = r.stdout
    insns = []
    for line in out.splitlines():
        m = re.match(r"\s*([0-9a-f]+):\s+((?:[0-9a-f]{2} )+)\s*(\S+)\s*(.*)", line)
        if m:
            insns.append((int(m.group(1), 16), m.group(3), m.group(4).split("#")[0].strip()))
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


def classify_window(insns, const):
    """Classify what the code right after the call does with the returned eax.

    Tracks the result as a small taint set of registers rather than looking only at eax: the
    common compiler output moves eax into another register and *then* zeroes eax as the enclosing
    function's own return value, so an eax-only reader calls a live gate "ignored" (measured on
    GTA V eboot+0x2d850cc: `mov ecx,eax; xor eax,eax; test ecx,ecx`).

    Buckets: const / nonzero / other-cmp / alu-gate all mean "branches on the result"; `forward`
    means the result left the window still live (returned, spilled, or tail-jumped) and needs a
    look by hand; `ignored` means it was dead before any read; `undecodable` means objdump could
    not decode the window, which is a VOID sample, not a negative one.
    """
    const_forms = {"0x%x" % const, str(const)}
    live = {"a"}
    spilled = False
    for va, mn, ops in insns:
        if mn.startswith("(bad)") or mn.startswith(".byte") or mn in ("data16", "rex.W", "rex.RB"):
            # The decode ran into data or a prefix objdump could not attach. Say so instead of
            # letting a broken window fall into `forward`, where it is indistinguishable from a
            # real forward and quietly weakens every count in the table.
            return "undecodable", (va, mn, ops)
        parts = [p.strip() for p in ops.split(",")] if ops else []
        dst = canon(parts[0]) if parts else None
        src = canon(parts[-1]) if len(parts) > 1 else None
        if mn == "cmp" and dst in live:
            imm = parts[-1]
            if imm in const_forms:
                return "const", (va, mn, ops)
            if imm in ("0x0", "0"):
                return "nonzero", (va, mn, ops)
            return "other-cmp", (va, mn, ops)
        if mn == "test" and dst in live and src in live:
            return "nonzero", (va, mn, ops)
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
                    return ("forward" if spilled else "ignored"), (va, mn, ops)
                continue
            return "alu-gate", (va, mn, ops)              # flags now derive from the result
        if mn == "ret":
            return ("forward" if ("a" in live or spilled) else "ignored"), (va, mn, ops)
        if mn == "jmp":
            # An unconditional jump ends this basic block; anything after it belongs to unrelated
            # code. Without this the scan walked straight on and could pick up a `cmp` from the
            # NEXT block, over-reporting a gate that the result never reaches.
            return ("forward" if (live or spilled) else "ignored"), (va, mn, ops)
        if mn == "call":
            live -= CALLER_SAVED
            if not live:
                return ("forward" if spilled else "ignored"), (va, mn, ops)
            continue
        if dst in live and mn not in ("cmp", "test", "push"):
            live.discard(dst)                             # overwritten from a non-result source
            if not live:
                return ("forward" if spilled else "ignored"), (va, mn, ops)
    return ("forward" if (live or spilled) else "ignored"), (insns[0] if insns else (0, "", ""))


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


def scan_module(path, nid, const, window, verbose=False):
    """Returns (status, detail) where status is 'no-import', 'no-slot', or a bucket census dict."""
    try:
        img = Image(flatten(path))
    except Exception as e:                                   # noqa: BLE001 — report, keep going
        return "unreadable: %s" % e, {}
    idx = img.symbol_index(nid)
    if idx is None:
        return "no-import", {}
    slots = img.jump_slots(idx)
    if not slots:
        return "import-no-slot", {}
    xref = scan_code(img)
    sites = []
    for slot in slots:
        for site, kind in xref.get(slot, []):
            if kind == "jmp*":
                for e in stub_entry_points(img, site):
                    sites += [(s, "call") for s, k in xref.get(e, []) if k == "call"]
            elif kind == "call*":                            # direct indirect call through the slot
                sites.append((site, "call*"))
    census, details = {}, []
    for site, kind in sorted(set(sites)):
        after = site + (6 if kind == "call*" else 5)
        fo = img.foff(after)
        if fo is None:
            continue
        bucket, ev = classify_window(disasm(img.raw[fo:fo + window], after), const)
        census[bucket] = census.get(bucket, 0) + 1
        details.append((site, bucket, ev, arg0_before(img.raw, img.foff(site))))
    if verbose:
        for site, bucket, ev, arg in details:
            argtxt = ("id=%#x%s" % (arg[0], "" if arg[1] else "?")) if arg else "id=?"
            print("    call at %#x %-10s -> %-9s %s %s" % (site, argtxt, bucket, ev[1], ev[2]))
    return "ok", census


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="SELF/PRX/ELF module, or an app0 directory")
    ap.add_argument("--nid", required=True)
    ap.add_argument("--const", default="0x805a1001")
    ap.add_argument("--window", type=lambda s: int(s, 0), default=48)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    const = int(a.const, 0)

    targets = []
    if os.path.isdir(a.target):
        for root, _dirs, files in os.walk(a.target):
            for f in files:
                if f == "eboot.bin" or f.lower().endswith((".prx", ".sprx")):
                    targets.append(os.path.join(root, f))
    else:
        targets = [a.target]

    total = {}
    for t in sorted(targets):
        status, census = scan_module(t, a.nid, const, a.window, a.verbose)
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
