#!/usr/bin/env python3
# xref.py — cross-reference finder for an (unsymbolicated) PS5 module.
#
# PS5 eboot/PRX code is position-independent: internal references appear as
#   - direct calls          e8 <rel32>
#   - rip-relative load/store REX.W/REX.WR 8d|8b|89 modrm(rip) <disp32>
#   - indirect call/jump     ff /2|/4 modrm(rip) <disp32>
#   - byte/dword store-imm   c6 05 <disp32> <imm8>  /  c7 05 <disp32> <imm32>   (storeb/stored)
#   - byte compare-imm       80 3d <disp32> <imm8>                              (cmpb)
#   - data pointers          absolute pointers stored in data, emitted as
#                            R_X86_64_RELATIVE relocations (DT_RELA / DT_SCE_RELA)
# The store-imm/compare-imm forms write/test a guest FLAG in place; without them the *writer* of a
# 1-byte state flag is invisible (it is not a reg-store) -- e.g. the Bendy Agc suspend-point SAFE
# flag (#1195), whose three `c6 05 .. 01` setters this decodes but objdump/readelf cannot name.
# so "grep for who references address X" is not a text search — it needs all three
# decoded. Standard tools don't read the Sony (DT_SCE_*) relocation encoding, and
# rip-relative refs never appear as literals at all. This tool builds a reverse
# index over all three and answers "who references <addr>" and "what does the
# function at <addr> reference".
#
# Input is a FLATTENED ELF (p_offset==p_vaddr) as produced by tools/il2cpp/prx_to_elf.py.
# Runtime address = module_load_base + VA (e.g. eboot base 0x410000000; it moved from
# 0x400000000 in #825 — see prosper/src/host/boot_program.hpp for the authoritative set).
#
# A SHORT string has no address to reference at all. Clang materialises a std::string built from a
# literal of <= 22 bytes into the small-buffer with `movabs`/`mov` IMMEDIATES, and then the .rodata
# copy of that literal can end up with zero references of every kind -- no rip-relative lea, no
# relocation, nothing. `to <string VA>` then answers 0 for a string the guest demonstrably uses, which
# reads as "the guest cannot be asking for this" (it cost the Sonic Origins lane a whole hypothesis:
# `ui/ui_startup` has zero references AND is requested on every boot -- #1905). `imm` closes that hole
# by searching the executable segments for the string's own bytes appearing as instruction operands.
#
# Usage:
#   xref.py <module.elf> to   <hexaddr>     # who references this address
#   xref.py <module.elf> from <hexaddr>     # what this function references (rip-lea/call, ~2KB window)
#   xref.py <module.elf> reloc <hexaddr>    # data-pointer relocations targeting this address only
#   xref.py <module.elf> imm  <string>      # who BUILDS this string from inline immediates

import struct, sys
from collections import defaultdict

MODRM_RIP = {0x05, 0x0d, 0x15, 0x1d, 0x25, 0x2d, 0x35, 0x3d}   # mod=00, rm=101 (rip-rel), reg=any


class Module:
    def __init__(self, path):
        self.raw = open(path, 'rb').read()
        raw = self.raw
        e_phoff, = struct.unpack_from('<Q', raw, 0x20)
        phentsize, phnum = struct.unpack_from('<HH', raw, 0x36)
        self.segs = []          # (va, foff, filesz, flags)
        self.dyn_va = 0
        for i in range(phnum):
            t, fl, off, va, pa, fs, ms, al = struct.unpack_from('<IIQQQQQQ', raw, e_phoff + i * phentsize)
            if t == 1:
                self.segs.append((va, off, fs, fl))
            elif t == 2:
                self.dyn_va = va
        self._build()

    def foff(self, va):
        for v, o, fs, fl in self.segs:
            if v <= va < v + fs:
                return o + (va - v)
        return None

    def va_at(self, fo):
        for v, o, fs, fl in self.segs:
            if o <= fo < o + fs:
                return v + (fo - o)
        return None

    def func_start(self, va):
        # nearest int3 (>=2x 0xcc) padding boundary at or before va
        fo = self.foff(va)
        if fo is None:
            return va
        p = fo
        while p > fo - 0x4000 and p > 1:
            if self.raw[p] == 0xcc and self.raw[p - 1] == 0xcc:
                return self.va_at(p + 1) or va
            p -= 1
        return va

    def _build(self):
        raw = self.raw
        self.data_xref = defaultdict(list)   # target VA -> [site VAs] (relocated data pointers)
        self.code_xref = defaultdict(list)   # target VA -> [(site VA, kind)]
        # --- relocations (standard DT_RELA + Sony DT_SCE_RELA) ---
        if self.dyn_va:
            o = self.foff(self.dyn_va)
            tags = {}
            while o is not None:
                tag, val = struct.unpack_from('<qQ', raw, o); o += 16
                if tag == 0:
                    break
                tags.setdefault(tag, val)
            for tv, sv in [(0x7, 0x8), (0x17, 0x2), (0x6100002f, 0x61000031), (0x61000029, 0x6100002d)]:
                if tv in tags and sv in tags and self.foff(tags[tv]) is not None:
                    base = self.foff(tags[tv])
                    for i in range(tags[sv] // 24):
                        r_off, r_info, r_add = struct.unpack_from('<QQq', raw, base + i * 24)
                        if (r_info & 0xffffffff) == 8:   # R_X86_64_RELATIVE: target VA = addend
                            self.data_xref[r_add].append(r_off)
        # --- code scan: direct calls, RIP-relative data accesses, and calls/jumps through slots ---
        for v, o, fs, fl in self.segs:
            if not (fl & 1):
                continue
            d = raw[o:o + fs]
            n = len(d)
            # Examine every possible instruction start. Each decoder below owns its exact bounds
            # check; a shared `range(n - 7)` dropped seven-byte store/cmp instructions that ended
            # exactly at the PT_LOAD boundary (#1314), as well as shorter tail calls.
            for i in range(n):
                b = d[i]
                if (i + 7 <= n and b in (0x48, 0x4c) and
                        d[i + 1] in (0x8d, 0x8b, 0x89) and d[i + 2] in MODRM_RIP):
                    disp = struct.unpack_from('<i', d, i + 3)[0]
                    site = v + i
                    kind = {0x8d: 'lea', 0x8b: 'load', 0x89: 'store'}[d[i + 1]]
                    self.code_xref[site + 7 + disp].append((site, kind))
                elif i + 6 <= n and b == 0xff and (d[i + 1] & 0xc7) == 0x05:
                    reg = (d[i + 1] >> 3) & 7
                    if reg in (2, 4):
                        disp = struct.unpack_from('<i', d, i + 2)[0]
                        site = v + i
                        self.code_xref[site + 6 + disp].append((site, 'call*' if reg == 2 else 'jmp*'))
                elif i + 5 <= n and b == 0xe8:
                    disp = struct.unpack_from('<i', d, i + 1)[0]
                    site = v + i
                    self.code_xref[site + 5 + disp].append((site, 'call'))
                elif i + 7 <= n and b == 0xc6 and d[i + 1] == 0x05:
                    # mov BYTE PTR [rip+disp], imm8  (c6 /0) -- 7 bytes. This is the byte-store form
                    # that sets a guest 1-byte flag; objdump/readelf and the loads/stores above all
                    # miss it, so it is how the *writer* of a flag byte is found (e.g. the Bendy Agc
                    # suspend-point SAFE flag setter, #1195).
                    disp = struct.unpack_from('<i', d, i + 2)[0]
                    site = v + i
                    self.code_xref[site + 7 + disp].append((site, 'storeb'))
                elif i + 10 <= n and b == 0xc7 and d[i + 1] == 0x05:
                    # mov DWORD PTR [rip+disp], imm32  (c7 /0) -- 10 bytes; the dword store-immediate.
                    disp = struct.unpack_from('<i', d, i + 2)[0]
                    site = v + i
                    self.code_xref[site + 10 + disp].append((site, 'stored'))
                elif i + 7 <= n and b == 0x80 and d[i + 1] == 0x3d:
                    # cmp BYTE PTR [rip+disp], imm8  (80 /7) -- 7 bytes; the paired flag-*read* form
                    # (a spin-loop testing a 1-byte flag, e.g. the Bendy suspend-point watchdog).
                    disp = struct.unpack_from('<i', d, i + 2)[0]
                    site = v + i
                    self.code_xref[site + 7 + disp].append((site, 'cmpb'))


def find_immediate_builds(m, needle, span=0x60):
    """Locate code that materialises `needle` (bytes) from instruction immediates.

    Returns [(first_site_va, last_site_va, [(va, lo, hi), ...])] — one entry per cluster of nearby
    executable-segment occurrences whose covered byte ranges together span the whole string.

    The chunk sizes are the ones a compiler can encode as an immediate: 8 (movabs) and 4 (mov
    imm32) carry the string, and at most a 3-byte tail may arrive as a 2- or 1-byte store.

    Only the >= 4-byte windows form clusters, and a cluster is reported only when they cover the
    whole string apart from such a tail. That bound is what makes the answer trustworthy: 1-byte
    windows match constantly in a dense instruction stream, so admitting them into the cover lets
    a stretch of code "build" any string. Requiring the 4- and 8-byte windows to do the work leaves
    the tail as the only concession, which is the shape the compiler actually emits.

    A needle shorter than 4 bytes is REFUSED (ValueError) rather than answered, because the floor
    that makes the answer meaningful cannot exist below the needle's own length: every plain
    occurrence of those bytes anywhere in the instruction stream would be reported as a
    construction, in the same confident format as a real one.

    Two precision limits the caller should read off the output rather than assume away:
      * a cluster is single-linkage over CONSECUTIVE matches no more than `span` bytes apart, so it
        can grow longer than `span` overall — the printed first..last range is what bounds it;
      * a tail-completed hit's small window is accepted anywhere inside the cluster's span, with no
        positional or origin relation to the strong stores, so its precision rests entirely on
        those. The reported [lo:hi] sizes are how a reader tells the two cases apart.
    """
    n = len(needle)
    if n == 0:
        return []
    if n < 4:
        raise ValueError(
            "needle %r is %d byte(s); `imm` needs at least 4. Below that the >=4-byte window floor "
            "that makes a match meaningful cannot apply, and every plain occurrence of these bytes "
            "in the instruction stream would be reported as a construction." % (needle, n))

    def scan(sizes):
        windows = [(lo, lo + s) for s in sizes if s <= n for lo in range(0, n - s + 1)]
        found = []
        for v, o, fs, fl in m.segs:
            if not (fl & 1):                    # executable segments only
                continue
            seg = m.raw[o:o + fs]
            for lo, hi in windows:
                chunk = needle[lo:hi]
                start = 0
                while True:
                    i = seg.find(chunk, start)
                    if i < 0:
                        break
                    found.append((v + i, lo, hi))
                    start = i + 1
        return sorted(found)

    strong = scan((8, 4))
    if not strong:
        return []
    weak = None                                 # scanned lazily; only a <= 3-byte tail may use it
    clusters, cur = [], [strong[0]]
    for h in strong[1:]:
        if h[0] - cur[-1][0] <= span:
            cur.append(h)
        else:
            clusters.append(cur)
            cur = [h]
    clusters.append(cur)

    out = []
    for c in clusters:
        covered = set()
        for _, lo, hi in c:
            covered.update(range(lo, hi))
        missing = set(range(n)) - covered
        parts = list(c)
        if missing:
            if len(missing) > 3:
                continue                        # partial match: not a construction of this string
            if weak is None:
                weak = scan((3, 2, 1))
            lo_va, hi_va = c[0][0] - span, c[-1][0] + span
            for va, lo, hi in weak:
                if lo_va <= va <= hi_va and missing & set(range(lo, hi)):
                    missing -= set(range(lo, hi))
                    parts.append((va, lo, hi))
            if missing:
                continue
        # Keep the largest window at each distinct site, then drop the windows that are merely
        # sub-ranges of a larger one at the same origin — an 8-byte movabs also matches its own
        # 4-byte prefixes at the next four addresses, and printing those hides the real stores.
        best = {}
        for va, lo, hi in parts:
            if va not in best or (hi - lo) > (best[va][1] - best[va][0]):
                best[va] = (lo, hi)
        cand = sorted((va, lo, hi) for va, (lo, hi) in best.items())
        merged = [w for w in cand
                  if not any(o is not w and o[0] - o[1] == w[0] - w[1]
                             and o[1] <= w[1] and w[2] <= o[2] and (o[2] - o[1]) > (w[2] - w[1])
                             for o in cand)]
        out.append((merged[0][0], merged[-1][0], merged))
    return out


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 1
    m = Module(sys.argv[1])
    mode = sys.argv[2]
    if mode == 'imm':
        text = sys.argv[3]
        needle = text.encode('utf-8')
        try:
            builds = find_immediate_builds(m, needle)
        except ValueError as exc:
            print(f"refused: {exc}")
            return 1
        print(f"inline-immediate constructions of {text!r} ({len(needle)} bytes): {len(builds)}")
        for first, last, parts in builds:
            where = ", ".join(f"0x{va:x}[{lo}:{hi}]" for va, lo, hi in parts)
            print(f"   0x{first:x}..0x{last:x}  (in func 0x{m.func_start(first):x})  {where}")
        # The .rodata copy and its (often zero) reference count — the other half of "who uses this
        # string", and the number that misleads on its own.
        lits = []
        for v, o, fs, fl in m.segs:
            if fl & 1:
                continue
            start = 0
            while True:
                i = m.raw[o:o + fs].find(needle + b'\0', start)
                if i < 0:
                    break
                if i == 0 or m.raw[o + i - 1] == 0:
                    lits.append(v + i)
                start = i + 1
        print(f"literal copies in non-executable segments: {len(lits)}")
        for va in lits[:16]:
            print(f"   0x{va:x}: {len(m.code_xref.get(va, []))} code refs, "
                  f"{len(m.data_xref.get(va, []))} data relocations")
        return 0
    addr = int(sys.argv[3], 0)
    if mode in ('to', 'reloc'):
        drefs = m.data_xref.get(addr, [])
        print(f"data-pointer relocations targeting 0x{addr:x}: {len(drefs)}")
        for s in drefs[:32]:
            print(f"   ptr stored at 0x{s:x}")
        if mode == 'to':
            crefs = m.code_xref.get(addr, [])
            print(f"code references to 0x{addr:x}: {len(crefs)}")
            for s, k in crefs[:32]:
                print(f"   {k:4s} at 0x{s:x}  (in func 0x{m.func_start(s):x})")
    elif mode == 'from':
        start = m.func_start(addr)
        print(f"references made by func 0x{start:x} (window from 0x{addr:x}):")
        fo = m.foff(addr)
        seen = set()
        for tgt, sites in m.code_xref.items():
            for s, k in sites:
                if addr <= s < addr + 0x800 and tgt not in seen:
                    seen.add(tgt)
                    print(f"   {k:4s} -> 0x{tgt:x}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
