#!/usr/bin/env python3
# xref.py — cross-reference finder for an (unsymbolicated) PS5 module.
#
# PS5 eboot/PRX code is position-independent: internal references appear as
#   - direct calls          e8 <rel32>
#   - rip-relative load/store REX.W/REX.WR 8d|8b|89 modrm(rip) <disp32>
#   - indirect call/jump     ff /2|/4 modrm(rip) <disp32>
#   - byte/dword store-imm   c6 05 <disp32> <imm8>  /  c7 05 <disp32> <imm32>   (storeb/stored)
#   - ALU reg,[rip+d32]      48/4c 2b|3b|03 /r <disp32>                        (sub/cmp/add, reads)
#   - ALU [rip+d32],imm8     48/4c 83 /r <disp32> <imm8>                       (addi..cmpi, mostly writes)
#   - VEX vmovups            c5 f8|fc 10|11 /r <disp32>                        (vload/vstore x/y)
# Every reference is printed with an access class (r / w / rw / & / x). The tool is usually asked
# "who WRITES this", and the forms on the three lines above were the ones missing (#2025) -- all
# four are mutators, so an answer made only of reads looked complete instead of partial.
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
# Handing it the SELF instead (`eboot.bin` as it ships) is REFUSED, not answered — see #2346. It used
# to parse the SCE container header as though it were an ELF and report 0 references for every
# address, which is the one answer this tool must never produce by accident: "who writes X" answered
# with 0 reads as "nobody writes X", and that is a conclusion strong enough to redirect an
# investigation. Flatten first:
#     python3 tools/il2cpp/prx_to_elf.py <DUMP>/eboot.bin /tmp/eboot.elf
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

# What each decoded form DOES to the referenced address. The tool's usual question is "who writes X",
# and #2025 was the case where that could not be answered: every missed form was a mutator, so an
# answer made only of reads looked complete. Printing the class makes a read-only answer visibly
# read-only instead of merely short.
#   r  reads it      w  writes it      rw  read-modify-write      &  takes its address
#   x  transfers control through it (the slot itself is read)
ACCESS = {
    'lea': '&', 'load': 'r', 'store': 'w',
    'call': 'x', 'call*': 'x', 'jmp*': 'x',
    'storeb': 'w', 'stored': 'w', 'cmpb': 'r',
    'sub': 'r', 'cmp': 'r', 'add': 'r',
    'addi': 'rw', 'ori': 'rw', 'adci': 'rw', 'sbbi': 'rw',
    'andi': 'rw', 'subi': 'rw', 'xori': 'rw', 'cmpi': 'r',
    'vloadx': 'r', 'vloady': 'r', 'vstorex': 'w', 'vstorey': 'w',
    'reloc': '&',
}


def access(kind):
    """Access class for a decoded kind; '?' for anything not classified rather than a guess."""
    return ACCESS.get(kind, '?')


ELF_MAGIC = b'\x7fELF'
SELF_MAGIC = b'\x4f\x15\x3d\x1d'   # PS5/PS4 SELF container (eboot.bin, *.sprx as shipped)


class BadModule(ValueError):
    """The input is not a module this tool can answer questions about.

    Raised instead of returning zeros. #2346: handed an `eboot.bin` (a SELF), this tool used to read
    e_phoff/phnum straight out of the SCE container header -- on a real one that yields phnum=45792
    with phentsize=0 -- build an empty segment table, and then answer **0 references** for every
    address, including the target of a direct `call`. That output is indistinguishable from a true
    negative, and a true negative ("nobody writes this address") is precisely what people run this
    tool to establish. One was nearly reported as a finding on #2345.
    """


class Module:
    def __init__(self, path):
        self.raw = open(path, 'rb').read()
        raw = self.raw
        # Refuse before parsing, not after. Everything below reads fixed offsets that exist in ANY
        # file long enough, so without this the tool cannot fail -- it can only answer wrongly.
        if len(raw) < 0x40:
            raise BadModule(f"{path}: {len(raw)} bytes is too short to be an ELF")
        if raw[:4] == SELF_MAGIC:
            raise BadModule(
                f"{path} is a SELF container, not a module ELF. Flatten it first:\n"
                f"    python3 tools/il2cpp/prx_to_elf.py {path} <out.elf>\n"
                f"and run this tool on <out.elf>. (Answering the question on the SELF would mean "
                f"reporting 0 references for every address -- see #2346.)")
        if raw[:4] != ELF_MAGIC:
            raise BadModule(f"{path}: not an ELF (magic {raw[:4].hex()}), and not a SELF either")
        e_phoff, = struct.unpack_from('<Q', raw, 0x20)
        phentsize, phnum = struct.unpack_from('<HH', raw, 0x36)
        # A well-formed ELF64 program header is exactly 56 bytes. A garbage phentsize (0 on a SELF)
        # makes the loop below re-read one offset phnum times and produce nonsense segments.
        if phentsize != 56:
            raise BadModule(f"{path}: e_phentsize is {phentsize}, not 56 — this is not an ELF64")
        if not phnum or e_phoff + phnum * phentsize > len(raw):
            raise BadModule(f"{path}: program header table (phoff=0x{e_phoff:x}, phnum={phnum}) "
                            f"does not fit in {len(raw)} bytes")
        self.segs = []          # (va, foff, filesz, flags)
        self.dyn_va = 0
        for i in range(phnum):
            t, fl, off, va, pa, fs, ms, al = struct.unpack_from('<IIQQQQQQ', raw, e_phoff + i * phentsize)
            if t == 1:
                self.segs.append((va, off, fs, fl))
            elif t == 2:
                self.dyn_va = va
        if not self.segs:
            raise BadModule(f"{path}: no PT_LOAD segments — nothing to search")
        if not any(fl & 1 for _, _, _, fl in self.segs):
            raise BadModule(f"{path}: no executable PT_LOAD segment — every code reference this "
                            f"tool decodes lives in one, so every answer would be 0")
        self._build()
        # The backstop, and the reason it is here rather than in a comment: the checks above reject
        # the input shapes we KNOW about, and a silent zero is exactly the failure that arrives in a
        # shape nobody anticipated. A real PS5 module has thousands of internal references; a decoder
        # that finds none of them has failed, whatever the file claimed to be. So make the tool
        # detect its own invalidity instead of trusting that the magic check was sufficient.
        #
        # The size threshold is load-bearing and was found by the guard firing on a legitimate input:
        # the premise "a module with no references at all has failed" is only true at MODULE scale.
        # A small hand-built ELF -- this repo's own `imm` test fixture is 256 bytes of executable
        # segment -- can honestly contain no reference of any decoded form. Refusing those would
        # trade a silent wrong answer for a loud wrong refusal, which is not an improvement. 64 KiB
        # is far below any real PS5 eboot/PRX and far above any fixture.
        exec_bytes = sum(fs for _, _, fs, fl in self.segs if fl & 1)
        if exec_bytes >= 0x10000 and not self.code_xref and not self.data_xref:
            raise BadModule(
                f"{path}: parsed as an ELF, but ZERO references of any kind were decoded across "
                f"{exec_bytes} bytes of executable segment. That is not a plausible module - "
                f"refusing rather than answering 0 for every address (#2346).")

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
                elif (i + 7 <= n and b in (0x48, 0x4c) and
                        d[i + 1] in (0x2b, 0x3b, 0x03) and d[i + 2] in MODRM_RIP):
                    # ALU reg, [rip+disp32] -- 7 bytes. `2b` sub, `3b` cmp, `03` add: the register is
                    # the destination, so the memory operand is READ. These are why "who touches X"
                    # answered with reads only in #2025 -- the container-size comparisons around a
                    # std::vector's _M_finish decode as none of the forms above.
                    disp = struct.unpack_from('<i', d, i + 3)[0]
                    site = v + i
                    kind = {0x2b: 'sub', 0x3b: 'cmp', 0x03: 'add'}[d[i + 1]]
                    self.code_xref[site + 7 + disp].append((site, kind))
                elif (i + 8 <= n and b in (0x48, 0x4c) and
                        d[i + 1] == 0x83 and d[i + 2] in MODRM_RIP):
                    # ALU [rip+disp32], imm8  (83 /r) -- 8 bytes. The read-modify-WRITE family, and
                    # the one that made #2025 biased: `add [rip+d32], imm8` is how a std::vector's
                    # _M_finish advances, so push_back/pop_back were invisible while every read of
                    # the same address was reported. /7 is cmp, which does not write.
                    reg = (d[i + 2] >> 3) & 7
                    disp = struct.unpack_from('<i', d, i + 3)[0]
                    site = v + i
                    kind = ('addi', 'ori', 'adci', 'sbbi', 'andi', 'subi', 'xori', 'cmpi')[reg]
                    self.code_xref[site + 8 + disp].append((site, kind))
                elif (i + 8 <= n and b == 0xc5 and (d[i + 1] & 0x7b) == 0x78 and
                        d[i + 2] in (0x10, 0x11) and d[i + 3] in MODRM_RIP):
                    # VEX.128/256 vmovups xmm/ymm, [rip+disp32] (0x10, load) and the store form
                    # (0x11) -- 8 bytes with the 2-byte VEX prefix. A vectorised struct copy writes
                    # its destination through exactly this and through none of the forms above.
                    #
                    # The second VEX byte is R vvvv L pp. Masking 0x7b keeps vvvv=1111 and pp=00
                    # (the unused-register, no-prefix encoding vmovups uses) while accepting BOTH
                    # values of R and L -- so c5 f8/fc (xmm0-7/ymm0-7) and c5 78/7c (xmm8-15/
                    # ymm8-15) all decode. Pinning R=1 by testing d[i+1] in (0xf8, 0xfc) missed the
                    # high registers: measured on a clean 34.1 MB PPSA24651 eboot, 876 of 14,722
                    # such instructions (6.0%) fell through, 869 of them loads (#2025 review).
                    disp = struct.unpack_from('<i', d, i + 4)[0]
                    site = v + i
                    wide = 'y' if (d[i + 1] & 0x04) else 'x'      # L bit, not equality
                    kind = ('vload' if d[i + 2] == 0x10 else 'vstore') + wide
                    self.code_xref[site + 8 + disp].append((site, kind))


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
    # The empty needle refuses through the SAME guard as 1-3 bytes (#2099). It used to return [],
    # which printed "inline-immediate constructions of '' (0 bytes): 0" -- a clean, confident zero,
    # indistinguishable in form from a real "nothing builds this string". That is precisely the
    # misleading zero this mode exists to eliminate, reproduced inside the mode itself, and it bites
    # hardest where the mode is most useful: an agent scripting `imm` over parsed names gets the same
    # answer for an unset variable as for a genuinely unreferenced string, with nothing separating
    # them. The message below already reads correctly at n == 0.
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
    try:
        m = Module(sys.argv[1])
    except BadModule as exc:
        # Non-zero exit as well as the message: a caller that pipes this into a report must not be
        # able to mistake a refusal for an empty result set (#2346).
        print(f"refused: {exc}", file=sys.stderr)
        return 2
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
            # Summarise by access class first. "who WRITES this" is the usual question, and an
            # answer made entirely of reads is the failure #2025 records -- it looked complete
            # because nothing said otherwise. Now it says so on its own line.
            writers = sum(1 for _, k in crefs if access(k) in ('w', 'rw'))
            readers = sum(1 for _, k in crefs if access(k) == 'r')
            taken = sum(1 for _, k in crefs if access(k) == '&')
            print(f"code references to 0x{addr:x}: {len(crefs)} "
                  f"({writers} write, {readers} read, {taken} address-taken)")
            if crefs and writers == 0:
                print("   note: NO writer found. Either the address is written by a form this "
                      "decoder does not know, or it is genuinely read-only (#2025).")
            for s, k in crefs[:32]:
                print(f"   [{access(k):2s}] {k:7s} at 0x{s:x}  (in func 0x{m.func_start(s):x})")
    elif mode == 'from':
        start = m.func_start(addr)
        print(f"references made by func 0x{start:x} (window from 0x{addr:x}):")
        fo = m.foff(addr)
        seen = set()
        for tgt, sites in m.code_xref.items():
            for s, k in sites:
                if addr <= s < addr + 0x800 and tgt not in seen:
                    seen.add(tgt)
                    print(f"   [{access(k):2s}] {k:7s} -> 0x{tgt:x}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
