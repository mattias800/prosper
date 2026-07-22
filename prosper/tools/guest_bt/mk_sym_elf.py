#!/usr/bin/env python3
# mk_sym_elf.py — turn a prosper-flattened guest module ELF (from tools/il2cpp/prx_to_elf.py,
# which drops the section-header table so p_offset==p_vaddr) into one gdb can actually UNWIND and
# SYMBOLICATE:
#
#   * a `.eh_frame` section header pointing at the module's real DWARF CFI (located through the
#     PT_GNU_EH_FRAME / .eh_frame_hdr segment) — without a section named ".eh_frame" gdb never
#     engages its DWARF unwinder, so `bt` stalls the moment frame pointers are absent (all of
#     Unity/IL2CPP), and
#   * an optional `.symtab` synthesised from an Il2CppDumper `script.json`, so every managed method
#     shows up by name in `bt` for free (no post-processing).
#
# The input must already be flattened (prx_to_elf.py output): p_offset==p_vaddr, e_shoff==0.
# Usage:
#   python3 mk_sym_elf.py <flattened.elf> <out.elf> [--symbols script.json]
#
# Adding sections is append-only (segments/LOAD data are untouched), so the result still loads at
# the same guest base via `add-symbol-file <out.elf> <base>`.
import struct, sys, json

def u16(b, o): return struct.unpack_from('<H', b, o)[0]
def u32(b, o): return struct.unpack_from('<I', b, o)[0]
def u64(b, o): return struct.unpack_from('<Q', b, o)[0]

# DWARF EH pointer-encoding decode, just enough for the .eh_frame_hdr's eh_frame_ptr field.
def read_encoded(buf, off, enc, pc):
    fmt = enc & 0x0f
    if   fmt == 0x03: v = u32(buf, off); n = 4          # DW_EH_PE_udata4
    elif fmt == 0x0b: v = struct.unpack_from('<i', buf, off)[0]; n = 4  # sdata4
    elif fmt == 0x04: v = u64(buf, off); n = 8          # udata8
    elif fmt == 0x0c: v = struct.unpack_from('<q', buf, off)[0]; n = 8  # sdata8
    else: raise SystemExit('mk_sym_elf: unsupported eh_frame_hdr ptr enc 0x%x' % enc)
    app = enc & 0x70
    if   app == 0x00: base = 0            # absptr
    elif app == 0x10: base = pc           # pcrel (relative to the field's own vaddr)
    else: raise SystemExit('mk_sym_elf: unsupported eh_frame_hdr ptr application 0x%x' % app)
    return base + v

def main(src, dst, script=None):
    b = bytearray(open(src, 'rb').read())
    assert b[:4] == b'\x7fELF', 'not an ELF'
    assert b[4] == 2, 'not ELF64'
    e_phoff = u64(b, 0x20); e_phnum = u16(b, 0x38); e_phentsize = u16(b, 0x36)
    if u64(b, 0x28) != 0:
        raise SystemExit('mk_sym_elf: input already has a section header table (not a flattened ELF)')

    phdrs = []
    for i in range(e_phnum):
        o = e_phoff + i * e_phentsize
        phdrs.append(dict(type=u32(b, o), flags=u32(b, o + 4), off=u64(b, o + 8),
                          vaddr=u64(b, o + 0x10), filesz=u64(b, o + 0x28), memsz=u64(b, o + 0x30)))
    PT_LOAD, PT_GNU_EH_FRAME = 1, 0x6474e550
    loads = [p for p in phdrs if p['type'] == PT_LOAD and p['filesz'] > 0]
    # .text := the first executable LOAD (PF_X == 1); its whole span is fine for symbol/addr lookup.
    exe = next((p for p in loads if p['flags'] & 1), loads[0])
    # gdb's single `add-symbol-file <elf> <base>` anchors ONE relocation delta on .text
    # (delta = base - sh_addr(.text)), which then places BOTH the symbols and .eh_frame. That is only
    # equal to the guest base when .text sits at vaddr 0 — which every PS5 SELF/PRX we load does. If a
    # module ever breaks that, symbols and CFI would silently shift by exe['vaddr'] and produce a
    # coherent-but-WRONG backtrace, so fail loudly instead.
    if exe['vaddr'] != 0:
        raise SystemExit('mk_sym_elf: executable LOAD is at vaddr 0x%x, not 0 — the single '
                         'add-symbol-file base anchor would misplace symbols/.eh_frame' % exe['vaddr'])

    # Locate .eh_frame via the .eh_frame_hdr the GNU_EH_FRAME segment points at. A module that ships no
    # CFI is not fatal: emit .text (+ .symtab) so it still symbolicates, just without an unwind section
    # (the walk simply ends at such a frame). guest_bt.py relies on this to never abort a whole run.
    eh_frame_vaddr = eh_frame_off = eh_frame_size = 0
    ehh = next((p for p in phdrs if p['type'] == PT_GNU_EH_FRAME), None)
    if ehh:
        ho = ehh['off']
        ver, ptr_enc = b[ho], b[ho + 1]
        assert ver == 1, 'eh_frame_hdr version %d' % ver
        eh_frame_vaddr = read_encoded(b, ho + 4, ptr_enc, ehh['vaddr'] + 4)
        # In a flattened ELF p_offset==p_vaddr, so file offset == vaddr for the loaded image.
        eh_frame_off = eh_frame_vaddr
        # Size: walk the CIE/FDE entries (each: u32 length, then body) until a 0-length terminator or
        # the end of the containing LOAD — the EXACT extent, so a linear parser (pyelftools) never
        # reads past the real CFI into unrelated segment bytes. .eh_frame is never 64-bit DWARF, so a
        # 0xffffffff length is treated as a stop.
        host = next(p for p in loads if p['vaddr'] <= eh_frame_vaddr < p['vaddr'] + p['filesz'])
        seg_end = host['vaddr'] + host['filesz']
        p = eh_frame_vaddr
        while p + 4 <= seg_end:
            ln = u32(b, p)
            if ln == 0 or ln == 0xffffffff:
                p += 4          # include the terminator word
                break
            p += 4 + ln
        eh_frame_size = p - eh_frame_vaddr

    # --- build appended data: shstrtab, and (optional) symtab+strtab -------------------------------
    def cstr_table(names):
        blob = b'\x00'; off = {}
        for n in names:
            off[n] = len(blob); blob += n.encode() + b'\x00'
        return blob, off

    have_eh = eh_frame_size > 0
    sec_names = ['', '.text'] + (['.eh_frame'] if have_eh else []) + ['.shstrtab']
    syms = []
    if script:
        sj = json.load(open(script))
        for m in sj.get('ScriptMethod', []):
            a = m.get('Address'); n = m.get('Name')
            if a is not None and n:
                syms.append((a, n))
        syms.sort()
        sec_names = ['', '.text'] + (['.eh_frame'] if have_eh else []) + \
                    ['.symtab', '.strtab', '.shstrtab']

    shstr_blob, shstr_off = cstr_table(sec_names)

    # Assemble the appended region: the section-name/string/symbol tables, then the section header
    # table. Everything is appended after the original file bytes (the LOAD segments are untouched),
    # each part 8-aligned. part_off records each part's absolute file offset for its section header.
    parts = []          # (name, bytes)
    strtab_len = 0
    if script:
        str_blob, str_o = cstr_table([n for _, n in syms])
        strtab_len = len(str_blob)
        # Elf64_Sym = name(4) info(1) other(1) shndx(2) value(8) size(8); index 0 is the null symbol.
        symtab = bytearray(24)
        STT_FUNC, STB_GLOBAL, text_shndx = 2, 1, 1
        for a, n in syms:
            symtab += struct.pack('<IBBHQQ', str_o[n], (STB_GLOBAL << 4) | STT_FUNC, 0, text_shndx, a, 0)
        parts += [('.strtab', bytes(str_blob)), ('.symtab', bytes(symtab))]
    parts.append(('.shstrtab', bytes(shstr_blob)))

    blob = bytearray()
    part_off = {}
    def align8():
        while (len(b) + len(blob)) % 8:
            blob.append(0)
    for name, data in parts:
        align8()
        part_off[name] = len(b) + len(blob)
        blob += data
    align8()
    shoff = len(b) + len(blob)
    blob_all = blob

    # Build section headers
    def sh(name, typ, flags, addr, off, size, link=0, info=0, align=1, entsize=0):
        return struct.pack('<IIQQQQIIQQ', shstr_off[name], typ, flags, addr, off, size,
                           link, info, align, entsize)

    idx = {n: i for i, n in enumerate(sec_names)}
    headers = [b'\x00' * 64]  # null
    headers.append(sh('.text', 1, 0x6, exe['vaddr'], exe['off'], exe['filesz'], align=16))          # PROGBITS AX
    if have_eh:
        headers.append(sh('.eh_frame', 1, 0x2, eh_frame_vaddr, eh_frame_off, eh_frame_size, align=8))  # PROGBITS A
    if script:
        nsym = 1 + len(syms)
        headers.append(sh('.symtab', 2, 0, 0, part_off['.symtab'], nsym * 24,
                          link=idx['.strtab'], info=1, align=8, entsize=24))                          # SYMTAB
        headers.append(sh('.strtab', 3, 0, 0, part_off['.strtab'], strtab_len, align=1))               # STRTAB
    headers.append(sh('.shstrtab', 3, 0, 0, part_off['.shstrtab'], len(shstr_blob), align=1))         # STRTAB

    shtab = b''.join(headers)
    out = bytes(b) + bytes(blob_all) + shtab
    # patch ELF header: e_shoff, e_shentsize, e_shnum, e_shstrndx
    out = bytearray(out)
    struct.pack_into('<Q', out, 0x28, shoff)
    struct.pack_into('<H', out, 0x3a, 64)              # e_shentsize
    struct.pack_into('<H', out, 0x3c, len(headers))    # e_shnum
    struct.pack_into('<H', out, 0x3e, len(sec_names) - 1)  # e_shstrndx (.shstrtab is last)
    open(dst, 'wb').write(out)
    print('mk_sym_elf: %s -> %s  .text@0x%x(0x%x) .eh_frame@0x%x(0x%x) syms=%d shoff=0x%x'
          % (src, dst, exe['vaddr'], exe['filesz'], eh_frame_vaddr, eh_frame_size,
             len(syms), shoff))

if __name__ == '__main__':
    args = sys.argv[1:]
    script = None
    if '--symbols' in args:
        i = args.index('--symbols'); script = args[i + 1]; del args[i:i + 2]
    if len(args) != 2:
        raise SystemExit('usage: mk_sym_elf.py <flattened.elf> <out.elf> [--symbols script.json]')
    main(args[0], args[1], script)
