#!/usr/bin/env python3
# prx_to_elf.py — flatten an unencrypted PS5 SELF/PRX into a loadable ET_DYN ELF
# (p_offset == p_vaddr), the input Il2CppDumper needs. Usage:
#   python3 prx_to_elf.py <in.prx> <out.elf> [--sections]
# Then:
#   printf '0\n0\n0\n' | dotnet Il2CppDumper.dll out.elf global-metadata.dat outdir
# (answer "0" to the "dump file" prompt to force the registration auto-scan).
# Produces dump.cs (method RVAs), script.json, il2cpp.h. Runtime addr of a method =
# module_base (e.g. 0x440000000 for Il2CppUserAssemblies.prx) + its RVA.
#
# --sections synthesizes a section header table so `objdump -d` can disassemble the result (#2154).
# OPT-IN, not the default: Il2CppDumper reads program headers and the dynamic segment and does not
# need them, but its ELF reader does consult sections on some paths and I could not run it here, so
# the default output is byte-for-byte what it was. Turn it on for disassembly, leave it off for a
# dump.
import struct, sys

def main(src, dst, want_sections=False):
    f = open(src, "rb").read()
    nseg, = struct.unpack_from("<H", f, 0x18)
    segs = [struct.unpack_from("<QQQQ", f, 0x20 + i*32) for i in range(nseg)]
    eb = f.find(b"\x7fELF", 0x20 + nseg*32)
    e_phoff, = struct.unpack_from("<Q", f, eb+0x20)
    e_phnum, = struct.unpack_from("<H", f, eb+0x38)
    # SELF data segments carry bit 0x800; phdr index = flags>>20; bytes at file_offset.
    data_seg = {(fl >> 20): (foff, fsize) for (fl, foff, fsize, ms) in segs if fl & 0x800}
    phdrs = [list(struct.unpack_from("<IIQQQQQQ", f, eb+e_phoff+i*56)) for i in range(e_phnum)]
    loads = [p for p in phdrs if p[0] == 1 and p[5] > 0]
    total = max(p[3]+p[5] for p in loads)
    buf = bytearray(total)
    for i, p in enumerate(phdrs):
        if p[0] == 1 and p[5] > 0 and i in data_seg:
            off, sz = data_seg[i]; sz = min(sz, p[5])
            buf[p[3]:p[3]+sz] = f[off:off+sz]
    buf[0:64] = f[eb:eb+64]
    struct.pack_into("<H", buf, 0x10, 3)       # e_type = ET_DYN
    struct.pack_into("<Q", buf, 0x20, 0x40)    # e_phoff
    struct.pack_into("<Q", buf, 0x28, 0)       # e_shoff = 0
    # The flattened image has no section header table, so all three fields describing one have to
    # say so together. e_shoff is zeroed above; the values the source module carries describe a
    # table that is not in this file (stripped PS5 modules keep stale ones — across 22 title
    # modules e_shoff is always a past-EOF offset, e_shstrndx is always 41-46, and e_shnum is
    # 0 or 43-48). Either survivor makes binutils reject the whole file with "file format not
    # recognized"; both must be cleared, and clearing only one is not enough.
    # e_phnum at 0x38 needs NO write: it survives the verbatim header copy above, and the loop
    # below emits exactly that many program headers. Writing it to 0x3c — the e_shnum slot — was
    # the bug in #2016.
    struct.pack_into("<H", buf, 0x3c, 0)       # e_shnum = 0
    struct.pack_into("<H", buf, 0x3e, 0)       # e_shstrndx = SHN_UNDEF
    for i, p in enumerate(phdrs):              # p_offset := p_vaddr for every phdr
        struct.pack_into("<IIQQQQQQ", buf, 0x40+i*56, p[0], p[1], p[3], p[3], p[4], p[5], p[6], p[7])
    if want_sections:
        total = add_sections(buf, phdrs)
    open(dst, "wb").write(buf)
    print("wrote %s size=%#x%s" % (dst, total, " (+sections)" if want_sections else ""))

# One SHT_PROGBITS section per PT_LOAD, plus .shstrtab, appended after the image (#2154).
#
# Without a section header table `objdump -f/-p/-T` work but `objdump -d` prints NOTHING: it
# disassembles SECTIONS and there are none. The documented workaround throws the addresses away
# (`objdump -D -b binary --start-address=...`), which defeats the point of flattening with
# p_offset == p_vaddr -- file offset and virtual address are already the same number, and a section
# table is what lets objdump use it.
#
# sh_addr == sh_offset == p_vaddr follows directly from that flattening, so nothing here is derived
# or guessed; each section is the segment restated in the form objdump reads.
def add_sections(buf, phdrs):
    SHT_PROGBITS, SHT_NOBITS, SHT_STRTAB = 1, 8, 3
    SHF_WRITE, SHF_ALLOC, SHF_EXECINSTR = 0x1, 0x2, 0x4
    secs, names, blob = [], bytearray(b"\0"), None
    def intern(nm):
        off = len(names); names.extend(nm.encode() + b"\0"); return off
    for i, p in enumerate(phdrs):
        if p[0] != 1 or (p[5] == 0 and p[6] == 0):     # PT_LOAD carrying something
            continue
        x, w = bool(p[1] & 1), bool(p[1] & 2)
        # A bss-only PT_LOAD is SHT_NOBITS so the file does not grow by p_memsz; everything else is
        # PROGBITS over the bytes actually present.
        bss = p[5] == 0
        base = ".bss" if bss else (".text" if x else (".data" if w else ".rodata"))
        flags = SHF_ALLOC | (SHF_EXECINSTR if x else 0) | (SHF_WRITE if w else 0)
        secs.append((intern("%s%d" % (base, i)), SHT_NOBITS if bss else SHT_PROGBITS, flags,
                     p[3], p[3], p[6] if bss else p[5], p[7] or 1))
    # .shstrtab last, and it is a real file section, so it needs an offset inside the file.
    shstr_name = intern(".shstrtab")
    shstr_off = len(buf)
    buf.extend(names)
    sh_off = (len(buf) + 7) & ~7                       # 8-align the table itself
    buf.extend(b"\0" * (sh_off - len(buf)))
    # Index 0 is the mandatory all-zero SHN_UNDEF entry.
    table = bytearray(struct.pack("<IIQQQQIIQQ", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    for (nm, typ, flags, addr, off, size, align) in secs:
        table.extend(struct.pack("<IIQQQQIIQQ", nm, typ, flags, addr, off, size, 0, 0, align, 0))
    table.extend(struct.pack("<IIQQQQIIQQ", shstr_name, SHT_STRTAB, 0, 0,
                             shstr_off, len(names), 0, 0, 1, 0))
    buf.extend(table)
    nsec = len(secs) + 2                               # + null + .shstrtab
    struct.pack_into("<Q", buf, 0x28, sh_off)          # e_shoff
    struct.pack_into("<H", buf, 0x3a, 64)              # e_shentsize (ELF64 section header)
    struct.pack_into("<H", buf, 0x3c, nsec)            # e_shnum
    struct.pack_into("<H", buf, 0x3e, nsec - 1)        # e_shstrndx = the .shstrtab we just wrote
    return len(buf)

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0], args[1], "--sections" in sys.argv[1:])
