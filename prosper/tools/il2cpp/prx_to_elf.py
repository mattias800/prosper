#!/usr/bin/env python3
# prx_to_elf.py — flatten an unencrypted PS5 SELF/PRX into a loadable ET_DYN ELF
# (p_offset == p_vaddr), the input Il2CppDumper needs. Usage:
#   python3 prx_to_elf.py <in.prx> <out.elf>
# Then:
#   printf '0\n0\n0\n' | dotnet Il2CppDumper.dll out.elf global-metadata.dat outdir
# (answer "0" to the "dump file" prompt to force the registration auto-scan).
# Produces dump.cs (method RVAs), script.json, il2cpp.h. Runtime addr of a method =
# module_base (e.g. 0x440000000 for Il2CppUserAssemblies.prx) + its RVA.
import struct, sys

def main(src, dst):
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
    struct.pack_into("<H", buf, 0x3c, e_phnum)
    for i, p in enumerate(phdrs):              # p_offset := p_vaddr for every phdr
        struct.pack_into("<IIQQQQQQ", buf, 0x40+i*56, p[0], p[1], p[3], p[3], p[4], p[5], p[6], p[7])
    open(dst, "wb").write(buf)
    print("wrote %s size=%#x" % (dst, total))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
