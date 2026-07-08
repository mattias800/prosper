// self_dump — Prospero (PS5) SELF/ELF inspector
// Part of the `prosper` PS5->PC compatibility layer.
//
// Parses an unencrypted PS5 SELF (Signed ELF wrapper), reconstructs the inner
// x86-64 FreeBSD-ABI ELF, walks PT_DYNAMIC + PT_SCE_DYNLIBDATA, and dumps:
//   - SELF header + segment table
//   - ELF header + program headers
//   - dynamic tags
//   - needed modules / import libraries / export libraries
//   - imported & exported symbols (NID#lib#module), resolved to lib/module names
//
// This is the intelligence-gathering front end for the loader: the imported
// module/function set defines the entire HLE surface we must implement.
//
// Build: g++ -O2 -std=c++20 self_dump.cpp -o self_dump
// Usage: self_dump <file.self|eboot.bin|*.prx> [--symbols]

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>

// ---- file IO -------------------------------------------------------------
static std::vector<uint8_t> read_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> b(n);
    if (fread(b.data(), 1, n, f) != (size_t)n) { fprintf(stderr, "short read\n"); exit(1); }
    fclose(f);
    return b;
}
template<class T> static T rd(const std::vector<uint8_t>& b, size_t off) {
    T v{}; if (off + sizeof(T) <= b.size()) memcpy(&v, b.data() + off, sizeof(T)); return v;
}

// ---- SELF ----------------------------------------------------------------
#pragma pack(push,1)
struct SelfHeader {
    uint32_t magic;        // 0x1D3D154F
    uint8_t  version, mode, endian, attr;
    uint32_t key_type;     // content type
    uint16_t header_size;
    uint16_t meta_size;
    uint64_t file_size;    // size of the logical (decrypted) ELF
    uint16_t num_segments;
    uint16_t flags;
    uint32_t pad;
};
struct SelfSegment {
    uint64_t flags;
    uint64_t file_offset;
    uint64_t file_size;    // stored size in the SELF
    uint64_t mem_size;     // uncompressed size
};
#pragma pack(pop)

// ---- ELF (x86-64) --------------------------------------------------------
struct Elf64_Ehdr {
    uint8_t  e_ident[16];
    uint16_t e_type, e_machine;
    uint32_t e_version;
    uint64_t e_entry, e_phoff, e_shoff;
    uint32_t e_flags;
    uint16_t e_ehsize, e_phentsize, e_phnum, e_shentsize, e_shnum, e_shstrndx;
};
struct Elf64_Phdr {
    uint32_t p_type, p_flags;
    uint64_t p_offset, p_vaddr, p_paddr, p_filesz, p_memsz, p_align;
};
struct Elf64_Sym {
    uint32_t st_name;
    uint8_t  st_info, st_other;
    uint16_t st_shndx;
    uint64_t st_value, st_size;
};
struct Elf64_Dyn { int64_t d_tag; uint64_t d_val; };

// ELF program header types (standard + Sony)
static const char* ptype_name(uint32_t t) {
    switch (t) {
        case 0: return "NULL";       case 1: return "LOAD";
        case 2: return "DYNAMIC";    case 3: return "INTERP";
        case 4: return "NOTE";       case 6: return "PHDR";
        case 7: return "TLS";
        case 0x6474e550: return "GNU_EH_FRAME";
        case 0x6474e551: return "GNU_STACK";
        case 0x6474e552: return "GNU_RELRO";
        case 0x61000000: return "SCE_DYNLIBDATA";
        case 0x61000001: return "SCE_PROCPARAM";
        case 0x61000002: return "SCE_MODULEPARAM";
        case 0x61000010: return "SCE_RELRO";
        case 0x6fffff00: return "SCE_COMMENT";
        case 0x6fffff01: return "SCE_VERSION";
        default: return "?";
    }
}
static const char* etype_name(uint16_t t) {
    switch (t) {
        case 0xfe00: return "ET_SCE_EXEC";
        case 0xfe04: return "ET_SCE_REPLAY_EXEC";
        case 0xfe0c: return "ET_SCE_STUBLIB";
        case 0xfe10: return "ET_SCE_DYNEXEC (main exe)";
        case 0xfe18: return "ET_SCE_DYNAMIC (PRX)";
        case 2: return "ET_EXEC"; case 3: return "ET_DYN";
        default: return "?";
    }
}

// Sony dynamic tags
static const char* dtag_name(int64_t t) {
    switch ((uint64_t)t) {
        case 0:  return "DT_NULL";     case 1: return "DT_NEEDED";
        case 3:  return "DT_PLTGOT";   case 5: return "DT_STRTAB";
        case 6:  return "DT_SYMTAB";   case 0x1e: return "DT_FLAGS";
        case 4:  return "DT_HASH";     case 0xa: return "DT_STRSZ";
        case 0xb: return "DT_SYMENT";  case 0x14: return "DT_PLTREL";
        case 0x17: return "DT_JMPREL";
        case 0x61000005: return "DT_SCE_MODULE_INFO";
        case 0x61000007: return "DT_SCE_FINGERPRINT";
        case 0x61000017: return "DT_SCE_EXPORT_LIB_ATTR";
        case 0x61000019: return "DT_SCE_IMPORT_LIB_ATTR";
        case 0x61000041: return "DT_SCE_ORIG_FILENAME";
        case 0x61000043: return "DT_SCE_MODULE_INFO";
        case 0x61000045: return "DT_SCE_NEEDED_MODULE";
        case 0x61000047: return "DT_SCE_MODULE_ATTR";
        case 0x61000049: return "DT_SCE_EXPORT_LIB";
        case 0x6100004b: return "DT_SCE_IMPORT_LIB";
        case 0x61000025: return "DT_SCE_HASH";
        case 0x61000027: return "DT_SCE_PLTGOT";
        case 0x61000029: return "DT_SCE_JMPREL";
        case 0x6100002b: return "DT_SCE_PLTREL";
        case 0x6100002d: return "DT_SCE_PLTRELSZ";
        case 0x6100002f: return "DT_SCE_RELA";
        case 0x61000031: return "DT_SCE_RELASZ";
        case 0x61000033: return "DT_SCE_RELAENT";
        case 0x61000035: return "DT_SCE_STRTAB";
        case 0x61000037: return "DT_SCE_STRSZ";
        case 0x61000039: return "DT_SCE_SYMTAB";
        case 0x6100003b: return "DT_SCE_SYMENT";
        case 0x6100003d: return "DT_SCE_HASHSZ";
        case 0x6100003f: return "DT_SCE_SYMTABSZ";
        default: return "?";
    }
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <file> [--symbols]\n", argv[0]); return 1; }
    bool dump_syms = (argc >= 3 && strcmp(argv[2], "--symbols") == 0);
    auto b = read_file(argv[1]);

    // -- SELF header --
    auto sh = rd<SelfHeader>(b, 0);
    bool is_self = (sh.magic == 0x1D3D154F || sh.magic == 0xEEF51454);   // both observed SELF variants
    size_t elf_base = 0;
    std::vector<SelfSegment> segs;
    printf("== FILE: %s (%zu bytes) ==\n", argv[1], b.size());
    if (is_self) {
        printf("[SELF] version=%u mode=%u endian=%u attr=0x%02x key_type=0x%x\n",
               sh.version, sh.mode, sh.endian, sh.attr, sh.key_type);
        printf("[SELF] header_size=0x%x meta_size=0x%x file_size=%llu num_segments=%u flags=0x%x\n",
               sh.header_size, sh.meta_size, (unsigned long long)sh.file_size, sh.num_segments, sh.flags);
        for (int i = 0; i < sh.num_segments; i++) {
            auto s = rd<SelfSegment>(b, 0x20 + i * sizeof(SelfSegment));
            segs.push_back(s);
            uint64_t idx = s.flags >> 20;
            bool comp = (s.flags >> 3) & 1, enc = (s.flags >> 1) & 1;
            printf("  seg[%2d] flags=0x%016llx phdr_idx=%llu%s%s off=0x%llx fsz=0x%llx msz=0x%llx\n",
                   i, (unsigned long long)s.flags, (unsigned long long)idx,
                   comp ? " COMPRESSED" : "", enc ? " ENCRYPTED" : "",
                   (unsigned long long)s.file_offset, (unsigned long long)s.file_size,
                   (unsigned long long)s.mem_size);
        }
        elf_base = 0x20 + (size_t)sh.num_segments * sizeof(SelfSegment);
    } else {
        printf("[note] no SELF magic; treating as raw ELF\n");
    }

    // -- inner ELF header --
    auto eh = rd<Elf64_Ehdr>(b, elf_base);
    if (memcmp(eh.e_ident, "\x7f""ELF", 4) != 0) {
        // fall back: scan
        for (size_t i = 0; i + 4 < b.size(); i++)
            if (!memcmp(&b[i], "\x7f""ELF", 4)) { elf_base = i; eh = rd<Elf64_Ehdr>(b, i); break; }
    }
    printf("\n[ELF] @0x%zx  class=%u data=%u osabi=%u  machine=0x%x(%s)  type=0x%x(%s)\n",
           elf_base, eh.e_ident[4], eh.e_ident[5], eh.e_ident[7],
           eh.e_machine, eh.e_machine == 0x3e ? "x86-64" : "?",
           eh.e_type, etype_name(eh.e_type));
    printf("[ELF] entry=0x%llx phoff=0x%llx phnum=%u phentsize=%u\n",
           (unsigned long long)eh.e_entry, (unsigned long long)eh.e_phoff, eh.e_phnum, eh.e_phentsize);

    // -- program headers --
    // Map ELF phdr index -> file bytes. In a SELF, phdr p_offset is logical; the
    // real bytes live in the SELF segment whose (flags>>20)==phdr_index.
    std::vector<Elf64_Phdr> phdrs;
    std::map<int, SelfSegment> phdr_to_seg;
    if (is_self)
        for (auto& s : segs) { uint64_t idx = s.flags >> 20; if (idx < (uint64_t)eh.e_phnum) phdr_to_seg[(int)idx] = s; }

    printf("\n[PROGRAM HEADERS]\n");
    Elf64_Phdr dynamic_ph{}, dynlib_ph{}; bool have_dyn = false, have_dld = false;
    for (int i = 0; i < eh.e_phnum; i++) {
        auto p = rd<Elf64_Phdr>(b, elf_base + eh.e_phoff + i * sizeof(Elf64_Phdr));
        phdrs.push_back(p);
        printf("  ph[%2d] %-14s flags=%c%c%c vaddr=0x%09llx filesz=0x%08llx memsz=0x%08llx align=0x%llx\n",
               i, ptype_name(p.p_type),
               (p.p_flags & 4) ? 'r' : '-', (p.p_flags & 2) ? 'w' : '-', (p.p_flags & 1) ? 'x' : '-',
               (unsigned long long)p.p_vaddr, (unsigned long long)p.p_filesz,
               (unsigned long long)p.p_memsz, (unsigned long long)p.p_align);
        if (p.p_type == 2)          { dynamic_ph = p; have_dyn = true; }
        if (p.p_type == 0x61000000) { dynlib_ph = p; have_dld = true; }
    }
    if (!have_dyn) { printf("\n[!] no DYNAMIC - static or malformed. done.\n"); return 0; }
    if (!have_dld) printf("\n[note] no PT_SCE_DYNLIBDATA phdr; DT_SCE_* offsets are relative to a synthetic dynlib blob (PS5 layout).\n");

    // -- build a virtual-address -> file-offset resolver from LOAD segments --
    // This mirrors what the real loader does: map PT_LOAD into a VA space, then
    // resolve DYNAMIC and the dynamic tables (strtab/symtab) by virtual address.
    // In the SELF segment table, the actual data segments have bit 0x800 set;
    // the companion "info/digest" segments (0x..10004) are skipped.
    struct Load { uint64_t vaddr, memsz, foff, fsz; uint32_t type; };
    std::vector<Load> loads;
    std::map<uint64_t, SelfSegment> data_seg_by_id;   // phdr index -> data segment
    if (is_self)
        for (auto& s : segs) if (s.flags & 0x800) data_seg_by_id[s.flags >> 20] = s;
    for (int i = 0; i < (int)phdrs.size(); i++) {
        auto& p = phdrs[i];
        if (p.p_filesz == 0) continue;
        uint64_t foff;
        if (is_self && data_seg_by_id.count(i)) foff = data_seg_by_id[i].file_offset;
        else                                    foff = elf_base + p.p_offset;
        loads.push_back({ p.p_vaddr, p.p_memsz, foff, p.p_filesz, p.p_type });
    }
    // PT_LOAD wins; non-LOAD file-backed phdrs (DYNAMIC/PROCPARAM/TLS) only resolve a
    // VA when no LOAD maps it. Mirrors Module::va2foff — without this, a DYNAMIC phdr
    // preceding its covering LOAD self-resolves to unmapped bytes (issue #113).
    auto va2foff = [&](uint64_t va) -> int64_t {
        for (auto& L : loads) if (L.type == 1 && va >= L.vaddr && va < L.vaddr + L.fsz) return (int64_t)(L.foff + (va - L.vaddr));
        for (auto& L : loads) if (L.type != 1 && va >= L.vaddr && va < L.vaddr + L.fsz) return (int64_t)(L.foff + (va - L.vaddr));
        return -1;
    };

    // -- read DYNAMIC via VA --
    int64_t dyn_foff = va2foff(dynamic_ph.p_vaddr);
    uint64_t dyn_sz = dynamic_ph.p_filesz;
    printf("\n[DATA] DYNAMIC vaddr=0x%llx -> file@0x%llx size=0x%llx\n",
           (unsigned long long)dynamic_ph.p_vaddr, (unsigned long long)dyn_foff, (unsigned long long)dyn_sz);
    if (dyn_foff < 0) { printf("[!] cannot map DYNAMIC vaddr\n"); return 1; }

    // -- walk dynamic tags -- (DT_SCE_STRTAB/SYMTAB are byte offsets from the
    // image's dynlib string/symbol area; on this PS5 layout they resolve as VAs.)
    uint64_t strtab_va = 0, strsz = 0, symtab_va = 0, symtabsz = 0, syment = 24;
    struct LibRec { std::string name; };
    std::map<int, LibRec> import_libs, export_libs, needed_mods;
    std::vector<Elf64_Dyn> dyns;
    for (size_t o = 0; o + sizeof(Elf64_Dyn) <= dyn_sz; o += sizeof(Elf64_Dyn)) {
        auto d = rd<Elf64_Dyn>(b, (size_t)dyn_foff + o);
        dyns.push_back(d);
        if (d.d_tag == 0) break;
        switch ((uint64_t)d.d_tag) {
            case 5:  strtab_va = d.d_val; break;              // DT_STRTAB (VA)
            case 0xa: strsz = d.d_val; break;                 // DT_STRSZ
            case 6:  symtab_va = d.d_val; break;              // DT_SYMTAB (VA)
            case 0xb: syment = d.d_val; break;                // DT_SYMENT
            case 0x61000035: if (!strtab_va) strtab_va = d.d_val; break;
            case 0x61000037: if (!strsz) strsz = d.d_val; break;
            case 0x61000039: if (!symtab_va) symtab_va = d.d_val; break;
            case 0x6100003f: symtabsz = d.d_val; break;       // DT_SCE_SYMTABSZ
            case 0x6100003b: if (!syment || syment==24) syment = d.d_val; break;
        }
    }
    // Resolve strtab: try as VA; if that fails, the value is an offset into the
    // first LOAD (image base 0) — resolve against file directly.
    int64_t strtab_foff = va2foff(strtab_va);
    if (strtab_foff < 0 && !loads.empty()) strtab_foff = loads[0].foff + strtab_va;
    int64_t symtab_foff = va2foff(symtab_va);
    if (symtab_foff < 0 && !loads.empty()) symtab_foff = loads[0].foff + symtab_va;
    printf("[DATA] strtab va=0x%llx -> file@0x%llx size=0x%llx | symtab va=0x%llx -> file@0x%llx size=0x%llx syment=%llu\n",
           (unsigned long long)strtab_va, (unsigned long long)strtab_foff, (unsigned long long)strsz,
           (unsigned long long)symtab_va, (unsigned long long)symtab_foff, (unsigned long long)symtabsz,
           (unsigned long long)syment);

    auto strtab_str = [&](uint64_t nameoff) -> const char* {
        if (strtab_foff < 0) return "<no-strtab>";
        uint64_t abs = (uint64_t)strtab_foff + nameoff;
        return (abs < b.size()) ? (const char*)(b.data() + abs) : "<oob>";
    };
    // decode lib/module records
    for (auto& d : dyns) {
        uint64_t v = d.d_val;
        uint32_t name_off = v & 0xffffffff;
        uint16_t id = (v >> 48) & 0xffff;
        switch ((uint64_t)d.d_tag) {
            case 0x61000045: needed_mods[id] = { strtab_str(name_off) }; break; // NEEDED_MODULE (1-based)
            case 0x61000049: import_libs[id] = { strtab_str(name_off) }; break; // library records (0-based id == #L# in NIDs)
            case 0x61000047: export_libs[id] = { strtab_str(name_off) }; break; // this module's own export lib
        }
    }

    printf("\n[DYNAMIC TAGS] (%zu)\n", dyns.size());
    for (auto& d : dyns) {
        const char* nm = dtag_name(d.d_tag);
        printf("  %-24s (0x%09llx) = 0x%llx\n", nm, (unsigned long long)d.d_tag, (unsigned long long)d.d_val);
    }

    printf("\n[NEEDED MODULES] (%zu)\n", needed_mods.size());
    for (auto& [id, r] : needed_mods) printf("  mod#%-3d %s\n", id, r.name.c_str());
    printf("\n[IMPORT LIBRARIES] (%zu)\n", import_libs.size());
    for (auto& [id, r] : import_libs) printf("  lib#%-3d %s\n", id, r.name.c_str());
    printf("\n[EXPORT LIBRARIES] (%zu)\n", export_libs.size());
    for (auto& [id, r] : export_libs) printf("  lib#%-3d %s\n", id, r.name.c_str());

    // -- symbols: name is "NID#libId#modId" (ids are Sony-base64) --
    static const char* B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-";
    auto b64val = [&](const std::string& s) -> int {
        int v = 0; for (char c : s) { const char* p = strchr(B64, c); if (!p) return -1; v = v * 64 + (int)(p - B64); } return v;
    };
    std::map<std::string, std::set<std::string>> by_lib; // libname -> set of NIDs
    size_t nsym = syment ? symtabsz / syment : 0, n_import = 0, n_export = 0;
    for (size_t i = 0; i < nsym && symtab_foff >= 0; i++) {
        auto sym = rd<Elf64_Sym>(b, (size_t)symtab_foff + i * syment);
        const char* raw = strtab_str(sym.st_name);
        std::string s = raw;
        // split nid#lib#mod
        auto h1 = s.find('#'); if (h1 == std::string::npos) continue;
        auto h2 = s.find('#', h1 + 1);
        std::string nid = s.substr(0, h1);
        std::string libid = (h2 == std::string::npos) ? "" : s.substr(h1 + 1, h2 - h1 - 1);
        int lid = b64val(libid);
        std::string libname = import_libs.count(lid) ? import_libs[lid].name
                              : export_libs.count(lid) ? export_libs[lid].name : ("lib#" + libid);
        bool is_import = (sym.st_shndx == 0 && sym.st_value == 0);
        if (is_import) { n_import++; by_lib[libname].insert(nid); }
        else n_export++;
    }
    printf("\n[SYMBOLS] total=%zu  imports=%zu  exports=%zu\n", nsym, n_import, n_export);
    printf("\n[IMPORTS BY LIBRARY]\n");
    for (auto& [lib, nids] : by_lib) {
        printf("  %-32s %zu functions\n", lib.c_str(), nids.size());
        if (dump_syms) for (auto& n : nids) printf("       %s\n", n.c_str());
    }
    return 0;
}
