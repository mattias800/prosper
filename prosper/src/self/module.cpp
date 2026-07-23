#include "module.hpp"
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <algorithm>

namespace prosper {

// ---- little helpers --------------------------------------------------------
template<class T> static T rd(const std::vector<uint8_t>& b, size_t off) {
    T v{}; if (self_read_ok(off, sizeof(T), b.size())) memcpy(&v, b.data() + off, sizeof(T)); return v;
}
// The SCE self-segment id (the phdr index a data segment maps to) is bits [31:20] of the segment
// flags — 12 bits; higher bits hold other block/property info and MUST be masked off. Without the
// mask, a data segment with any flag bit >= 32 keys the data-segment map by a garbage index, so every
// phdr lookup misses and every segment silently falls back to `elf_base + p_offset` (mapping garbage
// bytes — the fatal-but-silent failure the load() comment warns about). Matches Kyty Elf.cpp:270. #339
uint64_t self_segment_key(uint64_t flags) { return (flags >> 20) & 0xFFFu; }

static int b64val(const std::string& s) {
    static const char* B = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-";
    int v = 0; for (char c : s) { const char* p = strchr(B, c); if (!p) return -1; v = v * 64 + (int)(p - B); }
    return s.empty() ? -1 : v;
}

#pragma pack(push,1)
struct SelfHeader { uint32_t magic; uint8_t version, mode, endian, attr; uint32_t key_type;
                    uint16_t header_size, meta_size; uint64_t file_size; uint16_t num_segments, flags; uint32_t pad; };
struct SelfSegment { uint64_t flags, file_offset, file_size, mem_size; };
struct Ehdr { uint8_t e_ident[16]; uint16_t e_type, e_machine; uint32_t e_version;
              uint64_t e_entry, e_phoff, e_shoff; uint32_t e_flags;
              uint16_t e_ehsize, e_phentsize, e_phnum, e_shentsize, e_shnum, e_shstrndx; };
struct Phdr { uint32_t p_type, p_flags; uint64_t p_offset, p_vaddr, p_paddr, p_filesz, p_memsz, p_align; };
struct Sym  { uint32_t st_name; uint8_t st_info, st_other; uint16_t st_shndx; uint64_t st_value, st_size; };
struct Dyn  { int64_t d_tag; uint64_t d_val; };
struct Rela { uint64_t r_offset, r_info; int64_t r_addend; };
#pragma pack(pop)

int64_t Module::va2foff(uint64_t va) const {
    // Resolve against PT_LOAD segments FIRST — they define the actual mapped image,
    // and the dynamic tables (DYNAMIC/strtab/symtab/relocs) physically live inside
    // them. `loads` also carries non-LOAD file-backed phdrs (DYNAMIC/PROCPARAM/TLS)
    // as a fallback for layouts where those sit outside every LOAD; but such an entry
    // must NEVER shadow the LOAD that truly contains the VA. A DYNAMIC phdr whose own
    // logical p_offset addresses unmapped bytes would otherwise self-resolve when its
    // phdr precedes the covering LOAD in program-header order, returning garbage —
    // exactly the "0 imports" boot failure in issue #113 (PPSA02664). Two passes:
    // LOAD wins; non-LOAD only when no LOAD maps the address.
    for (auto& L : loads)
        if (L.type == PT_LOAD && va >= L.vaddr && va < L.vaddr + L.filesz)
            return (int64_t)(L.file_off + (va - L.vaddr));
    for (auto& L : loads)
        if (L.type != PT_LOAD && va >= L.vaddr && va < L.vaddr + L.filesz)
            return (int64_t)(L.file_off + (va - L.vaddr));
    return -1;
}
const char* Module::str_at(uint64_t off) const {
    int64_t f = va2foff(strtab_va);
    if (f < 0) return "";
    uint64_t uf = (uint64_t)f;
    if (off > file.size() || uf > file.size()) return "";   // guard the addition against wrap
    uint64_t abs = uf + off;                                 // abs <= 2*file.size() < 2^63, no wrap
    if (abs >= file.size()) return "";                       // need at least one byte in range
    // A consumer (std::string / strlen at module.cpp needed_files/imports, sym.raw) reads until a
    // NUL. On a malformed strtab with no terminator before EOF that would read past the buffer, so
    // require an in-range NUL and reject otherwise.
    if (!memchr(file.data() + abs, 0, file.size() - abs)) return "";
    return (const char*)(file.data() + abs);
}

std::optional<Module> Module::load(const std::string& path, std::string* err) {
    auto fail = [&](const char* m) -> std::optional<Module> { if (err) *err = m; return std::nullopt; };
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return fail("cannot open file");
    Module m; m.path = path;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    m.file.resize(n);
    if (fread(m.file.data(), 1, n, f) != (size_t)n) { fclose(f); return fail("short read"); }
    fclose(f);

    // SELF wrapper -> inner ELF, and SELF data segment map (flag 0x800 => real data).
    std::map<uint64_t, SelfSegment> data_seg; // phdr index -> data segment
    auto sh = rd<SelfHeader>(m.file, 0);
    // Two observed SELF magics with an identical header/segment-table layout from offset 4 on
    // (byte-compared across both dumps): 0x1D3D154F (The Messenger) and 0xEEF51454 (the UE4 title).
    // Missing this map is fatal-but-silent: the raw-ELF fallback below still finds the inner ELF, but
    // every segment file offset is then wrong, so the guest maps garbage bytes.
    if (sh.magic == 0x1D3D154F || sh.magic == 0xEEF51454) {
        for (int i = 0; i < sh.num_segments; i++) {
            auto s = rd<SelfSegment>(m.file, 0x20 + i * sizeof(SelfSegment));
            if (s.flags & 0x800) data_seg[self_segment_key(s.flags)] = s;
        }
        m.elf_base = 0x20 + (size_t)sh.num_segments * sizeof(SelfSegment);
    }
    auto eh = rd<Ehdr>(m.file, m.elf_base);
    if (memcmp(eh.e_ident, "\x7f""ELF", 4) != 0) {
        m.elf_base = std::string::npos;
        for (size_t i = 0; i + 4 < m.file.size(); i++)
            if (!memcmp(&m.file[i], "\x7f""ELF", 4)) { m.elf_base = i; break; }
        if (m.elf_base == std::string::npos) return fail("no ELF header");
        eh = rd<Ehdr>(m.file, m.elf_base);
    }
    m.e_type = eh.e_type; m.e_machine = eh.e_machine; m.e_entry = eh.e_entry;
    if (eh.e_machine != 0x3e) return fail("not x86-64");

    // program headers
    for (int i = 0; i < eh.e_phnum; i++) {
        auto p = rd<Phdr>(m.file, m.elf_base + eh.e_phoff + i * sizeof(Phdr));
        Segment s{ p.p_type, p.p_flags, p.p_vaddr, p.p_memsz, p.p_filesz, 0 };
        if (data_seg.count(i)) s.file_off = data_seg[i].file_offset;
        else                   s.file_off = m.elf_base + p.p_offset;
        m.segments.push_back(s);
        if (p.p_filesz > 0 && (p.p_type == PT_LOAD || p.p_type == PT_DYNAMIC ||
                               p.p_type == PT_SCE_PROCPARAM || p.p_type == PT_TLS))
            m.loads.push_back(s);
        if (p.p_type == PT_TLS) {   // TLS init template: tdata (filesz) + tbss (memsz-filesz)
            m.tls_vaddr = p.p_vaddr; m.tls_filesz = p.p_filesz;
            m.tls_memsz = p.p_memsz; m.tls_align = p.p_align;
        }
    }
    // Find DYNAMIC
    const Segment* dynseg = nullptr;
    for (auto& s : m.segments) if (s.type == PT_DYNAMIC) dynseg = &s;
    if (!dynseg) return fail("no PT_DYNAMIC");
    int64_t dyn_foff = m.va2foff(dynseg->vaddr);
    if (dyn_foff < 0) dyn_foff = (int64_t)dynseg->file_off; // dynamic may sit inside a LOAD
    // If DYNAMIC vaddr wasn't inside a load's filesz range, resolve via its own file_off.
    if (m.va2foff(dynseg->vaddr) < 0) dyn_foff = (int64_t)dynseg->file_off;

    // On some PS5 modules (e.g. Unity's Il2CppUserAssemblies.prx) the PT_DYNAMIC program header's
    // own file offset is mis-based: `elf_base + p_offset` points at garbage, so the DYNAMIC reads
    // as junk and the module yields 0 exports / 0 init (IL2CPP never bootstraps -> null-fn crash).
    // The .dynamic vaddr, however, also lies inside the sce_dynlibdata PT_LOAD, which IS mapped
    // correctly (its strtab/symtab/rela/jmprel resolve fine). va2foff returns the PT_DYNAMIC's own
    // (wrong) mapping because that segment is listed first; when the result isn't a valid run of DT
    // entries, re-derive the offset from the containing PT_LOAD. Only fires on a genuinely misread
    // DYNAMIC, so modules that already parse are untouched. CONFIDENCE: HIGH (byte-exact verified on
    // Il2CppUserAssemblies.prx: DYNAMIC/STRTAB/SYMTAB/JMPREL/RELA all resolve, IL2CPP API binds).
    auto tag_ok = [](uint64_t t){ return t == 0 || t < 0x100 || (t >= 0x61000000 && t <= 0x610000ffull)
                                       || t == 0x6ffffff9 || t == 0x6ffffffb; };
    auto dyn_run_ok = [&](int64_t off)->bool{
        if (off < 0 || (size_t)off + 16 * 8 > m.file.size()) return false;
        bool saw_sce = false;
        for (int i = 0; i < 8; i++) {
            uint64_t t = (uint64_t)rd<Dyn>(m.file, (size_t)off + i * 16).d_tag;
            if (!tag_ok(t)) return false;
            if (t >= 0x61000000 && t <= 0x610000ffull) saw_sce = true;
        }
        return saw_sce;   // a real PS5 .dynamic always carries DT_SCE_* tags
    };
    if (!dyn_run_ok(dyn_foff)) {
        for (auto& L : m.loads)
            if (L.type == PT_LOAD && dynseg->vaddr >= L.vaddr && dynseg->vaddr < L.vaddr + L.filesz) {
                int64_t cand = (int64_t)(L.file_off + (dynseg->vaddr - L.vaddr));
                if (dyn_run_ok(cand)) { dyn_foff = cand; break; }
            }
    }

    // Parse dynamic tags
    std::vector<Dyn> dyns;
    for (uint64_t o = 0; o + sizeof(Dyn) <= dynseg->filesz; o += sizeof(Dyn)) {
        auto d = rd<Dyn>(m.file, (size_t)dyn_foff + o);
        dyns.push_back(d);
        if (d.d_tag == 0) break;
        switch ((uint64_t)d.d_tag) {
            case 5:  m.strtab_va = d.d_val; break;
            case 0xa: m.strsz = d.d_val; break;
            case 6:  m.symtab_va = d.d_val; break;
            case 0xb: m.syment = d.d_val ? d.d_val : 24; break;
            case 7:  m.rela_va = d.d_val; break;
            case 8:  m.rela_sz = d.d_val; break;
            case 0x17: m.jmprel_va = d.d_val; break;
            case 2:  m.jmprel_sz = d.d_val; break;
            case 3:  m.pltgot_va = d.d_val; break;
            case 0xc:  m.init_va = d.d_val; break;         // DT_INIT
            case 0x19: m.init_array_va = d.d_val; break;   // DT_INIT_ARRAY
            case 0x1b: m.init_array_sz = d.d_val; break;   // DT_INIT_ARRAYSZ
            case 0x6100003f: m.symtabsz = d.d_val; break;
            case 0x61000035: if (!m.strtab_va) m.strtab_va = d.d_val; break;
            case 0x61000039: if (!m.symtab_va) m.symtab_va = d.d_val; break;
        }
    }
    // module/library records (need strtab resolved first, which it is)
    for (auto& d : dyns) {
        uint32_t name_off = d.d_val & 0xffffffff; uint16_t id = (d.d_val >> 48) & 0xffff;
        switch ((uint64_t)d.d_tag) {
            case 1:          m.needed_files.push_back(m.str_at(name_off)); break; // DT_NEEDED
            case 0x61000045: m.needed_mods[id] = m.str_at(name_off); break;
            case 0x61000049: m.import_libs[id] = m.str_at(name_off); break;
        }
    }
    // symbols
    int64_t sym_foff = m.va2foff(m.symtab_va);
    size_t nsym = m.syment ? (m.symtabsz / m.syment) : 0;
    if (nsym == 0 && sym_foff >= 0 && m.strtab_va > m.symtab_va) // fallback: symtab..strtab
        nsym = (m.strtab_va - m.symtab_va) / m.syment;
    for (size_t i = 0; i < nsym && sym_foff >= 0; i++) {
        auto s = rd<Sym>(m.file, (size_t)sym_foff + i * m.syment);
        Symbol sym; sym.raw = m.str_at(s.st_name); sym.shndx = s.st_shndx; sym.value = s.st_value;
        auto h1 = sym.raw.find('#');
        if (h1 != std::string::npos) {
            sym.nid = sym.raw.substr(0, h1);
            auto h2 = sym.raw.find('#', h1 + 1);
            std::string lib = (h2 == std::string::npos) ? "" : sym.raw.substr(h1 + 1, h2 - h1 - 1);
            sym.lib_id = b64val(lib);
        } else sym.nid = sym.raw;
        // A real Sony import is an undefined (shndx 0, value 0) NID-encoded symbol.
        // The null symbol (index 0) and any non-NID undefined symbols are not imports.
        sym.is_import = (s.st_shndx == 0 && s.st_value == 0 && h1 != std::string::npos);
        auto it = m.import_libs.find(sym.lib_id);
        sym.lib_name = (it != m.import_libs.end()) ? it->second : (sym.lib_id >= 0 ? "lib#" + std::to_string(sym.lib_id) : "");
        m.symbols.push_back(sym);
        if (sym.is_import) m.imports.push_back({ sym.nid, sym.lib_name, (uint32_t)i });
    }
    // relocations: RELA + JMPREL
    auto read_relocs = [&](uint64_t va, uint64_t sz, bool plt) {
        int64_t fo = m.va2foff(va);
        if (fo < 0) return;
        for (uint64_t o = 0; o + sizeof(Rela) <= sz; o += sizeof(Rela)) {
            auto r = rd<Rela>(m.file, (size_t)fo + o);
            m.relocs.push_back({ r.r_offset, (uint32_t)(r.r_info & 0xffffffff),
                                 (uint32_t)(r.r_info >> 32), r.r_addend, plt });
        }
    };
    read_relocs(m.rela_va, m.rela_sz, false);
    read_relocs(m.jmprel_va, m.jmprel_sz, true);
    return m;
}

// ---- image building --------------------------------------------------------
uint8_t* LoadedImage::at(uint64_t va) {
    if (va < base + min_vaddr || va >= base + max_vaddr) return nullptr;
    return mem.data() + (va - base - min_vaddr);
}
const uint8_t* LoadedImage::at(uint64_t va) const { return const_cast<LoadedImage*>(this)->at(va); }

LoadedImage build_image(const Module& m, uint64_t base) {
    LoadedImage img; img.base = base;
    uint64_t lo = UINT64_MAX, hi = 0;
    for (auto& s : m.segments)
        // Skip a segment whose vaddr+memsz overflows (a malformed memsz): including it would wrap
        // `hi` and mis-size the image. Well-formed segments have small extents and are unaffected.
        if (s.type == PT_LOAD && s.memsz <= UINT64_MAX - s.vaddr) {
            lo = std::min(lo, s.vaddr); hi = std::max(hi, s.vaddr + s.memsz);
        }
    if (lo == UINT64_MAX) lo = 0;
    auto align_dn = [](uint64_t v, uint64_t a){ return v & ~(a-1); };
    auto align_up = [](uint64_t v, uint64_t a){ return (v + a - 1) & ~(a-1); };
    img.min_vaddr = align_dn(lo, 0x4000);
    img.max_vaddr = align_up(hi, 0x4000);
    if (img.max_vaddr < img.min_vaddr) img.max_vaddr = img.min_vaddr;  // degenerate (align_up wrap) -> empty
    img.mem.assign(img.max_vaddr - img.min_vaddr, 0);
    for (auto& s : m.segments) {
        if (s.type != PT_LOAD || s.vaddr < img.min_vaddr) continue;   // outside the mapped window
        uint64_t dst = s.vaddr - img.min_vaddr;
        // Overflow-safe on BOTH the dest (img.mem) and source (m.file) — a malformed near-2^64
        // filesz/file_off wraps the naive `a + b <= size` check and drives a huge OOB memcpy.
        if (self_read_ok(dst, s.filesz, img.mem.size()) && self_read_ok(s.file_off, s.filesz, m.file.size()))
            memcpy(img.mem.data() + dst, m.file.data() + s.file_off, s.filesz);
        img.prot.push_back({ s.vaddr, s.memsz, (s.flags & 4) != 0, (s.flags & 2) != 0, (s.flags & 1) != 0 });
    }
    img.entry = base + m.e_entry;
    return img;
}

void bind_imports_to_stubs(const Module& m, LoadedImage& img, uint64_t stub_base, uint64_t stub_size) {
    for (size_t i = 0; i < m.imports.size(); i++)
        img.import_addr[m.imports[i].sym_index] = stub_base + i * stub_size;
}

size_t apply_relocations(const Module& m, LoadedImage& img,
                         const TlsSymbolMap* tls_symbols_by_nid,
                         const std::vector<uint64_t>* tls_module_below) {
    size_t applied = 0;
    auto write64 = [&](uint64_t va, uint64_t val) -> bool {
        uint8_t* p = img.at(va);
        if (!p || p + 8 > img.mem.data() + img.mem.size()) return false;
        memcpy(p, &val, 8); return true;
    };
    auto sym_addr = [&](uint32_t sym) -> uint64_t {
        auto it = img.import_addr.find(sym);
        if (it != img.import_addr.end()) return it->second;          // import -> stub
        if (sym < m.symbols.size()) return img.base + m.symbols[sym].value; // internal def
        return 0;
    };
    std::unordered_map<uint32_t, uint64_t> histo, unhandled;
    // PROSPER_GOT_NID=<image-relative GOT vaddr>[,<vaddr>...]: log which import NID fills each
    // listed slot. Used to map a PLT stub (jmp *GOT) seen in a disassembly/crash back to its Sony
    // symbol (e.g. #283). Comma-separated so one boot resolves a whole callsite cluster (#312).
    std::vector<uint64_t> want_gots;
    if (const char* g = getenv("PROSPER_GOT_NID"))
        for (const char* p = g; *p; ) {
            char* end = nullptr;
            uint64_t v = strtoull(p, &end, 0);
            if (end == p) break;
            if (v) want_gots.push_back(v);
            p = (*end == ',') ? end + 1 : end;
        }
    auto want_got_hit = [&](uint64_t off) {
        for (uint64_t w : want_gots) if (w == off) return true;
        return false;
    };
    for (auto& r : m.relocs) {
        uint64_t va = img.base + r.offset;
        histo[r.type]++;
        if (!want_gots.empty() && want_got_hit(r.offset) && r.sym < m.symbols.size())
            fprintf(stderr, "[got-nid] GOT reloc @0x%llx (loaded 0x%llx) type=%u -> import '%s'\n",
                    (unsigned long long)r.offset, (unsigned long long)va, r.type,
                    m.symbols[r.sym].nid.c_str());
        switch (r.type) {
            case R_X86_64_RELATIVE:  if (write64(va, img.base + (uint64_t)r.addend)) applied++; break;
            case R_X86_64_64:        if (write64(va, sym_addr(r.sym) + (uint64_t)r.addend)) applied++; break;
            case R_X86_64_GLOB_DAT:
            case R_X86_64_JUMP_SLOT:  if (write64(va, sym_addr(r.sym))) applied++; break;
            // TLS (general-dynamic model, used by .prx shared libs like libc.prx). A GOT
            // tls_index{module_id, offset} pair is patched by DTPMOD64 + DTPOFF64; the guest then
            // calls __tls_get_addr(tls_index*) which our HLE resolves to a per-thread block. For a
            // module-LOCAL TLS symbol the module id is THIS module's assigned id (the common case:
            // libc's own errno/locale state). A cross-module TLS ref (an IMPORT symbol defined in
            // another module) needs the DEFINING module's id and in-block symbol offset — resolved
            // together via tls_symbols_by_nid when the linker supplies it (#136/#338), else logged
            // unhandled with a best-effort local fallback. DTPOFF64 = offset within the block.
            case R_X86_64_DTPMOD64: {
                uint32_t modid = img.tls_modid;
                if (r.sym && r.sym < m.symbols.size() && m.symbols[r.sym].is_import) {
                    bool resolved = false;
                    if (tls_symbols_by_nid) {
                        auto it = tls_symbols_by_nid->find(m.symbols[r.sym].nid);
                        if (it != tls_symbols_by_nid->end()) { modid = it->second.modid; resolved = true; }
                    }
                    if (!resolved) {
                        unhandled[r.type]++;
                        if (getenv("PROSPER_RELOC_HISTO"))
                            fprintf(stderr, "[reloc]   DTPMOD64 cross-module TLS import '%s' unresolved "
                                    "-> best-effort local modid %u\n",
                                    m.symbols[r.sym].nid.c_str(), modid);
                    }
                }
                if (write64(va, modid)) applied++;
                break;
            }
            case R_X86_64_DTPOFF64: {
                // psABI: DTPOFF64 = S + A. The addend addresses a field INSIDE the TLS object
                // (e.g. a struct member); picking S *or* A drops it and resolves to the object base.
                uint64_t sv = (r.sym && r.sym < m.symbols.size()) ? m.symbols[r.sym].value : 0;
                if (r.sym && r.sym < m.symbols.size() && m.symbols[r.sym].is_import) {
                    bool resolved = false;
                    if (tls_symbols_by_nid) {
                        auto it = tls_symbols_by_nid->find(m.symbols[r.sym].nid);
                        if (it != tls_symbols_by_nid->end()) { sv = it->second.offset; resolved = true; }
                    }
                    if (!resolved) {
                        unhandled[r.type]++;
                        if (getenv("PROSPER_RELOC_HISTO"))
                            fprintf(stderr, "[reloc]   DTPOFF64 cross-module TLS import '%s' unresolved "
                                    "-> best-effort zero symbol offset\n",
                                    m.symbols[r.sym].nid.c_str());
                    }
                }
                if (write64(va, sv + (uint64_t)r.addend)) applied++;
                break;
            }
            case R_X86_64_TPOFF64: {
                // x86-64 Variant II: TPOFF64 = S + A - distance(TP, defining module block).
                // The distance comes from the same helper used by guest_tls_set_templates, so the
                // linker-baked offset and runtime allocation cannot independently drift (#338).
                uint32_t modid = img.tls_modid;
                uint64_t sv = 0;
                bool resolved = r.sym && r.sym < m.symbols.size();
                if (resolved) {
                    const Symbol& sym = m.symbols[r.sym];
                    sv = sym.value;
                    if (sym.is_import) {
                        resolved = false;
                        if (tls_symbols_by_nid) {
                            auto it = tls_symbols_by_nid->find(sym.nid);
                            if (it != tls_symbols_by_nid->end()) {
                                modid = it->second.modid;
                                sv = it->second.offset;
                                resolved = true;
                            }
                        }
                    }
                }
                resolved = resolved && modid != 0 && tls_module_below &&
                           modid < tls_module_below->size() && (*tls_module_below)[modid] != 0;
                if (!resolved) {
                    uint64_t& cnt = unhandled[r.type];
                    if (cnt == 0)
                        fprintf(stderr, "[reloc] WARNING: unresolved TPOFF64 in %s — "
                                "static TLS offset left unapplied (issue #338)\n", m.path.c_str());
                    cnt++;
                    break;
                }
                const uint64_t tpoff = sv + (uint64_t)r.addend - (*tls_module_below)[modid];
                if (write64(va, tpoff)) applied++;
                break;
            }
            default: {
                // Make an unhandled TLS relocation LOUD (once per type) even without
                // PROSPER_RELOC_HISTO: a silently-skipped TLS reloc leaves the GOT slot 0, so the guest
                // reads/writes thread-local storage at the wrong (often zero) offset and corrupts it
                // INVISIBLY. Types 19-23 are other unsupported TLS models. No current title emits
                // these (verified), so this never fires today; when one
                // does, it fails visibly instead of silently corrupting TLS. #338.
                uint64_t& cnt = unhandled[r.type];
                if (cnt == 0 && r.type >= 16 && r.type <= 23)
                    fprintf(stderr, "[reloc] WARNING: unhandled TLS relocation type %u in %s — "
                            "thread-local storage may be corrupted (issue #338)\n", r.type, m.path.c_str());
                cnt++;
                break;
            }
        }
    }
    if (getenv("PROSPER_RELOC_HISTO")) {
        fprintf(stderr, "[reloc] module base=0x%llx: %zu relocs, %zu applied\n",
                (unsigned long long)img.base, m.relocs.size(), applied);
        for (auto& kv : histo)
            fprintf(stderr, "[reloc]   type %u: %llu%s\n", kv.first, (unsigned long long)kv.second,
                    unhandled.count(kv.first) ? "  <<< UNHANDLED/FALLBACK" : "");
    }
    return applied;
}

} // namespace prosper
