#include "module.hpp"
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <algorithm>

namespace prosper {

// ---- little helpers --------------------------------------------------------
template<class T> static T rd(const std::vector<uint8_t>& b, size_t off) {
    T v{}; if (off + sizeof(T) <= b.size()) memcpy(&v, b.data() + off, sizeof(T)); return v;
}
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
    for (auto& L : loads)
        if (va >= L.vaddr && va < L.vaddr + L.filesz) return (int64_t)(L.file_off + (va - L.vaddr));
    return -1;
}
const char* Module::str_at(uint64_t off) const {
    int64_t f = va2foff(strtab_va);
    if (f < 0) return "";
    uint64_t abs = (uint64_t)f + off;
    return abs < file.size() ? (const char*)(file.data() + abs) : "";
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
    if (sh.magic == 0x1D3D154F) {
        for (int i = 0; i < sh.num_segments; i++) {
            auto s = rd<SelfSegment>(m.file, 0x20 + i * sizeof(SelfSegment));
            if (s.flags & 0x800) data_seg[s.flags >> 20] = s;
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
        if (s.type == PT_LOAD) { lo = std::min(lo, s.vaddr); hi = std::max(hi, s.vaddr + s.memsz); }
    if (lo == UINT64_MAX) lo = 0;
    auto align_dn = [](uint64_t v, uint64_t a){ return v & ~(a-1); };
    auto align_up = [](uint64_t v, uint64_t a){ return (v + a - 1) & ~(a-1); };
    img.min_vaddr = align_dn(lo, 0x4000);
    img.max_vaddr = align_up(hi, 0x4000);
    img.mem.assign(img.max_vaddr - img.min_vaddr, 0);
    for (auto& s : m.segments) {
        if (s.type != PT_LOAD) continue;
        uint64_t dst = s.vaddr - img.min_vaddr;
        if (dst + s.filesz <= img.mem.size() && s.file_off + s.filesz <= m.file.size())
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

size_t apply_relocations(const Module& m, LoadedImage& img) {
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
    for (auto& r : m.relocs) {
        uint64_t va = img.base + r.offset;
        histo[r.type]++;
        switch (r.type) {
            case R_X86_64_RELATIVE:  if (write64(va, img.base + (uint64_t)r.addend)) applied++; break;
            case R_X86_64_64:        if (write64(va, sym_addr(r.sym) + (uint64_t)r.addend)) applied++; break;
            case R_X86_64_GLOB_DAT:
            case R_X86_64_JUMP_SLOT:  if (write64(va, sym_addr(r.sym))) applied++; break;
            // TLS (general-dynamic model, used by .prx shared libs like libc.prx). A GOT
            // tls_index{module_id, offset} pair is patched by DTPMOD64 + DTPOFF64; the guest then
            // calls __tls_get_addr(tls_index*) which our HLE resolves to a per-thread block. We
            // resolve the module id to THIS module's assigned id (correct for module-local TLS,
            // which is the common case for libc's own errno/locale state); a cross-module TLS ref
            // (import symbol) would need the defining module's id — logged as unhandled if it ever
            // appears. DTPOFF64 = the TLS symbol's offset within its module's TLS block.
            case R_X86_64_DTPMOD64:  if (write64(va, img.tls_modid)) applied++; break;
            case R_X86_64_DTPOFF64: {
                uint64_t off = (r.sym && r.sym < m.symbols.size()) ? m.symbols[r.sym].value
                                                                   : (uint64_t)r.addend;
                if (write64(va, off)) applied++;
                break;
            }
            default: unhandled[r.type]++; break;
        }
    }
    if (getenv("PROSPER_RELOC_HISTO")) {
        fprintf(stderr, "[reloc] module base=0x%llx: %zu relocs, %zu applied\n",
                (unsigned long long)img.base, m.relocs.size(), applied);
        for (auto& kv : histo)
            fprintf(stderr, "[reloc]   type %u: %llu%s\n", kv.first, (unsigned long long)kv.second,
                    unhandled.count(kv.first) ? "  <<< UNHANDLED (silently skipped)" : "");
    }
    return applied;
}

} // namespace prosper
