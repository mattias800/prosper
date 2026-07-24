// module.hpp — Prospero SELF/ELF module: parse, build image, relocate, bind imports.
//
// Host-agnostic. The loaded image is a flat byte buffer plus a chosen guest base
// address; the real executable mmap/VirtualAlloc backing slots in at M2 behind the
// same offsets. Everything here is deterministic and unit-testable with no OS calls.
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <optional>

namespace prosper {

// Overflow-safe bounds check: can `need` bytes be read at `off` within a `total`-byte buffer?
// The obvious `off + need <= total` is NOT safe — a malformed/truncated dump can carry a file
// offset near 2^64 (e.g. an unvalidated e_phoff), and `off + need` then wraps to a small value and
// passes, letting the caller read from a wild pointer. Ordering `need <= total` first makes the
// `total - need` in the second term unable to underflow, and `off <= total - need` cannot wrap.
inline bool self_read_ok(uint64_t off, uint64_t need, uint64_t total) {
    return need <= total && off <= total - need;
}

// Clamp a guest-declared table size (DT_RELASZ / DT_PLTRELSZ, or a symtab span) to the bytes that
// actually exist in the file from the table's file offset. A relocation/symbol table cannot legitimately
// extend past the file, so a malformed dump declaring a huge size must not drive an enormous
// relocs/symbols vector (OOM). Returns 0 when the table's file offset is already at/after EOF; never
// underflows (the offset<size branch guarantees a positive remainder).
inline uint64_t clamp_table_bytes(uint64_t table_file_off, uint64_t declared_bytes, uint64_t file_size) {
    if (table_file_off >= file_size) return 0;
    uint64_t avail = file_size - table_file_off;
    return declared_bytes < avail ? declared_bytes : avail;
}

// ---- raw ELF/Sony structures we care about --------------------------------
enum : uint32_t {
    PT_LOAD = 1, PT_DYNAMIC = 2, PT_TLS = 7,
    PT_GNU_RELRO = 0x6474e552,
    PT_SCE_DYNLIBDATA = 0x61000000, PT_SCE_PROCPARAM = 0x61000001,
    PT_SCE_RELRO = 0x61000010,
};
enum : uint16_t { ET_SCE_DYNEXEC = 0xfe10, ET_SCE_DYNAMIC = 0xfe18 };
// x86-64 relocation types
enum : uint32_t {
    R_X86_64_64 = 1, R_X86_64_GLOB_DAT = 6, R_X86_64_JUMP_SLOT = 7,
    R_X86_64_RELATIVE = 8, R_X86_64_DTPMOD64 = 16, R_X86_64_DTPOFF64 = 17,
    R_X86_64_TPOFF64 = 18,
};

struct Segment {           // a PT_LOAD (or other file-backed) segment
    uint32_t type = 0, flags = 0;
    uint64_t vaddr = 0, memsz = 0, filesz = 0;
    uint64_t file_off = 0; // offset of this segment's bytes inside the container file
};
struct Reloc {
    uint64_t offset = 0;   // guest vaddr to patch
    uint32_t type = 0;
    uint32_t sym = 0;      // dynsym index (0 = none)
    int64_t  addend = 0;
    bool     is_plt = false;
};
struct Symbol {
    std::string raw;       // "NID#libId#modId" (or plain)
    std::string nid;       // the NID part
    int         lib_id = -1;
    uint16_t    shndx = 0;
    uint64_t    value = 0;
    bool        is_import = false; // undefined + value 0
    std::string lib_name;  // resolved from lib_id
};
struct Import {            // one unresolved external symbol
    std::string nid, lib_name;
    uint32_t    sym_index = 0;
};

// Defining-module identity for a TLS export. DTPMOD64 and DTPOFF64 form one tls_index pair, so
// they must resolve through the same record: the module id selects the per-thread block and the
// offset selects the exported object within that block.
struct TlsSymbolLocation {
    uint32_t modid = 0;
    uint64_t offset = 0;
};
using TlsSymbolMap = std::unordered_map<std::string, TlsSymbolLocation>;

// ---- a parsed module -------------------------------------------------------
struct Module {
    std::string path;
    std::vector<uint8_t> file;      // raw file bytes (owns the data)
    size_t   elf_base = 0;          // offset of inner ELF header in `file`
    uint16_t e_type = 0, e_machine = 0;
    uint64_t e_entry = 0;
    // Module initialization (C++ global ctors etc.): DT_INIT + DT_INIT_ARRAY.
    uint64_t init_va = 0, init_array_va = 0, init_array_sz = 0;
    // Thread-local storage template (PT_TLS): init image + sizes for per-thread block allocation.
    uint64_t tls_vaddr = 0, tls_filesz = 0, tls_memsz = 0, tls_align = 0;

    std::vector<Segment> segments;  // all program headers (file-backed ones have file_off)
    std::vector<Segment> loads;     // just the ones that occupy VA space

    // dynamic info
    uint64_t strtab_va = 0, strsz = 0, symtab_va = 0, symtabsz = 0, syment = 24;
    uint64_t rela_va = 0, rela_sz = 0, jmprel_va = 0, jmprel_sz = 0;
    uint64_t pltgot_va = 0;

    std::map<int, std::string> import_libs;   // lib_id (0-based) -> name
    std::map<int, std::string> needed_mods;   // mod_id -> name
    std::vector<std::string>   needed_files;  // DT_NEEDED module filenames

    std::vector<Symbol> symbols;
    std::vector<Reloc>  relocs;
    std::vector<Import> imports;

    // parse a SELF/ELF file from disk. Returns nullopt with *err set on failure.
    static std::optional<Module> load(const std::string& path, std::string* err);

    // virtual-address -> offset within `file` (via LOAD segments). -1 if unmapped.
    int64_t va2foff(uint64_t va) const;
    const char* str_at(uint64_t strtab_off) const; // string in strtab by offset
};

// ---- the built, relocatable image -----------------------------------------
// Flat image spanning [0, image_size). Guest content lives at (vaddr) within it;
// when placed at a host base B, guest address A == B + A_offset where A_offset is
// (A - min_vaddr). We keep min_vaddr==0 for these DYNEXEC modules.
struct LoadedImage {
    uint64_t base = 0;                 // chosen guest base (for relocation math)
    uint64_t min_vaddr = 0, max_vaddr = 0;
    std::vector<uint8_t> mem;          // the mapped image (size = max_vaddr - min_vaddr)
    struct Prot { uint64_t vaddr, size; bool r, w, x; };
    std::vector<Prot> prot;            // per-segment protections (for the real backend)
    uint64_t entry = 0;                // guest entry address (base + e_entry)
    uint32_t tls_modid = 0;            // TLS module id (assigned by the linker; 0 = none)

    // ELF-relative instruction boundaries independently verified for load-time Windows SSE4a
    // translation. Empty by default: the host must never infer boundaries by scanning arbitrary
    // executable bytes. Tests and exact-module manifests may populate this before map_image().
    std::vector<uint64_t> verified_sse4a_sites;

    // stub binding: import sym_index -> assigned guest address (trap stub)
    std::map<uint32_t, uint64_t> import_addr;

    uint8_t* at(uint64_t va);          // pointer into mem for a guest vaddr (or null)
    const uint8_t* at(uint64_t va) const;
};

// prosper materializes each ELF module as one contiguous, eagerly zero-filled host vector. Bound
// guest-declared PT_LOAD spans before allocation so malformed multi-terabyte extents cannot exploit
// virtual-memory overcommit and OOM-kill the process during zero fill. The current title corpus peaks
// at about 251 MiB (PPSA21564); 1 GiB retains substantial headroom without pretending this contiguous
// representation can safely model an arbitrary 64-bit ELF address span.
inline constexpr uint64_t kMaxLoadedImageBytes = 1ull << 30;

// Build a flat image from a parsed module at guest base `base`.
// Build a module image without allowing a malformed PT_LOAD extent to terminate the process through
// an uncaught vector allocation failure. Returns false and describes the load error on failure.
bool build_image(const Module& m, uint64_t base, LoadedImage& out, std::string* err = nullptr);

// The phdr-index key a SELF data segment maps to: bits [31:20] of its flags, masked to 12 bits (the
// SCE self-segment id). Higher flag bits must be masked off or the data-segment map is keyed wrong.
// Exposed for testing; see module.cpp / issue #339.
uint64_t self_segment_key(uint64_t flags);

// Bind each import to a synthetic stub address: stub_base + index*stub_size.
// (At M2 these become real logging-trap stubs.) Records into img.import_addr.
void bind_imports_to_stubs(const Module& m, LoadedImage& img,
                           uint64_t stub_base, uint64_t stub_size);

// Apply all relocations into img.mem. Symbol relocs use img.import_addr for imports
// and the module's own symbol values for internal defs. Returns #applied.
// `tls_symbols_by_nid` (optional) maps an exported TLS symbol's NID to its defining module id and
// in-block offset. Cross-module TLS relocations must use both halves of the same record. TPOFF64 also
// requires `tls_module_below`, the shared Variant-II distance below TP indexed by module id. Missing
// information is logged and left unapplied rather than baking a silently incorrect static offset.
size_t apply_relocations(const Module& m, LoadedImage& img,
                         const TlsSymbolMap* tls_symbols_by_nid = nullptr,
                         const std::vector<uint64_t>* tls_module_below = nullptr);

} // namespace prosper
