// test_runtime_prx_load — real runtime PRX loading (#639).
//
// The acceptance test deliberately does NOT reuse the original "Blasphemous 2 resolves
// FMOD5_Memory_GetStats" criterion: the boot-time Media/Plugins auto-link (#1609) satisfies that
// one WITHOUT any runtime loading, so it cannot discriminate. This uses the replacement criteria
// from #639's 2026-08-01 status comment, each of which auto-link fails by construction:
//
//   * the module lives OUTSIDE Media/Plugins, at a path composed while the "guest" runs;
//   * its module_start has NOT run before the guest asks for it, and runs exactly once, at the
//     load, with the guest's own sceKernelLoadStartModule arguments;
//   * its DT_INIT_ARRAY runs at the same point (relocated, so it proves relocation happened);
//   * its exports resolve through the returned handle, and the address is real, callable code;
//   * a second load of the same path returns the same handle and does not re-run initialisation;
//   * a genuinely missing path still returns ENOENT;
//   * an import no loaded module satisfies gets a NEW stub slot appended after boot, and the
//     module's GOT entry addresses that slot.
//
// The module is a synthetic ET_SCE_DYNAMIC ELF written by this test, so the check is hermetic:
// no game dump, no network, and the "guest" code is bytes this file emits and can therefore
// assert on exactly.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include "../src/host/boot_program.hpp"
#include "../src/host/exec_image.hpp"
#include "../src/host/runtime_module_load.hpp"
#include "../src/loader/linker.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// ---- synthetic module layout (file offset == vaddr throughout) --------------------------------
enum : uint64_t {
    kModuleStart = 0x100,   // DT_INIT
    kCtor        = 0x140,   // DT_INIT_ARRAY[0]
    kExportFn    = 0x160,   // the exported symbol
    kDynamic     = 0x200,
    kSymtab      = 0x300,
    kStrtab      = 0x400,
    kRela        = 0x500,
    kJmprel      = 0x580,
    kInitArray   = 0x600,
    kGot         = 0x700,
    kStartCount  = 0x800,
    kSavedArgs   = 0x808,
    kSavedArgp   = 0x810,
    kCtorCount   = 0x818,
    kFileSize    = 0x1000,
};

// memcpy, never a type-punned store: the buffer is std::vector<uint8_t> and a uint16_t*/uint32_t*
// write through it is a strict-aliasing violation even where the alignment happens to work out.
static void put16(std::vector<uint8_t>& f, uint64_t off, uint16_t v) { memcpy(&f[off], &v, 2); }
static void put32(std::vector<uint8_t>& f, uint64_t off, uint32_t v) { memcpy(&f[off], &v, 4); }
static void put64(std::vector<uint8_t>& f, uint64_t off, uint64_t v) { memcpy(&f[off], &v, 8); }

// `48 ff 05 d32` incq [rip+d] / `48 89 3d d32` mov [rip+d],rdi / `48 89 35 d32` mov [rip+d],rsi.
static uint64_t emit_riprel(std::vector<uint8_t>& f, uint64_t at, const uint8_t (&op)[3],
                            uint64_t target) {
    f[at] = op[0]; f[at + 1] = op[1]; f[at + 2] = op[2];
    put32(f, at + 3, (uint32_t)(int32_t)(int64_t)(target - (at + 7)));
    return at + 7;
}

static std::vector<uint8_t> build_module(const std::string& export_nid,
                                         const std::string& import_nid) {
    std::vector<uint8_t> f(kFileSize, 0);

    // --- ELF header (ET_SCE_DYNAMIC PRX, x86-64, FreeBSD ABI, program headers only) ---
    memcpy(&f[0], "\x7f" "ELF", 4);
    f[4] = 2; f[5] = 1; f[6] = 1; f[7] = 9;          // ELF64, LSB, version 1, ELFOSABI_FREEBSD
    put16(f, 0x10, 0xfe18);                           // e_type = ET_SCE_DYNAMIC
    put16(f, 0x12, 0x3e);                             // e_machine = x86-64
    put32(f, 0x14, 1);                                // e_version
    put64(f, 0x18, 0);                                // e_entry
    put64(f, 0x20, 0x40);                             // e_phoff
    put16(f, 0x34, 64);                               // e_ehsize
    put16(f, 0x36, 56);                               // e_phentsize
    put16(f, 0x38, 2);                                // e_phnum
    // e_shentsize / e_shnum / e_shstrndx stay zero: this module has no section header table.

    // --- program headers: one RWX PT_LOAD covering the file, plus PT_DYNAMIC ---
    auto phdr = [&](int i, uint32_t type, uint32_t flags, uint64_t off, uint64_t va,
                    uint64_t filesz, uint64_t align) {
        const uint64_t p = 0x40 + (uint64_t)i * 56;
        put32(f, p + 0, type); put32(f, p + 4, flags);
        put64(f, p + 8, off);  put64(f, p + 16, va); put64(f, p + 24, va);
        put64(f, p + 32, filesz); put64(f, p + 40, filesz); put64(f, p + 48, align);
    };
    phdr(0, PT_LOAD,    7, 0,        0,        kFileSize, 0x4000);
    phdr(1, PT_DYNAMIC, 6, kDynamic, kDynamic, 0x100,     8);

    // --- code ---
    // module_start(size_t args, const void* argp): record that it ran and what it was given.
    {
        static const uint8_t incq[3] = { 0x48, 0xff, 0x05 };
        static const uint8_t movrdi[3] = { 0x48, 0x89, 0x3d };
        static const uint8_t movrsi[3] = { 0x48, 0x89, 0x35 };
        uint64_t at = kModuleStart;
        at = emit_riprel(f, at, incq,   kStartCount);
        at = emit_riprel(f, at, movrdi, kSavedArgs);
        at = emit_riprel(f, at, movrsi, kSavedArgp);
        f[at++] = 0x31; f[at++] = 0xc0;                    // xor eax,eax
        f[at++] = 0xc3;                                    // ret
        // DT_INIT_ARRAY[0]: a plain ctor.
        at = kCtor;
        at = emit_riprel(f, at, incq, kCtorCount);
        f[at++] = 0xc3;
        // The exported symbol: real, callable code returning a value only this test knows.
        f[kExportFn + 0] = 0xb8; put32(f, kExportFn + 1, 0x5eed);   // mov eax, 0x5eed
        f[kExportFn + 5] = 0xc3;                                   // ret
    }

    // --- dynamic string + symbol tables ---
    // strtab: "\0<export_nid>\0<import_nid>#A#A\0"
    const std::string import_raw = import_nid + "#A#A";
    uint64_t so = kStrtab + 1;
    const uint64_t export_name_off = so - kStrtab;
    memcpy(&f[so], export_nid.data(), export_nid.size()); so += export_nid.size() + 1;
    const uint64_t import_name_off = so - kStrtab;
    memcpy(&f[so], import_raw.data(), import_raw.size()); so += import_raw.size() + 1;
    const uint64_t strsz = so - kStrtab;

    auto sym = [&](int i, uint64_t name_off, uint16_t shndx, uint64_t value) {
        const uint64_t p = kSymtab + (uint64_t)i * 24;
        put32(f, p + 0, (uint32_t)name_off);
        f[p + 4] = 0x12;                       // STB_GLOBAL | STT_FUNC
        put16(f, p + 6, shndx);
        put64(f, p + 8, value);
        put64(f, p + 16, 8);
    };
    // 0 = the null symbol (never an export), 1 = our export, 2 = an undefined Sony import.
    memset(&f[kSymtab], 0, 24);
    sym(1, export_name_off, 1, kExportFn);
    sym(2, import_name_off, 0, 0);
    const uint64_t nsym = 3;

    // --- relocations ---
    auto rela = [&](uint64_t at, uint64_t offset, uint32_t type, uint32_t symi, int64_t addend) {
        put64(f, at + 0, offset);
        put32(f, at + 8, type); put32(f, at + 12, symi);
        put64(f, at + 16, (uint64_t)addend);
    };
    rela(kRela,   kInitArray, R_X86_64_RELATIVE,  0, (int64_t)kCtor);   // init_array[0] = &ctor
    rela(kJmprel, kGot,       R_X86_64_JUMP_SLOT, 2, 0);                // GOT slot = the import

    // --- dynamic tags. DT_SCE_SYMTABSZ first so the parser's "this is a PS5 .dynamic" run check
    // sees an SCE tag within its 8-entry window. ---
    uint64_t d = kDynamic;
    auto dyn = [&](uint64_t tag, uint64_t val) { put64(f, d, tag); put64(f, d + 8, val); d += 16; };
    dyn(0x6100003f, nsym * 24);        // DT_SCE_SYMTABSZ
    dyn(6,  kSymtab);                  // DT_SYMTAB
    dyn(0xb, 24);                      // DT_SYMENT
    dyn(5,  kStrtab);                  // DT_STRTAB
    dyn(0xa, strsz);                   // DT_STRSZ
    dyn(7,  kRela);                    // DT_RELA
    dyn(8,  24);                       // DT_RELASZ
    dyn(0x17, kJmprel);                // DT_JMPREL
    dyn(2,  24);                       // DT_PLTRELSZ
    dyn(0xc, kModuleStart);            // DT_INIT
    dyn(0x19, kInitArray);             // DT_INIT_ARRAY
    dyn(0x1b, 8);                      // DT_INIT_ARRAYSZ
    dyn(0, 0);                         // DT_NULL
    return f;
}

int main(int argc, char** argv) {
    printf("== test_runtime_prx_load ==\n");
    // A scratch "dump root". argv[1] lets CTest place it on real disk (never /tmp on this project's
    // Linux box); default to the build directory the test is run from.
    const std::string root = (argc >= 2) ? argv[1] : std::string("./runtime-prx-test-root");
    const std::string prx_dir = root + "/prx";          // deliberately NOT Media/Plugins
    const std::string prx_path = prx_dir + "/prosper_runtime_test.prx";
    {
#ifdef _WIN32
        const int rc = system(("mkdir \"" + prx_dir + "\" 2>nul").c_str());
#else
        const int rc = system(("mkdir -p '" + prx_dir + "'").c_str());
#endif
        (void)rc;   // an already-existing directory is fine; the fopen below is the real check
    }

    const std::string export_nid = nid_hash("prosperRuntimeTestExport");
    const std::string import_nid = nid_hash("prosperRuntimeTestImport");
    const std::vector<uint8_t> image = build_module(export_nid, import_nid);
    if (FILE* f = fopen(prx_path.c_str(), "wb")) {
        fwrite(image.data(), 1, image.size(), f); fclose(f);
    } else { printf("  [FAIL] cannot write %s\n", prx_path.c_str()); return 1; }

    register_builtin_hle();
    set_app0_root(root);

    // Minimal booted program: no modules, no exports, an empty stub table at the usual base. Every
    // import the runtime module has must therefore become a NEW slot appended after "boot".
    static Program prog;
    prog.stub_base = BOOT_STUB;
    prog.stub_size = 96;
    std::string err;
    CHECK(install_stubs(prog.slots, prog.stub_base, prog.stub_size, &err),
          "install_stubs (empty table) succeeded");
    runtime_module_loader_init(&prog);

    auto load  = Hle::lookup(nid_hash("sceKernelLoadStartModule"));
    auto dlsym = Hle::lookup(nid_hash("sceKernelDlsym"));
    CHECK(load && dlsym, "LoadStartModule + Dlsym registered");
    if (fails) { printf("== FAIL ==\n"); return 1; }

    auto U = [](const void* p) { return (uint64_t)(uintptr_t)p; };

    // (1) Nothing has been initialised at boot: the module is untouched on disk.
    CHECK(runtime_loaded_module_count() == 0, "no module is loaded before the guest asks");

    // (2) Load it, through a path composed at run time, outside Media/Plugins.
    int32_t res = 0x7fffffff;
    const uint64_t kArgs = 0x10, kArgp = 0xdeadbeef;
    const uint64_t handle = load(U(("/app0/prx/" + std::string("prosper_runtime_test.prx")).c_str()),
                                 kArgs, kArgp, 0, 0, U(&res));
    CHECK(handle >= kSceModuleHandleBase && handle < 0x80000000ull,
          "a non-prelinked PRX outside Media/Plugins loads and returns a real handle");
    // Everything below reads the loaded module's memory, so a failed load must stop here rather
    // than dereference an address the loader never produced.
    if (handle < kSceModuleHandleBase || handle >= 0x80000000ull) {
        printf("== FAIL: load returned 0x%llx (no module was loaded) ==\n",
               (unsigned long long)handle);
        return 1;
    }
    CHECK(runtime_loaded_module_count() == 1, "exactly one module is now loaded");

    // (3) Its exports resolve through THAT handle, and the address is real, callable code.
    uint64_t addr = 0;
    CHECK(dlsym(handle, U("prosperRuntimeTestExport"), U(&addr), 0, 0, 0) == 0 && addr != 0,
          "dlsym through the returned handle resolves the module's export");
    const uint64_t base = addr - kExportFn;
    CHECK(base == BOOT_RUNTIME_MODULE_BASE, "the module is mapped in the runtime-module aperture");
    // Every stack-scan diagnostic that recovers a guest callsite filters on these two predicates.
    // Before #639 they stopped at the stub aperture, which now sits BELOW the runtime modules — a
    // runtime module's return addresses would have been invisible to every one of them.
    CHECK(guest_va_in_module(addr) && guest_va_in_module_code(addr) &&
          !guest_va_in_module_code(BOOT_STUB),
          "a runtime-module address classifies as guest module code, and a stub address does not");
    CHECK(addr && ((uint32_t (*)())(uintptr_t)addr)() == 0x5eed,
          "the resolved export is executable code with the expected behaviour");

    // (4) module_start ran exactly once, at the load, with the guest's own arguments — the
    // property boot-time auto-link cannot have.
    auto peek = [&](uint64_t off) { uint64_t v; memcpy(&v, (const void*)(uintptr_t)(base + off), 8); return v; };
    CHECK(peek(kStartCount) == 1, "module_start ran exactly once");
    CHECK(peek(kSavedArgs) == kArgs && peek(kSavedArgp) == kArgp,
          "module_start received the guest's (args, argp)");
    CHECK(res == 0, "sceKernelLoadStartModule reported module_start's result through pRes");
    CHECK(peek(kCtorCount) == 1, "DT_INIT_ARRAY ran (so the entry was relocated and executed)");

    // (5) An import nothing satisfies got a NEW stub slot, and the GOT addresses it.
    CHECK(prog.slots.size() == 1 && prog.slots[0].nid == import_nid,
          "the module's unsatisfied import appended one stub slot after boot");
    uint64_t got = 0; memcpy(&got, (const void*)(uintptr_t)(base + kGot), 8);
    CHECK(got == prog.stub_base, "the module's GOT entry addresses the newly appended stub");

    // (6) Repeat load: same handle, no re-initialisation.
    res = 0x7fffffff;
    const uint64_t again = load(U("/app0/prx/prosper_runtime_test.prx"), kArgs, kArgp, 0, 0, U(&res));
    CHECK(again == handle, "a repeat load returns the same handle");
    CHECK(peek(kStartCount) == 1, "a repeat load does not re-run module_start");
    CHECK(runtime_loaded_module_count() == 1, "a repeat load does not map a second copy");

    // (7) A genuinely absent path is still ENOENT — #146's contract is unchanged.
    CHECK(load(U("/app0/prx/no_such_module.prx"), 0, 0, 0, 0, 0) == 0x80020002ull,
          "a missing path still returns SCE_KERNEL_ERROR_ENOENT");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
