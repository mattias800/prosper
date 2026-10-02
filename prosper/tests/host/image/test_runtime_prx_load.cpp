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
// The Linux missing-data-base mode also pins unresolved OBJECT refusal and recovery against a
// real installed DATA backend, without following the wrong low GOT produced by the old loader.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "host/image/boot_program.hpp"
#include "host/image/exec_image.hpp"
#include "host/image/runtime_module_load.hpp"
#include "loader/linker.hpp"
#include "fixtures/test_scratch.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#ifdef __linux__
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#if defined(__linux__) && defined(__x86_64__)
#include <array>
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <system_error>
#include <sys/mman.h>
#endif

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
                                         const std::string& import_nid,
                                         uint8_t import_type = STT_FUNC) {
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
        f[p + 4] = 0x10 | (i == 2 ? import_type : STT_FUNC);
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
    rela(kJmprel, kGot, import_type == STT_OBJECT ? R_X86_64_GLOB_DAT : R_X86_64_JUMP_SLOT,
         2, 0);                                                      // GOT slot = the import

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

#if defined(__linux__) && defined(__x86_64__)
namespace {
constexpr size_t kStubPageBytes = 0x1000;
constexpr size_t kMappedImageBytes = 0x4000;
constexpr uint64_t kAppendArgs = 0x10, kAppendArgp = 0xdeadbeef;
constexpr int32_t kResultSentinel = 0x7fffffff;
constexpr uint64_t kImportMarker = 0xa550;

enum class AppendFailure { Map, Image, Emission };
enum class MmapFailure { None, Tail, Image };
struct MmapProbe {
    MmapFailure failure = MmapFailure::None;
    uint64_t image_address = 0;
    unsigned tail_requests = 0, tail_successes = 0, image_requests = 0, injected = 0;
    bool tail_is_mapped = false;
};
MmapProbe mmap_probe;

bool fixture_mapping(void* address, size_t length, int prot, int flags, int fd, off_t offset,
                     uint64_t expected_address, size_t expected_length) {
    return reinterpret_cast<uintptr_t>(address) == expected_address && length == expected_length &&
        prot == (PROT_READ | PROT_WRITE | PROT_EXEC) &&
        flags == (MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE) && fd == -1 && offset == 0;
}

void observe_append(MmapFailure failure = MmapFailure::None) {
    // Keep the mapping-ownership observation across phases; only counters/injection are reset.
    const bool mapped = mmap_probe.tail_is_mapped;
    mmap_probe = {};
    mmap_probe.tail_is_mapped = mapped;
    mmap_probe.failure = failure;
    mmap_probe.image_address = BOOT_RUNTIME_MODULE_BASE;
}

bool append_check(bool condition, const char* scope, const char* label) {
    printf("  [%s] %s: %s\n", condition ? "ok" : "FAIL", scope, label);
    if (!condition) ++fails;
    return condition;
}

PROSPER_SYSV_ABI uint64_t append_import_handler(uint64_t, uint64_t, uint64_t,
                                                uint64_t, uint64_t, uint64_t) {
    return kImportMarker;
}
void append_return_hook() {}
} // namespace

// Target-only --wrap=mmap observes the six-argument fixture tuple. All other host requests pass
// through, and a selected failure consumes exactly one request without creating a mapping.
extern "C" void* __real_mmap(void*, size_t, int, int, int, off_t);
extern "C" void* __wrap_mmap(void* address, size_t length, int prot, int flags, int fd, off_t offset) {
    const bool tail = fixture_mapping(address, length, prot, flags, fd, offset,
                                     BOOT_STUB + kStubPageBytes, kStubPageBytes);
    const bool image = fixture_mapping(address, length, prot, flags, fd, offset,
                                      mmap_probe.image_address, kMappedImageBytes);
    if (tail) ++mmap_probe.tail_requests;
    if (image) ++mmap_probe.image_requests;
    if ((tail && mmap_probe.failure == MmapFailure::Tail) ||
        (image && mmap_probe.failure == MmapFailure::Image)) {
        mmap_probe.failure = MmapFailure::None;
        ++mmap_probe.injected;
        errno = ENOMEM;
        return MAP_FAILED;
    }
    void* result = __real_mmap(address, length, prot, flags, fd, offset);
    if (tail && result == address) {
        ++mmap_probe.tail_successes;
        mmap_probe.tail_is_mapped = true;
    }
    return result;
}

namespace {
struct AppendFixture {
    Program* program = nullptr;
    std::string root;
    HleFn load = nullptr, dlsym = nullptr;
    size_t prefix_slots = 0;
    std::vector<uint8_t> prefix_bytes;
    std::array<uint8_t, kStubPageBytes> first_page{};
    std::vector<uint32_t> prefix_order;
};

uint64_t append_pointer(const void* p) { return reinterpret_cast<uintptr_t>(p); }

bool prepare_append_fixture(AppendFixture& fixture, size_t prefix_slots, uint64_t stride) {
    register_builtin_hle();
    const TlsModuleDesc empty_tls{};
    if (!append_check(setenv("PROSPER_NO_GUEST_FS", "1", 1) == 0, "append setup",
                      "select host TLS before configuring templates")) return false;
    guest_tls_set_templates(&empty_tls, 1);
    if (!append_check(!guest_tls_enabled(), "append setup", "host-mode stubs are selected"))
        return false;
    fixture.root = prosper_test::test_scratch_file("stub-recovery-root");
    std::error_code error;
    std::filesystem::create_directories(fixture.root + "/prx", error);
    if (!append_check(!error, "append setup", "create the owned synthetic module directory"))
        return false;
    set_app0_root(fixture.root);
    static Program program; // dispatcher/runtime registries retain this pointer until process exit
    fixture.program = &program;
    fixture.prefix_slots = prefix_slots;
    program.stub_base = BOOT_STUB;
    program.stub_size = stride;
    for (size_t i = 0; i < prefix_slots; ++i)
        program.slots.push_back({ "fixture", nid_hash("prosperAppendPrefix" + std::to_string(i)) });
    std::string err;
    if (!append_check(install_stubs(program.slots, program.stub_base, stride, &err),
                      "append setup", "install a real one-page prefix")) return false;
    runtime_module_loader_init(&program);
    fixture.load = Hle::lookup(nid_hash("sceKernelLoadStartModule"));
    fixture.dlsym = Hle::lookup(nid_hash("sceKernelDlsym"));
    if (!append_check(fixture.load && fixture.dlsym, "append setup",
                      "resolve the final registered LoadStartModule and Dlsym callers")) return false;
    // Index1 leaves sanitizer indirect-call metadata reads inside our owned mapped prefix page.
    const auto prefix = reinterpret_cast<HleFn>(BOOT_STUB + stride);
    append_check(prefix(0, 0, 0, 0, 0, 0) == 0 && prefix(0, 0, 0, 0, 0, 0) == 0,
                 "append setup", "the existing unresolved stub executes twice");
    fixture.prefix_order = call_order();
    append_check(fixture.prefix_order == std::vector<uint32_t>{1}, "append setup",
                 "two prefix calls produce one first-seen census entry");
    fixture.prefix_bytes.resize(prefix_slots * stride);
    memcpy(fixture.prefix_bytes.data(), reinterpret_cast<const void*>(BOOT_STUB),
           fixture.prefix_bytes.size());
    memcpy(fixture.first_page.data(), reinterpret_cast<const void*>(BOOT_STUB), kStubPageBytes);
    return fails == 0;
}

bool write_append_module(const AppendFixture& fixture, const char* basename,
                         const std::string& import_nid) {
    const std::vector<uint8_t> bytes = build_module(nid_hash("prosperRecoveryExport"), import_nid);
    const std::string path = fixture.root + "/prx/" + basename;
    FILE* file = fopen(path.c_str(), "wb");
    bool written = false;
    if (file) {
        const bool complete = fwrite(bytes.data(), 1, bytes.size(), file) == bytes.size();
        written = fclose(file) == 0 && complete;
    }
    if (!append_check(written, "append fixture", "write the complete synthetic module")) return false;
    std::string err;
    const auto module = Module::load(path, &err);
    if (!append_check(module && module->imports.size() == 1 && module->imports[0].nid == import_nid,
                      "append fixture", "parse a valid module with exactly the selected import"))
        return false;
    LoadedImage image;
    return append_check(build_image(*module, BOOT_RUNTIME_MODULE_BASE, image, &err) &&
                        image.min_vaddr == 0 && image.max_vaddr == kMappedImageBytes &&
                        image.mem.size() == kMappedImageBytes && bytes.size() == kFileSize,
                        "append fixture", "the valid 4 KiB file builds a 16 KiB image");
}

uint64_t call_append_load(const AppendFixture& fixture, const char* basename, int32_t& result) {
    const std::string path = "/app0/prx/" + std::string(basename);
    return fixture.load(append_pointer(path.c_str()), kAppendArgs, kAppendArgp, 0, 0,
                        append_pointer(&result));
}

void check_append_prefix(const AppendFixture& fixture, const char* scope) {
    append_check(memcmp(fixture.prefix_bytes.data(), reinterpret_cast<const void*>(BOOT_STUB),
                        fixture.prefix_bytes.size()) == 0, scope, "live prefix bytes remain unchanged");
    append_check(call_order() == fixture.prefix_order, scope, "the prefix census survives before another call");
    const auto prefix = reinterpret_cast<HleFn>(BOOT_STUB + fixture.program->stub_size);
    append_check(prefix(0, 0, 0, 0, 0, 0) == 0 && call_order() == fixture.prefix_order,
                 scope, "another prefix call preserves its first-seen count");
}

void check_append_refusal(const AppendFixture& fixture, const char* scope, bool committed) {
    int32_t result = kResultSentinel;
    append_check(call_append_load(fixture, "failure.prx", result) == 0x8002000cULL, scope,
                 "ENOMEM reaches the registered guest caller");
    append_check(result == kResultSentinel && runtime_loaded_module_count() == 0, scope,
                 "refusal leaves pRes untouched and publishes no module");
    append_check(fixture.program->slots.size() == fixture.prefix_slots + (committed ? 1 : 0), scope,
                 committed ? "the committed code slot is retained" : "uncommitted code slots are rolled back");
    check_append_prefix(fixture, scope);
    if (!committed)
        append_check(memcmp(fixture.first_page.data(), reinterpret_cast<const void*>(BOOT_STUB),
                            kStubPageBytes) == 0, scope, "refusal preserves the entire original page");
}

uint64_t append_peek(uint64_t base, uint64_t offset) {
    uint64_t value;
    memcpy(&value, reinterpret_cast<const void*>(base + offset), sizeof value);
    return value;
}

void check_append_repeat(const AppendFixture& fixture, const char* basename, uint64_t handle,
                         uint64_t base, size_t expected_count, const char* scope) {
    int32_t result = kResultSentinel;
    append_check(call_append_load(fixture, basename, result) == handle && result == 0 &&
                 runtime_loaded_module_count() == expected_count, scope,
                 "repeat load returns the same handle and successful pRes");
    append_check(append_peek(base, kStartCount) == 1 && append_peek(base, kCtorCount) == 1,
                 scope, "repeat load does not rerun either initializer");
}

bool check_append_loaded(const AppendFixture& fixture, const char* basename, uint64_t expected_base,
                         size_t expected_count, size_t expected_slots, size_t import_slot,
                         const char* scope) {
    int32_t result = kResultSentinel;
    const uint64_t handle = call_append_load(fixture, basename, result);
    if (!append_check(handle >= kSceModuleHandleBase && handle < 0x80000000ULL, scope,
                      "the registered caller returns a real handle")) return false;
    append_check(runtime_loaded_module_count() == expected_count, scope, "one new module is published");
    append_check(fixture.program->slots.size() == expected_slots, scope, "the exact slot count is preserved");
    uint64_t address = 0;
    if (!append_check(fixture.dlsym(handle, append_pointer("prosperRecoveryExport"),
                                   append_pointer(&address), 0, 0, 0) == 0 && address != 0,
                      scope, "Dlsym resolves this handle's real export")) return false;
    const uint64_t base = address - kExportFn;
    if (!append_check(base == expected_base, scope, "the successful load uses the next burned image base"))
        return false;
    append_check(reinterpret_cast<uint32_t (*)()>(address)() == 0x5eed, scope,
                 "the mapped export executes the synthetic positive control");
    append_check(result == 0 && append_peek(base, kStartCount) == 1 && append_peek(base, kCtorCount) == 1,
                 scope, "both initializers run once and pRes receives success");
    append_check(append_peek(base, kSavedArgs) == kAppendArgs && append_peek(base, kSavedArgp) == kAppendArgp,
                 scope, "module_start receives the original guest arguments");
    const uint64_t got = append_peek(base, kGot);
    append_check(got == BOOT_STUB + import_slot * fixture.program->stub_size, scope,
                 "the relocated GOT retains the exact import slot address");
    const bool mapped = got >= BOOT_STUB && got < BOOT_STUB + 2 * kStubPageBytes &&
        (got < BOOT_STUB + kStubPageBytes || mmap_probe.tail_is_mapped);
    if (append_check(mapped, scope, "the resolved import points into observed owned stub pages"))
        append_check(reinterpret_cast<HleFn>(got)(0, 0, 0, 0, 0, 0) == kImportMarker, scope,
                     "the resolved import stub executes its real handler");
    check_append_repeat(fixture, basename, handle, base, expected_count, scope);
    check_append_prefix(fixture, scope);
    return true;
}

void check_append_tail_recovery(const char* scope) {
    append_check(mmap_probe.tail_requests == 1 && mmap_probe.tail_successes == 1 &&
                 mmap_probe.injected == 0, scope, "one tail mapping succeeds without injection");
}

void exercise_append_map_failure(AppendFixture& fixture, const std::string& import_nid) {
    observe_append(MmapFailure::Tail);
    check_append_refusal(fixture, "map refusal", false);
    append_check(mmap_probe.injected == 1 && mmap_probe.tail_requests == 1 &&
                 mmap_probe.tail_successes == 0 && mmap_probe.image_requests == 0,
                 "map refusal", "one exact tail request is refused before image mapping");
    observe_append();
    const bool loaded = check_append_loaded(fixture, "failure.prx",
        BOOT_RUNTIME_MODULE_BASE + BOOT_RUNTIME_MODULE_STRIDE, 1, 43, 42, "map recovery");
    check_append_tail_recovery("map recovery");
    const std::string next_nid = nid_hash(import_nid + "Next");
    if (!loaded || !write_append_module(fixture, "followup.prx", next_nid)) return;
    Hle::register_fn(next_nid, append_import_handler, "fixture next import");
    observe_append();
    check_append_loaded(fixture, "followup.prx", BOOT_RUNTIME_MODULE_BASE + 2 * BOOT_RUNTIME_MODULE_STRIDE,
                        2, 44, 43, "map followup");
    append_check(mmap_probe.tail_requests == 0, "map followup", "the next slot reuses the mapped tail page");
}

void exercise_append_image_failure(AppendFixture& fixture, const std::string& import_nid) {
    observe_append(MmapFailure::Image);
    check_append_refusal(fixture, "image refusal", true);
    append_check(mmap_probe.injected == 1 && mmap_probe.image_requests == 1 &&
                 mmap_probe.tail_requests == 1 && mmap_probe.tail_successes == 1,
                 "image refusal", "the tail commits before one exact 16 KiB image refusal");
    observe_append();
    const bool loaded = check_append_loaded(fixture, "failure.prx",
        BOOT_RUNTIME_MODULE_BASE + BOOT_RUNTIME_MODULE_STRIDE, 1, 43, 42, "image recovery");
    append_check(mmap_probe.tail_requests == 0, "image recovery", "retry retains the already mapped tail page");
    if (!loaded || !write_append_module(fixture, "followup.prx", import_nid)) return;
    observe_append();
    check_append_loaded(fixture, "followup.prx", BOOT_RUNTIME_MODULE_BASE + 2 * BOOT_RUNTIME_MODULE_STRIDE,
                        2, 43, 42, "image followup");
    append_check(mmap_probe.tail_requests == 0, "image followup", "another module reuses the committed import");
}

void exercise_append_emission_failure(AppendFixture& fixture, const std::string& fitting_nid) {
    observe_append();
    check_append_refusal(fixture, "emission refusal", false);
    append_check(mmap_probe.tail_requests == 0 && mmap_probe.image_requests == 0 && mmap_probe.injected == 0,
                 "emission refusal", "oversized emission is refused before any tail or image mapping");
    if (!write_append_module(fixture, "recovery.prx", fitting_nid)) return;
    observe_append();
    check_append_loaded(fixture, "recovery.prx",
        BOOT_RUNTIME_MODULE_BASE + BOOT_RUNTIME_MODULE_STRIDE, 1, 171, 170, "emission recovery");
    check_append_tail_recovery("emission recovery");
}

int run_append_fixture(AppendFailure failure) {
    AppendFixture fixture;
    const bool emission = failure == AppendFailure::Emission;
    if (!prepare_append_fixture(fixture, emission ? 170 : 42, emission ? 24 : 96)) return 1;
    const std::string fitting_nid = nid_hash("prosperAppendFittingImport");
    const std::string hooked_nid = nid_hash("prosperAppendHookedImport");
    Hle::register_fn(fitting_nid, append_import_handler, "fixture fitting import");
    Hle::register_fn(hooked_nid, append_import_handler, "fixture hooked import", append_return_hook);
    append_check(Hle::return_hook_of(hooked_nid) == append_return_hook &&
                 !Hle::return_hook_of(fitting_nid), "append setup",
                 "the oversized hook and fitting plain handler have distinct emitter metadata");
    if (!write_append_module(fixture, "failure.prx", emission ? hooked_nid : fitting_nid)) return 1;
    if (failure == AppendFailure::Map) exercise_append_map_failure(fixture, fitting_nid);
    else if (failure == AppendFailure::Image) exercise_append_image_failure(fixture, fitting_nid);
    else exercise_append_emission_failure(fixture, fitting_nid);
    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}

// #3554: backend installation does not make Program.data_base valid. Keep the backend live so
// missing Program state cannot accidentally be rejected later by append_import_data instead.
constexpr uint64_t kDataPrefixMarker = 0x13572468ULL;
constexpr uint64_t kDataWriteMarker = 0x24681357ULL;
constexpr uint64_t kProviderObject = 0x850;

void add_fixture_object(std::vector<uint8_t>& bytes, const std::string& nid, bool provider) {
    uint64_t strsz = 0;
    memcpy(&strsz, bytes.data() + kDynamic + 4 * 16 + 8, sizeof strsz);
    const std::string raw = nid + (provider ? "" : "#A#A");
    memcpy(bytes.data() + kStrtab + strsz, raw.c_str(), raw.size() + 1);
    const uint64_t symbol = kSymtab + 3 * 24;
    put32(bytes, symbol, static_cast<uint32_t>(strsz));
    bytes[symbol + 4] = 0x11; // GLOBAL OBJECT, following the original FUNC import
    put16(bytes, symbol + 6, provider ? 1 : 0);
    put64(bytes, symbol + 8, provider ? kProviderObject : 0);
    put64(bytes, symbol + 16, 8);
    put64(bytes, kDynamic + 8, 4 * 24); // DT_SCE_SYMTABSZ
    put64(bytes, kDynamic + 4 * 16 + 8, strsz + raw.size() + 1); // DT_STRSZ
    if (provider) {
        put64(bytes, kProviderObject, kDataPrefixMarker);
    } else {
        put64(bytes, kJmprel + 24, kGot + 8);
        put32(bytes, kJmprel + 32, R_X86_64_GLOB_DAT);
        put32(bytes, kJmprel + 36, 3);
        put64(bytes, kDynamic + 8 * 16 + 8, 48); // DT_PLTRELSZ
    }
}

bool write_data_fixture(const AppendFixture& fixture, const char* basename,
                        const std::string& function, const std::string& object,
                        bool provider = false) {
    auto bytes = build_module(nid_hash("prosperRecoveryExport"),
                              function.empty() ? object : function,
                              function.empty() ? STT_OBJECT : STT_FUNC);
    if (!function.empty()) add_fixture_object(bytes, object, provider);
    const std::string path = fixture.root + "/prx/" + basename;
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    stream.close();
    if (!append_check(!stream.fail(), "data fixture", "write the complete synthetic DATA module"))
        return false;
    std::string err;
    const auto module = Module::load(path, &err);
    const bool mixed = !function.empty() && !provider;
    if (!append_check(module && module->imports.size() == (mixed ? 2 : 1) &&
                      module->imports[0].nid == (function.empty() ? object : function) &&
                      module->imports[0].elf_type == (function.empty() ? STT_OBJECT : STT_FUNC) &&
                      (!mixed || (module->imports[1].nid == object &&
                                  module->imports[1].elf_type == STT_OBJECT)),
                      "data fixture", "parse the intended import order and ELF types")) return false;
    LoadedImage image;
    return append_check(build_image(*module, BOOT_RUNTIME_MODULE_BASE, image, &err) &&
                        image.min_vaddr == 0 && image.max_vaddr == kMappedImageBytes &&
                        image.mem.size() == kMappedImageBytes, "data fixture",
                        "the DATA fixture independently builds a valid bounded image");
}

struct DataModuleView { uint64_t handle = 0, base = 0; bool mapped = false; };

DataModuleView load_data_fixture(const AppendFixture& fixture, const char* basename,
                                uint64_t expected_base, size_t expected_count, const char* scope) {
    int32_t result = kResultSentinel;
    DataModuleView view;
    view.handle = call_append_load(fixture, basename, result);
    if (!append_check(view.handle >= kSceModuleHandleBase && view.handle < 0x80000000ULL,
                      scope, "the registered caller returns a real handle")) return view;
    append_check(result == 0 && runtime_loaded_module_count() == expected_count, scope,
                 "successful pRes and exact module count reach the caller");
    uint64_t address = 0;
    if (!append_check(fixture.dlsym(view.handle, append_pointer("prosperRecoveryExport"),
                                   append_pointer(&address), 0, 0, 0) == 0 && address != 0,
                      scope, "Dlsym resolves this module's real export")) return view;
    view.base = address - kExportFn;
    view.mapped = view.base >= BOOT_RUNTIME_MODULE_BASE && view.base < BOOT_RUNTIME_MODULE_END &&
        (view.base - BOOT_RUNTIME_MODULE_BASE) % BOOT_RUNTIME_MODULE_STRIDE == 0 &&
        guest_va_in_module_code(address) && guest_va_in_module(view.base + kMappedImageBytes - 1);
    if (!append_check(view.mapped, scope, "the export belongs to an observed bounded runtime image"))
        return view;
    // A guard-disabled source may already have cached this path at the first base. Still inspect
    // its actual mapped image, report the wrong base/GOT normally, and never follow its low GOT.
    append_check(view.base == expected_base, scope, "the load preserves the failed-base-slot burn policy");
    append_check(reinterpret_cast<uint32_t (*)()>(address)() == 0x5eed, scope,
                 "the mapped export executes the independent positive control");
    append_check(append_peek(view.base, kStartCount) == 1 && append_peek(view.base, kCtorCount) == 1 &&
                 append_peek(view.base, kSavedArgs) == kAppendArgs &&
                 append_peek(view.base, kSavedArgp) == kAppendArgp, scope,
                 "initializers run once with the original guest arguments");
    check_append_repeat(fixture, basename, view.handle, view.base, expected_count, scope);
    check_append_prefix(fixture, scope);
    return view;
}

bool check_data_target(const DataModuleView& view, uint64_t offset, uint64_t expected,
                       const char* scope) {
    return append_check(view.mapped && append_peek(view.base, offset) == expected,
                        scope, "the OBJECT GOT addresses the exact owned writable storage");
}

int run_missing_data_base_fixture() {
    AppendFixture fixture;
    if (!prepare_append_fixture(fixture, 2, 96)) return 1;
    Program& program = *fixture.program;
    const std::string function = nid_hash("prosperDataRecoveryFunction");
    const std::string object = nid_hash("prosperDataRecoveryObject");
    const std::string prefix_nid = nid_hash("prosperDataPrefix");
    Hle::register_fn(function, append_import_handler, "fixture data-recovery function");
    program.data_slots.push_back({"fixture", prefix_nid});
    program.data_stride = kStubPageBytes;
    std::string err;
    if (!append_check(install_import_data(program.data_slots, BOOT_IMPORT_DATA,
                                          program.data_stride, &err), "data setup",
                      "install a real writable DATA prefix with nonzero stride")) return 1;
    memcpy(reinterpret_cast<void*>(BOOT_IMPORT_DATA), &kDataPrefixMarker, sizeof kDataPrefixMarker);
    std::array<uint8_t, kStubPageBytes> data_prefix{};
    memcpy(data_prefix.data(), reinterpret_cast<const void*>(BOOT_IMPORT_DATA), data_prefix.size());
    program.data_base = 0;
    runtime_module_loader_init(&program);
    append_check(import_data_addr(0) == BOOT_IMPORT_DATA && program.data_base == 0 &&
                 program.data_stride == kStubPageBytes, "data setup",
                 "only Program base is missing while the real backend remains installed");
    if (!write_data_fixture(fixture, "missing-data.prx", function, object)) return 1;

    observe_append();
    int32_t result = kResultSentinel;
    append_check(call_append_load(fixture, "missing-data.prx", result) == 0x8002000cULL,
                 "data refusal", "ENOMEM reaches the registered guest caller");
    append_check(result == kResultSentinel, "data refusal", "pRes is untouched on refusal");
    append_check(runtime_loaded_module_count() == 0 &&
                 module_handle_for_path("/app0/prx/missing-data.prx") == 0,
                 "data refusal", "no module or cached path handle is published");
    append_check(program.slots.size() == fixture.prefix_slots && program.data_slots.size() == 1 &&
                 program.data_slots[0].nid == prefix_nid, "data refusal",
                 "mixed import refusal preserves both slot prefixes");
    append_check(mmap_probe.image_requests == 0 && mmap_probe.tail_requests == 0 &&
                 mmap_probe.injected == 0, "data refusal", "binding refuses before image or stub-tail mapping");
    append_check(memcmp(fixture.first_page.data(), reinterpret_cast<const void*>(BOOT_STUB),
                        fixture.first_page.size()) == 0, "data refusal",
                 "the staged FUNC import leaves the entire code page unchanged");
    check_append_prefix(fixture, "data refusal");
    append_check(memcmp(data_prefix.data(), reinterpret_cast<const void*>(BOOT_IMPORT_DATA),
                        data_prefix.size()) == 0, "data refusal", "actual DATA prefix bytes survive refusal");

    program.data_base = BOOT_IMPORT_DATA; // The same backend is not reinstalled or remapped.
    const uint64_t slot = BOOT_IMPORT_DATA + kStubPageBytes;
    const auto retry = load_data_fixture(fixture, "missing-data.prx",
        BOOT_RUNTIME_MODULE_BASE + BOOT_RUNTIME_MODULE_STRIDE, 1, "data recovery");
    append_check(program.slots.size() == 3 && program.data_slots.size() == 2 &&
                 program.data_slots[1].nid == object, "data recovery",
                 "base-only retry claims exactly one FUNC and one DATA slot");
    const bool installed_function = retry.mapped && program.slots.size() == 3 &&
        program.slots[2].nid == function && append_peek(retry.base, kGot) == BOOT_STUB + 2 * 96;
    append_check(installed_function && reinterpret_cast<HleFn>(BOOT_STUB + 2 * 96)(0, 0, 0, 0, 0, 0) ==
                 kImportMarker, "data recovery", "the staged FUNC NID recovers a real installed handler");
    fixture.prefix_order = {1, 2};
    append_check(call_order() == fixture.prefix_order, "data recovery",
                 "recovery adds one first-seen FUNC census entry");
    const bool writable = check_data_target(retry, kGot + 8, slot, "data recovery");
    const uint64_t variable = retry.mapped ? append_peek(retry.base, kGot + 8) : 0;
    append_check(writable && append_peek(variable, 0) == 0, "data recovery", "new DATA storage is really zero-filled");
    if (writable) memcpy(reinterpret_cast<void*>(variable), &kDataWriteMarker, sizeof kDataWriteMarker);
    append_check(writable && append_peek(variable, 0) == kDataWriteMarker, "data recovery",
                 "a write through the resolved variable is retained");

    if (!write_data_fixture(fixture, "data-followup.prx", function, object)) return 1;
    const auto followup = load_data_fixture(fixture, "data-followup.prx",
        BOOT_RUNTIME_MODULE_BASE + 2 * BOOT_RUNTIME_MODULE_STRIDE, 2, "data followup");
    const bool shared = check_data_target(followup, kGot + 8, slot, "data followup");
    append_check(shared && append_peek(slot, 0) == kDataWriteMarker && program.data_slots.size() == 2 &&
                 program.slots.size() == 3, "data followup", "another module shares the retained DATA slot and bytes");

    program.data_base = 0;
    if (!write_append_module(fixture, "data-function-only.prx", function)) return 1;
    const auto plain = load_data_fixture(fixture, "data-function-only.prx",
        BOOT_RUNTIME_MODULE_BASE + 3 * BOOT_RUNTIME_MODULE_STRIDE, 3, "data FUNC control");
    append_check(plain.mapped && append_peek(plain.base, kGot) == BOOT_STUB + 2 * 96 &&
                 program.slots.size() == 3 && program.data_slots.size() == 2,
                 "data FUNC control", "function-only loading needs no Program DATA base");

    const std::string global_object = nid_hash("prosperDataGlobalObject");
    program.exports[global_object] = BOOT_IMPORT_DATA;
    if (!write_data_fixture(fixture, "data-global-object.prx", "", global_object)) return 1;
    const auto global = load_data_fixture(fixture, "data-global-object.prx",
        BOOT_RUNTIME_MODULE_BASE + 4 * BOOT_RUNTIME_MODULE_STRIDE, 4, "data global control");
    append_check(check_data_target(global, kGot, BOOT_IMPORT_DATA, "data global control") &&
                 append_peek(BOOT_IMPORT_DATA, 0) == kDataPrefixMarker && program.data_slots.size() == 2,
                 "data global control", "a real global OBJECT export wins over the missing DATA base");

    const std::string runtime_object = nid_hash("prosperDataRuntimeObject");
    if (!write_data_fixture(fixture, "data-provider.prx", function, runtime_object, true)) return 1;
    const auto provider = load_data_fixture(fixture, "data-provider.prx",
        BOOT_RUNTIME_MODULE_BASE + 5 * BOOT_RUNTIME_MODULE_STRIDE, 5, "data provider control");
    uint64_t provider_address = 0;
    const bool provider_export = provider.mapped && fixture.dlsym(provider.handle,
        append_pointer("prosperDataRuntimeObject"), append_pointer(&provider_address), 0, 0, 0) == 0 &&
        provider_address == provider.base + kProviderObject;
    append_check(provider_export && append_peek(provider_address, 0) == kDataPrefixMarker,
                 "data provider control", "Dlsym exposes the provider's real writable OBJECT bytes");
    if (!write_data_fixture(fixture, "data-runtime-object.prx", "", runtime_object)) return 1;
    const auto runtime = load_data_fixture(fixture, "data-runtime-object.prx",
        BOOT_RUNTIME_MODULE_BASE + 6 * BOOT_RUNTIME_MODULE_STRIDE, 6, "data runtime control");
    append_check(provider_export && check_data_target(runtime, kGot, provider_address, "data runtime control") &&
                 program.data_slots.size() == 2, "data runtime control",
                 "a real runtime OBJECT export wins over the missing DATA base");
    append_check(memcmp(data_prefix.data(), reinterpret_cast<const void*>(BOOT_IMPORT_DATA),
                        data_prefix.size()) == 0, "data final", "every valid load preserves actual DATA prefix bytes");
    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
} // namespace
#endif

int main(int argc, char** argv) {
#if defined(__linux__) && defined(__x86_64__)
    if (argc == 2 && strcmp(argv[1], "--append-map-failure") == 0)
        return run_append_fixture(AppendFailure::Map);
    if (argc == 2 && strcmp(argv[1], "--append-image-failure") == 0)
        return run_append_fixture(AppendFailure::Image);
    if (argc == 2 && strcmp(argv[1], "--append-emission-failure") == 0)
        return run_append_fixture(AppendFailure::Emission);
    if (argc == 2 && strcmp(argv[1], "--missing-data-base") == 0)
        return run_missing_data_base_fixture();
#endif
    printf("== test_runtime_prx_load ==\n");
    // A scratch "dump root", on real disk and never /tmp: tests/fixtures/test_scratch.h roots it at
    // PROSPER_TEST_SCRATCH_DIR (which prosper/CMakeLists.txt points into the build tree, per ctest
    // case) with a pid component beneath, and removes it at normal exit.
    //
    // #2582: it used to be the FIXED path `<build dir>/runtime-prx-test-root`, passed by the ctest
    // registration. Two ctest invocations against one build directory then wrote and mapped the same
    // synthetic .prx, and a load racing the other process's rewrite of those bytes fails to parse.
    // Measured: 6 concurrent runs sharing one root failed 4, all with `load returned 0x80020008 (no
    // module was loaded)`; 6 with distinct roots passed 6. Nothing in the failure names a file, so it
    // reads as a loader defect. argv[1] still overrides, for a hand-run that wants to keep the tree.
    const std::string root = (argc >= 2)
        ? std::string(argv[1])
        : prosper_test::test_scratch_file("runtime-prx-test-root");
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

#ifdef __linux__
    // #2687: a keeper opened nonblocking makes both runtime preflight fopen and Module::load's
    // fopen immediate. Put the same valid bytes in the pipe; only seekability differs from the
    // successful regular-file control above. One PIPE_BUF-sized nonblocking write is bounded.
    const std::string fifo_path = prx_dir + "/unseekable.prx";
    CHECK(::mkfifo(fifo_path.c_str(), 0600) == 0, "create a disposable unseekable module FIFO");
    const int keeper = ::open(fifo_path.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
    CHECK(keeper >= 0, "open the FIFO keeper without blocking");
    if (keeper >= 0) {
        const long pipe_buf = ::fpathconf(keeper, _PC_PIPE_BUF);
        const bool fits = pipe_buf > 0 && image.size() <= (size_t)pipe_buf;
        CHECK(fits && ::write(keeper, image.data(), image.size()) == (ssize_t)image.size(),
              "the FIFO contains the successful control's complete valid module bytes");
        res = 0x7fffffff;
        CHECK(load(U("/app0/prx/unseekable.prx"), 0, 0, 0, 0, U(&res)) == 0x80020008ull,
              "an unseekable module reaches the guest caller as SCE_KERNEL_ERROR_ENOEXEC");
        CHECK(runtime_loaded_module_count() == 1 && res == 0x7fffffff,
              "the rejected module publishes no handle and leaves the start result untouched");
        ::close(keeper);
    }
#endif

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
