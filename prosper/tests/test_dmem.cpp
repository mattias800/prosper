// test_dmem — guards sceKernelAvailableDirectMemorySize (issue #99). It must report the LARGEST
// free aligned direct-memory block within the caller's [searchStart, searchEnd) window and write
// BOTH out-params — it was aliased to the total-size stub, which returned success while leaving
// physAddrOut/sizeOut uninitialized (a caller sizing an allocation from *sizeOut read stack garbage).
// Drives the handlers through the NID registry exactly as the guest does. Each ctest binary is its
// own process, so the process-global direct-memory pool starts empty here.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include "../src/host/guest_write_watch.hpp"
#ifdef _WIN32
#include "../src/host/exec_image.hpp"
#include "../src/host/boot_program.hpp"   // BOOT_EBOOT / BOOT_STUB: the apertures the mprotect cases use
#include "../src/host/guest_memory_map.hpp"
#include "../src/gpu/gpu_execute.hpp"
#include <windows.h>
extern "C" int prosper_try_commit_dmem(uint64_t addr, uint64_t len, int write);
extern "C" int prosper_try_commit_reserved_placeholder(uint64_t addr, uint64_t len);
#endif
#include <cstdio>
#include <cstdint>
#include <cstring>
#ifdef __linux__
#include <csignal>
#include <sys/mman.h>
#include <unistd.h>
#endif

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// Pool bounds mirror hle_kernel_mem.cpp (kDmemBase / kDmemTotal).
static constexpr uint64_t kBase  = 0x10000000ull;
static constexpr uint64_t kTotal = 16ull * 1024 * 1024 * 1024;
static constexpr uint64_t kEnd   = kBase + kTotal;
using Hle7Fn = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t,
                            uint64_t, uint64_t, uint64_t);
#ifdef _WIN32
static constexpr uint64_t kGuestAutoVaMin = 0x2000000000ull;
static constexpr uint64_t kGuestAutoVaMax = 0xfbffffffffull;
#elif defined(__linux__)
namespace {
void texture_watch_segv_handler(int sig, siginfo_t* si, void*) {
    if (sig == SIGSEGV && si->si_addr &&
        host::guest_write_watch_handle_fault(reinterpret_cast<uint64_t>(si->si_addr)))
        return;
    const char message[] = "test_dmem: unexpected SIGSEGV\n";
    (void)!write(STDERR_FILENO, message, sizeof(message) - 1);
    _exit(2);
}

bool install_texture_watch_handler() {
    static uint8_t alternate_stack[128 * 1024];
    stack_t stack{};
    stack.ss_sp = alternate_stack;
    stack.ss_size = sizeof(alternate_stack);
    struct sigaction action{};
    action.sa_sigaction = texture_watch_segv_handler;
    action.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&action.sa_mask);
    if (sigaltstack(&stack, nullptr) != 0 || sigaction(SIGSEGV, &action, nullptr) != 0)
        return false;
    host::guest_write_watch_set_fault_onstack(true);
    return true;
}
} // namespace
#endif

int main() {
    printf("== test_dmem ==\n");
#ifdef _WIN32
    // Windows first-touch handling must commit one 16 KiB guest page, not a 64 KiB allocation-
    // granularity span. The latter crosses this reservation and fails with ERROR_INVALID_ADDRESS.
    register_builtin_hle();
    auto reserve = Hle::lookup(nid_hash("sceKernelReserveVirtualRange"));
    auto flexible = Hle::lookup(nid_hash("sceKernelMapNamedFlexibleMemory"));
    auto unmap   = Hle::lookup(nid_hash("sceKernelMunmap"));
    auto alloc   = Hle::lookup(nid_hash("sceKernelAllocateDirectMemory"));
    auto alloc_main = Hle::lookup(nid_hash("sceKernelAllocateMainDirectMemory"));
    auto map     = Hle::lookup(nid_hash("sceKernelMapDirectMemory"));
    auto map2    = reinterpret_cast<Hle7Fn>(
        Hle::lookup(nid_hash("sceKernelMapDirectMemory2")));
    auto protect = Hle::lookup(nid_hash("sceKernelMprotect"));
    auto mtypeprotect = Hle::lookup(nid_hash("sceKernelMtypeprotect"));
    auto release = Hle::lookup(nid_hash("sceKernelReleaseDirectMemory"));
    auto query   = Hle::lookup(nid_hash("sceKernelVirtualQuery"));
    auto get_type = Hle::lookup(nid_hash("sceKernelGetDirectMemoryType"));
    auto batch   = Hle::lookup(nid_hash("sceKernelBatchMap"));
    CHECK(nid_hash("sceKernelMapDirectMemory2") == "BQQniolj9tQ",
          "sceKernelMapDirectMemory2 hashes to the PS5 3.20 import NID");
    CHECK(nid_hash("sceKernelGetDirectMemoryType") == "BC+OG5m9+bw",
          "sceKernelGetDirectMemoryType hashes to the PS5 3.20 import NID");
    CHECK(reserve && flexible && unmap && alloc && alloc_main && map && map2 && protect &&
              mtypeprotect && release && query && get_type && batch,
          "memory HLE functions registered");
    if (fails) return 1;

    constexpr uint64_t len = 0x4000;
    CHECK((uint32_t)unmap(0, len, 0, 0, 0, 0) == 0x80020016u,
          "Munmap rejects a null address");
    CHECK((uint32_t)unmap(0x4000, 0, 0, 0, 0, 0) == 0x80020016u,
          "Munmap rejects a zero-length range");
    CHECK((uint32_t)unmap(UINT64_MAX & ~0x3fffull, 0x8000, 0, 0, 0, 0) ==
              0x80020016u,
          "Munmap rejects an overflowing guest range");
    uint64_t va = 0x30000000000ull;  // Fixed, 64 KiB-aligned, and above the VEH's heap threshold.
    CHECK(reserve((uint64_t)(uintptr_t)&va, len, 0x10 /* MAP_FIXED */, len, 0, 0) == 0,
          "ReserveVirtualRange creates an exact 16 KiB reservation");
    if (fails) return 1;

    install_trap_handler();
    volatile uint32_t* cell = (volatile uint32_t*)(uintptr_t)va;
    *cell = 0x6310CAFEu;
    CHECK(*cell == 0x6310CAFEu, "first touch commits one guest page and preserves the write");
    CHECK((uint32_t)unmap(va + 1, len, 0, 0, 0, 0) == 0x80020016u,
          "Munmap reports an unaligned host-unmap failure");
    CHECK((uint32_t)unmap(va, 1, 0, 0, 0, 0) == 0x80020016u,
          "Munmap rejects a non-page-multiple length before host rounding");
    uint8_t failed_unmap_info[0x48]{};
    CHECK(query(va, 0, (uint64_t)(uintptr_t)failed_unmap_info,
                sizeof(failed_unmap_info), 0, 0) == 0 &&
              *(uint64_t*)(failed_unmap_info + 0x00) == va &&
              *(uint64_t*)(failed_unmap_info + 0x08) == va + len &&
              *cell == 0x6310CAFEu,
          "failed Munmap preserves VA tracking and mapped contents");
    CHECK(unmap(va, len, 0, 0, 0, 0) == 0, "reserved page unmaps cleanly");

    // Placeholder-backed flexible memory must retain the old VirtualAlloc partial-unmap
    // semantics: decommit only the requested guest page, preserve neighboring data, and allow
    // the hole to be committed again in place.
    constexpr uint64_t flexible_len = 0x10000;
    uint64_t flexible_base = 0;
    CHECK(reserve((uint64_t)(uintptr_t)&flexible_base, flexible_len, 0, 0x4000, 0, 0) == 0 &&
              flexible_base,
          "reserve a multi-page placeholder-backed flexible range");
    uint64_t flexible_full = flexible_base;
    const bool flexible_full_mapped = flexible_base &&
        flexible((uint64_t)(uintptr_t)&flexible_full, flexible_len, 0x2, 0,
                 (uint64_t)(uintptr_t)"partial-flexible", 0) == 0 &&
        flexible_full == flexible_base;
    CHECK(flexible_full_mapped,
          "commit the full placeholder-backed flexible range");
    if (flexible_full_mapped) {
        *(volatile uint64_t*)(uintptr_t)(flexible_base + 0x0100) = 0x1111aaaabbbb2222ull;
        *(volatile uint64_t*)(uintptr_t)(flexible_base + 0x4100) = 0x3333ccccdddd4444ull;
        *(volatile uint64_t*)(uintptr_t)(flexible_base + 0xc100) = 0x5555eeeeffff6666ull;
        CHECK(unmap(flexible_base + 0x4000, 0x4000, 0, 0, 0, 0) == 0,
              "partial unmap decommits one page of a placeholder-backed flexible mapping");
        MEMORY_BASIC_INFORMATION flexible_hole_info{}, flexible_prefix_info{},
                                 flexible_suffix_info{};
        VirtualQuery((void*)(uintptr_t)(flexible_base + 0x4000), &flexible_hole_info,
                     sizeof(flexible_hole_info));
        VirtualQuery((void*)(uintptr_t)flexible_base, &flexible_prefix_info,
                     sizeof(flexible_prefix_info));
        VirtualQuery((void*)(uintptr_t)(flexible_base + 0xc000), &flexible_suffix_info,
                     sizeof(flexible_suffix_info));
        CHECK(flexible_hole_info.State == MEM_RESERVE &&
                  flexible_prefix_info.State == MEM_COMMIT &&
                  flexible_suffix_info.State == MEM_COMMIT,
              "partial flexible unmap preserves committed prefix and suffix pages");
        CHECK(*(volatile uint64_t*)(uintptr_t)(flexible_base + 0x0100) ==
                  0x1111aaaabbbb2222ull &&
              *(volatile uint64_t*)(uintptr_t)(flexible_base + 0xc100) ==
                  0x5555eeeeffff6666ull,
              "partial flexible unmap preserves neighboring contents");
        auto flexible_prefix_watch = host::GuestWriteWatch::create(flexible_base, 0x4000);
        auto flexible_suffix_watch = host::GuestWriteWatch::create(
            flexible_base + 0xc000, 0x4000);
        CHECK(!flexible_prefix_watch && !flexible_suffix_watch &&
                  flexible_prefix_watch.query() == host::GuestWriteWatchQuery::Unknown &&
                  flexible_suffix_watch.query() == host::GuestWriteWatchQuery::Unknown,
              "partial flexible unmap retains the safe exact-comparison fallback");
        uint64_t flexible_hole = flexible_base + 0x4000;
        const bool flexible_hole_mapped =
            flexible((uint64_t)(uintptr_t)&flexible_hole, 0x4000, 0x2, 0,
                     (uint64_t)(uintptr_t)"partial-flexible-hole", 0) == 0 &&
            flexible_hole == flexible_base + 0x4000;
        CHECK(flexible_hole_mapped,
              "MapFlexible recommits the decommitted private page in place");
        if (flexible_hole_mapped) {
            *(volatile uint64_t*)(uintptr_t)(flexible_hole + 0x100) =
                0x7777000011118888ull;
            CHECK(*(volatile uint64_t*)(uintptr_t)(flexible_hole + 0x100) ==
                      0x7777000011118888ull,
                  "recommitted flexible page is writable");
        }
        CHECK(unmap(flexible_base, flexible_len, 0, 0, 0, 0) == 0,
              "partially unmapped flexible allocation releases cleanly as one range");
    } else if (flexible_base) {
        unmap(flexible_base, flexible_len, 0, 0, 0, 0);
    }

    alignas(8) uint8_t invalid_unmap[0x20]{};
    *(uint64_t*)(invalid_unmap + 0x00) = 0x30000020000ull;
    *(uint64_t*)(invalid_unmap + 0x10) = len;
    *(int32_t*)(invalid_unmap + 0x1c) = 1; // UNMAP
    int32_t batch_done = -1;
    CHECK(batch((uint64_t)(uintptr_t)invalid_unmap, 1,
                (uint64_t)(uintptr_t)&batch_done, 0, 0, 0) != 0 && batch_done == 0,
          "BatchMap reports an invalid host unmap instead of counting it as complete");

    // Direct memory is one physical pool, not private memory per virtual mapping. Dead Cells
    // releases and reuses small physical ranges aggressively; independent Windows allocations
    // let its allocator observe different metadata through two aliases and corrupt adjacent VA.
    constexpr uint64_t dlen = 0x10000;
    uint64_t invalid_phys = 0xfeedfacecafebeefull;
    CHECK((uint32_t)alloc(0, kEnd, dlen, dlen, 0, 0) == 0x80020016u,
          "AllocateDirectMemory(null physAddrOut) -> EINVAL");
    CHECK((uint32_t)alloc_main(dlen, dlen, 0, 0, 0, 0) == 0x80020016u,
          "AllocateMainDirectMemory(null physAddrOut) -> EINVAL");
    CHECK((uint32_t)alloc(0, kEnd, 0, dlen, 0, (uint64_t)(uintptr_t)&invalid_phys) ==
              0x80020016u && invalid_phys == 0xfeedfacecafebeefull,
          "AllocateDirectMemory(zero length) -> EINVAL without writing output");
    CHECK((uint32_t)alloc(0, kEnd, dlen - 1, dlen, 0,
                          (uint64_t)(uintptr_t)&invalid_phys) == 0x80020016u &&
              invalid_phys == 0xfeedfacecafebeefull,
          "AllocateDirectMemory(non-page length) -> EINVAL without writing output");
    CHECK((uint32_t)alloc(0, kEnd, dlen, 0x2000, 0,
                          (uint64_t)(uintptr_t)&invalid_phys) == 0x80020016u &&
              invalid_phys == 0xfeedfacecafebeefull,
          "AllocateDirectMemory(non-page alignment) -> EINVAL without writing output");
    CHECK((uint32_t)alloc(0, kEnd, UINT64_MAX & ~0x3fffull, dlen, 0,
                          (uint64_t)(uintptr_t)&invalid_phys) == 0x8002000cu &&
              invalid_phys == 0xfeedfacecafebeefull,
          "AllocateDirectMemory(overflowing size) fails without writing output");
    CHECK((uint32_t)alloc(0, kEnd, dlen, dlen, 0x80000000ull,
                          (uint64_t)(uintptr_t)&invalid_phys) == 0x80020016u &&
              invalid_phys == 0xfeedfacecafebeefull,
          "AllocateDirectMemory(non-int memory type) -> EINVAL without writing output");
    CHECK((uint32_t)alloc(0, kEnd, dlen, dlen, 0, 1) == 0x80020016u,
          "AllocateDirectMemory(low invalid physAddrOut) -> EINVAL");
    CHECK((uint32_t)alloc_main(dlen - 1, dlen, 0,
                               (uint64_t)(uintptr_t)&invalid_phys, 0, 0) == 0x80020016u &&
              invalid_phys == 0xfeedfacecafebeefull,
          "AllocateMainDirectMemory validates the shared direct-memory request contract");
    CHECK((uint32_t)alloc_main(UINT64_MAX & ~0x3fffull, dlen, 0,
                               (uint64_t)(uintptr_t)&invalid_phys, 0, 0) == 0x8002000cu &&
              invalid_phys == 0xfeedfacecafebeefull,
          "AllocateMainDirectMemory(overflowing size) fails without writing output");
    uint64_t multiple_align_phys = 0;
    CHECK(alloc(0, kEnd, 0x4000, 0xc000, 0,
                (uint64_t)(uintptr_t)&multiple_align_phys) == 0 &&
              multiple_align_phys % 0xc000 == 0,
          "AllocateDirectMemory accepts a 16-KiB-multiple non-power-of-two alignment");
    if (multiple_align_phys)
        CHECK(release(multiple_align_phys, 0x4000, 0, 0, 0, 0) == 0,
              "release the non-power-of-two-aligned allocation");
    uint64_t phys = 0, va1 = 0, va2 = 0, va_map2 = 0;
    CHECK(alloc(0, kEnd, dlen, dlen, 12, (uint64_t)(uintptr_t)&phys) == 0 && phys == kBase,
          "PS5 memory type 12 succeeds after invalid allocations leave the pool untouched");
    int32_t direct_type = -1;
    uint64_t direct_start = UINT64_MAX, direct_end = UINT64_MAX;
    CHECK(get_type(phys + 0x4000, (uint64_t)(uintptr_t)&direct_type,
                   (uint64_t)(uintptr_t)&direct_start,
                   (uint64_t)(uintptr_t)&direct_end, 0, 0) == 0 &&
              direct_type == 12 && direct_start == phys && direct_end == phys + dlen,
          "GetDirectMemoryType returns the containing physical allocation and exact type");
    CHECK((uint32_t)get_type(phys, 0, (uint64_t)(uintptr_t)&direct_start,
                             (uint64_t)(uintptr_t)&direct_end, 0, 0) == 0x80020016u,
          "GetDirectMemoryType requires all output pointers");
    direct_type = -1; direct_start = UINT64_MAX; direct_end = UINT64_MAX;
    CHECK((uint32_t)get_type(phys + dlen, (uint64_t)(uintptr_t)&direct_type,
                             (uint64_t)(uintptr_t)&direct_start,
                             (uint64_t)(uintptr_t)&direct_end, 0, 0) == 0x80020002u &&
              direct_type == -1 && direct_start == UINT64_MAX && direct_end == UINT64_MAX,
          "GetDirectMemoryType returns ENOENT without changing outputs for a free offset");
    CHECK(map((uint64_t)(uintptr_t)&va1, dlen, 0x2, 0, phys, dlen) == 0 && va1,
          "map first direct-memory view");
#ifdef _WIN32
    CHECK(va1 >= kGuestAutoVaMin && va1 <= kGuestAutoVaMax - dlen + 1,
          "automatic direct view lands in PS5 libc's valid low VA aperture");
#endif
    CHECK(map((uint64_t)(uintptr_t)&va2, dlen, 0x2, 0, phys, dlen) == 0 && va2 && va2 != va1,
          "map second direct-memory view");
    CHECK(map2((uint64_t)(uintptr_t)&va_map2, dlen, 3 /* type */, 0x2 /* RW */, 0,
               phys, dlen) == 0 && va_map2 && va_map2 != va1 && va_map2 != va2,
          "MapDirectMemory2 consumes shifted prot/flags/phys/alignment arguments");
    if (va1 && va2) {
        *(volatile uint64_t*)(uintptr_t)(va1 + 0x1230) = 0x6310CAFEDEADC0DEull;
        CHECK(*(volatile uint64_t*)(uintptr_t)(va2 + 0x1230) == 0x6310CAFEDEADC0DEull,
              "two virtual mappings of one physical offset alias the same bytes");
        CHECK(va_map2 && *(volatile uint64_t*)(uintptr_t)(va_map2 + 0x1230) ==
                          0x6310CAFEDEADC0DEull,
               "MapDirectMemory2 view aliases the requested physical range");
    }

    uint8_t direct_info[0x48]{};
    CHECK(query(va1 + 0x2000, 0, (uint64_t)(uintptr_t)direct_info,
                sizeof(direct_info), 0, 0) == 0 &&
              *(uint64_t*)(direct_info + 0x00) == va1 &&
              *(uint64_t*)(direct_info + 0x08) == va1 + dlen &&
              *(uint64_t*)(direct_info + 0x10) == phys &&
              *(int32_t*)(direct_info + 0x1c) == 12 &&
              direct_info[0x20] == 0x12,
          "VirtualQuery reports ordinary direct mapping offset/type/classification");
    uint8_t short_query_info[0x22];
    memset(short_query_info, 0xa5, sizeof(short_query_info));
    CHECK(query(va1 + 0x2000, 0, (uint64_t)(uintptr_t)short_query_info,
                0x21, 0, 0) == 0 &&
              short_query_info[0x20] == 0x12 && short_query_info[0x21] == 0xa5,
          "VirtualQuery writes the one-byte classification at exact infoSize 0x21");
    uint8_t exact_query_info[0x42];
    memset(exact_query_info, 0xa5, sizeof(exact_query_info));
    CHECK(query(va1 + 0x2000, 0, (uint64_t)(uintptr_t)exact_query_info,
                0x41, 0, 0) == 0 &&
              exact_query_info[0x20] == 0x12 &&
              memcmp(exact_query_info + 0x21, "direct", 7) == 0 &&
              exact_query_info[0x41] == 0xa5,
          "VirtualQuery writes name[32] at 0x21 within the exact ABI size");
    memset(direct_info, 0, sizeof(direct_info));
    CHECK(query(va_map2 + 0x2000, 0, (uint64_t)(uintptr_t)direct_info,
                sizeof(direct_info), 0, 0) == 0 &&
              *(uint64_t*)(direct_info + 0x10) == phys &&
              *(int32_t*)(direct_info + 0x1c) == 3 &&
              direct_info[0x20] == 0x12,
          "MapDirectMemory2 publishes its explicit type in VirtualQuery");
    direct_type = -1; direct_start = direct_end = 0;
    CHECK(get_type(phys + 0x2000, (uint64_t)(uintptr_t)&direct_type,
                   (uint64_t)(uintptr_t)&direct_start,
                   (uint64_t)(uintptr_t)&direct_end, 0, 0) == 0 &&
              direct_type == 3 && direct_start == phys && direct_end == phys + dlen,
          "MapDirectMemory2 updates the physical allocation type");

    CHECK(mtypeprotect(va1 + 0x4000, 0x4000, 9, 0x1, 0, 0) == 0,
          "Mtypeprotect changes one direct-mapping page");
    memset(direct_info, 0, sizeof(direct_info));
    CHECK(query(va1 + 0x5000, 0, (uint64_t)(uintptr_t)direct_info,
                sizeof(direct_info), 0, 0) == 0 &&
              *(uint64_t*)(direct_info + 0x00) == va1 + 0x4000 &&
              *(uint64_t*)(direct_info + 0x08) == va1 + 0x8000 &&
              *(uint64_t*)(direct_info + 0x10) == phys + 0x4000 &&
              *(int32_t*)(direct_info + 0x18) == 0x1 &&
              *(int32_t*)(direct_info + 0x1c) == 9 &&
              direct_info[0x20] == 0x12,
          "Mtypeprotect preserves the carved page's physical offset and publishes its type");
    CHECK(mtypeprotect(va1 + 0x4000, 0, 14, 0x2, 0, 0) == 0,
          "aligned zero-length Mtypeprotect is a successful no-op");
    memset(direct_info, 0, sizeof(direct_info));
    CHECK(query(va1 + 0x5000, 0, (uint64_t)(uintptr_t)direct_info,
                sizeof(direct_info), 0, 0) == 0 &&
              *(int32_t*)(direct_info + 0x18) == 0x1 &&
              *(int32_t*)(direct_info + 0x1c) == 9,
          "zero-length Mtypeprotect leaves protection and type metadata unchanged");

    alignas(8) uint8_t type_entry[0x20]{};
    *(uint64_t*)(type_entry + 0x00) = va1 + 0x8000;
    type_entry[0x18] = 0x2;
    type_entry[0x19] = 11;
    *(int32_t*)(type_entry + 0x1c) = 4; // TYPE_PROTECT
    int32_t type_done = -1;
    CHECK((uint32_t)batch((uint64_t)(uintptr_t)type_entry, 1,
                          (uint64_t)(uintptr_t)&type_done, 0, 0, 0) == 0x80020016u &&
              type_done == 0,
          "BatchMap rejects a zero-length type-protect entry without counting it");
    *(uint64_t*)(type_entry + 0x00) = va1 + 0x8123;
    *(uint64_t*)(type_entry + 0x10) = 0x100;
    type_done = -1;
    CHECK(batch((uint64_t)(uintptr_t)type_entry, 1,
                (uint64_t)(uintptr_t)&type_done, 0, 0, 0) == 0 && type_done == 1,
          "BatchMap TYPE_PROTECT accepts an unaligned sub-page range");
    memset(direct_info, 0, sizeof(direct_info));
    CHECK(query(va1 + 0x9000, 0, (uint64_t)(uintptr_t)direct_info,
                sizeof(direct_info), 0, 0) == 0 &&
              *(uint64_t*)(direct_info + 0x00) == va1 + 0x8000 &&
              *(uint64_t*)(direct_info + 0x08) == va1 + 0xc000 &&
              *(uint64_t*)(direct_info + 0x10) == phys + 0x8000 &&
              *(int32_t*)(direct_info + 0x18) == 0x2 &&
              *(int32_t*)(direct_info + 0x1c) == 11 &&
              direct_info[0x20] == 0x12,
          "BatchMap TYPE_PROTECT normalizes protection and metadata to the guest page");
    memset(direct_info, 0, sizeof(direct_info));
    CHECK(query(va1 + 0xd000, 0, (uint64_t)(uintptr_t)direct_info,
                sizeof(direct_info), 0, 0) == 0 &&
              *(uint64_t*)(direct_info + 0x10) == phys + 0xc000 &&
              *(int32_t*)(direct_info + 0x1c) == 12,
          "tracker suffix keeps its original type and rebased physical offset");
    direct_type = -1; direct_start = direct_end = 0;
    CHECK(get_type(phys + 0x9000, (uint64_t)(uintptr_t)&direct_type,
                   (uint64_t)(uintptr_t)&direct_start,
                   (uint64_t)(uintptr_t)&direct_end, 0, 0) == 0 &&
              direct_type == 11 && direct_start == phys + 0x8000 &&
              direct_end == phys + 0xc000,
          "TYPE_PROTECT carves the same type range in physical allocation queries");

    CHECK(mtypeprotect(va1 + 0xc123, 0x100, 13, 0x1, 0, 0) == 0,
          "Mtypeprotect accepts an unaligned sub-page range");
    memset(direct_info, 0, sizeof(direct_info));
    CHECK(query(va1 + 0xd000, 0, (uint64_t)(uintptr_t)direct_info,
                sizeof(direct_info), 0, 0) == 0 &&
              *(uint64_t*)(direct_info + 0x00) == va1 + 0xc000 &&
              *(uint64_t*)(direct_info + 0x08) == va1 + 0x10000 &&
              *(uint64_t*)(direct_info + 0x10) == phys + 0xc000 &&
              *(int32_t*)(direct_info + 0x18) == 0x1 &&
              *(int32_t*)(direct_info + 0x1c) == 13 &&
              direct_info[0x20] == 0x12,
          "unaligned Mtypeprotect normalizes VA metadata to the 16 KiB guest page");
    direct_type = -1; direct_start = direct_end = 0;
    CHECK(get_type(phys + 0xd000, (uint64_t)(uintptr_t)&direct_type,
                   (uint64_t)(uintptr_t)&direct_start,
                   (uint64_t)(uintptr_t)&direct_end, 0, 0) == 0 &&
              direct_type == 13 && direct_start == phys + 0xc000 &&
              direct_end == phys + 0x10000,
          "unaligned Mtypeprotect normalizes physical type metadata to the same page");
    CHECK((uint32_t)mtypeprotect(UINT64_MAX - 0x1000, 0x2000, 13, 0x1, 0, 0) ==
              0x80020016u,
          "Mtypeprotect rejects an overflowing guest range");

    uint64_t hinted = va1;
    CHECK(map((uint64_t)(uintptr_t)&hinted, dlen, 0x2, 0, phys, dlen) == 0 &&
              hinted && hinted != va1,
          "non-fixed occupied hint relocates to a shared direct-memory view");
    if (hinted) {
        CHECK(*(volatile uint64_t*)(uintptr_t)(hinted + 0x1230) == 0x6310CAFEDEADC0DEull,
              "relocated hinted mapping retains physical aliasing");
    }

    CHECK(release(phys, dlen, 0, 0, 0, 0) == 0, "release direct-memory page");
    direct_type = -1; direct_start = UINT64_MAX; direct_end = UINT64_MAX;
    CHECK((uint32_t)get_type(phys, (uint64_t)(uintptr_t)&direct_type,
                             (uint64_t)(uintptr_t)&direct_start,
                             (uint64_t)(uintptr_t)&direct_end, 0, 0) == 0x80020002u &&
              direct_type == -1 && direct_start == UINT64_MAX && direct_end == UINT64_MAX,
          "GetDirectMemoryType stops reporting a released allocation");
    uint64_t reused_phys = 0, va3 = 0;
    CHECK(alloc(0, kEnd, dlen, dlen, 0, (uint64_t)(uintptr_t)&reused_phys) == 0 && reused_phys == phys,
          "released physical range is reused");
    CHECK(map((uint64_t)(uintptr_t)&va3, dlen, 0x2, 0, reused_phys, dlen) == 0 && va3,
          "map reused direct-memory page");
    if (va3) {
        CHECK(*(volatile uint64_t*)(uintptr_t)(va3 + 0x1230) == 0,
              "fresh direct-memory allocation is zeroed after physical reuse");
    }
    if (va1) CHECK(unmap(va1, dlen, 0, 0, 0, 0) == 0, "unmap first direct-memory view");
    if (va2) CHECK(unmap(va2, dlen, 0, 0, 0, 0) == 0, "unmap second direct-memory view");
    if (va3) CHECK(unmap(va3, dlen, 0, 0, 0, 0) == 0, "unmap reused direct-memory view");
    if (va_map2) CHECK(unmap(va_map2, dlen, 0, 0, 0, 0) == 0,
                       "unmap MapDirectMemory2 view");
    if (hinted) CHECK(unmap(hinted, dlen, 0, 0, 0, 0) == 0, "unmap relocated direct-memory view");
    if (reused_phys) release(reused_phys, dlen, 0, 0, 0, 0);

    // Large zero-hint alignments use a placeholder-backed sparse-file view. Dead Cells requests
    // one 3 GiB mapping at 2 MiB alignment; a paging-file mapping would charge that whole range as
    // private commit. File-backed demand paging keeps untouched storage sparse without guest-time
    // access violations, and a second VA must still alias the same physical bytes.
    constexpr uint64_t sparse_len = 0x02000000;
    constexpr uint64_t sparse_align = 0x00200000;
    constexpr uint64_t sparse_window = kBase + 0x10000000;
    uint64_t sparse_phys = 0, sparse_va1 = 0, sparse_va2 = 0;
    CHECK(alloc(sparse_window, sparse_window + sparse_len, sparse_len, sparse_align, 0,
                (uint64_t)(uintptr_t)&sparse_phys) == 0 && sparse_phys == sparse_window,
          "allocate a virgin 32 MiB sparse direct-memory range");
    CHECK(map((uint64_t)(uintptr_t)&sparse_va1, sparse_len, 0x2, 0,
              sparse_phys, sparse_align) == 0 && sparse_va1 &&
              (sparse_va1 & (sparse_align - 1)) == 0,
          "large-alignment direct view uses a 2 MiB-aligned address");
    MEMORY_BASIC_INFORMATION near_before{}, far_before{};
    if (sparse_va1) {
        uint8_t guest_query[0x48]{};
        CHECK(query(sparse_va1 + 0x01000000, 0,
                    (uint64_t)(uintptr_t)guest_query, sizeof(guest_query), 0, 0) == 0 &&
                  *(uint64_t*)(guest_query + 0x10) == sparse_phys &&
                  guest_query[0x20] == 0x12,
              "guest VirtualQuery reports sparse direct backing and classification");
        host::GuestReadableRange persistent_range{};
        CHECK(host::guest_readable_mapping_containing(
                  sparse_va1 + 0x4000, sparse_va1 + 0x8000, persistent_range),
              "file-backed direct view supports cross-submit host-readability reuse");
        VirtualQuery((void*)(uintptr_t)(sparse_va1 + 0x4000), &near_before,
                     sizeof(near_before));
        VirtualQuery((void*)(uintptr_t)(sparse_va1 + 0x01000000), &far_before,
                     sizeof(far_before));
        CHECK(near_before.State == MEM_COMMIT && far_before.State == MEM_COMMIT,
              "untouched direct pages need no guest-time lazy-commit fault");
        CHECK(gpu::guest_readable(sparse_va1 + 0x4000, 0x4000),
              "GPU guest-read guard accepts an untouched file-backed zero page");
        MEMORY_BASIC_INFORMATION near_after{}, far_after{};
        VirtualQuery((void*)(uintptr_t)(sparse_va1 + 0x4000), &near_after,
                     sizeof(near_after));
        VirtualQuery((void*)(uintptr_t)(sparse_va1 + 0x01000000), &far_after,
                     sizeof(far_after));
        CHECK(near_after.State == MEM_COMMIT && far_after.State == MEM_COMMIT,
              "host read leaves the fault-free direct mapping committed");
        CHECK(*(volatile uint32_t*)(uintptr_t)(sparse_va1 + 0x4120) == 0,
              "virgin sparse direct-memory page reads as zero");

        const uint64_t protected_page = sparse_va1 + 0x01000000;
        CHECK(protect(protected_page, 0x4000, 0x1, 0, 0, 0) == 0,
              "mprotect accepts an untouched sparse page");
        MEMORY_BASIC_INFORMATION protected_before{};
        VirtualQuery((void*)(uintptr_t)protected_page, &protected_before,
                     sizeof(protected_before));
        CHECK(protected_before.State == MEM_COMMIT &&
                  (protected_before.Protect & 0xff) == PAGE_READONLY,
              "mprotect updates an untouched file-backed page without materialization");
        MEMORY_BASIC_INFORMATION protected_after{};
        VirtualQuery((void*)(uintptr_t)protected_page, &protected_after,
                     sizeof(protected_after));
        CHECK(protected_after.State == MEM_COMMIT &&
                  (protected_after.Protect & 0xff) == PAGE_READONLY,
              "read-only protection remains visible without a first-touch fault");
        CHECK(protect(protected_page, 0x4000, 0x2, 0, 0, 0) == 0,
              "mprotect restores read/write on a materialized sparse page");
        MEMORY_BASIC_INFORMATION writable_after{};
        VirtualQuery((void*)(uintptr_t)protected_page, &writable_after,
                     sizeof(writable_after));
        CHECK(writable_after.State == MEM_COMMIT &&
                  (writable_after.Protect & 0xff) == PAGE_READWRITE,
              "mprotect updates an existing sparse host page");
    }
    // #2101: memory the guest did NOT get from the memory HLE is still real guest memory, and the
    // guest may protect it. A module image is mapped by the exec substrate's VirtualAlloc, so it
    // never enters g_maps; requiring tracker coverage made sceKernelMprotect refuse a title's
    // mprotect of its OWN eboot segment with EINVAL. The Messenger treats that as fatal and
    // abandons AGC device init, losing its register-offset table (upside-down, world-less frames).
    // POSIX mprotect accepts this, so Windows must too. Stand in for the module image with a plain
    // VirtualAlloc the memory HLE has never heard of.
    // The allowance is deliberately narrow: a MODULE aperture, not "any committed page". Commit
    // inside the eboot aperture exactly as map_image does, and separately confirm that an ordinary
    // committed allocation outside every module aperture is still refused -- that second half is the
    // guest-does-not-own-host-memory property test_retrack_reservation guards from the other side.
    {
        constexpr SIZE_T kLen = 0x4000;
        void* image = VirtualAlloc((void*)(uintptr_t)(prosper::BOOT_EBOOT + 0x2000000ull), kLen,
                                   MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        CHECK(image != nullptr, "commit inside the eboot module aperture, as map_image does");
        if (image) {
            const uint64_t va = (uint64_t)(uintptr_t)image;
            CHECK(protect(va, kLen, 0x1, 0, 0, 0) == 0,
                  "sceKernelMprotect accepts a module image the VA tracker does not own");
            MEMORY_BASIC_INFORMATION after{};
            VirtualQuery(image, &after, sizeof(after));
            CHECK(after.State == MEM_COMMIT && (after.Protect & 0xff) == PAGE_READONLY,
                  "the module image's host protection actually changed");
            CHECK(protect(va, kLen, 0x2, 0, 0, 0) == 0, "and it can be restored to read/write");
            // The OS stays the authority on existence: a freed range must still be refused.
            VirtualFree(image, 0, MEM_RELEASE);
            CHECK(protect(va, kLen, 0x2, 0, 0, 0) != 0,
                  "sceKernelMprotect still refuses a freed (MEM_FREE) range");
        }
        void* foreign = VirtualAlloc(nullptr, kLen, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        CHECK(foreign != nullptr, "commit host memory outside every module aperture");
        if (foreign) {
            CHECK((uint32_t)protect((uint64_t)(uintptr_t)foreign, kLen, 0x1, 0, 0, 0) == 0x80020016u,
                  "sceKernelMprotect still refuses committed memory the guest does not own");
            MEMORY_BASIC_INFORMATION unchanged{};
            VirtualQuery(foreign, &unchanged, sizeof(unchanged));
            CHECK((unchanged.Protect & 0xff) == PAGE_READWRITE,
                  "the refused foreign page keeps its original protection");
            VirtualFree(foreign, 0, MEM_RELEASE);
        }
    }
    // #2144 F1: the tracker snapshot win_protect works from must be the COMPLETE coverage of the
    // span, not the prefix before the first untracked hole. #2117 made the caller read that vector
    // as coverage, so a span BEGINNING in an untracked page produced an EMPTY vector and every
    // tracked slice above it was misread as an untracked gap -- which silently voided the refusals
    // that only apply to tracked slices, and made rollback restore the raw mbi.Protect (under an
    // armed write-watch, the transient read-only value the tracker split exists to avoid).
    // Both #2117 arms are wholly inside or wholly outside a module image; this one is MIXED, which
    // is the only shape that reaches the truncation. Mutation killed: restoring the `return false`
    // on the first hole makes the refusal below turn into an accept that reprotects both halves.
    {
        constexpr uint64_t kGranule = 0x10000;   // Windows allocation granularity
        const uint64_t image_va = prosper::BOOT_EBOOT + 0x3000000ull;
        const uint64_t reserved_va = image_va + kGranule;
        // Prefix: a module image the memory HLE has never heard of, exactly as in the #2101 arm.
        void* image = VirtualAlloc((void*)(uintptr_t)image_va, kGranule,
                                   MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        CHECK(image != nullptr, "commit an untracked module-image prefix in the eboot aperture");
        // Suffix: a range the tracker DOES own and calls uncommitted, over host pages that are in
        // fact committed -- the bookkeeping disagreement the merged comment says must keep failing.
        // The fixed reserve hands back a Windows PLACEHOLDER, which plain VirtualAlloc(MEM_COMMIT)
        // cannot back (ERROR_INVALID_ADDRESS). prosper's own lazy-commit entry point -- the one its
        // VEH uses when a guest first touches a reserved page -- replaces the placeholder with
        // private pages under g_dview_mx and never touches g_maps, so the tracker keeps calling the
        // range uncommitted. That is precisely the disagreement this arm needs.
        uint64_t reserved = reserved_va;
        const bool suffix_reserved =
            image && reserve((uint64_t)(uintptr_t)&reserved, kGranule,
                             0x10 /* SCE_KERNEL_MAP_FIXED */, kGranule, 0, 0) == 0 &&
            reserved == reserved_va &&
            prosper_try_commit_reserved_placeholder(reserved_va, kGranule) == 1;
        CHECK(suffix_reserved, "track an uncommitted reservation directly above it");
        uint8_t reserved_info[0x48]{};
        MEMORY_BASIC_INFORMATION suffix_host{};
        if (suffix_reserved)
            VirtualQuery((void*)(uintptr_t)reserved_va, &suffix_host, sizeof(suffix_host));
        // Self-invalidating precondition: without BOTH halves of the disagreement (tracker says
        // uncommitted, host says committed) the refusal below would prove nothing.
        const bool disagreement =
            suffix_reserved && suffix_host.State == MEM_COMMIT &&
            query(reserved_va, 0, (uint64_t)(uintptr_t)reserved_info, sizeof(reserved_info),
                  0, 0) == 0 && reserved_info[0x20] == 0;
        CHECK(disagreement, "the suffix is tracked-uncommitted over committed host pages");
        if (disagreement) {
            CHECK((uint32_t)protect(image_va, kGranule * 2, 0x1, 0, 0, 0) == 0x80020016u,
                  "mprotect still refuses a tracked-uncommitted suffix behind an untracked prefix");
            MEMORY_BASIC_INFORMATION prefix_after{}, suffix_after{};
            VirtualQuery((void*)(uintptr_t)image_va, &prefix_after, sizeof(prefix_after));
            VirtualQuery((void*)(uintptr_t)reserved_va, &suffix_after, sizeof(suffix_after));
            CHECK((prefix_after.Protect & 0xff) == PAGE_READWRITE &&
                      (suffix_after.Protect & 0xff) == PAGE_READWRITE,
                  "the refused mixed span leaves both halves' host protection alone");
            // Positive control for the arm itself: the untracked prefix ALONE is still accepted and
            // still moves, so the refusal above came from the tracked suffix and not from a blanket
            // refusal of the aperture or of this address.
            CHECK(protect(image_va, kGranule, 0x1, 0, 0, 0) == 0,
                  "the untracked module prefix alone is still accepted");
            VirtualQuery((void*)(uintptr_t)image_va, &prefix_after, sizeof(prefix_after));
            CHECK((prefix_after.Protect & 0xff) == PAGE_READONLY,
                  "and the prefix's host protection actually changed");
        }
        if (suffix_reserved) unmap(reserved_va, kGranule, 0, 0, 0, 0);
        if (image) VirtualFree(image, 0, MEM_RELEASE);
    }
    // #2144 F2: [BOOT_STUB, BOOT_STUB_END) is labelled "STUB" rather than "mapped/host", so the
    // merged aperture test ("the label is not mapped/host") admitted it -- but install_stubs
    // reserves and commits that window with a plain VirtualAlloc, so it is untracked exactly like a
    // module image while holding PROSPER's OWN emitted trampolines. A guest could therefore
    // reprotect every HLE entry point, which the pre-#2117 code refused. guest_va_in_module_code()
    // already draws the line. Mutation killed: reverting to the strcmp("mapped/host") test makes
    // both refusals below turn into accepts that strip execute from the stub table.
    {
        constexpr uint64_t kGranule = 0x10000;
        // Mixed span: ONE committed host region whose low half is the real guest plugin pool and
        // whose high half is the stub aperture, so the region walk sees a single gap spanning both.
        const uint64_t straddle = prosper::BOOT_STUB - kGranule;
        void* mixed = VirtualAlloc((void*)(uintptr_t)straddle, kGranule * 2,
                                   MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        CHECK(mixed != nullptr, "commit one region straddling the plugin/import-stub boundary");
        if (mixed) {
            CHECK((uint32_t)protect(straddle, kGranule * 2, 0x1, 0, 0, 0) == 0x80020016u,
                  "mprotect refuses a span running out of a guest module into the stub aperture");
            MEMORY_BASIC_INFORMATION after{};
            VirtualQuery((void*)(uintptr_t)prosper::BOOT_STUB, &after, sizeof(after));
            CHECK((after.Protect & 0xff) == PAGE_READWRITE,
                  "the refused straddling span leaves the stub half's protection alone");
            // Positive control: the guest-module half of the same allocation is still accepted, so
            // the refusal is about the stub aperture and not about this allocation.
            CHECK(protect(straddle, kGranule, 0x1, 0, 0, 0) == 0,
                  "the guest-module half of the same allocation is still accepted");
            VirtualQuery((void*)(uintptr_t)straddle, &after, sizeof(after));
            CHECK((after.Protect & 0xff) == PAGE_READONLY,
                  "and the guest-module half's host protection actually changed");
            VirtualFree(mixed, 0, MEM_RELEASE);
        }
        // And wholly inside the aperture, committed exactly as install_stubs commits the table.
        void* stub_table = VirtualAlloc((void*)(uintptr_t)(prosper::BOOT_STUB + 0x100000ull),
                                        kGranule, MEM_RESERVE | MEM_COMMIT,
                                        PAGE_EXECUTE_READWRITE);
        CHECK(stub_table != nullptr, "commit an import-stub table page as install_stubs does");
        if (stub_table) {
            CHECK((uint32_t)protect((uint64_t)(uintptr_t)stub_table, kGranule, 0x1, 0, 0, 0) ==
                      0x80020016u,
                  "mprotect refuses prosper's own import-stub aperture");
            MEMORY_BASIC_INFORMATION after{};
            VirtualQuery(stub_table, &after, sizeof(after));
            CHECK((after.Protect & 0xff) == PAGE_EXECUTE_READWRITE,
                  "the stub table keeps execute permission, so every HLE entry point still runs");
            VirtualFree(stub_table, 0, MEM_RELEASE);
        }
    }
    CHECK(map((uint64_t)(uintptr_t)&sparse_va2, sparse_len, 0x1, 0,
              sparse_phys, sparse_align) == 0 && sparse_va2 && sparse_va2 != sparse_va1,
          "map a second read-only sparse view of the same physical range");
    if (sparse_va1 && sparse_va2) {
        *(volatile uint64_t*)(uintptr_t)(sparse_va1 + 0x5120) = 0x5A17C0DE6310CAFEull;
        CHECK(*(volatile uint64_t*)(uintptr_t)(sparse_va2 + 0x5120) ==
                  0x5A17C0DE6310CAFEull,
              "sparse views retain physical alias coherence");

        constexpr uint64_t late_commit_offset = 0x01800000;
        MEMORY_BASIC_INFORMATION late_rw_before{}, late_ro_before{};
        VirtualQuery((void*)(uintptr_t)(sparse_va1 + late_commit_offset),
                     &late_rw_before, sizeof(late_rw_before));
        VirtualQuery((void*)(uintptr_t)(sparse_va2 + late_commit_offset),
                     &late_ro_before, sizeof(late_ro_before));
        CHECK(late_rw_before.State == MEM_COMMIT && late_ro_before.State == MEM_COMMIT,
              "inverse-order alias test starts with fault-free mapped pages");
        *(volatile uint64_t*)(uintptr_t)(sparse_va1 + late_commit_offset + 0x120) =
            0x1a7ec0de6310cafeull;
        MEMORY_BASIC_INFORMATION late_ro_after{};
        VirtualQuery((void*)(uintptr_t)(sparse_va2 + late_commit_offset),
                     &late_ro_after, sizeof(late_ro_after));
        CHECK(late_ro_after.State == MEM_COMMIT &&
                  (late_ro_after.Protect & 0xff) == PAGE_READONLY,
              "write through one alias preserves the other alias's read-only protection");
        CHECK(*(volatile uint64_t*)(uintptr_t)(sparse_va2 + late_commit_offset + 0x120) ==
                  0x1a7ec0de6310cafeull,
              "late-committed read-only alias remains physically coherent");
    }
    if (sparse_va1) CHECK(unmap(sparse_va1, sparse_len, 0, 0, 0, 0) == 0,
                          "unmap first sparse direct-memory view");
    if (sparse_va2) CHECK(unmap(sparse_va2, sparse_len, 0, 0, 0, 0) == 0,
                          "unmap second sparse direct-memory view");
    if (sparse_phys) release(sparse_phys, sparse_len, 0, 0, 0, 0);

    // MapViewOfFileEx requires the virtual base and physical offset to have the same 64 KiB
    // remainder. Placeholder replacement is page-granular, so an exact 16 KiB fixed mapping with
    // incompatible remainders must still be a real shared section view. Partially unmapping its
    // middle page must preserve both neighboring aliases and permit an exact remap of the hole.
    constexpr uint64_t fixed_len = 0x10000;
    constexpr uint64_t fixed_base = 0x30000104000ull; // 16 KiB into a free 64 KiB host granule
    constexpr uint64_t readonly_base = 0x30000304000ull;
    uint64_t fixed_phys = 0, fixed_va = fixed_base, fixed_alias = 0;
    uint64_t readonly_va = readonly_base;
    bool fixed_mapped = false, hole_remapped = false;
    CHECK(alloc(0, kEnd, fixed_len, 0x4000, 0,
                (uint64_t)(uintptr_t)&fixed_phys) == 0 && fixed_phys,
          "allocate physical range for incompatible fixed mapping");
    CHECK(reserve((uint64_t)(uintptr_t)&fixed_va, fixed_len,
                  0x10 /* SCE_KERNEL_MAP_FIXED */, 0x4000, 0, 0) == 0 &&
              fixed_va == fixed_base,
          "reserve the exact range used by a fixed direct mapping");
    fixed_mapped = map((uint64_t)(uintptr_t)&fixed_va, fixed_len, 0x2,
                       0x10 /* SCE_KERNEL_MAP_FIXED */, fixed_phys, 0x4000) == 0 &&
                   fixed_va == fixed_base;
    CHECK(fixed_mapped,
          "fixed direct mapping replaces its own reservation at 16 KiB granularity");
    CHECK(map((uint64_t)(uintptr_t)&fixed_alias, fixed_len, 0x2, 0,
              fixed_phys, 0x4000) == 0 && fixed_alias,
          "map ordinary alias for incompatible fixed view");
    if (fixed_mapped && fixed_alias) {
        *(volatile uint64_t*)(uintptr_t)(fixed_va + 0x0100) = 0x1111222233334444ull;
        *(volatile uint64_t*)(uintptr_t)(fixed_va + 0x4100) = 0x5555666677778888ull;
        *(volatile uint64_t*)(uintptr_t)(fixed_va + 0xc100) = 0x9999aaaabbbbccccull;
        CHECK(*(volatile uint64_t*)(uintptr_t)(fixed_alias + 0x0100) ==
                  0x1111222233334444ull &&
              *(volatile uint64_t*)(uintptr_t)(fixed_alias + 0x4100) ==
                  0x5555666677778888ull &&
              *(volatile uint64_t*)(uintptr_t)(fixed_alias + 0xc100) ==
                  0x9999aaaabbbbccccull,
              "incompatible fixed view preserves physical alias coherence");

        CHECK(reserve((uint64_t)(uintptr_t)&readonly_va, fixed_len,
                      0x10 /* SCE_KERNEL_MAP_FIXED */, 0x4000, 0, 0) == 0 &&
                  readonly_va == readonly_base,
              "reserve an exact range for a read-only physical alias");
        const bool readonly_mapped =
            map((uint64_t)(uintptr_t)&readonly_va, fixed_len, 0x1,
                0x10 /* SCE_KERNEL_MAP_FIXED */, fixed_phys, 0x4000) == 0 &&
            readonly_va == readonly_base;
        CHECK(readonly_mapped,
              "map a fixed read-only alias after the physical pages are committed");
        if (readonly_mapped) {
            MEMORY_BASIC_INFORMATION readonly_info{};
            VirtualQuery((void*)(uintptr_t)readonly_va, &readonly_info,
                         sizeof(readonly_info));
            CHECK(readonly_info.State == MEM_COMMIT &&
                      (readonly_info.Protect & 0xff) == PAGE_READONLY,
                  "new sparse alias applies its requested protection to existing pages");
            CHECK(*(volatile uint64_t*)(uintptr_t)(readonly_va + 0x4100) ==
                      0x5555666677778888ull,
                  "read-only fixed alias sees the precommitted physical contents");
            CHECK(unmap(readonly_va, fixed_len, 0, 0, 0, 0) == 0,
                  "unmap the fixed read-only alias");
            readonly_va = 0;
        }

        CHECK((uint32_t)unmap(fixed_va + 0x4000, 0x4001, 0, 0, 0, 0) == 0x80020016u,
              "invalid partial unmap fails after restoring the original shared view");
        CHECK(*(volatile uint64_t*)(uintptr_t)(fixed_va + 0x0100) ==
                  0x1111222233334444ull &&
              *(volatile uint64_t*)(uintptr_t)(fixed_va + 0x4100) ==
                  0x5555666677778888ull &&
              *(volatile uint64_t*)(uintptr_t)(fixed_alias + 0xc100) ==
                  0x9999aaaabbbbccccull,
              "failed partial-unmap transaction preserves data and physical aliases");

        CHECK(protect(fixed_va, 0x4000, 0x1, 0, 0, 0) == 0,
              "make fixed-view prefix read-only before partial unmap");

        const uint64_t hole = fixed_va + 0x4000;
        CHECK(unmap(hole, 0x4000, 0, 0, 0, 0) == 0,
              "partial unmap removes one page from a shared section view");
        MEMORY_BASIC_INFORMATION hole_info{};
        VirtualQuery((void*)(uintptr_t)hole, &hole_info, sizeof(hole_info));
        CHECK(hole_info.State == MEM_RESERVE &&
                  !prosper_try_commit_dmem(hole, 0x4000, 0),
              "partially unmapped page is an inaccessible, untracked placeholder");
        MEMORY_BASIC_INFORMATION prefix_info{};
        VirtualQuery((void*)(uintptr_t)fixed_va, &prefix_info, sizeof(prefix_info));
        CHECK(prefix_info.State == MEM_COMMIT &&
                  (prefix_info.Protect & 0xff) == PAGE_READONLY,
              "partial unmap preserves the retained prefix protection");
        CHECK(*(volatile uint64_t*)(uintptr_t)(fixed_va + 0x0100) ==
                  0x1111222233334444ull &&
              *(volatile uint64_t*)(uintptr_t)(fixed_va + 0xc100) ==
                  0x9999aaaabbbbccccull,
              "partial unmap preserves the prefix and suffix views");
        uint8_t suffix_query[0x48]{};
        CHECK(query(fixed_va + 0xc000, 0, (uint64_t)(uintptr_t)suffix_query,
                    sizeof(suffix_query), 0, 0) == 0 &&
                  *(uint64_t*)(suffix_query + 0x00) == fixed_va + 0x8000 &&
                  *(uint64_t*)(suffix_query + 0x08) == fixed_va + fixed_len &&
                  *(uint64_t*)(suffix_query + 0x10) == fixed_phys + 0x8000 &&
                  *(int32_t*)(suffix_query + 0x1c) == 0 &&
                  suffix_query[0x20] == 0x12,
              "partial direct unmap rebases the retained suffix's physical offset");

        uint64_t flexible_hole = hole;
        const bool flexible_mapped =
            flexible((uint64_t)(uintptr_t)&flexible_hole, 0x4000, 0x2, 0,
                     (uint64_t)(uintptr_t)"placeholder-hole", 0) == 0 &&
            flexible_hole == hole;
        CHECK(flexible_mapped,
              "flexible memory replaces a 16 KiB hole left by partial direct unmap");
        if (flexible_mapped) {
            *(volatile uint64_t*)(uintptr_t)(flexible_hole + 0x100) =
                0xf1e81b1ecafebeefull;
            CHECK(*(volatile uint64_t*)(uintptr_t)(fixed_alias + 0x4100) ==
                      0x5555666677778888ull,
                  "private reuse of the hole does not overwrite its physical page");
            CHECK(unmap(flexible_hole, 0x4000, 0, 0, 0, 0) == 0,
                  "flexible hole returns to an exact replaceable placeholder");
        }

        uint64_t remapped_hole = hole;
        hole_remapped = map((uint64_t)(uintptr_t)&remapped_hole, 0x4000, 0x2,
                            0x10 /* SCE_KERNEL_MAP_FIXED */,
                            fixed_phys + 0x4000, 0x4000) == 0 &&
                        remapped_hole == hole;
        CHECK(hole_remapped,
              "fixed direct mapping replaces the exact 16 KiB placeholder hole");
        if (hole_remapped) {
            CHECK(*(volatile uint64_t*)(uintptr_t)(remapped_hole + 0x100) ==
                      0x5555666677778888ull,
                  "partial unmap/remap retains the physical page contents");
            *(volatile uint64_t*)(uintptr_t)(remapped_hole + 0x100) =
                0xd00df00dcafef00dull;
            CHECK(*(volatile uint64_t*)(uintptr_t)(fixed_alias + 0x4100) ==
                      0xd00df00dcafef00dull,
                  "remapped placeholder hole remains coherent with its physical alias");
        }
    }
    if (fixed_mapped)
        CHECK(unmap(fixed_va, fixed_len, 0, 0, 0, 0) == 0,
              "split fixed mapping unmaps cleanly as one guest range");
    if (fixed_alias) CHECK(unmap(fixed_alias, fixed_len, 0, 0, 0, 0) == 0,
                           "unmap incompatible-view alias");
    if (readonly_va) CHECK(unmap(readonly_va, fixed_len, 0, 0, 0, 0) == 0,
                           "clean up reserved read-only alias range");
    if (fixed_phys) release(fixed_phys, fixed_len, 0, 0, 0, 0);

    if (fails) { printf("== FAIL: %d check(s) ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
#elif !defined(__linux__)
    // The direct-memory HLE (hle_kernel_mem.cpp) is #ifdef __linux__ — its functions aren't
    // registered on this platform, so there is nothing to exercise here. Skip cleanly.
    printf("  [skip] direct-memory HLE is Linux-only on this build\n== PASS ==\n");
    return 0;
#else
    register_builtin_hle();

    auto avail   = Hle::lookup(nid_hash("sceKernelAvailableDirectMemorySize"));
    auto flexible = Hle::lookup(nid_hash("sceKernelMapFlexibleMemory"));
    auto alloc   = Hle::lookup(nid_hash("sceKernelAllocateDirectMemory"));
    auto alloc_main = Hle::lookup(nid_hash("sceKernelAllocateMainDirectMemory"));
    auto map     = Hle::lookup(nid_hash("sceKernelMapDirectMemory"));
    auto map2    = reinterpret_cast<Hle7Fn>(
        Hle::lookup(nid_hash("sceKernelMapDirectMemory2")));
    auto query   = Hle::lookup(nid_hash("sceKernelVirtualQuery"));
    auto unmap   = Hle::lookup(nid_hash("sceKernelMunmap"));
    auto mtypeprotect = Hle::lookup(nid_hash("sceKernelMtypeprotect"));
    auto batch   = Hle::lookup(nid_hash("sceKernelBatchMap"));
    auto release = Hle::lookup(nid_hash("sceKernelReleaseDirectMemory"));
    auto get_type = Hle::lookup(nid_hash("sceKernelGetDirectMemoryType"));
    CHECK(nid_hash("sceKernelMapDirectMemory2") == "BQQniolj9tQ",
          "sceKernelMapDirectMemory2 hashes to the PS5 3.20 import NID");
    CHECK(nid_hash("sceKernelGetDirectMemoryType") == "BC+OG5m9+bw",
          "sceKernelGetDirectMemoryType hashes to the PS5 3.20 import NID");
    CHECK(avail && flexible && alloc && alloc_main && map && map2 && query && unmap && mtypeprotect && batch &&
              release && get_type,
          "dmem fns registered");
    if (!(avail && flexible && alloc && alloc_main && map && map2 && query && unmap && mtypeprotect && batch &&
          release && get_type)) {
        printf("== FAIL ==\n"); return 1;
    }

    // Persistent texture reuse must be able to trust unchanged flexible-memory assets too.  Linux
    // previously registered only direct-memory mappings with the page-fault write watcher, forcing
    // every flexible texture to take the exact byte-comparison fallback on every submit.
    CHECK(install_texture_watch_handler(),
          "install red-zone-safe texture write-watch handler");
    uint64_t flexible_va = 0;
    CHECK(flexible((uint64_t)(uintptr_t)&flexible_va, 0x10000, 0x2 /* CPU_RW */,
                   0, 0, 0) == 0 && flexible_va,
          "map flexible memory for texture write-watch coverage");
    if (flexible_va) {
        auto watch = host::GuestWriteWatch::create(flexible_va + 0x100, 0x2000);
        CHECK(static_cast<bool>(watch),
              "flexible guest mapping is registered with the texture write-watch");
        CHECK(watch.query() == host::GuestWriteWatchQuery::Unchanged,
              "fresh flexible texture watch is unchanged");
        *(volatile uint8_t*)(uintptr_t)(flexible_va + 0x180) = 0x5a;
        CHECK(watch.query() == host::GuestWriteWatchQuery::Dirty,
              "CPU write dirties a watched flexible texture");
        watch.reset();
        CHECK(unmap(flexible_va, 0x10000, 0, 0, 0, 0) == 0,
              "unmap watched flexible memory cleanly");
    }

    // sceKernelDirectMemoryQuery enumeration contract (issue #1129). GTA V (PPSA04263) walks
    // allocated direct memory at boot as query(offset, flags=1) / offset = info.end, and its ONLY
    // loop exit is return code 0x8002000d (EACCES). Returning any other error here (the old
    // 0x8002000e) reads as success with a zeroed info block, so the guest re-queries offset 0
    // forever. The pool is empty at this point in the test, so find-next must terminate at once.
    auto dm_query = Hle::lookup(nid_hash("sceKernelDirectMemoryQuery"));
    CHECK(nid_hash("sceKernelDirectMemoryQuery") == "BHouLQzh0X0",
          "sceKernelDirectMemoryQuery hashes to the PS5 3.20 import NID");
    CHECK(dm_query != nullptr, "DirectMemoryQuery registered");
    uint8_t dmq_info[0x18];
    memset(dmq_info, 0xa5, sizeof(dmq_info));
    CHECK((uint32_t)dm_query(0, 1, (uint64_t)(uintptr_t)dmq_info, 0x18, 0, 0) == 0x8002000du,
          "DirectMemoryQuery(0, find-next) on an empty pool -> EACCES terminator");
    CHECK((uint32_t)dm_query(0, 0, (uint64_t)(uintptr_t)dmq_info, 0x18, 0, 0) == 0x8002000du,
          "DirectMemoryQuery(0, exact) on an empty pool -> EACCES");

    CHECK((uint32_t)unmap(0, 0x4000, 0, 0, 0, 0) == 0x80020016u,
          "Munmap rejects a null address");
    CHECK((uint32_t)unmap(0x4000, 0, 0, 0, 0, 0) == 0x80020016u,
          "Munmap rejects a zero-length range");
    CHECK((uint32_t)unmap(UINT64_MAX & ~0x3fffull, 0x8000, 0, 0, 0, 0) ==
              0x80020016u,
          "Munmap rejects an overflowing guest range");

    // Both allocation APIs validate the complete 16 KiB-granular request before taking from the
    // pool, so a rejected call cannot write its output or leak capacity.
    uint64_t invalid_phys = 0xfeedfacecafebeefull;
    CHECK((uint32_t)alloc(0, kEnd, 0x4000, 0x4000, 0, 0) == 0x80020016u,
          "AllocateDirectMemory(null physAddrOut) -> EINVAL");
    CHECK((uint32_t)alloc_main(0x4000, 0x4000, 0, 0, 0, 0) == 0x80020016u,
          "AllocateMainDirectMemory(null physAddrOut) -> EINVAL");
    CHECK((uint32_t)alloc(0, kEnd, 0, 0x4000, 0,
                          (uint64_t)(uintptr_t)&invalid_phys) == 0x80020016u &&
              invalid_phys == 0xfeedfacecafebeefull,
          "AllocateDirectMemory(zero length) -> EINVAL without writing output");
    CHECK((uint32_t)alloc(0, kEnd, 0x4001, 0x4000, 0,
                          (uint64_t)(uintptr_t)&invalid_phys) == 0x80020016u &&
              invalid_phys == 0xfeedfacecafebeefull,
          "AllocateDirectMemory(non-page length) -> EINVAL without writing output");
    CHECK((uint32_t)alloc(0, kEnd, 0x4000, 0x2000, 0,
                          (uint64_t)(uintptr_t)&invalid_phys) == 0x80020016u &&
              invalid_phys == 0xfeedfacecafebeefull,
          "AllocateDirectMemory(non-page alignment) -> EINVAL without writing output");
    CHECK((uint32_t)alloc(0, kEnd, UINT64_MAX & ~0x3fffull, 0x4000, 0,
                          (uint64_t)(uintptr_t)&invalid_phys) == 0x8002000cu &&
              invalid_phys == 0xfeedfacecafebeefull,
          "AllocateDirectMemory(overflowing size) fails without writing output");
    CHECK((uint32_t)alloc(0, kEnd, 0x4000, 0x4000, 0x80000000ull,
                          (uint64_t)(uintptr_t)&invalid_phys) == 0x80020016u &&
              invalid_phys == 0xfeedfacecafebeefull,
          "AllocateDirectMemory(non-int memory type) -> EINVAL without writing output");
    CHECK((uint32_t)alloc(0, kEnd, 0x4000, 0x4000, 0, 1) == 0x80020016u,
          "AllocateDirectMemory(low invalid physAddrOut) -> EINVAL");
    CHECK((uint32_t)alloc_main(0x4001, 0x4000, 0,
                               (uint64_t)(uintptr_t)&invalid_phys, 0, 0) == 0x80020016u &&
              invalid_phys == 0xfeedfacecafebeefull,
          "AllocateMainDirectMemory validates the shared direct-memory request contract");
    CHECK((uint32_t)alloc_main(UINT64_MAX & ~0x3fffull, 0x4000, 0,
                               (uint64_t)(uintptr_t)&invalid_phys, 0, 0) == 0x8002000cu &&
              invalid_phys == 0xfeedfacecafebeefull,
          "AllocateMainDirectMemory(overflowing size) fails without writing output");
    uint64_t multiple_align_phys = 0;
    CHECK(alloc(0, kEnd, 0x4000, 0xc000, 0,
                (uint64_t)(uintptr_t)&multiple_align_phys) == 0 &&
              multiple_align_phys % 0xc000 == 0,
          "AllocateDirectMemory accepts a 16-KiB-multiple non-power-of-two alignment");
    if (multiple_align_phys)
        CHECK(release(multiple_align_phys, 0x4000, 0, 0, 0, 0) == 0,
              "release the non-power-of-two-aligned allocation");

    // Fresh pool: the largest free block is the whole pool, and BOTH out-params are written.
    uint64_t phys = 0xdead, size = 0xdead;
    uint64_t r = avail(0, kEnd, 0x4000, (uint64_t)(uintptr_t)&phys, (uint64_t)(uintptr_t)&size, 0);
    CHECK(r == 0, "available(fresh) returns success");
    CHECK(phys == kBase, "available(fresh) phys == pool base");
    CHECK(size == kTotal, "available(fresh) size == whole pool");

    // Null out-pointers -> EINVAL, and it must NOT be the old success-with-garbage.
    CHECK((uint32_t)avail(0, kEnd, 0x4000, 0, (uint64_t)(uintptr_t)&size, 0) == 0x80020016u,
          "available(null physAddrOut) -> EINVAL");
    CHECK((uint32_t)avail(0, kEnd, 0x4000, (uint64_t)(uintptr_t)&phys, 0, 0) == 0x80020016u,
          "available(null sizeOut) -> EINVAL");

    // Allocate 1 MiB (args: searchStart, searchEnd, len, align, type, physOut) at the pool base.
    uint64_t ap = 0;
    uint64_t ar = alloc(0, kEnd, 0x100000, 0x4000, 12, (uint64_t)(uintptr_t)&ap);
    CHECK(ar == 0 && ap == kBase,
          "PS5 memory type 12 succeeds after invalid allocations leave the pool untouched");
    // With exactly one region allocated ([kBase, kBase+0x100000) type 12), the GTA V-style walk
    // must visit it and then terminate: find-next from 0 returns the region, find-next from its
    // end returns the EACCES terminator (issue #1129).
    memset(dmq_info, 0, sizeof(dmq_info));
    CHECK(dm_query(0, 1, (uint64_t)(uintptr_t)dmq_info, 0x18, 0, 0) == 0 &&
              *(uint64_t*)(dmq_info + 0x00) == ap &&
              *(uint64_t*)(dmq_info + 0x08) == ap + 0x100000 &&
              *(int32_t*)(dmq_info + 0x10) == 12,
          "DirectMemoryQuery(0, find-next) returns the sole region with exact bounds and type");
    CHECK((uint32_t)dm_query(*(uint64_t*)(dmq_info + 0x08), 1,
                             (uint64_t)(uintptr_t)dmq_info, 0x18, 0, 0) == 0x8002000du,
          "DirectMemoryQuery(end, find-next) past the last region -> EACCES terminator");

    int32_t direct_type = -1;
    uint64_t direct_start = UINT64_MAX, direct_end = UINT64_MAX;
    CHECK(get_type(ap + 0x80000, (uint64_t)(uintptr_t)&direct_type,
                   (uint64_t)(uintptr_t)&direct_start,
                   (uint64_t)(uintptr_t)&direct_end, 0, 0) == 0 &&
              direct_type == 12 && direct_start == ap && direct_end == ap + 0x100000,
          "GetDirectMemoryType returns the containing physical allocation and exact type");
    CHECK((uint32_t)get_type(ap, (uint64_t)(uintptr_t)&direct_type, 0,
                             (uint64_t)(uintptr_t)&direct_end, 0, 0) == 0x80020016u,
          "GetDirectMemoryType requires all output pointers");
    direct_type = -1; direct_start = UINT64_MAX; direct_end = UINT64_MAX;
    CHECK((uint32_t)get_type(ap + 0x100000, (uint64_t)(uintptr_t)&direct_type,
                             (uint64_t)(uintptr_t)&direct_start,
                             (uint64_t)(uintptr_t)&direct_end, 0, 0) == 0x80020002u &&
              direct_type == -1 && direct_start == UINT64_MAX && direct_end == UINT64_MAX,
          "GetDirectMemoryType returns ENOENT without changing outputs for a free offset");

    // Now the largest free block starts just past the allocation and is the pool minus 1 MiB.
    phys = size = 0;
    avail(0, kEnd, 0x4000, (uint64_t)(uintptr_t)&phys, (uint64_t)(uintptr_t)&size, 0);
    CHECK(phys == kBase + 0x100000, "available after alloc -> phys past the allocation");
    CHECK(size == kTotal - 0x100000, "available after alloc -> size shrank by the allocation");

    // A search window entirely within free space clamps the reported block to the window.
    uint64_t wlo = 0x20000000ull, wsz = 0x40000ull;
    phys = size = 0;
    r = avail(wlo, wlo + wsz, 0x4000, (uint64_t)(uintptr_t)&phys, (uint64_t)(uintptr_t)&size, 0);
    CHECK(r == 0 && phys == wlo && size == wsz, "available(window in free space) clamps to the window");

    // A search window entirely inside the allocated region has nothing free -> ENOMEM.
    r = avail(kBase, kBase + 0x80000, 0x4000, (uint64_t)(uintptr_t)&phys, (uint64_t)(uintptr_t)&size, 0);
    CHECK((uint32_t)r == 0x8002000Cu, "available(window fully allocated) -> ENOMEM");

    // Alignment is honored: a 1 MiB-aligned search over free space returns a 1 MiB-aligned phys.
    phys = 0;
    avail(0x20000123ull, kEnd, 0x100000, (uint64_t)(uintptr_t)&phys, (uint64_t)(uintptr_t)&size, 0);
    CHECK((phys & 0xfffff) == 0 && phys >= 0x20000123ull, "available honors the requested alignment");

    // Release the allocation -> the whole pool is available again.
    release(kBase, 0x100000, 0, 0, 0, 0);
    direct_type = -1; direct_start = UINT64_MAX; direct_end = UINT64_MAX;
    CHECK((uint32_t)get_type(kBase, (uint64_t)(uintptr_t)&direct_type,
                             (uint64_t)(uintptr_t)&direct_start,
                             (uint64_t)(uintptr_t)&direct_end, 0, 0) == 0x80020002u &&
              direct_type == -1 && direct_start == UINT64_MAX && direct_end == UINT64_MAX,
          "GetDirectMemoryType stops reporting a released allocation");
    phys = size = 0;
    avail(0, kEnd, 0x4000, (uint64_t)(uintptr_t)&phys, (uint64_t)(uintptr_t)&size, 0);
    CHECK(phys == kBase && size == kTotal, "available after release -> whole pool free again");

    // --- Allocate HONORS the search window (issue #108): a request constrained to a window well
    //     above the pool base must land INSIDE that window, not at the base as before. ---
    {
        uint64_t wlo = 0x40000000ull;                 // 1 GiB into the pool
        uint64_t p = 0;
        uint64_t rr = alloc(wlo, wlo + 0x200000, 0x100000, 0x4000, 0, (uint64_t)(uintptr_t)&p);
        CHECK(rr == 0 && p >= wlo && p + 0x100000 <= wlo + 0x200000,
              "allocate honors [searchStart,searchEnd) -> phys inside the window");
        release(p, 0x100000, 0, 0, 0, 0);
    }
    // A window too small for the request -> ENOMEM (not a base-of-pool fallback).
    {
        uint64_t wlo = 0x50000000ull; uint64_t p = 0xdead;
        uint64_t rr = alloc(wlo, wlo + 0x1000 /*window < 0x100000 request*/, 0x100000, 0x4000, 0,
                            (uint64_t)(uintptr_t)&p);
        CHECK((uint32_t)rr == 0x8002000Cu, "allocate with too-small window -> ENOMEM (no out-of-window fallback)");
    }

    // A zero-hint direct mapping must honor its explicit VA alignment. Linux mmap only promises
    // 4 KiB, which made HashLink reject a mapping requested at 64 KiB alignment during startup.
    {
        uint64_t p = 0, va = 0;
        uint64_t rr = alloc(0, kEnd, 0x10000, 0x10000, 0, (uint64_t)(uintptr_t)&p);
        CHECK(rr == 0, "allocate 64KiB-aligned direct page succeeds");
        rr = map((uint64_t)(uintptr_t)&va, 0x10000, 0x2 /*RW*/, 0, p, 0x10000);
        CHECK(rr == 0 && va && (va & 0xffff) == 0,
              "map direct memory honors requested 64KiB VA alignment");
        if (va) CHECK(unmap(va, 0x10000, 0, 0, 0, 0) == 0,
                      "unmap aligned direct view through the HLE path");
        if (p) release(p, 0x10000, 0, 0, 0, 0);
    }

    // MapDirectMemory2 inserts `type` before prot/flags/phys/alignment. The explicit type must
    // reach both VirtualQuery and the physical allocation table. Later type-protect operations
    // carve metadata without losing the virtual-to-physical offset of suffix pages.
    {
        uint64_t p = 0, va = 0, alias = 0;
        uint64_t rr = alloc(0, kEnd, 0x10000, 0x10000, 6,
                            (uint64_t)(uintptr_t)&p);
        CHECK(rr == 0, "allocate direct page for MapDirectMemory2");
        rr = map2((uint64_t)(uintptr_t)&va, 0x10000, 3 /* type */, 0x2 /* RW */, 0,
                  p, 0x10000);
        CHECK(rr == 0 && va && (va & 0xffff) == 0,
              "MapDirectMemory2 consumes shifted arguments and seventh-argument alignment");
        rr = map((uint64_t)(uintptr_t)&alias, 0x10000, 0x2, 0, p, 0x10000);
        CHECK(rr == 0 && alias, "map ordinary alias for MapDirectMemory2 physical range");
        CHECK((uint32_t)unmap(va + 1, 0x4000, 0, 0, 0, 0) == 0x80020016u,
              "Munmap reports an unaligned host-unmap failure");
        CHECK((uint32_t)unmap(va, 1, 0, 0, 0, 0) == 0x80020016u,
              "Munmap rejects a non-page-multiple length before host rounding");
        uint8_t failed_unmap_info[0x48]{};
        CHECK(query(va + 0x2000, 0, (uint64_t)(uintptr_t)failed_unmap_info,
                    sizeof(failed_unmap_info), 0, 0) == 0 &&
                  *(uint64_t*)(failed_unmap_info + 0x00) == va &&
                  *(uint64_t*)(failed_unmap_info + 0x08) == va + 0x10000,
              "failed Munmap preserves VA tracking");
        if (va && alias) {
            *(volatile uint64_t*)(uintptr_t)(va + 0x120) = 0xB002D1EC7A11A5ull;
            CHECK(*(volatile uint64_t*)(uintptr_t)(alias + 0x120) == 0xB002D1EC7A11A5ull,
                   "MapDirectMemory2 maps the requested physical offset as writable shared memory");
        }

        uint8_t info[0x48]{};
        CHECK(query(va + 0x2000, 0, (uint64_t)(uintptr_t)info, sizeof(info), 0, 0) == 0 &&
                  *(uint64_t*)(info + 0x00) == va &&
                  *(uint64_t*)(info + 0x08) == va + 0x10000 &&
                  *(uint64_t*)(info + 0x10) == p &&
                  *(int32_t*)(info + 0x1c) == 3 &&
                  info[0x20] == 0x12,
              "VirtualQuery reports direct offset, explicit type, and classification");
        uint8_t short_query_info[0x22];
        memset(short_query_info, 0xa5, sizeof(short_query_info));
        CHECK(query(va + 0x2000, 0, (uint64_t)(uintptr_t)short_query_info,
                    0x21, 0, 0) == 0 &&
                  short_query_info[0x20] == 0x12 && short_query_info[0x21] == 0xa5,
              "VirtualQuery writes the one-byte classification at exact infoSize 0x21");
        uint8_t exact_query_info[0x42];
        memset(exact_query_info, 0xa5, sizeof(exact_query_info));
        CHECK(query(va + 0x2000, 0, (uint64_t)(uintptr_t)exact_query_info,
                    0x41, 0, 0) == 0 &&
                  exact_query_info[0x20] == 0x12 &&
                  memcmp(exact_query_info + 0x21, "direct", 7) == 0 &&
                  exact_query_info[0x41] == 0xa5,
              "VirtualQuery writes name[32] at 0x21 within the exact ABI size");
        direct_type = -1; direct_start = direct_end = 0;
        CHECK(get_type(p + 0x2000, (uint64_t)(uintptr_t)&direct_type,
                       (uint64_t)(uintptr_t)&direct_start,
                       (uint64_t)(uintptr_t)&direct_end, 0, 0) == 0 &&
                  direct_type == 3 && direct_start == p && direct_end == p + 0x10000,
              "MapDirectMemory2 updates the physical allocation type");

        CHECK(mtypeprotect(va + 0x4000, 0x4000, 9, 0x1, 0, 0) == 0,
              "Mtypeprotect changes one direct-mapping page");
        memset(info, 0, sizeof(info));
        CHECK(query(va + 0x5000, 0, (uint64_t)(uintptr_t)info, sizeof(info), 0, 0) == 0 &&
                  *(uint64_t*)(info + 0x00) == va + 0x4000 &&
                  *(uint64_t*)(info + 0x08) == va + 0x8000 &&
                  *(uint64_t*)(info + 0x10) == p + 0x4000 &&
                  *(int32_t*)(info + 0x18) == 0x1 &&
                  *(int32_t*)(info + 0x1c) == 9 &&
                  info[0x20] == 0x12,
              "Mtypeprotect publishes the carved page's rebased offset and type");
        CHECK(mtypeprotect(va + 0x4000, 0, 14, 0x2, 0, 0) == 0,
              "aligned zero-length Mtypeprotect is a successful no-op");
        memset(info, 0, sizeof(info));
        CHECK(query(va + 0x5000, 0, (uint64_t)(uintptr_t)info, sizeof(info), 0, 0) == 0 &&
                  *(int32_t*)(info + 0x18) == 0x1 &&
                  *(int32_t*)(info + 0x1c) == 9,
              "zero-length Mtypeprotect leaves protection and type metadata unchanged");

        alignas(8) uint8_t type_entry[0x20]{};
        *(uint64_t*)(type_entry + 0x00) = va + 0x8000;
        type_entry[0x18] = 0x2;
        type_entry[0x19] = 11;
        *(int32_t*)(type_entry + 0x1c) = 4; // TYPE_PROTECT
        int32_t type_done = -1;
        CHECK((uint32_t)batch((uint64_t)(uintptr_t)type_entry, 1,
                              (uint64_t)(uintptr_t)&type_done, 0, 0, 0) == 0x80020016u &&
                  type_done == 0,
              "BatchMap rejects a zero-length type-protect entry without counting it");
        *(uint64_t*)(type_entry + 0x00) = va + 0x8123;
        *(uint64_t*)(type_entry + 0x10) = 0x100;
        type_done = -1;
        CHECK(batch((uint64_t)(uintptr_t)type_entry, 1,
                    (uint64_t)(uintptr_t)&type_done, 0, 0, 0) == 0 && type_done == 1,
              "BatchMap TYPE_PROTECT accepts an unaligned sub-page range");
        memset(info, 0, sizeof(info));
        CHECK(query(va + 0x9000, 0, (uint64_t)(uintptr_t)info, sizeof(info), 0, 0) == 0 &&
                  *(uint64_t*)(info + 0x00) == va + 0x8000 &&
                  *(uint64_t*)(info + 0x08) == va + 0xc000 &&
                  *(uint64_t*)(info + 0x10) == p + 0x8000 &&
                  *(int32_t*)(info + 0x18) == 0x2 &&
                  *(int32_t*)(info + 0x1c) == 11,
              "BatchMap TYPE_PROTECT normalizes protection and metadata to the guest page");
        memset(info, 0, sizeof(info));
        CHECK(query(va + 0xd000, 0, (uint64_t)(uintptr_t)info, sizeof(info), 0, 0) == 0 &&
                  *(uint64_t*)(info + 0x10) == p + 0xc000 &&
                  *(int32_t*)(info + 0x1c) == 3,
              "tracker suffix keeps its original type and rebased physical offset");
        direct_type = -1; direct_start = direct_end = 0;
        CHECK(get_type(p + 0x9000, (uint64_t)(uintptr_t)&direct_type,
                       (uint64_t)(uintptr_t)&direct_start,
                       (uint64_t)(uintptr_t)&direct_end, 0, 0) == 0 &&
                  direct_type == 11 && direct_start == p + 0x8000 &&
                  direct_end == p + 0xc000,
              "TYPE_PROTECT carves the same type range in physical allocation queries");

        CHECK(mtypeprotect(va + 0xc123, 0x100, 13, 0x1, 0, 0) == 0,
              "Mtypeprotect accepts an unaligned sub-page range");
        memset(info, 0, sizeof(info));
        CHECK(query(va + 0xd000, 0, (uint64_t)(uintptr_t)info, sizeof(info), 0, 0) == 0 &&
                  *(uint64_t*)(info + 0x00) == va + 0xc000 &&
                  *(uint64_t*)(info + 0x08) == va + 0x10000 &&
                  *(uint64_t*)(info + 0x10) == p + 0xc000 &&
                  *(int32_t*)(info + 0x18) == 0x1 &&
                  *(int32_t*)(info + 0x1c) == 13 &&
                  info[0x20] == 0x12,
              "unaligned Mtypeprotect normalizes VA metadata to the 16 KiB guest page");
        direct_type = -1; direct_start = direct_end = 0;
        CHECK(get_type(p + 0xd000, (uint64_t)(uintptr_t)&direct_type,
                       (uint64_t)(uintptr_t)&direct_start,
                       (uint64_t)(uintptr_t)&direct_end, 0, 0) == 0 &&
                  direct_type == 13 && direct_start == p + 0xc000 &&
                  direct_end == p + 0x10000,
              "unaligned Mtypeprotect normalizes physical type metadata to the same page");
        CHECK((uint32_t)mtypeprotect(UINT64_MAX - 0x1000, 0x2000, 13, 0x1, 0, 0) ==
                  0x80020016u,
              "Mtypeprotect rejects an overflowing guest range");
        if (va) CHECK(unmap(va, 0x10000, 0, 0, 0, 0) == 0,
                      "unmap MapDirectMemory2 view through the HLE path");
        if (alias) CHECK(unmap(alias, 0x10000, 0, 0, 0, 0) == 0,
                         "unmap ordinary alias through the HLE path");
        if (p) release(p, 0x10000, 0, 0, 0, 0);
    }

    // A non-fixed direct-memory address is a search hint. Astro Bot supplies the eboot base as
    // its preferred address; treating every non-zero input as fixed returned ENOMEM instead of
    // relocating the mapping. A genuinely fixed collision must still fail without clobbering it.
    {
        constexpr uint64_t len = 0x10000;
        uint64_t p = 0;
        void* live = mmap(nullptr, len, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        CHECK(live != MAP_FAILED, "create a live mapping for direct-memory hint collision");
        if (live != MAP_FAILED) {
            *(volatile uint32_t*)live = 0xA57B0782u;
            uint64_t hinted = (uint64_t)(uintptr_t)live;
            uint64_t rr = alloc(0, kEnd, len, len, 0, (uint64_t)(uintptr_t)&p);
            CHECK(rr == 0 && p, "allocate direct memory for occupied-hint test");
            if (rr == 0 && p) {
                rr = map((uint64_t)(uintptr_t)&hinted, len, 0x2 /*RW*/, 0,
                         p, len);
                CHECK(rr == 0 && hinted && hinted != (uint64_t)(uintptr_t)live,
                      "non-fixed occupied direct-memory hint relocates");
                CHECK(hinted >= 0x2000000000ull && hinted < 0x40000000000ull,
                      "relocated direct mapping stays in the guest user-VA range");
                CHECK(*(volatile uint32_t*)live == 0xA57B0782u,
                      "relocated direct mapping does not clobber the live hint");
                if (hinted && hinted != (uint64_t)(uintptr_t)live)
                    munmap((void*)(uintptr_t)hinted, len);

                uint64_t fixed_hint = (uint64_t)(uintptr_t)live;
                rr = map((uint64_t)(uintptr_t)&fixed_hint, len, 0x2 /*RW*/,
                         0x10 /*SCE_KERNEL_MAP_FIXED*/, p, len);
                CHECK(rr != 0, "fixed occupied direct-memory hint fails");
                CHECK(*(volatile uint32_t*)live == 0xA57B0782u,
                      "failed fixed direct mapping preserves the live range");
                release(p, len, 0, 0, 0, 0);
            }
            munmap(live, len);
        }
    }

    if (fails) { printf("== FAIL: %d check(s) ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
#endif
}
