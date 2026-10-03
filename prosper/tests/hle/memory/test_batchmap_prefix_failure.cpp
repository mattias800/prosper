// BatchMap reports a completed prefix when a later entry fails. Callers use the count to know
// which mappings took effect; returning the wrong count can make cleanup unmap the wrong pages.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

using namespace prosper;

namespace {
constexpr uint64_t kSpan = 0x10000;
constexpr uint64_t kPoolEnd = 0x10000ull + 16ull * 1024 * 1024 * 1024;

void* occupy_aligned_span() {
#ifdef _WIN32
    return VirtualAlloc(nullptr, kSpan, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    const long host_page = sysconf(_SC_PAGESIZE);
    if (host_page <= 0) return MAP_FAILED;
    // Aligning up to kSpan can skip up to kSpan - host_page bytes, so reserve a whole extra kSpan.
    // Two host pages of slack let the span run past the reservation, and the suffix size then
    // underflowed, unmapping unrelated memory (heap included).
    const size_t reserve_size = static_cast<size_t>(2 * kSpan);
    void* raw = mmap(nullptr, reserve_size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) return MAP_FAILED;
    const uintptr_t begin = reinterpret_cast<uintptr_t>(raw);
    const uintptr_t aligned = (begin + kSpan - 1) & ~(kSpan - 1);
    const size_t prefix = aligned - begin;
    const size_t suffix = reserve_size - prefix - kSpan;
    if (prefix) munmap(raw, prefix);
    if (suffix) munmap(reinterpret_cast<void*>(aligned + kSpan), suffix);
    return reinterpret_cast<void*>(aligned);
#endif
}

void release_occupied_span(void* span) {
    if (!span) return;
#ifdef _WIN32
    VirtualFree(span, 0, MEM_RELEASE);
#else
    if (span == MAP_FAILED) return;
    munmap(span, kSpan);
#endif
}

bool occupied_span_was_created(void* span) {
#ifdef _WIN32
    return span != nullptr;
#else
    return span != MAP_FAILED;
#endif
}
} // namespace

TEST(BatchMapPrefixFailure, PreservesSuccessfulPrefixAndReportsItsCount) {
    register_builtin_hle();
    const auto allocate = Hle::lookup(nid_hash("sceKernelAllocateDirectMemory"));
    const auto reserve = Hle::lookup(nid_hash("sceKernelReserveVirtualRange"));
    const auto batch = Hle::lookup(nid_hash("sceKernelBatchMap"));
    const auto map = Hle::lookup(nid_hash("sceKernelMapDirectMemory"));
    const auto query = Hle::lookup(nid_hash("sceKernelVirtualQuery"));
    const auto unmap = Hle::lookup(nid_hash("sceKernelMunmap"));
    const auto release = Hle::lookup(nid_hash("sceKernelReleaseDirectMemory"));
    ASSERT_TRUE(allocate && reserve && batch && map && query && unmap && release)
        << "the direct-memory HLE entry points are registered";

    uint64_t phys = 0, first_va = 0;
    void* collision = occupy_aligned_span();
    ASSERT_TRUE(occupied_span_was_created(collision))
        << "reserve a live host span for the later collision";
    const uint64_t collision_va = reinterpret_cast<uint64_t>(collision);
    *reinterpret_cast<uint32_t*>(collision_va) = 0xBA7C4A11u;
    *reinterpret_cast<uint32_t*>(collision_va + kSpan - sizeof(uint32_t)) = 0xE0DCA11Du;

    const auto cleanup = [&] {
        if (first_va) unmap(first_va, kSpan, 0, 0, 0, 0);
        if (phys) release(phys, kSpan, 0, 0, 0, 0);
        release_occupied_span(collision);
    };

    const bool allocation_ok = allocate(0, kPoolEnd, kSpan, kSpan, 0,
                                         reinterpret_cast<uint64_t>(&phys)) == 0 && phys != 0;
    EXPECT_TRUE(allocation_ok) << "allocate one physical span for the successful prefix";
    const bool reservation_ok = allocation_ok &&
        reserve(reinterpret_cast<uint64_t>(&first_va), kSpan, 0, kSpan, 0, 0) == 0 && first_va;
    EXPECT_TRUE(reservation_ok) << "reserve the exact target of the first entry";
    if (!reservation_ok) {
        cleanup();
        return;
    }

    struct Entry {
        uint64_t start, phys, length;
        uint8_t protection, type;
        uint16_t padding;
        int32_t operation;
    };
    static_assert(sizeof(Entry) == 0x20);
    Entry entries[2] = {
        {first_va, phys, kSpan, 3, 0, 0, 0},
        {collision_va, phys, kSpan, 3, 0, 0, 0},
    };
    int completed = -1;
    const uint64_t result = batch(reinterpret_cast<uint64_t>(entries), 2,
                                  reinterpret_cast<uint64_t>(&completed), 0x10, 0, 0);

    EXPECT_EQ(static_cast<uint32_t>(result), 0x8002000cu)
        << "the occupied second fixed target fails with ENOMEM";
    EXPECT_EQ(completed, 1) << "BatchMap counts only the successful first entry";

    uint8_t info[0x48]{};
    const bool first_is_tracked = query(first_va, 0, reinterpret_cast<uint64_t>(info),
                                        sizeof(info), 0, 0) == 0 &&
        *reinterpret_cast<uint64_t*>(info) == first_va &&
        *reinterpret_cast<uint64_t*>(info + 8) == first_va + kSpan &&
        info[0x20] == 0x12;
    EXPECT_TRUE(first_is_tracked) << "the successful prefix remains a tracked direct mapping";

    uint64_t alias = 0;
    const bool alias_ok = first_is_tracked &&
        map(reinterpret_cast<uint64_t>(&alias), kSpan, 3, 0, phys, kSpan) == 0 && alias;
    EXPECT_TRUE(alias_ok) << "map an ordinary alias of the prefix's physical backing";
    if (alias_ok) {
        *reinterpret_cast<volatile uint32_t*>(first_va) = 0x51A7E123u;
        EXPECT_EQ(*reinterpret_cast<volatile uint32_t*>(alias), 0x51A7E123u)
            << "the successful prefix still aliases its allocated physical pages";
        unmap(alias, kSpan, 0, 0, 0, 0);
    }
    EXPECT_EQ(*reinterpret_cast<volatile uint32_t*>(collision_va), 0xBA7C4A11u)
        << "the refused collision preserves the live mapping's first canary";
    EXPECT_EQ(*reinterpret_cast<volatile uint32_t*>(collision_va + kSpan - sizeof(uint32_t)),
              0xE0DCA11Du) << "the refused collision preserves the live mapping's last canary";
    cleanup();
}
