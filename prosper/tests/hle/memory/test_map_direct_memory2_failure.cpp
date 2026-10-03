// A refused MapDirectMemory2 must not publish a mapping or apply its requested memory type. A
// partially applied type change makes later physical-memory queries disagree with the real view.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

using namespace prosper;

namespace {
using Hle7Fn = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t,
                            uint64_t, uint64_t, uint64_t);
constexpr uint64_t kSpan = 0x10000;
constexpr uint64_t kPoolEnd = 0x10000ull + 16ull * 1024 * 1024 * 1024;
constexpr uint32_t kFirstCanary = 0xA11DCA7Eu;
constexpr uint32_t kLastCanary = 0x51A7E123u;

void* occupy_aligned_span() {
#ifdef _WIN32
    return VirtualAlloc(nullptr, kSpan, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    const long host_page = sysconf(_SC_PAGESIZE);
    if (host_page <= 0) return MAP_FAILED;
    const size_t reserve_size = static_cast<size_t>(kSpan + 2 * host_page);
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
    if (span != MAP_FAILED) munmap(span, kSpan);
#endif
}

bool occupied_span_was_created(void* span) {
#ifdef _WIN32
    return span != nullptr;
#else
    return span != MAP_FAILED;
#endif
}

bool direct_memory_type_is(HleFn get_type, uint64_t phys, int32_t expected_type) {
    int32_t type = -1;
    uint64_t start = UINT64_MAX, end = UINT64_MAX;
    return get_type(phys + 0x1000, reinterpret_cast<uint64_t>(&type),
                    reinterpret_cast<uint64_t>(&start), reinterpret_cast<uint64_t>(&end), 0, 0) == 0 &&
           type == expected_type && start == phys && end == phys + kSpan;
}
} // namespace

TEST(MapDirectMemory2Failure, RefusalLeavesOutputAndAllocationTypeUnchanged) {
    register_builtin_hle();
    const auto allocate = Hle::lookup(nid_hash("sceKernelAllocateDirectMemory"));
    const auto map2 = reinterpret_cast<Hle7Fn>(Hle::lookup(nid_hash("sceKernelMapDirectMemory2")));
    const auto get_type = Hle::lookup(nid_hash("sceKernelGetDirectMemoryType"));
    const auto unmap = Hle::lookup(nid_hash("sceKernelMunmap"));
    const auto release = Hle::lookup(nid_hash("sceKernelReleaseDirectMemory"));
    ASSERT_TRUE(allocate && map2 && get_type && unmap && release)
        << "the direct-memory HLE entry points are registered";

    void* collision = occupy_aligned_span();
    ASSERT_TRUE(occupied_span_was_created(collision))
        << "reserve a live host span for the fixed-map refusal";
    const uint64_t collision_va = reinterpret_cast<uint64_t>(collision);
    *reinterpret_cast<uint32_t*>(collision_va) = kFirstCanary;
    *reinterpret_cast<uint32_t*>(collision_va + kSpan - sizeof(uint32_t)) = kLastCanary;

    uint64_t phys = 0, successful_view = 0;
    const auto cleanup = [&] {
        if (successful_view) unmap(successful_view, kSpan, 0, 0, 0, 0);
        if (phys) release(phys, kSpan, 0, 0, 0, 0);
        release_occupied_span(collision);
    };

    const bool allocated = allocate(0, kPoolEnd, kSpan, kSpan, 12,
                                    reinterpret_cast<uint64_t>(&phys)) == 0 && phys != 0;
    ASSERT_TRUE(allocated) << "allocate direct memory with a known initial type";
    ASSERT_TRUE(direct_memory_type_is(get_type, phys, 12))
        << "the physical allocation begins with the requested type";

    uint64_t failed_output = collision_va;
    const uint64_t failed_result = map2(reinterpret_cast<uint64_t>(&failed_output), kSpan,
                                        3, 3, 0x10, phys, kSpan);
    EXPECT_EQ(static_cast<uint32_t>(failed_result), 0x8002000cu)
        << "the occupied fixed target is refused with ENOMEM";
    EXPECT_EQ(failed_output, collision_va)
        << "a failed map leaves its in/out address untouched";
    EXPECT_TRUE(direct_memory_type_is(get_type, phys, 12))
        << "a refused mapping does not apply its requested type to the allocation";
    EXPECT_EQ(*reinterpret_cast<volatile uint32_t*>(collision_va), kFirstCanary)
        << "the failed fixed map preserves the live target's first canary";
    EXPECT_EQ(*reinterpret_cast<volatile uint32_t*>(collision_va + kSpan - sizeof(uint32_t)),
              kLastCanary) << "the failed fixed map preserves the live target's last canary";

    const uint64_t success_result = map2(reinterpret_cast<uint64_t>(&successful_view), kSpan,
                                         3, 3, 0, phys, kSpan);
    EXPECT_EQ(success_result, 0u) << "the positive control maps the same allocation successfully";
    EXPECT_NE(successful_view, 0u) << "the successful call publishes its view address";
    EXPECT_TRUE(direct_memory_type_is(get_type, phys, 3))
        << "a successful MapDirectMemory2 applies its requested type";
    cleanup();
}
