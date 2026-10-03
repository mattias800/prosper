// A refused fixed map must leave its physical allocation available for a later retry.
// If failure partially publishes the view or releases the allocation, guest memory can be lost.
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

namespace prosper {
namespace {

constexpr uint64_t kGuestPage = 0x4000;
constexpr uint64_t kLength = 0x10000;
constexpr uint64_t kSearchEnd = 16ull * 1024 * 1024 * 1024;
constexpr uint64_t kNoMemory = 0x8002000c;

TEST(DirectMemoryMapFailure, FixedCollisionPreservesAllocationForRetry) {
    register_builtin_hle();
    const HleFn allocate = Hle::lookup(nid_hash("sceKernelAllocateDirectMemory"));
    const HleFn map = Hle::lookup(nid_hash("sceKernelMapDirectMemory"));
    const HleFn unmap = Hle::lookup(nid_hash("sceKernelMunmap"));
    const HleFn release = Hle::lookup(nid_hash("sceKernelReleaseDirectMemory"));
    const HleFn get_type = Hle::lookup(nid_hash("sceKernelGetDirectMemoryType"));
    ASSERT_TRUE(allocate && map && unmap && release && get_type);

    uint64_t physical = 0;
    ASSERT_EQ(allocate(0, kSearchEnd, kLength, kLength, 0,
                       reinterpret_cast<uint64_t>(&physical)), 0u);
    ASSERT_NE(physical, 0u);

    uint64_t alias = 0;
    ASSERT_EQ(map(reinterpret_cast<uint64_t>(&alias), kLength, 3, 0, physical, kLength), 0u);
    ASSERT_NE(alias, 0u);

    uintptr_t target = 0;
#ifdef _WIN32
    void* collision = VirtualAlloc(nullptr, kLength, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    ASSERT_NE(collision, nullptr);
    target = reinterpret_cast<uintptr_t>(collision);  // VirtualAlloc is 64 KiB aligned.
#else
    const long host_page = sysconf(_SC_PAGESIZE);
    ASSERT_GT(host_page, 0);
    const uint64_t reserve_length = kLength + 2 * kGuestPage;
    void* reservation = mmap(nullptr, reserve_length, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(reservation, MAP_FAILED);
    const uintptr_t raw = reinterpret_cast<uintptr_t>(reservation);
    target = (raw + kGuestPage - 1) & ~(kGuestPage - 1);
    const size_t prefix = target - raw;
    const size_t suffix = static_cast<size_t>(raw + reserve_length - (target + kLength));
    ASSERT_EQ(prefix % static_cast<size_t>(host_page), 0u);
    ASSERT_EQ(suffix % static_cast<size_t>(host_page), 0u);
    if (prefix) ASSERT_EQ(munmap(reinterpret_cast<void*>(raw), prefix), 0);
    if (suffix) ASSERT_EQ(munmap(reinterpret_cast<void*>(target + kLength), suffix), 0);
#endif

    std::memset(reinterpret_cast<void*>(target), 0x5a, kLength);
    uint64_t mapped_address = target;
    EXPECT_EQ(map(reinterpret_cast<uint64_t>(&mapped_address), kLength, 3, 0x10,
                  physical, kLength), kNoMemory)
        << "fixed mapping over a live host range must be refused";
    EXPECT_EQ(mapped_address, target)
        << "a refused map must not publish a different or partial address";
    EXPECT_EQ(*reinterpret_cast<volatile uint8_t*>(target), 0x5a)
        << "the refused map must preserve the occupied range";
    EXPECT_EQ(*reinterpret_cast<volatile uint8_t*>(target + kLength - 1), 0x5a)
        << "the refusal must preserve the whole mapping, including its tail";

    int32_t type = -1;
    uint64_t allocation_start = UINT64_MAX;
    uint64_t allocation_end = UINT64_MAX;
    EXPECT_EQ(get_type(physical + kGuestPage,
                       reinterpret_cast<uint64_t>(&type),
                       reinterpret_cast<uint64_t>(&allocation_start),
                       reinterpret_cast<uint64_t>(&allocation_end), 0, 0), 0u);
    EXPECT_EQ(type, 0);
    EXPECT_EQ(allocation_start, physical);
    EXPECT_EQ(allocation_end, physical + kLength);

#ifdef _WIN32
    ASSERT_NE(VirtualFree(reinterpret_cast<void*>(target), 0, MEM_RELEASE), 0);
#else
    ASSERT_EQ(munmap(reinterpret_cast<void*>(target), kLength), 0);
#endif
    mapped_address = target;
    ASSERT_EQ(map(reinterpret_cast<uint64_t>(&mapped_address), kLength, 3, 0x10,
                  physical, kLength), 0u)
        << "the same physical allocation must map successfully after the collision is removed";
    EXPECT_EQ(mapped_address, target);

    *reinterpret_cast<volatile uint8_t*>(mapped_address) = 0xc3;
    EXPECT_EQ(*reinterpret_cast<volatile uint8_t*>(alias), 0xc3)
        << "the retry must alias the original allocation rather than a replacement";

    EXPECT_EQ(unmap(mapped_address, kLength, 0, 0, 0, 0), 0u);
    EXPECT_EQ(unmap(alias, kLength, 0, 0, 0, 0), 0u);
    EXPECT_EQ(release(physical, kLength, 0, 0, 0, 0), 0u);
}

}  // namespace
}  // namespace prosper
