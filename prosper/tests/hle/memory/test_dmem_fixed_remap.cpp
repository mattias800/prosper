// test_dmem_fixed_remap -- sceKernelMapDirectMemory with MAP_FIXED over an address that is already
// mapped REPLACES the old mapping, as mmap(MAP_FIXED) does on the kernel the PS5 derives from.
//
// THE DEFECT (Windows). A Win32 view cannot be mapped over committed pages, so the fixed request
// failed with ERROR_INVALID_ADDRESS and the guest saw ENOMEM. Assassin's Creed Black Flag Resynced's
// allocator maps a large piece of direct memory, releases most of it and maps a smaller piece fixed
// at the same address, then aborts with "Out of memory". The POSIX arm always got the replacement
// from mmap itself.
//
// WHAT EACH ASSERTION KILLS:
//   FixedRemapOverCommittedMappingSucceeds  the Windows arm stops unmapping before the retry
//   ReplacementIsAtRequestedAddress          the replacement maps at a different address
//   ReplacementIsUsable                      the replacement is not readable and writable
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <gtest/gtest.h>

#include <cstdint>

using namespace prosper;

namespace {

constexpr uint64_t kPage = 0x10000;
constexpr uint64_t kFixed = 0x10;

class DmemFixedRemap : public ::testing::Test {
protected:
    void SetUp() override {
        register_builtin_hle();
        allocate_ = Hle::lookup(nid_hash("sceKernelAllocateDirectMemory"));
        map_ = Hle::lookup(nid_hash("sceKernelMapDirectMemory"));
        release_ = Hle::lookup(nid_hash("sceKernelReleaseDirectMemory"));
        ASSERT_NE(allocate_, nullptr);
        ASSERT_NE(map_, nullptr);
        ASSERT_NE(release_, nullptr);
    }
    HleFn allocate_ = nullptr, map_ = nullptr, release_ = nullptr;
};

}  // namespace

TEST_F(DmemFixedRemap, FixedRemapOverCommittedMappingSucceeds) {
    uint64_t phys = 0;
    ASSERT_EQ(allocate_(0, 0x200000000ull, 4 * kPage, kPage, 0, (uint64_t)(uintptr_t)&phys), 0u);
    ASSERT_NE(phys, 0u);
    uint64_t addr = 0;
    ASSERT_EQ(map_((uint64_t)(uintptr_t)&addr, 4 * kPage, 3, 0, phys, kPage), 0u);
    ASSERT_NE(addr, 0u);
    *(volatile uint8_t*)addr = 0x11;

    // Release the tail the way the title does, then remap the head FIXED at the same address.
    release_(phys + kPage, 3 * kPage, 0, 0, 0, 0);
    uint64_t again = addr;
    const uint64_t rc = map_((uint64_t)(uintptr_t)&again, kPage, 3, kFixed, phys, kPage);
    EXPECT_EQ(rc, 0u) << "a fixed map over an existing mapping must replace it, not fail with ENOMEM";
    EXPECT_EQ(again, addr) << "the replacement must be at the requested address";
    if (rc == 0 && again == addr) {
        volatile uint8_t* p = (volatile uint8_t*)addr;
        p[0] = 0x22;
        EXPECT_EQ(p[0], 0x22) << "the replacement must be readable and writable";
    }
}
