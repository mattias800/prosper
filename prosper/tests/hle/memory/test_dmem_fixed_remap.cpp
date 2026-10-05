// test_dmem_fixed_remap -- sceKernelMapDirectMemory with MAP_FIXED over tracked DIRECT memory replaces it,
// as mmap(MAP_FIXED) does on the kernel the PS5 derives from; over anything else it is refused and that
// memory is left untouched.
//
// THE DEFECT. Assassin's Creed Black Flag Resynced's allocator maps a large direct view, then maps a
// smaller piece with different physical backing FIXED over its head, and aborts with "Out of memory"
// when the map answers ENOMEM. A Win32 view cannot be placed over a live view, and the POSIX arm refused
// any fixed range holding committed memory before it reached mmap.
//
// THE CONTRACT. Only ranges the registries prove are direct memory are replaced; the part of an old
// view outside the request keeps its contents. Plain flexible memory and anything untracked stay refused
// exactly as before and untouched (the first revision released the range through a fallback that blindly
// decommitted it and then failed anyway). The guest asks to keep what is mapped with
// SCE_KERNEL_MAP_NO_OVERWRITE; its value is unverified and it stays unhonoured (#3819).
//
// WHAT EACH TEST KILLS:
//   WholeViewRemapWithNewBacking         a view re-mapped at its exact extent is refused or keeps the old backing
//   TitleShapedCarveHeadOfLiveView       the title's smaller fixed map over a live view's head is refused
//   CarveKeepsLivePrefixAndSuffix        a partial replace damages the untouched prefix or suffix
//   FixedMapOverFlexibleMemoryIsRefused  flexible memory is replaced, or wiped by a failing retry
//   RangeSpanningDirectAndFlexibleIsRefused  a range accepted after its FIRST direct cover clobbers the flexible page
//   RangeSpanningTwoDirectViewsIsReplaced    a range spanning two old views replaces only the first
//   AmprMapOverLiveDirectMemoryIsRefused     a placement that never asked for MAP_FIXED (the Ampr push-map)
//                                            replaces a live direct view and zeroes it (#88 / #107 class)
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <gtest/gtest.h>

#include <cstdint>

using namespace prosper;

namespace {

constexpr uint64_t kPage = 0x10000;
constexpr uint64_t kFixed = 0x10;
constexpr uint64_t kEnomem = 0x8002000Cull;

class DmemFixedRemap : public ::testing::Test {
protected:
    void SetUp() override {
        register_builtin_hle();
        allocate_ = Hle::lookup(nid_hash("sceKernelAllocateDirectMemory"));
        map_ = Hle::lookup(nid_hash("sceKernelMapDirectMemory"));
        release_ = Hle::lookup(nid_hash("sceKernelReleaseDirectMemory"));
        flex_ = Hle::lookup(nid_hash("sceKernelMapNamedFlexibleMemory"));
        reserve_ = Hle::lookup(nid_hash("sceKernelReserveVirtualRange"));
        ASSERT_NE(reserve_, nullptr);
        ASSERT_NE(allocate_, nullptr);
        ASSERT_NE(map_, nullptr);
        ASSERT_NE(release_, nullptr);
        ASSERT_NE(flex_, nullptr);
    }

    uint64_t alloc(uint64_t pages) {
        uint64_t phys = 0;
        EXPECT_EQ(allocate_(0, 0x200000000ull, pages * kPage, kPage, 0, (uint64_t)(uintptr_t)&phys), 0u);
        EXPECT_NE(phys, 0u);
        return phys;
    }
    uint64_t map_anywhere(uint64_t pages, uint64_t phys) {
        uint64_t addr = 0;
        EXPECT_EQ(map_((uint64_t)(uintptr_t)&addr, pages * kPage, 3, 0, phys, kPage), 0u);
        EXPECT_NE(addr, 0u);
        return addr;
    }
    uint64_t reserve(uint64_t pages) {
        uint64_t addr = 0;
        EXPECT_EQ(reserve_((uint64_t)(uintptr_t)&addr, pages * kPage, 0, kPage, 0, 0), 0u);
        EXPECT_NE(addr, 0u);
        return addr;
    }
    uint64_t map_fixed(uint64_t addr, uint64_t pages, uint64_t phys) {
        uint64_t at_addr = addr;
        return map_((uint64_t)(uintptr_t)&at_addr, pages * kPage, 3, kFixed, phys, kPage);
    }
    static volatile uint8_t* at(uint64_t addr, uint64_t page = 0) {
        return reinterpret_cast<volatile uint8_t*>(addr + page * kPage);
    }

    HleFn allocate_ = nullptr, map_ = nullptr, release_ = nullptr, flex_ = nullptr, reserve_ = nullptr;
};

}  // namespace

TEST_F(DmemFixedRemap, WholeViewRemapWithNewBacking) {
    const uint64_t a = alloc(1);
    const uint64_t addr = map_anywhere(1, a);
    *at(addr) = 0x11;
    const uint64_t b = alloc(1);
    const uint64_t b_addr = map_anywhere(1, b);
    *at(b_addr) = 0xBB;

    uint64_t again = addr;
    ASSERT_EQ(map_((uint64_t)(uintptr_t)&again, kPage, 3, kFixed, b, kPage), 0u)
        << "re-mapping a view's exact extent with new backing must replace it, not fail with ENOMEM";
    ASSERT_EQ(again, addr);
    EXPECT_EQ(*at(addr), 0xBB) << "the view must now alias the new physical page, not keep the old one";
    *at(addr) = 0x22;
    EXPECT_EQ(*at(b_addr), 0x22) << "the replacement is a real alias of the new backing";
}

TEST_F(DmemFixedRemap, TitleShapedCarveHeadOfLiveView) {
    // A large live view, then a smaller piece with DIFFERENT backing mapped fixed over its head.
    const uint64_t a = alloc(8);
    const uint64_t addr = map_anywhere(8, a);
    for (uint64_t i = 0; i < 8; ++i) *at(addr, i) = static_cast<uint8_t>(0xA0 + i);
    const uint64_t b = alloc(2);
    const uint64_t b_addr = map_anywhere(2, b);
    *at(b_addr) = 0xBB;
    *at(b_addr, 1) = 0xBC;

    uint64_t again = addr;
    ASSERT_EQ(map_((uint64_t)(uintptr_t)&again, 2 * kPage, 3, kFixed, b, kPage), 0u)
        << "a fixed map over the head of a live direct view must replace it, not fail with ENOMEM";
    ASSERT_EQ(again, addr);
    EXPECT_EQ(*at(addr, 0), 0xBB) << "the head aliases the new backing";
    EXPECT_EQ(*at(addr, 1), 0xBC);
    for (uint64_t i = 2; i < 8; ++i)
        EXPECT_EQ(*at(addr, i), static_cast<uint8_t>(0xA0 + i)) << "the untouched tail keeps page " << i;
}

TEST_F(DmemFixedRemap, CarveKeepsLivePrefixAndSuffix) {
    const uint64_t a = alloc(4);
    const uint64_t addr = map_anywhere(4, a);
    for (uint64_t i = 0; i < 4; ++i) *at(addr, i) = static_cast<uint8_t>(0xA0 + i);
    const uint64_t b = alloc(1);
    const uint64_t b_addr = map_anywhere(1, b);
    *at(b_addr) = 0xBB;

    uint64_t again = addr + kPage;   // replace page 1 only
    ASSERT_EQ(map_((uint64_t)(uintptr_t)&again, kPage, 3, kFixed, b, kPage), 0u);
    EXPECT_EQ(*at(addr, 0), 0xA0) << "the live prefix must keep its contents";
    EXPECT_EQ(*at(addr, 1), 0xBB) << "the replaced page aliases the new backing";
    EXPECT_EQ(*at(addr, 2), 0xA2) << "the live suffix must keep its contents";
    EXPECT_EQ(*at(addr, 3), 0xA3);
}

TEST_F(DmemFixedRemap, FixedMapOverFlexibleMemoryIsRefused) {
    uint64_t flex = 0;
    ASSERT_EQ(flex_((uint64_t)(uintptr_t)&flex, 4 * kPage, 3, 0, (uint64_t)(uintptr_t)"flex", 0), 0u);
    ASSERT_NE(flex, 0u);
    for (uint64_t i = 0; i < 4; ++i) *at(flex, i) = static_cast<uint8_t>(0x50 + i);

    const uint64_t b = alloc(1);
    uint64_t again = flex + kPage;
    EXPECT_EQ(map_((uint64_t)(uintptr_t)&again, kPage, 3, kFixed, b, kPage), kEnomem)
        << "flexible memory is not direct memory; a fixed direct map must not replace it";
    for (uint64_t i = 0; i < 4; ++i)
        EXPECT_EQ(*at(flex, i), static_cast<uint8_t>(0x50 + i))
            << "a refused map must leave the flexible memory intact and accessible (page " << i << ")";
}

TEST_F(DmemFixedRemap, RangeSpanningDirectAndFlexibleIsRefused) {
    const uint64_t base = reserve(2);
    const uint64_t a = alloc(1);
    ASSERT_EQ(map_fixed(base, 1, a), 0u);
    uint64_t flex = base + kPage;
    ASSERT_EQ(flex_((uint64_t)(uintptr_t)&flex, kPage, 3, kFixed, (uint64_t)(uintptr_t)"flex", 0), 0u);
    *at(base) = 0xD1;
    *at(base, 1) = 0xF1;

    const uint64_t b = alloc(2);
    EXPECT_EQ(map_fixed(base, 2, b), kEnomem)
        << "a range that spans direct AND flexible memory must be refused as a whole";
    EXPECT_EQ(*at(base), 0xD1) << "the direct page keeps its contents";
    EXPECT_EQ(*at(base, 1), 0xF1) << "the flexible page must not be clobbered";
}

TEST_F(DmemFixedRemap, RangeSpanningTwoDirectViewsIsReplaced) {
    const uint64_t base = reserve(2);
    const uint64_t a = alloc(1);
    const uint64_t b = alloc(1);
    ASSERT_EQ(map_fixed(base, 1, a), 0u);
    ASSERT_EQ(map_fixed(base + kPage, 1, b), 0u);
    *at(base) = 0x11;
    *at(base, 1) = 0x22;

    const uint64_t c = alloc(2);
    const uint64_t c_addr = map_anywhere(2, c);
    *at(c_addr) = 0xC0;
    *at(c_addr, 1) = 0xC1;

    ASSERT_EQ(map_fixed(base, 2, c), 0u) << "one fixed map across two direct views must replace both";
    EXPECT_EQ(*at(base), 0xC0);
    EXPECT_EQ(*at(base, 1), 0xC1) << "the second view must be replaced too";
}

TEST_F(DmemFixedRemap, AmprMapOverLiveDirectMemoryIsRefused) {
    // sceAmprCommandBufferSetBuffer's map flavor places a physical range at a guest VA without
    // MAP_FIXED and zeroes it after a successful map. Over a live direct view it must keep refusing
    // (#88 / #107; docs/games/KHAZAN_STATUS.md), whatever a guest MAP_FIXED direct map may replace.
    HleFn set_buffer = Hle::lookup("N-FSPA4S3nI");   // sceAmprCommandBufferSetBuffer
    ASSERT_NE(set_buffer, nullptr);
    const uint64_t a = alloc(1);
    const uint64_t addr = map_anywhere(1, a);
    *(volatile uint64_t*)(uintptr_t)addr = 0x5eedfacecafe01ull;
    set_buffer(0x7f0000ab0000ull, addr, kPage, addr + 8, 0xffffffffull, 0);
    EXPECT_EQ(*(volatile uint64_t*)(uintptr_t)addr, 0x5eedfacecafe01ull)
        << "the Ampr map flavor must not replace (and zero) a live direct view";
}
