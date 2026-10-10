// #2873: a released AMM range must become an inaccessible, reusable reservation, and its
// physical pages must return to AMM rather than alias another live range or leak permanently.
// Evidence: the guest's Unmap(cb, va, size) at eboot+0xd9f7ac and the refused re-Maps in #2873;
// the destructor wrapper at +0xcc4bd0 passes only cb and chains to the base destructor.
// Invalid/unowned-span errors are Prosper's conservative EINVAL contract (CONFIDENCE: MED).
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "host/memory/guest_memory_map.hpp"

#include <gtest/gtest.h>

#include <cstdint>

using namespace prosper;
extern "C" int prosper_reserved_range_state(uint64_t addr);

namespace {
constexpr uint64_t kPage = 0x4000;
constexpr uint64_t kEinval = 0x80020016;
constexpr uint64_t kEnomem = 0x8002000c;
constexpr uint64_t kCb = 0x7f0000030000;

class AmprAmmUnmap : public ::testing::Test {
protected:
    void SetUp() override {
        register_builtin_hle();
        map_ = Hle::lookup("JEVYGhDc97M");
        unmap_ = Hle::lookup("M-VFI2DJWQA");
        destroy_ = Hle::lookup("pvUFDOHilnE");
        ASSERT_NE(map_, nullptr);
        ASSERT_NE(unmap_, nullptr);
        ASSERT_NE(destroy_, nullptr);
        const auto ranges = Hle::lookup("wkQR9+xTFKY");
        const auto give = Hle::lookup("Q07J7XpvhrU");
        ASSERT_NE(ranges, nullptr);
        ASSERT_NE(give, nullptr);
        uint64_t window[4]{};
        ASSERT_EQ(ranges((uint64_t)&window[0], (uint64_t)&window[1], (uint64_t)&window[2],
                         (uint64_t)&window[3], 0, 0),
                  0u);
        // CTest runs each case in a new process. Also support running all cases together:
        // retain one small pool and use fresh VA slices, returning live AMM pages in TearDown.
        static bool have_pool = false;
        if (!have_pool) {
            uint64_t physical = 0;
            ASSERT_EQ(give(0, 16ull << 30, 4 * kPage, kPage, 1, (uint64_t)&physical), 0u);
            have_pool = true;
        }
        static uint64_t next_slice = 0;
        base_ = window[0] + next_slice++ * 32 * kPage;
        window_end_ = window[1];
        const auto construct = Hle::lookup("EDq5bqCqYpA");
        ASSERT_NE(construct, nullptr);
        ASSERT_EQ(construct(kCb, 0, 0, 0, 0, 0), 0u);
    }

    void TearDown() override {
        if (!base_ || !unmap_) return;
        for (uint64_t i = 0; i < 16; ++i)
            if (prosper_reserved_range_state(base_ + i * kPage) == 2)
                EXPECT_EQ(unmap_(kCb, base_ + i * kPage, kPage, 0, 0, 0), 0u);
        if (destroy_) EXPECT_EQ(destroy_(kCb, 0, 0, 0, 0, 0), 0u);
    }

    uint64_t map(uint64_t page, uint64_t pages) {
        return map_(kCb, base_ + page * kPage, pages * kPage, 0xb, 0xc3, 0);
    }
    uint64_t unmap(uint64_t page, uint64_t pages) {
        return unmap_(kCb, base_ + page * kPage, pages * kPage, 0, 0, 0);
    }
    uint8_t& byte(uint64_t page) { return *reinterpret_cast<uint8_t*>(base_ + page * kPage); }
    HleFn map_ = nullptr, unmap_ = nullptr, destroy_ = nullptr;
    uint64_t base_ = 0, window_end_ = 0;
};
}   // namespace

TEST_F(AmprAmmUnmap, MapUnmapRemapReusesAnExhaustedPool) {
    ASSERT_EQ(map(0, 4), 0u);
    byte(0) = 0x5a;
    byte(3) = 0xa5;
    ASSERT_EQ(map(8, 1), kEnomem) << "the four-page pool really is exhausted";
    const auto generation = host::guest_mapping_generation();
    ASSERT_EQ(unmap(0, 4), 0u);
    EXPECT_GT(host::guest_mapping_generation(), generation);
    EXPECT_EQ(prosper_reserved_range_state(base_), 4) << "no committed or lazy-commit backing";
    EXPECT_EQ(prosper_reserved_range_state(base_ + 3 * kPage), 4);
    host::GuestReadableRange readable;
    EXPECT_FALSE(host::guest_readable_mapping_containing(base_, base_ + 4 * kPage, readable));
    EXPECT_DEATH(
        { (void)*reinterpret_cast<volatile uint8_t*>(base_); }, "")
        << "Unmap must remove CPU access, not merely retag a still-readable mapping";
    ASSERT_EQ(map(0, 4), 0u) << "the SAME VA and the entire physical pool are reusable";
    byte(0) = 0x33;
    byte(3) = 0xcc;
    EXPECT_EQ(byte(0), 0x33);
    EXPECT_EQ(byte(3), 0xcc);
}

TEST_F(AmprAmmUnmap, ReusesNonTailPagesWithoutAliasingLiveNeighbors) {
    ASSERT_EQ(map(0, 1), 0u);
    ASSERT_EQ(map(1, 3), 0u);
    byte(1) = 0x11;
    byte(3) = 0x33;
    ASSERT_EQ(unmap(0, 1), 0u);
    ASSERT_EQ(map(8, 1), 0u) << "returning a non-tail carve must recover capacity";
    byte(8) = 0xaa;
    EXPECT_EQ(byte(1), 0x11);
    EXPECT_EQ(byte(3), 0x33);
}

TEST_F(AmprAmmUnmap, CoalescesFreeRangesOnBothSides) {
    for (uint64_t i = 0; i < 4; ++i) ASSERT_EQ(map(i, 1), 0u);
    byte(3) = 0x77;
    ASSERT_EQ(unmap(0, 1), 0u);
    ASSERT_EQ(unmap(2, 1), 0u);
    ASSERT_EQ(unmap(1, 1), 0u);
    ASSERT_EQ(map(8, 3), 0u) << "three neighboring physical carves must coalesce";
    byte(8) = 0x11;
    byte(10) = 0x22;
    EXPECT_EQ(byte(3), 0x77);
}

TEST_F(AmprAmmUnmap, PartialUnmapSurvivesProtectionRetagsAndKeepsNeighborOffsets) {
    ASSERT_EQ(map(0, 4), 0u);
    byte(0) = 0x11;
    byte(3) = 0x44;
    const auto protect = Hle::lookup(nid_hash("sceKernelMprotect"));
    ASSERT_NE(protect, nullptr);
    ASSERT_EQ(protect(base_ + kPage, 2 * kPage, 3, 0, 0, 0), 0u);
    ASSERT_EQ(unmap(1, 2), 0u) << "a protection retag must preserve AMM ownership";
    EXPECT_EQ(prosper_reserved_range_state(base_ + kPage), 4);
    ASSERT_EQ(map(1, 2), 0u);
    byte(1) = 0x22;
    byte(2) = 0x33;
    EXPECT_EQ(byte(0), 0x11) << "released physical offsets must account for the partial VA";
    EXPECT_EQ(byte(3), 0x44);
}

TEST_F(AmprAmmUnmap, UnmapCanSpanAdjacentMappings) {
    ASSERT_EQ(map(0, 2), 0u);
    ASSERT_EQ(map(2, 2), 0u);
    ASSERT_EQ(unmap(0, 4), 0u);
    ASSERT_EQ(map(0, 4), 0u) << "all physical slices of the requested span must be returned";
}

TEST_F(AmprAmmUnmap, BadArgumentsLeaveLiveMappingsAndPoolUntouched) {
    ASSERT_EQ(map(0, 4), 0u);
    byte(0) = 0x5a;
    EXPECT_EQ(unmap_(0, base_, kPage, 0, 0, 0), kEinval);
    EXPECT_EQ(unmap_(0xffff, base_, kPage, 0, 0, 0), kEinval);
    EXPECT_EQ(unmap_(kCb, 0, kPage, 0, 0, 0), kEinval);
    EXPECT_EQ(unmap_(kCb, base_, 0, 0, 0, 0), kEinval);
    EXPECT_EQ(unmap_(kCb, base_ + 1, kPage, 0, 0, 0), kEinval);
    EXPECT_EQ(unmap_(kCb, base_, kPage + 1, 0, 0, 0), kEinval);
    EXPECT_EQ(unmap_(kCb, window_end_ - kPage, 2 * kPage, 0, 0, 0), kEinval);
    EXPECT_EQ(unmap_(kCb, UINT64_MAX - kPage + 1, 2 * kPage, 0, 0, 0), kEinval);
    EXPECT_EQ(map_(0, base_ + 8 * kPage, kPage, 0xb, 0xc3, 0), kEinval);
    EXPECT_EQ(map(8, 1), kEnomem) << "no rejected request returned live physical pages";
    EXPECT_EQ(prosper_reserved_range_state(base_), 2);
    EXPECT_EQ(byte(0), 0x5a);
}

TEST_F(AmprAmmUnmap, RefusesAHoleBeforeMutatingTheMappedPrefix) {
    ASSERT_EQ(map(0, 1), 0u);
    ASSERT_EQ(map(2, 1), 0u);
    byte(0) = 0x11;
    byte(2) = 0x33;
    ASSERT_EQ(unmap(0, 3), kEinval);
    EXPECT_EQ(prosper_reserved_range_state(base_), 2);
    EXPECT_EQ(byte(0), 0x11);
    EXPECT_EQ(byte(2), 0x33);
    ASSERT_EQ(map(8, 2), 0u);
    byte(8) = 0xaa;
    byte(9) = 0xbb;
    EXPECT_EQ(byte(0), 0x11);
    EXPECT_EQ(byte(2), 0x33);
}

TEST_F(AmprAmmUnmap, RefusesForeignDirectMemoryInsideTheWindow) {
    const auto allocate = Hle::lookup(nid_hash("sceKernelAllocateDirectMemory"));
    const auto kernel_map = Hle::lookup(nid_hash("sceKernelMapDirectMemory"));
    const auto kernel_unmap = Hle::lookup(nid_hash("sceKernelMunmap"));
    const auto release = Hle::lookup(nid_hash("sceKernelReleaseDirectMemory"));
    ASSERT_TRUE(allocate && kernel_map && kernel_unmap && release);
    uint64_t physical = 0, target = base_;
    ASSERT_EQ(allocate(0, 16ull << 30, kPage, kPage, 1, (uint64_t)&physical), 0u);
    ASSERT_EQ(kernel_map((uint64_t)&target, kPage, 3, 0x10, physical, kPage), 0u);
    ASSERT_EQ(target, base_);
    byte(0) = 0x5a;
    EXPECT_EQ(unmap(0, 1), kEinval) << "the window alone is insufficient proof of ownership";
    EXPECT_EQ(byte(0), 0x5a);
    EXPECT_EQ(kernel_unmap(base_, kPage, 0, 0, 0, 0), 0u);
    EXPECT_EQ(release(physical, kPage, 0, 0, 0, 0), 0u);
}

TEST_F(AmprAmmUnmap, DestructorRetiresOnlyCommandBufferBookkeeping) {
    const auto construct = Hle::lookup("8aI7R7WaOlc");
    const auto attach = Hle::lookup("N-FSPA4S3nI");
    const auto get_base = Hle::lookup("RPCAhx-aabE");
    ASSERT_TRUE(construct && attach && get_base);
    // A legacy constructor shape does not set tracks_offset: teardown must still erase it.
    ASSERT_EQ(construct(kCb, 0x80, 1, 0, 0, 0), 0u);
    uint64_t storage[16]{};
    const uint64_t buffer = (uint64_t)storage;
    ASSERT_EQ(attach(kCb, buffer, sizeof(storage), buffer, 0, 3), 0u);
    ASSERT_EQ(get_base(kCb, 0, 0, 0, 0, 0), buffer);
    ASSERT_EQ(map(0, 4), 0u);
    byte(0) = 0x5a;
    EXPECT_EQ(destroy_(0, 0, 0, 0, 0, 0), kEinval);
    EXPECT_EQ(get_base(kCb, 0, 0, 0, 0, 0), buffer);
    ASSERT_EQ(destroy_(kCb, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(get_base(kCb, 0, 0, 0, 0, 0), 0u) << "dead buffer must not retain attached storage";
    storage[0] = 0x1234;
    EXPECT_EQ(storage[0], 0x1234) << "SetBuffer storage remains caller-owned";
    EXPECT_EQ(prosper_reserved_range_state(base_), 2);
    EXPECT_EQ(byte(0), 0x5a) << "completed mappings survive destruction of their recorder";
    EXPECT_EQ(map(8, 1), kEnomem) << "destructor must not return live backing to the pool";
    ASSERT_EQ(unmap_(kCb + 0x10000, base_, 4 * kPage, 0, 0, 0), 0u);
    ASSERT_EQ(map(0, 4), 0u) << "another command buffer can release the global heap's mapping";
}

TEST_F(AmprAmmUnmap, ARefusedMapReturnsItsPhysicalClaim) {
    ASSERT_EQ(map(0, 1), 0u);
    byte(0) = 0x5a;
    ASSERT_EQ(map(0, 3), kEnomem) << "a map must not overwrite a committed range";
    ASSERT_EQ(map(8, 3), 0u) << "the failed three-page claim must be returned";
    byte(8) = 0xaa;
    EXPECT_EQ(byte(0), 0x5a);
}
