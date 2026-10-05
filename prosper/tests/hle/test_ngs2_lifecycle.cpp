// test_ngs2_lifecycle — the NGS2 allocator-create/destroy/grain/lock remainder.
//
// These six exports were unregistered, so the dispatcher answered `0`: a system create that
// wrote no handle, a destroy that freed nothing, locks nobody holds. Every TEST drives the
// real NIDs through the same table/tag/error model as the merged create/query paths —
// handles validate, double-free fails, and destroying a system reclaims its racks.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

using namespace prosper;

static uint64_t addr(const void* p) { return (uint64_t)(uintptr_t)p; }
static constexpr uint64_t sx(uint32_t v) { return (uint64_t)(int64_t)(int32_t)v; }

static constexpr uint64_t kInvalidSystem = sx(0x804a0230u);
static constexpr uint64_t kInvalidRack = sx(0x804a0261u);
static constexpr uint64_t kInvalidOut = sx(0x804a0053u);
static constexpr uint64_t kInvalidAllocator = sx(0x804a020au);
static constexpr uint64_t kInvalidGrain = sx(0x804a0051u);

TEST(Ngs2Lifecycle, AllNidsBound) {
    register_builtin_hle();
    static const char* table[] = {
        "sceNgs2SystemCreateWithAllocator", "sceNgs2SystemDestroy",
        "sceNgs2SystemSetGrainSamples",     "sceNgs2RackCreateWithAllocator",
        "sceNgs2RackLock",                  "sceNgs2RackUnlock",
    };
    static_assert(sizeof(table) / sizeof(table[0]) == 6, "the 6 lifecycle exports");
    for (const char* name : table) {
        EXPECT_NE(Hle::lookup(nid_hash(name)), nullptr) << name << " is not registered";
    }
}

TEST(Ngs2Lifecycle, NidsResolveToFirmwareValues) {
    // All six NIDs reproduce the PS5 3.20 firmware export set verbatim.
    EXPECT_EQ(nid_hash("sceNgs2SystemCreateWithAllocator"), "mPYgU4oYpuY");
    EXPECT_EQ(nid_hash("sceNgs2RackCreateWithAllocator"), "U546k6orxQo");
    EXPECT_EQ(nid_hash("sceNgs2SystemDestroy"), "u-WrYDaJA3k");
    EXPECT_EQ(nid_hash("sceNgs2SystemSetGrainSamples"), "l4Q2dWEH6UM");
    EXPECT_EQ(nid_hash("sceNgs2RackLock"), "MzTa7VLjogY");
    EXPECT_EQ(nid_hash("sceNgs2RackUnlock"), "++YZ7P9e87U");
}

TEST(Ngs2Lifecycle, SystemCreateDestroyReclaimsRacks) {
    register_builtin_hle();
    HleFn create = Hle::lookup(nid_hash("sceNgs2SystemCreateWithAllocator"));
    HleFn destroy = Hle::lookup(nid_hash("sceNgs2SystemDestroy"));
    HleFn grain = Hle::lookup(nid_hash("sceNgs2SystemSetGrainSamples"));
    HleFn rack_create = Hle::lookup(nid_hash("sceNgs2RackCreateWithAllocator"));
    HleFn lock = Hle::lookup(nid_hash("sceNgs2RackLock"));
    HleFn unlock = Hle::lookup(nid_hash("sceNgs2RackUnlock"));
    for (HleFn f : {create, destroy, grain, rack_create, lock, unlock}) ASSERT_NE(f, nullptr);

    uint64_t sys = 0;
    // An allocator whose allocHandler (+0) is set; the callbacks are never invoked.
    uint64_t alloc[3] = {0x1000, 0x2000, 0};
    uint64_t no_handler[3] = {0, 0x2000, 0};
    EXPECT_EQ(create(0, 0, 0, 0, 0, 0), kInvalidAllocator)
        << "the allocator is checked before the handle out";
    EXPECT_EQ(create(0, 0, addr(&sys), 0, 0, 0), kInvalidAllocator) << "null allocator";
    EXPECT_EQ(create(0, addr(no_handler), addr(&sys), 0, 0, 0), kInvalidAllocator)
        << "an allocator without an allocHandler";
    EXPECT_EQ(create(0, addr(alloc), 0, 0, 0, 0), kInvalidOut) << "null handle out";
    EXPECT_EQ(sys, 0u) << "a refused create writes no handle";
    ASSERT_EQ(create(0, addr(alloc), addr(&sys), 0, 0, 0), 0u);
    EXPECT_NE(sys, 0u) << "CreateWithAllocator writes a tagged handle";
    EXPECT_EQ(grain(sys, 4096, 0, 0, 0, 0), 0u);
    EXPECT_EQ(grain(sys, 100, 0, 0, 0, 0), kInvalidGrain) << "not a multiple of 64";
    EXPECT_EQ(grain(sys, 32, 0, 0, 0, 0), kInvalidGrain) << "below the minimum";
    EXPECT_EQ(grain(sys, 16384, 0, 0, 0, 0), kInvalidGrain) << "above the maximum";
    EXPECT_EQ(grain(0xDEADu, 4096, 0, 0, 0, 0), kInvalidSystem) << "foreign system refused";

    uint64_t rack = 0;
    EXPECT_EQ(rack_create(sys, 1, 0, 0, addr(&rack), 0), kInvalidAllocator) << "null allocator";
    EXPECT_EQ(rack_create(sys, 1, 0, addr(no_handler), addr(&rack), 0), kInvalidAllocator);
    EXPECT_EQ(rack_create(sys, 1, 0, addr(alloc), 0, 0), kInvalidOut) << "null handle out";
    ASSERT_EQ(rack_create(sys, 1, 0, addr(alloc), addr(&rack), 0), 0u);
    // A second system with its own rack: destroying the first must leave it alone.
    uint64_t sys2 = 0, rack2 = 0;
    ASSERT_EQ(create(0, addr(alloc), addr(&sys2), 0, 0, 0), 0u);
    ASSERT_EQ(rack_create(sys2, 1, 0, addr(alloc), addr(&rack2), 0), 0u);
    EXPECT_NE(rack, 0u);
    EXPECT_EQ(lock(rack, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(unlock(rack, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(lock(0xDEADu, 0, 0, 0, 0, 0), kInvalidRack) << "foreign rack refused";
    EXPECT_EQ(destroy(0xDEADu, 0, 0, 0, 0, 0), kInvalidSystem);
    uint8_t released[0x40];
    memset(released, 0xAB, sizeof(released));  // nonzero sentinel proves the zero-write
    ASSERT_EQ(destroy(sys, addr(released), 0, 0, 0, 0), 0u);
    for (size_t i = 0; i < sizeof(released); ++i) {
        EXPECT_EQ(released[i], 0u) << "released-context OUT block is zeroed at byte " << i;
    }
    EXPECT_EQ(destroy(sys, 0, 0, 0, 0, 0), kInvalidSystem) << "destroy frees exactly once";
    EXPECT_EQ(lock(rack, 0, 0, 0, 0, 0), kInvalidRack)
        << "destroying the system reclaims its racks";
    EXPECT_EQ(lock(rack2, 0, 0, 0, 0, 0), 0u) << "another system's rack survives the destroy";
    EXPECT_EQ(destroy(sys2, 0, 0, 0, 0, 0), 0u);
}
