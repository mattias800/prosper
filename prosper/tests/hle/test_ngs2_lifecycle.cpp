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
    // Both allocator-create NIDs reproduce the firmware set verbatim.
    EXPECT_EQ(nid_hash("sceNgs2SystemCreateWithAllocator"), "mPYgU4oYpuY");
    EXPECT_EQ(nid_hash("sceNgs2RackCreateWithAllocator"), "U546k6orxQo");
    EXPECT_NE(nid_hash("sceNgs2SystemCreateWithAllocator"), "AAAAAAAAAAA")
        << "positive control: the discriminator rejects a wrong NID";
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
    uint64_t alloc = 0;  // required input pointer; its callbacks are never invoked
    EXPECT_EQ(create(0, 0, 0, 0, 0, 0), kInvalidOut) << "null handle out is refused";
    EXPECT_EQ(create(0, 0, addr(&sys), 0, 0, 0), kInvalidOut)
        << "null allocator is refused like SystemCreate's null buffer_info";
    ASSERT_EQ(create(0, addr(&alloc), addr(&sys), 0, 0, 0), 0u);
    EXPECT_NE(sys, 0u) << "CreateWithAllocator writes a tagged handle";
    EXPECT_EQ(grain(sys, 4096, 0, 0, 0, 0), 0u);
    EXPECT_EQ(grain(0xDEADu, 4096, 0, 0, 0, 0), kInvalidSystem) << "foreign system refused";

    uint64_t rack = 0;
    EXPECT_EQ(rack_create(sys, 1, 0, 0, addr(&rack), 0), kInvalidOut)
        << "null allocator is refused";
    ASSERT_EQ(rack_create(sys, 1, 0, addr(&alloc), addr(&rack), 0), 0u);
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
}
