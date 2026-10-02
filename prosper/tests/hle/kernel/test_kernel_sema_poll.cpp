// #1640: a guest drains a semaphore until PollSema reports the encoded EBUSY value.
// Exercise the registered import, including partial acquisitions and count preservation.
#include "hle/dispatch/dispatch.hpp"
#include <gtest/gtest.h>
#include "hle/dispatch/nid.hpp"
#include "hle/kernel/sce_errno.hpp"

#include <cstdint>
#include <cstdio>

using namespace prosper;

#define CHECK(c, m) EXPECT_TRUE(c) << (m)

TEST(KernelSemaPoll, Contract) {
    register_builtin_hle();
    const HleFn create = Hle::lookup(nid_hash("sceKernelCreateSema").c_str());
    const HleFn poll = Hle::lookup("12wOHk8ywb0");
    const HleFn signal = Hle::lookup(nid_hash("sceKernelSignalSema").c_str());
    const HleFn destroy = Hle::lookup(nid_hash("sceKernelDeleteSema").c_str());
    CHECK(create && poll && signal && destroy, "semaphore imports are registered");
    CHECK(poll == Hle::lookup(nid_hash("sceKernelPollSema").c_str()),
          "actual guest PollSema NID resolves to the named entry point");
    if (!create || !poll || !signal || !destroy) FAIL() << "legacy early exit";

    void* sema = nullptr;
    CHECK(create(reinterpret_cast<uintptr_t>(&sema), 0, 0, 0, 8, 0) == 0 && sema,
          "empty semaphore is created");
    if (!sema) FAIL() << "legacy early exit";
    const uint64_t handle = reinterpret_cast<uintptr_t>(sema);
    constexpr uint64_t busy = 0x80020010ull; // The guest drain loop compares this literal.
    static_assert(hle::kSceKernelErrorEBUSY == busy);

    CHECK(poll(handle, 1, 0, 0, 0, 0) == busy, "empty poll returns encoded EBUSY");
    CHECK(signal(handle, 3, 0, 0, 0, 0) == 0, "three units are signaled");
    CHECK(poll(handle, 4, 0, 0, 0, 0) == busy, "insufficient count returns EBUSY");
    CHECK(poll(handle, 2, 0, 0, 0, 0) == 0, "failed poll preserves all three units");
    CHECK(poll(handle, 2, 0, 0, 0, 0) == busy, "successful poll consumes exactly two units");
    CHECK(poll(handle, 1, 0, 0, 0, 0) == 0, "one remaining unit is acquired");
    CHECK(poll(handle, 1, 0, 0, 0, 0) == busy, "consumed semaphore is empty");

    CHECK(signal(handle, 3, 0, 0, 0, 0) == 0, "drain starts with three units");
    unsigned acquired = 0;
    unsigned calls = 0;
    bool terminated = false;
    // The finite bound lets the old error value fail without hanging the test process.
    for (; calls < 8; ++calls) {
        const uint64_t result = poll(handle, 1, 0, 0, 0, 0);
        if (result == busy) { terminated = true; ++calls; break; }
        if (result == 0) ++acquired;
    }
    CHECK(terminated && calls == 4 && acquired == 3,
          "drain acquires every unit then terminates on the first empty poll");
    CHECK(poll(handle, 1, 0, 0, 0, 0) == busy, "drain leaves no phantom units");
    CHECK(destroy(handle, 0, 0, 0, 0, 0) == 0, "semaphore is deleted");
}
