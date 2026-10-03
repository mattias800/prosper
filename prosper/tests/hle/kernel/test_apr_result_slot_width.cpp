// APR result is status32 + failing-offset32 (8 bytes); the separate submit ID is 4 bytes.
// A token in status turns a successful read into a guest fatal. Widening the ID corrupts
// the adjacent live flag/canary. Original callers establish both widths independently.
#include "hle/dispatch/dispatch.hpp"
#include <gtest/gtest.h>
#include "hle/dispatch/nid.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>

using namespace prosper;

#define CHECK(c, m) EXPECT_TRUE(c) << (m)

namespace {
bool all_bytes(const uint8_t* p, size_t n, uint8_t v) {
    for (size_t i = 0; i < n; ++i)
        if (p[i] != v) return false;
    return true;
}
}   // namespace

TEST(AprResultSlotWidth, Contract) {
    printf("== test_apr_result_slot_width ==\n");
    register_builtin_hle();

    HleFn submit = Hle::lookup(nid_hash("sceKernelAprSubmitCommandBufferAndGetResult"));
    CHECK(submit != nullptr, "sceKernelAprSubmitCommandBufferAndGetResult is registered");
    if (!submit) FAIL() << "legacy early exit";

    // The command buffer is never bound to an event queue, so the submit takes the path that
    // writes the result and ID. Its contents are not read by that path.
    alignas(16) static uint8_t cb[256] = {};
    alignas(16) uint8_t region[24];
    memset(region, 0xA5, sizeof region);

    const uint64_t rc = submit((uint64_t)(uintptr_t)cb, /*ring, 1-based=*/1,
                               (uint64_t)(uintptr_t)(region + 4), (uint64_t)(uintptr_t)(region + 12),
                               0, 0);
    CHECK(rc == 0, "the submit succeeds");

    uint32_t out1 = 0, offset = 0, out2 = 0;
    memcpy(&out1, region + 4, sizeof out1);
    memcpy(&offset, region + 8, sizeof offset);
    memcpy(&out2, region + 12, sizeof out2);

    CHECK(out1 != 0xA5A5A5A5u, "out1 received a result (M2)");
    CHECK(out2 != 0xA5A5A5A5u, "out2 received a result (M2, M3)");
    CHECK(out1 == 0 && offset == 0, "successful execution writes both result words");
    CHECK(all_bytes(region, 4, 0xA5), "bytes before out1 are untouched");
    HleFn wait = Hle::lookup("rqwFKI4PAiM");
    ASSERT_NE(wait, nullptr);
    EXPECT_EQ(wait(out2, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(wait(out2, 0, 0, 0, 0, 0), 0u) << "repeat waits retain the completed handle";
    CHECK(all_bytes(region + 16, 8, 0xA5),
          "the bytes after out2 (the stack canary's place in the failing guest) are untouched (M1)");

}
