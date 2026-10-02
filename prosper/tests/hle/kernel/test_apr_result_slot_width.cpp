// test_apr_result_slot_width — sceKernelAprSubmitCommandBufferAndGetResult writes 32-bit results.
//
// THE DEFECT. The call is (cb, ring, uint32_t* out1, uint32_t* out2). prosper stored an 8-byte token
// through each pointer. A guest that keeps the two slots as adjacent 4-byte stack locals therefore
// had the four bytes after each slot overwritten; for the slot sitting directly below its stack
// canary that was the low half of the canary, and the function's epilogue called __stack_chk_fail
// (Assassin's Creed Black Flag Resynced, host SIGILL right after an APR read; issue #4138).
//
// WHAT THE ARMS KILL:
//   M1  apr_write_result_slot stores 8 bytes again              -> the guard bytes after each slot
//   M2  the slot is not written at all                          -> the "result is delivered" arms
//   M3  only the first slot is written                          -> the out2 arm
//
// The layout is the failing guest's: out1 at +4 and out2 at +12 of a 24-byte region (both 4 mod 8,
// 8 bytes apart), every other byte pre-filled with a sentinel.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

namespace {
bool all_bytes(const uint8_t* p, size_t n, uint8_t v) {
    for (size_t i = 0; i < n; ++i)
        if (p[i] != v) return false;
    return true;
}
}   // namespace

int main() {
    printf("== test_apr_result_slot_width ==\n");
    register_builtin_hle();

    HleFn submit = Hle::lookup(nid_hash("sceKernelAprSubmitCommandBufferAndGetResult"));
    CHECK(submit != nullptr, "sceKernelAprSubmitCommandBufferAndGetResult is registered");
    if (!submit) return 1;

    // The command buffer is never bound to an event queue, so the submit takes the path that
    // writes the result slots. Its contents are not read by that path.
    alignas(16) static uint8_t cb[256] = {};
    alignas(16) uint8_t region[24];
    memset(region, 0xA5, sizeof region);

    const uint64_t rc = submit((uint64_t)(uintptr_t)cb, /*ring, 1-based=*/1,
                               (uint64_t)(uintptr_t)(region + 4), (uint64_t)(uintptr_t)(region + 12),
                               0, 0);
    CHECK(rc == 0, "the submit succeeds");

    uint32_t out1 = 0, out2 = 0;
    memcpy(&out1, region + 4, sizeof out1);
    memcpy(&out2, region + 12, sizeof out2);

    CHECK(out1 != 0xA5A5A5A5u, "out1 received a result (M2)");
    CHECK(out2 != 0xA5A5A5A5u, "out2 received a result (M2, M3)");
    CHECK(out1 == out2, "both slots carry the same submission token");
    CHECK(all_bytes(region, 4, 0xA5), "bytes before out1 are untouched");
    CHECK(all_bytes(region + 8, 4, 0xA5),
          "the four bytes after out1 (the pad before out2) are untouched (M1)");
    CHECK(all_bytes(region + 16, 8, 0xA5),
          "the bytes after out2 (the stack canary's place in the failing guest) are untouched (M1)");

    printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
