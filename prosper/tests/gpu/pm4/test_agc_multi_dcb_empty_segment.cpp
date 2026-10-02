// test_agc_multi_dcb_empty_segment -- sceAgcDriverSubmitMultiDcbs accepts a batch in which some
// segments carry zero dwords, instead of ending the process.
//
// THE DEFECT. The handler treated a zero-length segment as unsupported input and abort()ed.
// Assassin's Creed Black Flag Resynced submits 27 segments in one call and leaves some empty (segment
// 14: a valid stream address, zero dwords), so the title died right after its first frame setup. A
// segment with no dwords holds no packets: there is nothing to execute.
//
// WHAT EACH ASSERTION KILLS:
//   AllEmptyBatchIsANoOp  a zero-length segment aborts again (the process dies before any
//                         assertion) or an all-empty batch is not treated as a no-op
#include "hle/dispatch/dispatch.hpp"

#include <gtest/gtest.h>

#include <cstdint>

using namespace prosper;

TEST(AgcMultiDcb, AllEmptyBatchIsANoOp) {
    register_builtin_hle();
    HleFn multi = Hle::lookup("6UzEidRZwkg");   // sceAgcDriverSubmitMultiDcbs
    ASSERT_NE(multi, nullptr);

    // Descriptor arrays: stream pointers and dword counts. Streams are never read when empty.
    alignas(16) static uint32_t unused[4] = {};
    uint64_t streams[2] = { (uint64_t)(uintptr_t)unused, (uint64_t)(uintptr_t)unused };
    uint32_t words[2] = { 0, 0 };
    EXPECT_EQ(multi((uint64_t)(uintptr_t)streams, (uint64_t)(uintptr_t)words, 2, 0, 0, 0), 0u)
        << "a batch of zero-length segments has nothing to execute and must succeed";
}
