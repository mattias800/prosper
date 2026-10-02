// test_agc_multi_dcb_empty_segment -- sceAgcDriverSubmitMultiDcbs skips a zero-length segment in a
// batch that also carries real ones, instead of ending the process.
//
// THE DEFECT. The handler treated a zero-length segment as unsupported input and abort()ed.
// Assassin's Creed Black Flag Resynced submits 27 segments in one call and leaves some empty (segment
// 14: a valid stream address, zero dwords), so the title died right after its first frame setup. A
// segment with no dwords holds no packets: there is nothing to execute for it.
//
// WHAT EACH ASSERTION KILLS:
//   MixedBatchSkipsTheEmptySegment  a zero-length segment aborts again (the process dies before any
//                                   assertion), or skipping it drops or reorders the real segment's
//                                   work (no submission, no draw, register not applied).
// An ALL-empty batch is unobserved and stays fatal; tests/hle/test_agc_submit_multi.cpp's
// `zero-length` death arm keeps that pinned.
#include "hle/dispatch/dispatch.hpp"
#include "gpu/pm4/command_processor.hpp"

#include <gtest/gtest.h>

#include <cstdint>

using namespace prosper;
extern "C" void prosper_agc_submit_stats(uint64_t* submits, uint64_t* draws);
extern "C" bool prosper_agc_submit_sh_reg(uint64_t queue, uint32_t offset, uint32_t* value);

namespace {
uint64_t U(const void* p) { return reinterpret_cast<uintptr_t>(p); }
uint32_t header(uint32_t words, uint32_t op, uint32_t sub = 0) {
    return 0xc0000000u | ((words - 2) << 16) | (op << 8) | (sub << 2);
}
}  // namespace

TEST(AgcMultiDcb, MixedBatchSkipsTheEmptySegment) {
    register_builtin_hle();
    HleFn multi = Hle::lookup("6UzEidRZwkg");   // sceAgcDriverSubmitMultiDcbs
    ASSERT_NE(multi, nullptr);
    auto hook = Hle::return_hook_of("6UzEidRZwkg");
    ASSERT_NE(hook, nullptr);

    // The title's shape: an empty segment (a valid address, zero dwords) next to a real one.
    alignas(16) static uint32_t empty[4] = {};
    uint32_t real[] = {header(3, gpu::IT_SET_SH_REG), 0x123, 0x0badf00d,
                       header(3, gpu::IT_NOP, gpu::R_DRAW_INDEX_AUTO), 3, 0};
    uint64_t streams[2] = {U(empty), U(real)};
    uint32_t words[2] = {0, 6};

    uint64_t before = 0, draws_before = 0, after = 0, draws_after = 0;
    prosper_agc_submit_stats(&before, &draws_before);
    EXPECT_EQ(multi(U(streams), U(words), 2, 0, 0, 0), 0u);
    prosper_agc_submit_stats(&after, &draws_after);
    hook();
    EXPECT_EQ(after, before + 1) << "the batch is still one submission";
    EXPECT_EQ(draws_after, draws_before + 1) << "the real segment's draw still executes";
    uint32_t value = 0;
    EXPECT_TRUE(prosper_agc_submit_sh_reg(0, 0x123, &value) && value == 0x0badf00d)
        << "the real segment's register write still lands";
}
