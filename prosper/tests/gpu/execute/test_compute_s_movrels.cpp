// s_movrels_b32 (SOP1 0x2E) execution: dst = SGPR[src0# + M0], through both lowering paths.
//
// The program seeds s4 = 100, s5 = 200, s6 = 300 (s7 is never written, so it reads the unwritten
// SGPR placeholder 0), reads s[4 + M0] into s0 and returns it in v0. M0 comes either from the
// lane input through v_readfirstlane_b32 + s_mov_b32 m0 (the dynamic select over s4..s105) or
// from an inline constant (the compile-time fold to one register). Each M0 value must return its
// OWN register, so a lowering that ignores M0, folds to the wrong register, or swaps candidates
// is caught. Every word below was assembled with llvm-mc -mcpu=gfx1030.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "fixtures/compute_runner.h"
#include <gtest/gtest.h>
#include <bit>
#include <cstdint>
#include <vector>

namespace {
constexpr uint32_t kLanes = 64;

const std::vector<uint32_t> kSeed = {
    0xbe8403ffu, 0x00000064u,   // s_mov_b32 s4, 0x64
    0xbe8503ffu, 0x000000c8u,   // s_mov_b32 s5, 0xc8
    0xbe8603ffu, 0x0000012cu,   // s_mov_b32 s6, 0x12c
};
const std::vector<uint32_t> kReadAndReturn = {
    0xbe802e04u,   // s_movrels_b32 s0, s4
    0x7e000200u,   // v_mov_b32_e32 v0, s0
    0xbf810000u,   // s_endpgm
};

std::vector<uint32_t> program(const std::vector<uint32_t>& set_m0) {
    std::vector<uint32_t> code = kSeed;
    code.insert(code.end(), set_m0.begin(), set_m0.end());
    code.insert(code.end(), kReadAndReturn.begin(), kReadAndReturn.end());
    return code;
}

// Runs `code` with every lane's v0 holding the raw word `m0_input`, and returns lane results.
std::vector<uint32_t> run(const std::vector<uint32_t>& code, uint32_t m0_input,
                          bool force_dispatcher = false) {
    const auto module =
        prosper::gpu::recompile_valu(code.data(), code.size(), 1, 0, nullptr, 0,
                                     prosper::gpu::kDefaultComputePgmRsrc1, force_dispatcher);
    EXPECT_FALSE(module.empty()) << "s_movrels_b32 must compile on this path";
    if (module.empty()) return {};
    const std::vector<float> input(kLanes, std::bit_cast<float>(m0_input));
    const auto output = prosper::test::run_compute(module, input, kLanes, kLanes);
    std::vector<uint32_t> words;
    for (float value : output) words.push_back(std::bit_cast<uint32_t>(value));
    return words;
}

void expect_every_lane(const std::vector<uint32_t>& words, uint32_t expected) {
    ASSERT_EQ(words.size(), kLanes) << "dispatch and host readback must complete";
    for (uint32_t lane = 0; lane < kLanes; ++lane)
        EXPECT_EQ(words[lane], expected) << "lane " << lane;
}
}   // namespace

TEST(ComputeSMovrelsExecution, DynamicM0SelectsItsOwnRegister) {
    const auto code = program({
        0x7e000500u,   // v_readfirstlane_b32 s0, v0
        0xbefc0300u,   // s_mov_b32 m0, s0
    });
    constexpr uint32_t kExpected[] = {100u, 200u, 300u, 0u};   // s4, s5, s6, unwritten s7
    for (uint32_t m0 = 0; m0 < 4; ++m0) {
        SCOPED_TRACE(m0);
        expect_every_lane(run(code, m0), kExpected[m0]);
    }
}

TEST(ComputeSMovrelsExecution, AScratchWriteSkippedByAnEmptyExecIsNotObserved) {
    // #4538. s5 is rewritten inside a block that s_cbranch_execz skips, and the relative read
    // after the merge returns s5 (M0 = 1). With EXEC empty at the branch the write never runs on
    // hardware, so the read returns the seeded 200. The recompiler used to prove s5 dead at the
    // merge -- it charged the relative read s4 alone -- drop the branch, and run the write
    // unconditionally: 500.
    const auto skipped = [](uint32_t exec_before_branch) {
        return program({
            0xbefc0381u,   // s_mov_b32 m0, 1
            0xbe8a047eu,   // s_mov_b64 s[10:11], exec
            exec_before_branch,
            0xbf880002u,   // s_cbranch_execz +2
            0xbe8503ffu,
            0x000001f4u,   // s_mov_b32 s5, 0x1f4
            0xbefe040au,   // s_mov_b64 exec, s[10:11]
        });
    };
    {
        SCOPED_TRACE("EXEC empty at the branch: the write is skipped");
        expect_every_lane(run(skipped(0xbefe0480u /* s_mov_b64 exec, 0 */), 0u), 200u);
    }
    {
        // Control: the same program with EXEC left alone runs the write, so the 200 above is the
        // skip and not a read that ignores s5 altogether.
        SCOPED_TRACE("EXEC full at the branch: the write runs");
        expect_every_lane(run(skipped(0xbf800000u /* s_nop */), 0u), 500u);
    }
}

TEST(ComputeSMovrelsExecution, ResultSurvivesABranchJoin) {
    // The relative read's result is consumed after a scalar branch has rejoined. Each shape
    // must return s5 (M0 = 1): 200. The forced arm runs the CFG dispatcher, which reloads scalar
    // registers at every case boundary; in this harness nothing else selects it, and without it
    // these shapes passed even when the dispatcher dropped the result.
    const std::vector<std::vector<uint32_t>> set_m0 = {
        {0xbefc0381u},   // s_mov_b32 m0, 1
        {0x7e000500u, 0xbefc0300u},   // v_readfirstlane_b32 s0, v0; s_mov_b32 m0, s0
    };
    const std::vector<std::vector<uint32_t>> between = {
        {0xbf008005u, 0xbf850001u, 0xbf800000u},   // s_cmp_eq_i32 s5,0; scc1 +1; nop
        {0xbf008005u, 0xbf840001u, 0xbf800000u},   // ... scc0 +1 (taken)
        {0xbf008005u, 0xbf850002u, 0xbe8903ffu, 0x00000005u},   // scc1 +2; s_mov_b32 s9, 5
        {0xbf008005u, 0xbf840002u, 0xbe8903ffu, 0x00000005u},   // scc0 +2 (taken); s_mov_b32 s9, 5
    };
    for (const bool force_dispatcher : {false, true})
        for (size_t index = 0; index < set_m0.size(); ++index)
            for (size_t shape = 0; shape < between.size(); ++shape) {
                SCOPED_TRACE((force_dispatcher ? 100 : 0) + index * 10 + shape);
                std::vector<uint32_t> code = kSeed;
                code.insert(code.end(), set_m0[index].begin(), set_m0[index].end());
                code.push_back(0xbe802e04u);   // s_movrels_b32 s0, s4
                code.insert(code.end(), between[shape].begin(), between[shape].end());
                code.insert(code.end(), {0x7e000200u, 0xbf810000u});   // v_mov_b32 v0, s0; s_endpgm
                expect_every_lane(run(code, 1u, force_dispatcher), 200u);
            }
}

TEST(ComputeSMovrelsExecution, ResultSurvivesABranchJoinWhenTheBaseWasNeverWritten) {
    // The same, with s4 never written: M0 = 1 still selects s5.
    for (const bool force_dispatcher : {false, true}) {
        SCOPED_TRACE(force_dispatcher);
        const std::vector<uint32_t> code = {
            0xbe8503ffu, 0x000000c8u,   // s_mov_b32 s5, 0xc8
            0xbefc0381u,   // s_mov_b32 m0, 1
            0xbe802e04u,   // s_movrels_b32 s0, s4
            0xbf008005u, 0xbf850001u, 0xbf800000u,   // s_cmp_eq_i32 s5, 0; s_cbranch_scc1 +1; s_nop
            0x7e000200u, 0xbf810000u,   // v_mov_b32 v0, s0; s_endpgm
        };
        expect_every_lane(run(code, 1u, force_dispatcher), 200u);
    }
}

TEST(ComputeSMovrelsExecution, AnSccBranchAfterTheReadFollowsTheCompareBeforeIt) {
    // s_cmp ... ; s_movrels_b32 s0, s4 ; (block boundary) ; s_cbranch_scc1 over a rewrite of s0.
    // The relative read leaves SCC alone, so the branch is the compare's. The boundary matters:
    // within one block the branch consumes the compare's SCC directly, and only an SCC that
    // crosses into another dispatcher case is reloaded on the strength of the transfer that did
    // not know the relative read keeps it (#4559).
    const auto code = [](uint32_t compare, bool boundary) {
        std::vector<uint32_t> words = kSeed;
        words.insert(words.end(), {0xbefc0381u, compare, 0xbe802e04u});   // m0 = 1; cmp; s0 = s5
        if (boundary) words.push_back(0xbf880000u);   // s_cbranch_execz +0: the next pc is a target
        words.insert(words.end(), {
                                      0xbf850002u,   // s_cbranch_scc1 +2
                                      0xbe8003ffu,
                                      0x00000309u,   // s_mov_b32 s0, 0x309
                                      0x7e000200u,   // v_mov_b32 v0, s0
                                      0xbf810000u,   // s_endpgm
                                  });
        return words;
    };
    for (const bool force_dispatcher : {false, true})
        for (const bool boundary : {false, true}) {
            SCOPED_TRACE((force_dispatcher ? 10 : 0) + (boundary ? 1 : 0));
            // s_cmp_lg_i32 s5, 0: SCC = 1, the rewrite is skipped and the relative read survives.
            expect_every_lane(run(code(0xbf018005u, boundary), 0u, force_dispatcher), 200u);
            // s_cmp_eq_i32 s5, 0: SCC = 0, the rewrite runs.
            expect_every_lane(run(code(0xbf008005u, boundary), 0u, force_dispatcher), 0x309u);
        }
}

TEST(ComputeSMovrelsExecution, ConstantM0FoldsToOneRegister) {
    // The lane input is ignored here; a stray dependence on it would show up as a wrong value.
    {
        SCOPED_TRACE("m0 = 0");
        expect_every_lane(run(program({0xbefc0380u /* s_mov_b32 m0, 0 */}), 2u), 100u);
    }
    {
        SCOPED_TRACE("m0 = 2");
        expect_every_lane(run(program({0xbefc0382u /* s_mov_b32 m0, 2 */}), 0u), 300u);
    }
}
