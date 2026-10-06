// test_counted_loop_prelude_exec_loop — a counted SCC loop preceded by a bottom-tested EXEC loop.
//
// Kena's compute program 0x5006fb0000 runs a per-lane do-while before its counted loop:
//
//     L:     s_cbranch_execz DONE           ; header exit
//            v_cmpx_lt_u32 sI, sN           ; uniform bound
//            s_cbranch_execz DONE           ; break
//            ...                            ; body
//            s_andn2_b64 exec, exec, vcc    ; lanes leave one by one
//            s_cbranch_execnz L             ; bottom-tested back-edge
//     DONE:  s_mov_b64 exec, saved
//            ... s_cmp / s_cbranch_scc0 / s_branch counted loop ...
//
// `detect_counted_loop` counts only s_branch and SCC back-edges, so it claims the SCC loop and
// never sees the EXEC loop. The counted-loop route then scans its prelude for forward ifs WITHOUT
// loop information, which declines `unclaimed execnz`, and that refusal used to be final although
// nothing had been emitted and the general route (divergent loops, then forward ifs) handles both
// loops. The route now declines instead of refusing, and the general route compiles the program.
//
// The kernels are hand-written and assembled with llvm-mc -mcpu=gfx1030 -mattr=+wavefrontsize64
// (encodings and branch displacements are the assembler's). Every lane executes, and the result is
// checked per lane: a lane that ran the EXEC loop the wrong number of times, or skipped the counted
// loop, reads a different number.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <gtest/gtest.h>
#include "fixtures/compute_runner.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <unordered_set>
#include <vector>

using namespace prosper::gpu;

namespace {

constexpr uint32_t kLanes = 64;

// x = lane; acc = 0; save EXEC; i = 0; n = 6
// L:    execz DONE; v_cmpx_lt_u32 i, n; execz DONE; acc += 1; i += 1;
//       vcc = (x < i); exec &= ~vcc; execnz L
// DONE: restore EXEC; c = 0
// C:    s_cmp_lt_u32 c, 3; scc0 END; acc += 10; c += 1; s_branch C
// END:  out = float(acc)
// Lane x runs the EXEC loop min(x + 1, 6) times, then the counted loop adds 30.
const uint32_t kExecLoopThenCountedLoop[] = {
    0x7E000F00u, 0x7E060280u, 0xBE8E047Eu, 0xBE960380u, 0xBE970386u, 0xBF880008u,
    0xD4D1007Eu, 0x00002E16u, 0xBF880005u, 0x4A060681u, 0x80168116u, 0x7D880016u,
    0x8AFE6A7Eu, 0xBF89FFF7u, 0xBEFE040Eu, 0xBE940380u, 0xBF0A8314u, 0xBF840003u,
    0x4A06068Au, 0x80148114u, 0xBF82FFFBu, 0x7E060D03u, 0xBF810000u,
};
constexpr uint32_t kCountedHeaderPc = 16;   // s_cmp_lt_u32 s20, 3
constexpr uint32_t kExecBackedgePc = 13;   // s_cbranch_execnz L

// Control: the same EXEC loop with no counted loop after it. It never takes the counted-loop route,
// so it shows the general route can lower the prelude's loop on its own.
const uint32_t kExecLoopAlone[] = {
    0x7E000F00u, 0x7E060280u, 0xBE8E047Eu, 0xBE960380u, 0xBE970386u, 0xBF880008u,
    0xD4D1007Eu, 0x00002E16u, 0xBF880005u, 0x4A060681u, 0x80168116u, 0x7D880016u,
    0x8AFE6A7Eu, 0xBF89FFF7u, 0xBEFE040Eu, 0x7E060D03u, 0xBF810000u,
};

template <size_t N>
std::vector<Rdna2Inst> decode(const uint32_t (&code)[N]) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, N, ins);
    return ins;
}

template <size_t N>
std::vector<uint32_t> compile(const uint32_t (&code)[N]) {
    return recompile_valu(code, N, /*num_inputs*/ 1, /*out_vgpr*/ 3);
}

std::vector<float> lane_indices() {
    std::vector<float> input(kLanes);
    for (uint32_t lane = 0; lane < kLanes; ++lane) input[lane] = static_cast<float>(lane);
    return input;
}

}   // namespace

// The fixture must reach the route this test is about. If either premise stops holding, the
// executed test below could pass without ever exercising the decline.
TEST(CountedLoopPreludeExecLoop, FixtureTakesTheCountedRouteAndItsPreludeScanRefuses) {
    const std::vector<Rdna2Inst> ins = decode(kExecLoopThenCountedLoop);
    const CountedLoop loop = detect_counted_loop(ins);
    ASSERT_TRUE(loop.found) << "the SCC loop must be claimed by the counted-loop route";
    EXPECT_EQ(loop.header_pc, kCountedHeaderPc);

    // The prelude as the counted-loop route scans it: everything before the header, ended by an
    // artificial end marker, and no loop information.
    std::vector<Rdna2Inst> preloop;
    for (const auto& in : ins)
        if (!in.is_end && in.pc < loop.header_pc) preloop.push_back(in);
    Rdna2Inst end;
    end.pc = loop.header_pc;
    end.is_end = true;
    preloop.push_back(end);
    bool rejected = false;
    std::unordered_set<uint32_t> safe;
    detect_forward_ifs(preloop, /*allow_vcc*/ false, kExecLoopThenCountedLoop,
                       std::size(kExecLoopThenCountedLoop), &safe, nullptr, &rejected,
                       /*compute_wave_branches*/ true);
    EXPECT_TRUE(rejected) << "the prelude's execnz back-edge at pc " << kExecBackedgePc
                          << " must be what the counted-loop route cannot structure";

    ASSERT_FALSE(detect_counted_loop(decode(kExecLoopAlone)).found)
        << "control: without the SCC loop the counted-loop route is never taken";
}

TEST(CountedLoopPreludeExecLoop, ControlTheGeneralRouteLowersTheExecLoopAlone) {
    const std::vector<uint32_t> spv = compile(kExecLoopAlone);
    ASSERT_FALSE(spv.empty()) << "the general route must lower the EXEC do-while, or the next "
                                 "test's fall-back has nothing to fall back to";
    const std::vector<float> got = prosper::test::run_compute(spv, lane_indices(), kLanes, kLanes);
    if (got.empty()) GTEST_SKIP() << "no Vulkan compute device";
    for (uint32_t lane = 0; lane < kLanes; ++lane)
        EXPECT_FLOAT_EQ(got[lane], static_cast<float>(std::min(lane + 1u, 6u))) << "lane " << lane;
}

TEST(CountedLoopPreludeExecLoop, FallsBackToTheGeneralRouteAndRunsBothLoops) {
    const std::vector<uint32_t> spv = compile(kExecLoopThenCountedLoop);
    ASSERT_FALSE(spv.empty())
        << "a prelude the counted-loop route cannot structure must fall back, not refuse";
    const std::vector<float> got = prosper::test::run_compute(spv, lane_indices(), kLanes, kLanes);
    if (got.empty()) GTEST_SKIP() << "no Vulkan compute device";
    for (uint32_t lane = 0; lane < kLanes; ++lane)
        EXPECT_FLOAT_EQ(got[lane], static_cast<float>(std::min(lane + 1u, 6u) + 30u))
            << "lane " << lane;
}
