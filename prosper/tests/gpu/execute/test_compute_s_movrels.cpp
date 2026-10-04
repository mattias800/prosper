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
std::vector<uint32_t> run(const std::vector<uint32_t>& code, uint32_t m0_input) {
    const auto module = prosper::gpu::recompile_valu(code.data(), code.size(), 1, 0);
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
