// s_wqm_b64 into VCC is a VCC write.
//
// The lowering recorded the new mask under register 106 and left the value that branches and the
// e32 selects read (RegState::vcc) at whatever VCC held before. s_mov_b64 and s_not_b64 into VCC
// already updated both; S_WQM did not, so the instruction after it read a stale VCC.
//
// The raw-wide classifier treats all three as mask transfers that replace what VCC held (#4555),
// which is only right if the emitted code agrees. This is the emitted code's half of that.
//
//   v_mov_b32 v1, 7
//   v_cmp_eq_u32 vcc, v0, v0             VCC = every lane
//   v_cmp_ne_u32 s[4:5], v0, v0          s[4:5] = no lane
//   (the instruction under test)         VCC <- s[4:5]
//   v_cndmask_b32 v0, 5, v1              v0 = VCC ? 7 : 5
//   s_endpgm
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "fixtures/compute_runner.h"
#include <gtest/gtest.h>
#include <bit>
#include <cstdint>
#include <vector>

using namespace prosper::gpu;

namespace {
constexpr uint32_t kLanes = 64;

std::vector<uint32_t> program(uint32_t transfer) {
    return {
        0x7e020287u,                // v_mov_b32 v1, 7
        0x7d840100u,                // v_cmp_eq_u32 vcc, v0, v0
        0xd4c50004u, 0x00020100u,   // v_cmp_ne_u32 s[4:5], v0, v0
        transfer,
        0x02000285u,                // v_cndmask_b32 v0, 5, v1
        0xbf810000u,                // s_endpgm
    };
}

std::vector<uint32_t> run(const std::vector<uint32_t>& code) {
    const auto module = recompile_valu(code.data(), code.size(), 1, 0);
    EXPECT_FALSE(module.empty()) << "the program must compile";
    if (module.empty()) return {};
    const std::vector<float> input(kLanes, std::bit_cast<float>(3u));
    const auto output = prosper::test::run_compute(module, input, kLanes, kLanes);
    std::vector<uint32_t> words;
    words.reserve(output.size());
    for (float value : output) words.push_back(std::bit_cast<uint32_t>(value));
    return words;
}

void expect_every_lane(const std::vector<uint32_t>& words, uint32_t expected) {
    ASSERT_EQ(words.size(), kLanes);
    for (uint32_t lane = 0; lane < kLanes; ++lane)
        EXPECT_EQ(words[lane], expected) << "lane " << lane;
}
}   // namespace

TEST(ComputeWqmIntoVcc, FixtureDecodesAsDescribed) {
    const auto code = program(0xbeea0a04u);
    std::vector<Rdna2Inst> ins;
    ASSERT_EQ(rdna2_walk(code.data(), code.size(), ins), code.size());
    ASSERT_EQ(ins.size(), 6u);
    EXPECT_EQ(ins[1].fmt, Rdna2Format::VOPC);
    EXPECT_EQ(ins[1].dst.value, 106) << "the first compare writes VCC";
    EXPECT_EQ(ins[2].dst.kind, OperandKind::SGPR);
    EXPECT_EQ(ins[2].dst.value, 4) << "the second compare writes s[4:5]";
    EXPECT_EQ(ins[3].fmt, Rdna2Format::SOP1);
    EXPECT_EQ(ins[3].opcode, kSop1OpcodeWqmB64);
    EXPECT_EQ(ins[3].dst.value, 106);
    EXPECT_EQ(ins[3].src[0].value, 4);
    EXPECT_EQ(ins[4].fmt, Rdna2Format::VOP2);
    EXPECT_EQ(ins[4].opcode, 0x01u) << "v_cndmask_b32: reads VCC implicitly";
}

TEST(ComputeWqmIntoVcc, TheSelectAfterItReadsTheNewMask) {
    // s_wqm_b64 vcc, s[4:5]: VCC is now "no lane", so the select takes its first operand.
    expect_every_lane(run(program(0xbeea0a04u)), 5u);
}

TEST(ComputeWqmIntoVcc, TheOtherUnaryTransfersAgree) {
    // The two forms that already did this, so the 5 above is the transfer and not the fixture:
    // s_mov_b64 vcc, s[4:5] gives the same "no lane", and s_not_b64 vcc, s[4:5] its complement.
    expect_every_lane(run(program(0xbeea0404u)), 5u);
    expect_every_lane(run(program(0xbeea0804u)), 7u);
    // And with VCC left alone the select still sees the first compare.
    expect_every_lane(run(program(0xbf800000u)), 7u);
}
