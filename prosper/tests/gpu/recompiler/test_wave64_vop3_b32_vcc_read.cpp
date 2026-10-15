// test_wave64_vop3_b32_vcc_read: V_BFE_U32 reads VCC_LO as ONE scalar dword in the Wave64 CFG
// dispatcher's block-entry MUST analysis (#4848).
//
// House of the Dead 2's fragment programs 0x41c7dfd500 and 0x41d05bd900 run a scalar table search
// that recycles VCC as scratch, then a bit test on a freshly loaded VCC_LO:
//
//     s_cbranch_scc1  ...                        skip the search: VCC is still a Wave64 mask
//     ...  s_add_i32 vcc_lo, s5, 1 / s_buffer_load_dword vcc_hi, ...   VCC as scalar scratch
//     s_buffer_load_dword vcc_lo, s[0:3], 0xae4  join block: VCC_LO is scalar data again
//     v_bfe_u32       v0, vcc_lo, s4, 1          a 32-BIT read of that dword
//     v_cmp_ne_u32    vcc, 0, v0                 complete VCC replacement
//
// `scalar_alu_source_words` had no entry for V_BFE_U32, so the VOP3 fail-closed default charged
// the read the whole PAIR and demanded VCC_HI be scalar data too. VCC_HI is a mask half on the
// skip path and scalar data on the search path, so the dispatcher declined with
// `wave64-ambiguous-mask-read` and the straight-line fallback reported its own first reject, an
// `s_cbranch_vccz` at pc71/72. Every draw using either program was dropped.
//
// V_BFE_U32 is `D.u32 = (S0.u32 >> S1.u32[4:0]) & ((1 << S2.u32[4:0]) - 1)`: three 32-bit
// operands (RDNA2 ISA reference, document 70648), and the emitter lowers S0 with a one-dword read.
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

using namespace prosper::gpu;

namespace {

// The live V_BFE_U32 words from 0x41c7dfd500 pc694: v_bfe_u32 v0, vcc_lo, s4, 1.
constexpr uint32_t kBfeVccLo0 = 0xd5480000u;
constexpr uint32_t kBfeVccLo1 = 0x0204086au;

// The two facts the guest arrives with at the consumer, both load-bearing:
//   * VCC is AMBIGUOUS at the join: the vccz edge leaves the pair a mask, the fall-through leaves
//     VCC_LO scalar data;
//   * VCC_LO is then republished as a MUST scalar word by a ONE-dword write, which by design cannot
//     clear the pair's ambiguity. VCC_LO is provably scalar; VCC_HI has no MUST scalar-data proof.
// `republish_vcc_lo = false` drops the second fact, leaving the dword itself ambiguous.
std::vector<uint32_t> program(uint32_t site0, uint32_t site1, bool republish_vcc_lo) {
    std::vector<uint32_t> code{
        0x7c020300u,   // v_cmp_lt_f32 vcc, v0, v1 -> VCC is a real Wave64 mask
        0xbf860001u,   // s_cbranch_vccz +1: two edges reach the next block
        0xbeea0380u,   // s_mov_b32 vcc_lo, 0 -> scalar on the fall-through path only
        0xbe840380u,   // s_mov_b32 s4, 0 (join block; the proved scalar offset)
    };
    if (republish_vcc_lo) code.push_back(0xbeea0385u);   // s_mov_b32 vcc_lo, 5
    code.insert(
        code.end(),
        {
            site0,
            site1,
            0x7d840100u,   // v_cmp_eq_u32 vcc, v0, v0: complete VCC replacement
            // A varying branch plus a VCC loop: the structured emitter declines the back edge, so the
            // whole program goes through the generic Wave64 CFG dispatcher and its MUST analysis.
            0x7e040280u,   // v_mov_b32 v2, 0
            0x7c020300u,   // v_cmp_lt_f32 vcc, v0, v1
            0xbf860001u,   // s_cbranch_vccz +1
            0x7e040281u,   // v_mov_b32 v2, 1
            0x7d840100u,   // v_cmp_eq_u32 vcc, v0, v0 (always true)
            0xbf870001u,   // s_cbranch_vccnz +1
            0xbf82fffdu,   // s_branch -3
            0x7e040d02u,   // v_cvt_f32_u32 v2, v2
            0xbf810000u,   // s_endpgm
        });
    return code;
}

bool has_opcode(const std::vector<uint32_t>& spirv, uint32_t opcode) {
    if (spirv.size() < 5) return false;
    for (size_t i = 5; i < spirv.size();) {
        const uint32_t words = spirv[i] >> 16u;
        if (words == 0 || words > spirv.size() - i) return false;
        if ((spirv[i] & 0xffffu) == opcode) return true;
        i += words;
    }
    return false;
}

using Rejections = std::vector<std::pair<std::string, std::string>>;

std::vector<uint32_t> compile(const std::vector<uint32_t>& code, Rejections* rejections = nullptr,
                              uint32_t tidig_comp_cnt = 1) {
    ComputeShaderConfig config;
    config.local_x = 64;
    config.local_y = 1;
    config.wave_size = 64;
    config.native_subgroup_size = 64;
    // The first compare reads v0/v1; local Y is a genuine input even for a one-row group.
    config.tidig_comp_cnt = tidig_comp_cnt;
    // The later straight-line fallback also refuses at PC1. Capture all terminal attempts so
    // that its final diagnostic cannot conceal a missing CFG ambiguity guard.
    // Without local Y, v1 is never loaded and so arbitrary under #5185 on every route; give it a
    // hard value from s30, which no launch loads either (#5039). Shifts every pc by one.
    std::vector<uint32_t> words = code;
    if (tidig_comp_cnt == 0) words.insert(words.begin(), 0x7e02021eu);   // v_mov_b32 v1, s30
    TerminalRejectCapture capture;
    auto spirv = recompile_compute(words.data(), words.size(), nullptr, config);
    if (rejections) *rejections = capture.take();
    return spirv;
}

}   // namespace

TEST(Wave64Vop3B32VccRead, BfeU32ChargesOneDwordPerSource) {
    const uint32_t words[] = {kBfeVccLo0, kBfeVccLo1};
    std::vector<Rdna2Inst> decoded;
    ASSERT_EQ(rdna2_walk(words, 2, decoded), 2u);
    ASSERT_EQ(decoded.size(), 1u);
    const Rdna2Inst& bfe = decoded.front();
    ASSERT_EQ(bfe.fmt, Rdna2Format::VOP3);
    ASSERT_EQ(bfe.opcode, 0x148u);
    ASSERT_EQ(bfe.src[0].value, 106);   // VCC_LO
    for (uint32_t source = 0; source < 3; ++source)
        EXPECT_EQ(scalar_alu_source_words(bfe, source), 1u) << source;
}

// Discriminator. Mutation: drop 0x148 from the B32 VOP3 allowlist in `scalar_alu_source_words`
// (the state before #4848) and this program is refused `wave64-ambiguous-mask-read`.
TEST(Wave64Vop3B32VccRead, ScalarVccLoBesideAmbiguousVccHiCompiles) {
    const auto spirv = compile(program(kBfeVccLo0, kBfeVccLo1, true));
    ASSERT_FALSE(spirv.empty());
    EXPECT_TRUE(has_opcode(spirv, 251u)) << "OpSwitch proves the CFG dispatcher route";
}

// Positive control for the FIXTURE, from a different source than the subject: V_LSHL_ADD_U32 was
// already classified B32 (#2820), so the same site compiles before and after #4848. It shows the
// prefix really presents a scalar VCC_LO beside an ambiguous VCC_HI and the dispatcher accepts it.
TEST(Wave64Vop3B32VccRead, ControlAlreadyClassifiedSiblingCompiles) {
    // v_lshl_add_u32 v7, v6, 2, vcc_lo
    const auto spirv = compile(program(0xd7460007u, 0x01a90506u, true));
    ASSERT_FALSE(spirv.empty());
    EXPECT_TRUE(has_opcode(spirv, 251u)) << "the sibling must exercise the same CFG dispatcher";
}

TEST(Wave64Vop3B32VccRead, MissingLocalYCannotFabricateCompareMask) {
    Rejections rejections;
    EXPECT_TRUE(compile(program(kBfeVccLo0, kBfeVccLo1, true), &rejections, 0).empty());
    const std::pair<std::string, std::string> expected{
        "fabricated-fresh-mask-source", "pc=1 a fresh predicate requires defined inputs"};
    EXPECT_NE(std::find(rejections.begin(), rejections.end(), expected), rejections.end())
        << testing::PrintToString(rejections);
    EXPECT_TRUE(std::any_of(rejections.begin(), rejections.end(), [](const auto& rejection) {
        return rejection.first == "cfg-recompile-reject" &&
               rejection.second.find("pc=1 words=7c020300 ") != std::string::npos;
    })) << testing::PrintToString(rejections);
}

// The unsafe direction. A genuine PAIR consumer at the same site still reads the ambiguous VCC_HI.
// Mutation: undercharge V_CNDMASK's condition as one word. Its emission still refuses the
// unresolved operand, so an empty module alone would miss the weakened CFG proof.
TEST(Wave64Vop3B32VccRead, PairConsumerAtTheSameSiteStaysRefused) {
    // v_cndmask_b32_e64 v7, v6, v6, vcc
    Rejections rejections;
    EXPECT_TRUE(compile(program(0xd5010007u, 0x01aa0d06u, true), &rejections).empty());
    const std::pair<std::string, std::string> expected{"compute-cfg-reject",
                                                       "pc=5 reason=wave64-ambiguous-mask-read"};
    EXPECT_NE(std::find(rejections.begin(), rejections.end(), expected), rejections.end())
        << testing::PrintToString(rejections);
}

// A one-dword read is only as good as that dword's own proof. Without the republishing write
// VCC_LO itself is a mask half on one edge and scalar on the other, and nothing may pick a domain.
// Mutation: skip the ambiguity check for B32 reads. The later operand lowering still refuses
// this fixture, so assert the originating CFG guard rather than just an empty module.
TEST(Wave64Vop3B32VccRead, AmbiguousVccLoItselfStaysRefused) {
    Rejections rejections;
    EXPECT_TRUE(compile(program(kBfeVccLo0, kBfeVccLo1, false), &rejections).empty());
    const std::pair<std::string, std::string> expected{"compute-cfg-reject",
                                                       "pc=4 reason=wave64-ambiguous-mask-read"};
    EXPECT_NE(std::find(rejections.begin(), rejections.end(), expected), rejections.end())
        << testing::PrintToString(rejections);
}
