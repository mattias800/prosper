// test_bcnt_source_width: V_BCNT_U32_B32 reads ONE scalar dword per source (#5146, #5039).
//
// `D.u32 = CountOneBits(S0.u32) + S1.u32`: two 32-bit operands (RDNA2 ISA reference, document
// 70648). `scalar_alu_source_words` had no entry for it, so the VOP3 fail-closed default charged
// an SGPR source the whole PAIR. The population-count idiom over a B64 mask counts each half
// separately:
//
//     v_cmp_eq_u32_e64 s[8:9], 0, v4      a fresh B64 predicate
//     v_bcnt_u32_b32   v7, s8, 0          low half
//     v_bcnt_u32_b32   v7, s9, v7         high half -- charged s9 AND s10
//     v_cmp_eq_u32     vcc, 1, v7         a fresh predicate over that count
//
// The phantom s10 is a never-written SGPR, so v7 carried a "fabricated" mark and the fresh-mask
// guard (#4943) refused the compare: `fabricated-fresh-mask-source` at pc15 of
// tests/gpu/execute/test_fragment_partial_wave_exec.cpp's fixture, which held main's CI red.
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <iterator>
#include <vector>

using namespace prosper::gpu;

namespace {

// v_bcnt_u32_b32 v7, s9, v7 (the live high-half word of the fixture).
constexpr uint32_t kBcntS9V7_0 = 0xd7640007u;
constexpr uint32_t kBcntS9V7_1 = 0x00020e09u;

// The fixture's population-count prefix, then an export of the count. `high_sgpr` selects the
// SGPR the high-half count reads: 9 is the compare's own high word; 10 is never written.
std::vector<uint32_t> program(uint32_t high_sgpr) {
    return {
        0xbe80047eu,                // s_mov_b64 s[0:1], exec
        0xd7650004u, 0x00010000u,   // v_mbcnt_lo_u32_b32 v4, s0, 0
        0xd7660004u, 0x00020801u,   // v_mbcnt_hi_u32_b32 v4, s1, v4
        0xd4c20008u, 0x00020880u,   // v_cmp_eq_u32_e64 s[8:9], 0, v4
        0xd7640007u, 0x00010008u,   // v_bcnt_u32_b32 v7, s8, 0
        0xd7640007u, 0x00020e00u | high_sgpr,   // v_bcnt_u32_b32 v7, s<high>, v7
        0x7d840e81u,                // v_cmp_eq_u32 vcc, 1, v7
        0xd5010008u, 0x01a90480u,   // v_cndmask_b32_e64 v8, 0, 2, vcc
        0x7e000d08u,                // v_cvt_f32_u32 v0, v8
        0x7e020280u,                // v_mov_b32 v1, 0
        0x7e040280u,                // v_mov_b32 v2, 0
        0x7e0602f2u,                // v_mov_b32 v3, 1.0
        0xf800180fu, 0x03020100u,   // exp mrt0 v0, v1, v2, v3 done vm
        0xbf810000u,                // s_endpgm
    };
}

std::vector<uint32_t> compile(const std::vector<uint32_t>& code) {
    return recompile_fragment(code.data(), code.size());
}

} // namespace

TEST(BcntSourceWidth, EachSourceIsOneDword) {
    const uint32_t code[] = {kBcntS9V7_0, kBcntS9V7_1};
    std::vector<Rdna2Inst> decoded;
    ASSERT_EQ(rdna2_walk(code, std::size(code), decoded), std::size(code));
    ASSERT_EQ(decoded.size(), 1u);
    const Rdna2Inst& bcnt = decoded[0];
    ASSERT_EQ(bcnt.fmt, Rdna2Format::VOP3);
    ASSERT_EQ(bcnt.opcode, 0x364u);
    ASSERT_EQ(bcnt.src[0].kind, OperandKind::SGPR);
    ASSERT_EQ(bcnt.src[0].value, 9);
    EXPECT_EQ(scalar_alu_source_words(bcnt, 0), 1u);
    EXPECT_EQ(scalar_alu_source_words(bcnt, 1), 1u);
}

TEST(BcntSourceWidth, PopulationCountOfAFreshMaskFormsAFreshPredicate) {
    // Both halves of s[8:9] are the compare's genuine words; nothing above s9 is consumed.
    EXPECT_FALSE(compile(program(9)).empty());
}

TEST(BcntSourceWidth, ANeverWrittenSourceStillRefusesTheFreshPredicate) {
    // Control: the same idiom whose high-half count reads s10, which nothing wrote. The guard
    // must still refuse it, so the one-dword width did not switch the fresh-mask proof off.
    EXPECT_TRUE(compile(program(10)).empty());
}
