// #4429: a B32 VOP3 that reads the SGPR just below a register-offset descriptor load reads ONE
// dword. Charged the pair, its phantom second word is the descriptor's first, and descriptor
// assembly was reclassified as numeric raw data that needs a backing it can never be given.
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

using namespace prosper::gpu;

namespace {
std::vector<Rdna2Inst> decode(const std::vector<uint32_t>& code) {
    std::vector<Rdna2Inst> result;
    EXPECT_EQ(rdna2_walk(code.data(), code.size(), result), code.size());
    EXPECT_FALSE(result.empty());
    return result;
}

// The Unity vertex-fetch shape, with the live instruction words: a V# selected from a table by a
// runtime VCC_LO offset, one VOP3 in between, then the fetch that consumes the V# as a descriptor.
std::vector<Rdna2Inst> fetch_with(uint32_t vop3_word0, uint32_t vop3_word1, bool literal) {
    std::vector<uint32_t> code{
        0xf408020cu,
        0xd4000000u, // s_load_dwordx4 s[8:11], s[24:25], vcc_lo
        vop3_word0,
        vop3_word1,
    };
    if (literal) code.push_back(0x509502f9u);
    code.insert(code.end(),
                {
                    0xe00c2000u,
                    0x0c020001u, // buffer_load_format_xyzw v[0:3], v1, s[8:11], s12 idxen
                    0xbf810000u,              // s_endpgm
                });
    return decode(code);
}
} // namespace

TEST(RawWideVop3SourceWidth, Med3F32ReadsOneDwordPerSource) {
    const auto instructions = fetch_with(0xd5570010u, 0x04000effu, true);
    const Rdna2Inst& med3 = instructions[1];
    ASSERT_EQ(med3.fmt, Rdna2Format::VOP3);
    ASSERT_EQ(med3.opcode, 0x157u); // v_med3_f32 v16, 0x509502f9, s7, v0
    ASSERT_EQ(med3.src[1].kind, OperandKind::SGPR);
    ASSERT_EQ(med3.src[1].value, 7);
    for (uint32_t source = 0; source < 3; ++source)
        EXPECT_EQ(scalar_alu_source_words(med3, source), 1u) << source;
}

TEST(RawWideVop3SourceWidth, B32NeighbourBelowTheDescriptorStaysDescriptorOnly) {
    // s7 sits directly below the V#'s first word s8. A one-dword read of it observes nothing the
    // load produced, so the established exact-fetch-PC descriptor route stays open.
    const auto instructions = fetch_with(0xd5570010u, 0x04000effu, true);
    ASSERT_EQ(instructions[2].fmt, Rdna2Format::MUBUF);
    ASSERT_EQ(instructions[2].src[1].value, 8);
    EXPECT_TRUE(rdna2_raw_wide_data_loads(instructions).empty());
}

TEST(RawWideVop3SourceWidth, RealObserversOfTheDescriptorStillNeedBacking) {
    // The same v_med3_f32 reading s8 itself is a genuine numeric observer of the loaded word.
    const auto direct = fetch_with(0xd5570010u, 0x040010ffu, true);
    ASSERT_EQ(direct[1].src[1].value, 8);
    EXPECT_EQ(rdna2_raw_wide_data_loads(direct), std::vector<uint32_t>{0u});
    // And a true pair reader below it still reaches s8: v_cndmask_b32 v1, v8, v5, s[7:8] reads its
    // condition as a mask pair, so the fail-closed width is intact for the forms that need it.
    const auto pair = fetch_with(0xd5010001u, 0x001e0b08u, false);
    ASSERT_EQ(pair[1].opcode, 0x101u);
    ASSERT_EQ(pair[1].src[2].value, 7);
    EXPECT_EQ(scalar_alu_source_words(pair[1], 2), 2u);
    EXPECT_EQ(rdna2_raw_wide_data_loads(pair), std::vector<uint32_t>{0u});

    // The fail-closed default itself: v_mad_u64_u32 v1, vcc, v8, v5, s[7:8] is not on the one-dword
    // allowlist and really does read its addend as a pair, so widening the allowlist into "every
    // VOP3 reads one dword" would lose this observer.
    const auto wide = fetch_with(0xd5766a01u, 0x001e0b08u, false);
    ASSERT_EQ(wide[1].fmt, Rdna2Format::VOP3);
    ASSERT_EQ(wide[1].opcode, 0x176u);
    ASSERT_EQ(wide[1].src[2].value, 7);
    EXPECT_EQ(scalar_alu_source_words(wide[1], 2), 2u);
    EXPECT_EQ(rdna2_raw_wide_data_loads(wide), std::vector<uint32_t>{0u});
}
