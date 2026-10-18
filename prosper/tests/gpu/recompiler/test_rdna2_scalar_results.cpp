// Pins the newly described VALU scalar writes without making existing scalar-ALU queries see
// them. Confusing VDST with an SGPR or writing VCC for CMPX loses a real state word at a join.
#include "gpu/recompiler/rdna2_instruction_effects.hpp"

#include "gpu/recompiler/ir/effects/operand_effects.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <set>

using namespace prosper::gpu;
namespace {
using S = Rdna2ScalarDestination;

void result(Rdna2Format fmt, uint32_t op, S destination, uint8_t words, bool mask) {
    const auto& e = rdna2_instruction_effects(fmt, op);
    ASSERT_EQ(e.status, Rdna2EffectsStatus::Valu);
    EXPECT_EQ(e.scalar_result.destination, destination) << e.name;
    EXPECT_EQ(e.scalar_result.dwords, words) << e.name;
    EXPECT_EQ(e.scalar_result.lane_mask, mask) << e.name;
    // The old scalar-ALU contract is deliberately unchanged, even for these newly pinned writes.
    EXPECT_FALSE(e.known());
    EXPECT_EQ(e.sdst_dwords, 0);
    EXPECT_EQ(e.implicit_writes, 0);
}
}   // namespace

TEST(Rdna2ScalarResults, VopcAndCmpxHaveDifferentDestinations) {
    unsigned count = 0;
    for (const auto& row : rdna2_instruction_effects_rows()) {
        if (row.fmt != Rdna2Format::VOPC || row.effects.status != Rdna2EffectsStatus::Valu)
            continue;
        ++count;
        // ISA 70648 3.9 / 12.9: CMP writes decoded VCC/SDST, CMPX writes EXEC only on gfx10.
        result(row.fmt, row.opcode, (row.opcode & 0x10) ? S::Exec : S::Dst, 2, true);
    }
    EXPECT_EQ(count, 190u);
}

TEST(Rdna2ScalarResults, Vop3bWritesItsSeparateSdst) {
    // ISA 70648 6.2.3 / 12.12: Wave64 pair, Wave32 word; neither result is the VGPR VDST.
    for (uint32_t op :
         {0x128u, 0x129u, 0x12au, 0x16du, 0x16eu, 0x176u, 0x177u, 0x30fu, 0x310u, 0x319u})
        result(Rdna2Format::VOP3, op, S::Sdst, 2, true);
    for (uint32_t op : {0x28u, 0x29u, 0x2au}) result(Rdna2Format::VOP2, op, S::Vcc, 2, true);
}

TEST(Rdna2ScalarResults, LaneReadsWriteOneScalarWord) {
    result(Rdna2Format::VOP1, 2, S::Dst, 1, false);
    result(Rdna2Format::VOP3, 0x360, S::Dst, 1, false);
    EXPECT_EQ(rdna2_instruction_effects(Rdna2Format::VOP1, 2).vdst_dwords, 0);
    EXPECT_EQ(rdna2_instruction_effects(Rdna2Format::VOP3, 0x360).vdst_dwords, 0);
}

TEST(Rdna2ScalarResults, OtherValuRowsNeverAcquireAScalarDestination) {
    const std::set<uint32_t> carry{0x128, 0x129, 0x12a, 0x16d, 0x16e,
                                   0x176, 0x177, 0x30f, 0x310, 0x319};
    for (const auto& row : rdna2_instruction_effects_rows()) {
        if (row.effects.status != Rdna2EffectsStatus::Valu || row.fmt == Rdna2Format::VOPC)
            continue;
        if (row.fmt == Rdna2Format::VOP3 && (carry.count(row.opcode) || row.opcode == 0x360))
            continue;
        if (row.fmt == Rdna2Format::VOP1 && row.opcode == 2) continue;
        if (row.fmt == Rdna2Format::VOP2 && row.opcode >= 0x28 && row.opcode <= 0x2a) continue;
        result(row.fmt, row.opcode, S::None, 0, false);
    }
}

TEST(Rdna2ScalarResults, DecoderFieldsDistinguishVdstSdstAndImplicitVcc) {
    const uint32_t cmp32[] = {0x7c040102u};   // v_cmp_eq_f32 vcc, v2, v0
    const uint32_t cmp64[] = {0xd4020004u, 258u | (259u << 9)};
    const auto compact = rdna2_decode_one(cmp32, 1);
    const auto extended = rdna2_decode_one(cmp64, 2);
    ASSERT_EQ(compact.fmt, Rdna2Format::VOPC);
    ASSERT_EQ(extended.fmt, Rdna2Format::VOPC);
    EXPECT_EQ(compact.dst.value, 106);
    EXPECT_EQ(extended.dst.value, 4);
    result(compact.fmt, compact.opcode, S::Dst, 2, true);
    result(extended.fmt, extended.opcode, S::Dst, 2, true);

    const uint32_t carry[] = {0xd70f0401u, 258u | (259u << 9)};
    const auto co = rdna2_decode_one(carry, 2);
    ASSERT_EQ(co.fmt, Rdna2Format::VOP3);
    EXPECT_EQ(co.dst.value, 1);
    EXPECT_EQ(co.sdst.value, 4);
    result(co.fmt, co.opcode, S::Sdst, 2, true);

    const uint32_t lane[] = {0x7e080502u};   // v_readfirstlane_b32 s4, v2
    const auto read = rdna2_decode_one(lane, 1);
    ASSERT_EQ(read.fmt, Rdna2Format::VOP1);
    EXPECT_EQ(read.dst.value, 4);
    EXPECT_EQ(read.dst.kind, OperandKind::VGPR);   // storage kind alone is not the write file
    result(read.fmt, read.opcode, S::Dst, 1, false);
}
