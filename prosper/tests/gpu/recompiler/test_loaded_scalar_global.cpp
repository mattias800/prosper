// Loaded pointers need a complete readpoint proof, including later control and partial aliases.
#include "gpu/recompiler/rdna2_loaded_scalar_global.hpp"
#include <gtest/gtest.h>
#include <bit>

using namespace prosper::gpu;
namespace {
Rdna2Inst scalar_move(uint32_t reg) {
    Rdna2Inst in;
    in.fmt = Rdna2Format::SOP1;
    in.opcode = kSop1OpcodeMovB32;
    in.dst = {OperandKind::SGPR, int(reg)};
    in.src[0] = {OperandKind::InlineInt, 0};
    in.n_src = 1;
    return in;
}
void number(std::vector<Rdna2Inst>& ins) {
    uint32_t pc = 0;
    for (auto& in : ins) { in.pc = pc; pc += in.len_dwords; }
}
std::vector<Rdna2Inst> program() {
    std::vector<Rdna2Inst> ins(8);
    auto& parent = ins[0];
    parent.fmt = Rdna2Format::SMEM;
    parent.opcode = kSmemOpcodeLoadDwordX2;
    parent.len_dwords = 2;
    parent.dst = {OperandKind::SGPR, 16};
    parent.src[0] = {OperandKind::SGPR, 10};
    parent.src[1] = {OperandKind::Special, 125};
    parent.n_src = 2;
    ins[1].fmt = Rdna2Format::SOPP;
    ins[1].opcode = 0x0c;
    ins[2].fmt = Rdna2Format::VOP2;
    ins[2].opcode = 1;
    ins[2].dst = {OperandKind::VGPR, 9};
    ins[2].src[0] = {OperandKind::InlineInt, 0};
    ins[2].src[1] = {OperandKind::InlineFloat, 242};
    ins[2].n_src = 2;
    ins[3].fmt = Rdna2Format::VOP1;
    ins[3].opcode = 8;
    ins[3].dst = {OperandKind::VGPR, 2};
    ins[3].src[0] = {OperandKind::VGPR, 9};
    ins[3].n_src = 1;
    ins[4].fmt = Rdna2Format::VOP2;
    ins[4].opcode = 0x1a;
    ins[4].dst = {OperandKind::VGPR, 9};
    ins[4].src[0] = {OperandKind::InlineInt, 4};
    ins[4].src[1] = {OperandKind::VGPR, 2};
    ins[4].n_src = 2;
    ins[5].fmt = Rdna2Format::FLAT;
    ins[5].opcode = 0x0e;
    ins[5].flat_segment = 2;
    ins[5].len_dwords = 2;
    ins[5].dst = {OperandKind::VGPR, 8}; // Overlaps address v9; evaluate before writes.
    ins[5].src[0] = {OperandKind::VGPR, 9};
    ins[5].src[1] = {OperandKind::SGPR, 16};
    ins[5].n_src = 2;
    ins[6].fmt = Rdna2Format::SOPP;
    ins[6].opcode = 0x0c;
    ins[7].fmt = Rdna2Format::SOPP;
    ins[7].opcode = 1;
    ins[7].is_end = true;
    number(ins);
    return ins;
}
void refused(const std::vector<Rdna2Inst>& ins) {
    EXPECT_TRUE(rdna2_loaded_scalar_global_reads(ins).empty());
}
} // namespace

TEST(LoadedScalarGlobal, FiniteFloatSelectConversionShiftBoundsOriginalRead) {
    const auto ins = program();
    const auto proof = rdna2_loaded_scalar_global_reads(ins);
    ASSERT_EQ(proof.size(), 1u);
    EXPECT_EQ(proof[0], (LoadedScalarGlobalRead{0, 6, 10, 16, 9, 0, 0, 32}));
}
TEST(LoadedScalarGlobal, WholeDwordSdwaPreservesFiniteDefinition) {
    auto ins = program();
    ins[2].has_sdwa = true;
    ASSERT_EQ(rdna2_loaded_scalar_global_reads(ins).size(), 1u);
    ins[2].sdwa_dst_sel = 4;
    refused(ins);
}
TEST(LoadedScalarGlobal, EntryPointerReuseAfterParentReadIsHarmless) {
    auto ins = program();
    ins.insert(ins.begin() + 2, scalar_move(10));
    number(ins);
    ASSERT_EQ(rdna2_loaded_scalar_global_reads(ins).size(), 1u);
    ins.insert(ins.begin(), scalar_move(11));
    number(ins);
    refused(ins);
}
TEST(LoadedScalarGlobal, LaterLoadedPairOverwriteDoesNotChangeEarlierReadpoint) {
    auto ins = program();
    ins.insert(ins.end() - 1, scalar_move(17));
    number(ins);
    ASSERT_EQ(rdna2_loaded_scalar_global_reads(ins).size(), 1u);
    ins.insert(ins.begin() + 5, scalar_move(17));
    number(ins);
    refused(ins);
}
TEST(LoadedScalarGlobal, KnownNegativeImmediateRebasesAnAlignedWindow) {
    auto ins = program();
    ins[2].src[0] = {OperandKind::InlineInt, 16};
    ins[2].src[1] = {OperandKind::InlineInt, 32};
    ins[3].opcode = 1; // v_mov_b32 keeps integer bit patterns.
    ins[4].src[0].value = 0;
    ins[5].literal = uint32_t(-16);
    auto proof = rdna2_loaded_scalar_global_reads(ins);
    ASSERT_EQ(proof.size(), 1u);
    EXPECT_EQ(proof[0].window_offset, 0u);
    EXPECT_EQ(proof[0].window_bytes, 32u);
    EXPECT_EQ(proof[0].global_offset, -16);
    ins[5].literal = uint32_t(-20);
    refused(ins);
}
TEST(LoadedScalarGlobal, UnknownLaneInputAndClobberCannotSupplyBounds) {
    auto ins = program();
    ins[2].src[0] = {OperandKind::VGPR, 250};
    refused(ins);
    ins = program();
    ins[4].opcode = 3; // Unknown result replaces v9; prior finite value is dead.
    refused(ins);
}
TEST(LoadedScalarGlobal, ExecChangeInvalidatesPreviouslyDefinedLanes) {
    auto ins = program();
    auto change = scalar_move(126);
    ins.insert(ins.begin() + 5, change);
    number(ins);
    refused(ins);
    ins = program();
    ins.insert(ins.begin() + 2, change); // New VALU definitions cover the new active mask.
    number(ins);
    ASSERT_EQ(rdna2_loaded_scalar_global_reads(ins).size(), 1u);
}
TEST(LoadedScalarGlobal, LaterBackedgeAndDebugBranchRefuseWholeBody) {
    auto ins = program();
    Rdna2Inst branch;
    branch.fmt = Rdna2Format::SOPP;
    branch.opcode = kSoppOpcodeBranch;
    branch.simm16 = -7;
    ins.insert(ins.end() - 1, branch);
    number(ins);
    refused(ins);
    ins[ins.size() - 2].opcode = kSoppOpcodeCbranchCdbguser;
    refused(ins);
}
TEST(LoadedScalarGlobal, IndirectControlAndRelativeScalarWritesRefuse) {
    for (uint32_t opcode : {kSop1OpcodeSetpcB64, kSop1OpcodeSwappcB64,
                           kSop1OpcodeRfeB64, kSop1OpcodeMovreldB32,
                           kSop1OpcodeMovreldB64, kSop1OpcodeMovrelsd2B32}) {
        auto ins = program();
        auto uncertain = scalar_move(0);
        uncertain.opcode = opcode;
        ins.insert(ins.end() - 1, uncertain);
        number(ins);
        refused(ins);
    }
}
TEST(LoadedScalarGlobal, LaterGuestWriterAndUnknownEncodingRefuse) {
    auto ins = program();
    auto writer = ins[5];
    writer.opcode = 0x1c;
    ins.insert(ins.end() - 1, writer);
    number(ins);
    refused(ins);
    ins[ins.size() - 2].fmt = Rdna2Format::Unknown;
    refused(ins);
}
TEST(LoadedScalarGlobal, NonfiniteConversionAndOversizedWindowRefuse) {
    auto ins = program();
    ins[2].src[1] = {OperandKind::Literal, 0};
    ins[2].has_literal = true;
    ins[2].literal = 0x7fc00000;
    refused(ins);
    ins = program();
    ins[4].src[0].value = 5; // Addresses {0,32}: 48-byte span exceeds this contract.
    refused(ins);
}
TEST(LoadedScalarGlobal, SlicedOrUnterminatedBodyCannotGrantOriginalProof) {
    auto ins = program();
    ins.erase(ins.begin() + 1); // Keep original PCs so the omitted code is visible.
    refused(ins);
    ins = program();
    ins.back().synthetic_terminator = true;
    refused(ins);
    ins.back().synthetic_terminator = false;
    ins.back().is_end = false;
    refused(ins);
}
TEST(LoadedScalarGlobal, UnknownScalarPairAndSecondaryDestinationAliasesRefuse) {
    auto ins = program();
    auto unknown = scalar_move(9);
    unknown.fmt = Rdna2Format::SOP2;
    unknown.opcode = 0x7f; // Canonical destination inventory conservatively includes s10.
    ins.insert(ins.begin(), unknown);
    number(ins);
    refused(ins);
    ins = program();
    Rdna2Inst secondary;
    secondary.fmt = Rdna2Format::VOP3;
    secondary.opcode = 0x128;
    secondary.dst = {OperandKind::VGPR, 0};
    secondary.sdst = {OperandKind::SGPR, 17};
    ins.insert(ins.begin() + 5, secondary);
    number(ins);
    refused(ins);
}
TEST(LoadedScalarGlobal, GsParameterCacheRequestHasNoGlobalPointerEffect) {
    auto ins = program();
    Rdna2Inst message;
    message.fmt = Rdna2Format::SOPP;
    message.opcode = 0x10;
    message.simm16 = 9;
    ins.insert(ins.begin(), message);
    number(ins);
    ASSERT_EQ(rdna2_loaded_scalar_global_reads(ins).size(), 1u);
    ins[0].simm16 = 6; // Halt Waves is not a hint or a straight-line admission.
    refused(ins);
    ins[0].simm16 = 9;
    ins[0].opcode = 0x11; // SENDMSGHALT is also outside this code proof.
    refused(ins);
}

TEST(LoadedScalarGlobal, ParentMustCompleteBeforeAResultCanBeUsed) {
    auto ins = program();
    ins[1].simm16 = int16_t(0xc07f); // LGKM=0; unrelated VM work need not complete.
    ASSERT_EQ(rdna2_loaded_scalar_global_reads(ins).size(), 1u);
    ins[1].simm16 = 0x017f; // LGKM=1 does not complete this potentially out-of-order load.
    refused(ins);
    ins[1].opcode = 0; // A missing wait cannot establish scalar memory completion.
    refused(ins);
    ins = program();
    ins.insert(ins.begin() + 1, scalar_move(16)); // Even a later full wait is too late.
    number(ins);
    refused(ins);
}

TEST(LoadedScalarGlobal, GlobalMustCompleteBeforeAResultCanBeUsed) {
    auto ins = program();
    ins[6].simm16 = 0x3f70; // VM=0; unrelated LGKM work need not complete.
    ASSERT_EQ(rdna2_loaded_scalar_global_reads(ins).size(), 1u);
    ins[6].simm16 = 0x3f71;
    refused(ins);
    ins[6].simm16 = int16_t(0x7f70); // High VM counter bits are load bearing too.
    refused(ins);
    ins[6].opcode = 0;
    refused(ins);
    ins = program();
    ins.insert(ins.begin() + 6, ins[3]); // A dependent instruction before completion.
    number(ins);
    refused(ins);
}

TEST(LoadedScalarGlobal, HintsCanPrecedeAnImmediateCompletionWait) {
    auto ins = program();
    Rdna2Inst hint;
    hint.fmt = Rdna2Format::SOPP;
    hint.opcode = 0x20;
    ins.insert(ins.begin() + 6, hint);
    ins.insert(ins.begin() + 1, hint);
    number(ins);
    ASSERT_EQ(rdna2_loaded_scalar_global_reads(ins).size(), 1u);
    ins[1].opcode = 0x22; // WAIT_IDLE is not completion authority for either counter.
    refused(ins);
}

TEST(LoadedScalarGlobal, ParentImmediateMustFitTheOwnedReaderOffsetContract) {
    auto ins = program();
    ins[0].literal = 4;
    const auto proof = rdna2_loaded_scalar_global_reads(ins);
    ASSERT_EQ(proof.size(), 1u);
    EXPECT_EQ(proof[0].parent_offset, 4u);
    ins[0].literal = uint32_t(-4); // Signed SMEM immediate is outside the initial owner contract.
    refused(ins);
}
