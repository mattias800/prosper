// #4884: B64 logical masks accept real trailing literal dwords without sign-extending bit 31.
// Execute production ALU output on a closed CPU oracle; missing/fabricated peers still refuse.
#include "bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

using namespace prosper::gpu;

namespace {
constexpr uint64_t kPeer = 0x55555555aaaaaaaaull;

Rdna2Inst instruction(uint32_t opcode, uint32_t value, bool first = false, uint32_t peer = 0u,
                      bool literal = true) {
    const uint32_t immediate = literal ? 255u : value;
    const std::vector<uint32_t> words{0x80000000u | (opcode << 23u) | (106u << 16u) |
                                          ((first ? peer : immediate) << 8u) |
                                          (first ? immediate : peer),
                                      value};
    std::vector<Rdna2Inst> decoded;
    EXPECT_EQ(rdna2_walk(words.data(), literal ? 2u : 1u, decoded), literal ? 2u : 1u);
    EXPECT_EQ(decoded.size(), 1u);
    return decoded.empty() ? Rdna2Inst{} : decoded[0];
}

struct Emitted {
    std::vector<uint32_t> module;
    bool ok = false;
    bool scc_poisoned = false;
};

Emitted emit(const Rdna2Inst& in, bool missing_high = false, bool marked = false,
             uint32_t wave = 64u, uint32_t native = 0u) {
    SpirvCompute b;
    static uint64_t next_program = 0x48840000u;
    b.diagnostic = {RecompileDiagnosticStage::Compute, next_program++};
    b.native_subgroup_size = native;
    b.begin(4u, nullptr, 128u, 1u, 1u, wave, 0u, true, true);
    RegState state;
    state.exec = b.btrue();
    state.vcc = b.bfalse();
    state.scc = b.btrue();
    state.absent_sgpr_reads_fabricated_zero = true;
    state.sreg[0] = b.uconst(static_cast<uint32_t>(kPeer));
    if (!missing_high) state.sreg[1] = b.uconst(static_cast<uint32_t>(kPeer >> 32u));
    if (marked) state.sreg_merge_placeholder.insert(1);
    Emitted result;
    result.ok = true;
    EXPECT_TRUE(emit_alu(b, state, in, result.ok, true));
    result.scc_poisoned = state.scc == 0;
    if (result.ok) {
        b.store_output_word(b.sel(state.vcc, b.uconst(1u), b.uconst(0u)), 1u, 0u);
        result.module = b.finish();
    }
    return result;
}

uint64_t logical(uint32_t opcode, uint64_t a, uint64_t b) {
    switch (opcode) {
        case 0x0f: return a & b;
        case 0x11: return a | b;
        case 0x13: return a ^ b;
        case 0x15: return a & ~b;
        case 0x17: return a | ~b;
        case 0x19: return ~(a & b);
        case 0x1b: return ~(a | b);
        case 0x1d: return ~(a ^ b);
        default: ADD_FAILURE() << "unexpected logical opcode"; return 0;
    }
}

void expect_bits(const Emitted& result, uint64_t expected, uint32_t wave = 64u) {
    ASSERT_TRUE(result.ok);
    ASSERT_FALSE(result.module.empty());
    EXPECT_TRUE(result.scc_poisoned) << "this change must not invent wave-level SCC";
    bpermute_oracle::Interpreter vm(result.module);
    const auto actual = vm.run_buffers(128u, {{1u, std::vector<uint32_t>(128u, 0xdeadbeefu)}}, 1u);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    std::vector<uint32_t> wanted;
    wanted.reserve(128u);
    for (uint32_t lane = 0; lane < 128; ++lane)
        wanted.push_back(static_cast<uint32_t>((expected >> (lane % wave)) & 1u));
    EXPECT_EQ(actual, wanted);
}

TEST(LiteralB64Mask, DecoderKeepsTheRealTrailingLiteral) {
    const auto in = instruction(0x0f, 0xffu);
    EXPECT_EQ(in.src[0].kind, OperandKind::SGPR);
    EXPECT_EQ(in.src[0].value, 0);
    EXPECT_EQ(in.src[1].kind, OperandKind::Literal);
    EXPECT_TRUE(in.has_literal);
    EXPECT_EQ(in.literal, 0xffu);
    EXPECT_EQ(in.len_dwords, 2u);
}

TEST(LiteralB64Mask, EveryLogicalOperationExecutesTheDecodedLiteral) {
    for (uint32_t op : {0x0fu, 0x11u, 0x13u, 0x15u, 0x17u, 0x19u, 0x1bu, 0x1du}) {
        SCOPED_TRACE(op);
        for (uint32_t value : {0u, 0xffu, 0x80000000u, UINT32_MAX}) {
            SCOPED_TRACE(value);
            expect_bits(emit(instruction(op, value)), logical(op, kPeer, value));
        }
    }
}

TEST(LiteralB64Mask, LiteralInTheFirstOperandKeepsOperandOrder) {
    for (uint32_t op : {0x0fu, 0x11u, 0x13u, 0x15u, 0x17u, 0x19u, 0x1bu, 0x1du}) {
        SCOPED_TRACE(op);
        expect_bits(emit(instruction(op, 0x800000ffu, true)), logical(op, 0x800000ffu, kPeer));
    }
}

TEST(LiteralB64Mask, LiteralBit31DoesNotSetBits32Or63) {
    expect_bits(emit(instruction(0x0f, 0x80000000u, false, 126u)), 0x80000000u);
    expect_bits(emit(instruction(0x0f, UINT32_MAX, false, 126u)), UINT32_MAX);
}

TEST(LiteralB64Mask, InlineMinusOneStillSetsTheEntireWave) {
    // Inline -1 is operand encoding 193, not a trailing literal 0xffffffff.
    const auto in = instruction(0x0f, 193u, false, 126u, false);
    ASSERT_EQ(in.src[1].kind, OperandKind::InlineInt);
    ASSERT_EQ(in.src[1].value, -1);
    expect_bits(emit(in), UINT64_MAX);
}

TEST(LiteralB64Mask, OtherNegativeInlineIntegersRetainTheirHighSignBits) {
    const auto in = instruction(0x0f, 194u, false, 126u, false);   // inline -2
    ASSERT_EQ(in.src[1].kind, OperandKind::InlineInt);
    ASSERT_EQ(in.src[1].value, -2);
    expect_bits(emit(in), UINT64_MAX - 1u);
}

TEST(LiteralB64Mask, ZeroLiteralDoesNotFabricateAMissingPeerWord) {
    EXPECT_FALSE(emit(instruction(0x0f, 0u), true).ok);
    expect_bits(emit(instruction(0x0f, 0u)), 0u);
}

TEST(LiteralB64Mask, PresentButFabricatedPeerWordStillRefuses) {
    EXPECT_FALSE(emit(instruction(0x0f, 0xffu), false, true).ok);
    expect_bits(emit(instruction(0x0f, 0xffu)), kPeer & 0xffu);
}

TEST(LiteralB64Mask, MissingLiteralStorageStillRefuses) {
    auto in = instruction(0x0f, 0xffu);
    in.has_literal = false;
    EXPECT_FALSE(emit(in).ok);
    expect_bits(emit(instruction(0x0f, 0xffu)), kPeer & 0xffu);
}

TEST(LiteralB64Mask, Wave32ProjectsOnlyItsLowWord) {
    for (uint32_t value : {0u, 0xffu, 0x80000000u, UINT32_MAX}) {
        SCOPED_TRACE(value);
        expect_bits(emit(instruction(0x0f, value, false, 126u), false, false, 32u), value, 32u);
    }
}

TEST(LiteralB64Mask, NativeWave64CompilesTheSameLiteralAndStillGuardsPeers) {
    EXPECT_TRUE(emit(instruction(0x0f, 0xffu), false, false, 64u, 64u).ok);
    EXPECT_FALSE(emit(instruction(0x0f, 0xffu), true, false, 64u, 64u).ok);
    EXPECT_FALSE(emit(instruction(0x0f, 0xffu), false, true, 64u, 64u).ok);
}

TEST(LiteralB64Mask, OrdinaryVertexCannotInventAPartialMaskLane) {
    SpirvCompute b;
    b.begin_vertex();
    RegState state;
    state.exec = b.btrue();
    state.vcc = b.bfalse();
    bool ok = true;
    EXPECT_TRUE(emit_alu(b, state, instruction(0x0f, 0xffu, false, 126u), ok, true));
    EXPECT_FALSE(ok);
    // A uniform zero needs no lane identity; it is the paired stage control.
    ok = true;
    EXPECT_TRUE(emit_alu(b, state, instruction(0x0f, 0u, false, 126u), ok, true));
    EXPECT_TRUE(ok);
}

TEST(LiteralB64Mask, NggOneLaneNeedsNoComputeLaneBuiltin) {
    for (uint32_t value : {0xffu, 0x80000000u, UINT32_MAX}) {
        SCOPED_TRACE(value);
        SpirvCompute b;
        b.begin_vertex();
        b.ngg_one_lane = true;
        RegState state;
        state.exec = b.btrue();
        state.vcc = b.bfalse();
        bool ok = true;
        EXPECT_TRUE(emit_alu(b, state, instruction(0x0f, value, false, 126u), ok, true));
        ASSERT_TRUE(ok);
        const auto module = b.finish();
        ASSERT_FALSE(module.empty());
        for (size_t at = 5; at < module.size();) {
            const uint32_t count = module[at] >> 16u;
            ASSERT_GT(count, 0u);
            ASSERT_LE(count, module.size() - at);
            if ((module[at] & 0xffffu) == Op_Decorate && count == 4u &&
                module[at + 2] == Dec_BuiltIn) {
                EXPECT_NE(module[at + 3], BI_LocalInvocationId);
                EXPECT_NE(module[at + 3], 29u);   // SPIR-V LocalInvocationIndex
            }
            at += count;
        }
    }
}

TEST(LiteralB64Mask, FragmentLiteralUsesTheExistingMaskLaneContract) {
    SpirvCompute b;
    b.wave_size = 64u;
    b.begin_fragment();
    RegState state;
    state.exec = b.btrue();
    state.vcc = b.bfalse();
    bool ok = true;
    EXPECT_TRUE(emit_alu(b, state, instruction(0x0f, 0x80000000u, false, 126u), ok, true));
    EXPECT_TRUE(ok);
    EXPECT_FALSE(b.finish().empty());
}
}   // namespace
