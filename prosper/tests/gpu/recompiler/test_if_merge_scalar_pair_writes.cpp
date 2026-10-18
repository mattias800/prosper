// #5436: a 64-bit scalar-data SOP2 result written in a structured if arm must be phi'd at the
// merge in BOTH words. The structured emitter's phi scan charged the pair only for s_cselect_b64,
// s_lshl_b64 and s_lshr_b64, so s_bfe_u64, s_ashr_i64 and s_bfm_b64 left their high SGPR to the
// arm's SSA value, which does not dominate the merge: a read of it after the merge took the arm's
// value on the edge that never ran it.
//
// Each program runs the production emitter (emit_body) and executes the emitted module per lane
// on the closed SPIR-V interpreter, which is a value oracle, not a live title or GPU claim. Every
// lane has its own sources and its own pre-branch s[6:7], and the expected values are computed
// here from the ISA's pseudo-code (RDNA2 ISA 70648, SOP2), not from the recompiler's helpers.
#include "bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

using namespace prosper::gpu;

namespace {
constexpr uint32_t kBfeU64 = 0x29, kAshrI64 = 0x23, kBfmB64 = 0x25;
constexpr uint32_t kEntry6 = 0x600dcafeu, kEntry7 = 0x7ee1f00du;

// SOP2 `op sdst, ssrc0, ssrc1` with SGPR operands.
uint32_t sop2(uint32_t op, uint32_t sdst, uint32_t ssrc0, uint32_t ssrc1) {
    return 0x80000000u | (op << 23) | (sdst << 16) | (ssrc1 << 8) | ssrc0;
}

// The pair op under test into s[6:7]. s_bfe_u64 and s_ashr_i64 read the data pair s[4:5] and a
// 32-bit control; s_bfm_b64 reads two 32-bit operands (width, offset).
uint32_t pair_op(uint32_t op, uint32_t control, uint32_t other) {
    return op == kBfmB64 ? sop2(op, 6, control, other) : sop2(op, 6, 4, control);
}

// The per-lane launch values the program starts from (set in compile() below).
struct LaneValues {
    uint32_t s4, s5, s8, s9, s6, s7;
};
LaneValues lane_values(uint32_t lane) {
    return {lane ^ 0x89abcdefu,
            (lane << 26) ^ 0x70e1d2c3u,              // both signs of the high word
            lane | (((lane ^ 0x2au) | 1u) << 16),     // bfe: offset = lane, width odd, nonzero
            (lane ^ 0x1fu) | ((lane | 1u) << 16),     // the else arm's control
            lane ^ kEntry6,
            lane ^ kEntry7};
}

// RDNA2 ISA 70648 SOP2 pseudo-code.
uint64_t reference(uint32_t op, uint32_t control, uint32_t other, const LaneValues& v) {
    const uint64_t data = (uint64_t(v.s5) << 32) | v.s4;
    switch (op) {
        case kBfeU64: {   // D.u64 = (S0.u64 >> S1.u[5:0]) & ((1 << S1.u[22:16]) - 1)
            const uint32_t offset = control & 63u, width = (control >> 16) & 127u;
            const uint64_t mask = width >= 64 ? ~0ull : (1ull << width) - 1ull;
            return (data >> offset) & mask;
        }
        case kAshrI64:   // D.i64 = signext(S0.i64) >> S1.u[5:0]
            return uint64_t(int64_t(data) >> (control & 63u));
        case kBfmB64:   // D.u64 = ((1ULL << S0.u[5:0]) - 1) << S1.u[5:0]
            return ((1ull << (control & 63u)) - 1ull) << (other & 63u);
    }
    ADD_FAILURE() << "no reference for op " << op;
    return 0;
}

// s_cmp_eq_u32 s0, 0 | s_cbranch_scc1 <past the then arm> | then arm [| s_branch | else arm]
// | s_mov_b32 s11, s6 | s_mov_b32 s10, s7 | s_endpgm. The arm runs when s0 != 0; the reads after
// the merge are guest instructions, so they see exactly what the merge left in s6 and s7.
std::vector<uint32_t> program(uint32_t op, bool has_else) {
    std::vector<uint32_t> code{0xbf068000u, has_else ? 0xbf850002u : 0xbf850001u,
                               pair_op(op, 8, 9)};
    if (has_else) code.insert(code.end(), {0xbf820001u, pair_op(op, 9, 8)});
    code.insert(code.end(), {0xbe8b0306u, 0xbe8a0307u, 0xbf810000u});
    return code;
}

std::vector<uint32_t> compile(const std::vector<uint32_t>& code, uint32_t selector) {
    std::vector<Rdna2Inst> ins;
    EXPECT_EQ(rdna2_walk(code.data(), code.size(), ins), code.size());
    SpirvCompute b;
    b.diagnostic = {RecompileDiagnosticStage::Compute, 0x54360001u};
    b.begin(4u, nullptr, 64u, 1u, 1u, 64u, 0u, true, true);
    RegState state;
    state.exec = b.btrue();
    state.vcc = b.bfalse();
    state.scc = b.bfalse();
    state.absent_sgpr_reads_fabricated_zero = true;
    const uint32_t lane = b.linear_localid;
    const auto shl = [&](uint32_t value, uint32_t by) {
        return b.ibin(Op_ShiftLeftLogical, value, b.uconst(by));
    };
    const auto bxor = [&](uint32_t value, uint32_t k) {
        return b.ibin(Op_BitwiseXor, value, b.uconst(k));
    };
    const auto bor = [&](uint32_t x, uint32_t y) { return b.ibin(Op_BitwiseOr, x, y); };
    state.sreg[0] = b.uconst(selector);
    state.sreg[4] = bxor(lane, 0x89abcdefu);
    state.sreg[5] = bxor(shl(lane, 26), 0x70e1d2c3u);
    state.sreg[8] = bor(lane, shl(bor(bxor(lane, 0x2au), b.uconst(1)), 16));
    state.sreg[9] = bor(bxor(lane, 0x1fu), shl(bor(lane, b.uconst(1)), 16));
    state.sreg[6] = bxor(lane, kEntry6);
    state.sreg[7] = bxor(lane, kEntry7);
    if (!emit_body(
            b, state, ins, {}, nullptr, true, false,
            [](RegState&, const Rdna2Inst&) { return false; }, code.data(), code.size(), nullptr,
            false))
        return {};
    b.store_output_word(sreg_seed(b, state, 11), 2u, 0u);
    b.store_output_word(sreg_seed(b, state, 10), 2u, 1u);
    return b.finish();
}

// selector != 0 runs the then arm; with an else, selector == 0 runs the else arm, and without
// one the pre-branch s[6:7] must survive.
void expect_merge(uint32_t op, bool has_else, uint32_t selector) {
    SCOPED_TRACE(std::string("op 0x") + std::to_string(op) + (has_else ? " if/else" : " if") +
                 " selector " + std::to_string(selector));
    const auto module = compile(program(op, has_else), selector);
    ASSERT_FALSE(module.empty()) << "the structured emitter refused the program";
    bpermute_oracle::Interpreter vm(module);
    const auto actual = vm.run_buffers(64u, {{1u, std::vector<uint32_t>(128u, 0xdeadbeefu)}}, 1u);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_EQ(actual.size(), 128u);
    for (uint32_t lane = 0; lane < 64; ++lane) {
        const LaneValues v = lane_values(lane);
        uint64_t expected = (uint64_t(v.s7) << 32) | v.s6;
        if (selector)
            expected = reference(op, v.s8, v.s9, v);
        else if (has_else)
            expected = reference(op, v.s9, v.s8, v);
        EXPECT_EQ(actual[size_t(2) * lane], uint32_t(expected)) << "low word, lane " << lane;
        EXPECT_EQ(actual[size_t(2) * lane + 1], uint32_t(expected >> 32))
            << "high word, lane " << lane;
    }
}

class IfMergeScalarPairWrites : public testing::TestWithParam<uint32_t> {};

TEST_P(IfMergeScalarPairWrites, SkippedArmKeepsThePreBranchPair) {
    expect_merge(GetParam(), false, 0u);
}

TEST_P(IfMergeScalarPairWrites, TakenArmPublishesBothWords) {
    expect_merge(GetParam(), false, 1u);
}

TEST_P(IfMergeScalarPairWrites, IfElseMergesBothWordsFromEachArm) {
    expect_merge(GetParam(), true, 0u);
    expect_merge(GetParam(), true, 1u);
}

INSTANTIATE_TEST_SUITE_P(Sop2PairResults, IfMergeScalarPairWrites,
                         testing::Values(kBfeU64, kAshrI64, kBfmB64),
                         [](const testing::TestParamInfo<uint32_t>& info) {
                             return info.param == kBfeU64    ? std::string("s_bfe_u64")
                                    : info.param == kAshrI64 ? std::string("s_ashr_i64")
                                                             : std::string("s_bfm_b64");
                         });
}   // namespace
