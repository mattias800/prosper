// #4774: a structured scalar merge must retain real launch words on skipped edges.
// Execute production emitted PHIs with nonzero inputs so a fabricated zero cannot pass.
// The closed CPU interpreter is a value oracle, not a live title or GPU claim.
#include "bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <string>
#include <vector>

using namespace prosper::gpu;

namespace {
constexpr uint32_t kLo = 0x13579bdfu, kHi = 0x2468ace0u;

std::vector<uint32_t> one_arm(bool loop = false) {
    // s_cmp_eq_u32 s0,0 | s_cbranch_scc1 +1 | s_mov_b32 s12,-1
    std::vector<uint32_t> code{0xbf068000u, 0xbf850001u, 0xbe8c03c1u};
    if (loop) {
        // i=0; while (i<2) ++i. The preceding if is the counted emitter's prelude.
        code.insert(code.end(),
                    {0xb0020002u, 0xbe840380u, 0xbf0a0204u, 0xbf840002u, 0x81048104u, 0xbf82fffcu});
    }
    code.push_back(0xbf810000u);
    return code;
}

struct Compiled {
    std::vector<uint32_t> module;
    bool lo_fabricated = false, hi_fabricated = false;
};

Compiled compile(const std::vector<uint32_t>& code, uint32_t selector, bool absent = false,
                 bool marked = false) {
    std::vector<Rdna2Inst> ins;
    EXPECT_EQ(rdna2_walk(code.data(), code.size(), ins), code.size());
    SpirvCompute b;
    b.diagnostic = {RecompileDiagnosticStage::Compute, 0x47740001u};
    b.begin(4u, nullptr, 64u, 1u, 1u, 64u, 0u, true, true);
    RegState state;
    state.exec = b.btrue();
    state.vcc = b.bfalse();
    state.scc = b.bfalse();
    state.absent_sgpr_reads_fabricated_zero = true;
    state.sreg[0] = b.uconst(selector);
    if (!absent) state.sreg_input[12] = b.uconst(kLo);
    state.sreg_input[13] = b.uconst(kHi);
    if (marked) state.sreg_merge_placeholder.insert(12);
    Compiled result;
    if (emit_body(
            b, state, ins, {}, nullptr, true, false,
            [](RegState&, const Rdna2Inst&) { return false; }, code.data(), code.size(), nullptr,
            false)) {
        result.lo_fabricated = sreg_word_may_be_fabricated(state, 12);
        result.hi_fabricated = sreg_word_may_be_fabricated(state, 13);
        b.store_output_word(sreg_seed(b, state, 12), 2u, 0u);
        b.store_output_word(sreg_seed(b, state, 13), 2u, 1u);
        result.module = b.finish();
    }
    return result;
}

bool has_op(const std::vector<uint32_t>& module, uint32_t op) {
    for (size_t at = 5; at < module.size();) {
        const uint32_t count = module[at] >> 16u;
        if (!count || count > module.size() - at) return false;
        if ((module[at] & 0xffffu) == op) return true;
        at += count;
    }
    return false;
}

void expect_values(const Compiled& result, uint32_t lo, uint32_t hi, bool loop = false,
                   bool phi = true) {
    ASSERT_FALSE(result.module.empty());
    EXPECT_EQ(has_op(result.module, Op_Phi), phi);
    EXPECT_FALSE(has_op(result.module, Op_Switch)) << "this must exercise structured PHIs";
    EXPECT_EQ(has_op(result.module, Op_LoopMerge), loop);
    bpermute_oracle::Interpreter vm(result.module);
    const auto actual = vm.run_buffers(64u, {{1u, std::vector<uint32_t>(128u, 0xdeadbeefu)}}, 1u);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    std::vector<uint32_t> expected;
    for (uint32_t lane = 0; lane < 64; ++lane) {
        expected.push_back(lo);
        expected.push_back(hi);
    }
    EXPECT_EQ(actual, expected);
}

TEST(DescriptorIfMerge, SkippedArmRetainsBothGenuineLaunchWords) {
    const auto result = compile(one_arm(), 0u);
    expect_values(result, kLo, kHi);
    EXPECT_FALSE(result.lo_fabricated);
    EXPECT_FALSE(result.hi_fabricated);
}

TEST(DescriptorIfMerge, TakenArmRetainsTheActualOverwriteAndUntouchedWord) {
    const auto result = compile(one_arm(), 1u);
    expect_values(result, UINT32_MAX, kHi);
    EXPECT_FALSE(result.lo_fabricated);
    EXPECT_FALSE(result.hi_fabricated);
}

TEST(DescriptorIfMerge, EqualIncomingValuesSurviveLaunchAliasRetirement) {
    auto code = one_arm();
    code[2] = 0xbe8c030cu;   // s_mov_b32 s12,s12: both incoming SSA IDs are identical
    // The general one-arm path deliberately still emits an equal-value PHI.
    expect_values(compile(code, 0u), kLo, kHi);
    expect_values(compile(code, 1u), kLo, kHi);
}

TEST(DescriptorIfMerge, CountedPreludeRetainsEqualValuesWhenItsPhiIsElided) {
    auto code = one_arm(true);
    code[2] = 0xbe8c030cu;
    // The loop counter has PHIs; the equal-valued descriptor prelude does not.
    expect_values(compile(code, 0u), kLo, kHi, true);
    expect_values(compile(code, 1u), kLo, kHi, true);
}

TEST(DescriptorIfMerge, BothArmSelfMovesRetainEqualValuesWithoutAnyPhi) {
    const std::vector<uint32_t> code{0xbf068000u, 0xbf850002u, 0xbe8c030cu,
                                     0xbf820001u, 0xbe8d030du, 0xbf810000u};
    expect_values(compile(code, 0u), kLo, kHi, false, false);
    expect_values(compile(code, 1u), kLo, kHi, false, false);
}

TEST(DescriptorIfMerge, BothArmMergePreservesEachSkippedDescriptorWord) {
    const std::vector<uint32_t> code{0xbf068000u, 0xbf850002u, 0xbe8c03c1u,
                                     0xbf820001u, 0xbe8d03c2u, 0xbf810000u};
    expect_values(compile(code, 0u), kLo, UINT32_MAX - 1u);
    expect_values(compile(code, 1u), UINT32_MAX, kHi);
}

TEST(DescriptorIfMerge, CountedLoopPreludeRetainsTheSkippedLaunchWord) {
    const auto result = compile(one_arm(true), 0u);
    expect_values(result, kLo, kHi, true);
    EXPECT_FALSE(result.lo_fabricated);
}

TEST(DescriptorIfMerge, CountedLoopPreludeRetainsTheTakenOverwrite) {
    expect_values(compile(one_arm(true), 1u), UINT32_MAX, kHi, true);
}

TEST(DescriptorIfMerge, MissingLaunchWordStillMarksTheMergeAsFabricated) {
    const auto result = compile(one_arm(), 0u, true);
    expect_values(result, 0u, kHi);
    EXPECT_TRUE(result.lo_fabricated);
    EXPECT_FALSE(result.hi_fabricated);
}

TEST(DescriptorIfMerge, AnExplicitMarkWinsOverLaunchWordPresence) {
    const auto result = compile(one_arm(), 0u, false, true);
    expect_values(result, kLo, kHi);
    EXPECT_TRUE(result.lo_fabricated);
    EXPECT_FALSE(result.hi_fabricated);
}

TEST(DescriptorIfMerge, ScalarOverwriteTakesPrecedenceOverLaunchInput) {
    SpirvCompute b;
    b.begin(4u);
    RegState state;
    const auto input = b.uconst(kLo), overwrite = b.uconst(42u);
    state.sreg_input[12] = input;
    EXPECT_EQ(sreg_seed(b, state, 12), input);
    state.sreg[12] = overwrite;
    EXPECT_EQ(sreg_seed(b, state, 12), overwrite);
    EXPECT_EQ(sreg_seed(b, state, 14), b.uconst(0u));
    EXPECT_TRUE(merge_edge_word_fabricated(state, 14));
}

TEST(DescriptorIfMerge, ExpiredLaunchAliasCannotResurrectAnUnrepresentedOverwrite) {
    SpirvCompute b;
    b.begin(4u);
    RegState state;
    state.absent_sgpr_reads_fabricated_zero = true;
    state.sreg_input[12] = b.uconst(kLo);
    const uint32_t code[]{0xbe8c03c1u};
    std::vector<Rdna2Inst> ins;
    ASSERT_EQ(rdna2_walk(code, std::size(code), ins), std::size(code));
    ASSERT_EQ(ins.size(), 1u);
    record_scalar_write(state, ins[0], false, snapshot_saved_b64_masks(state, ins[0]));
    state.sreg.erase(12);   // an unrepresented new value cannot reuse the retired input
    EXPECT_EQ(sreg_seed(b, state, 12), b.uconst(0u));
    EXPECT_FALSE(state.sreg_input.count(12));
    EXPECT_TRUE(merge_edge_word_fabricated(state, 12));
}

TEST(DescriptorIfMerge, CpuOracleExecutesTheLoopsIntegerNotAndRejectsBooleanOperands) {
    SpirvCompute b;
    b.begin(4u, nullptr, 64u, 1u, 1u, 64u, 0u, true, true);
    b.store_output_word(b.iun(Op_Not, b.uconst(kLo)), 2u, 0u);
    b.store_output_word(b.iun(Op_Not, b.uconst(kHi)), 2u, 1u);
    expect_values({b.finish()}, ~kLo, ~kHi, false, false);

    SpirvCompute invalid;
    invalid.begin(4u, nullptr, 64u, 1u, 1u, 64u, 0u, true, true);
    invalid.store_output_word(invalid.iun(Op_Not, invalid.btrue()), 1u, 0u);
    bpermute_oracle::Interpreter vm(invalid.finish());
    vm.run_buffers(64u, {{1u, std::vector<uint32_t>(64u, 0u)}}, 1u);
    EXPECT_EQ(vm.error, "Not operand type mismatch");
}
}   // namespace
