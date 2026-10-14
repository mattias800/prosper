// These producing compiler tests retain the complete legacy fixture shapes. Missing words or
// malformed packet edits must be fixed in fixture construction while old compiler guards remain.
#include "fixtures/legacy_coverage_fixtures.hpp"
#include "fixtures/compute_coverage_inputs.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

namespace {
using namespace prosper::gpu;
namespace fixtures = prosper::test::coverage_fixtures;
std::vector<uint32_t> compile(const std::vector<uint32_t>& code, const ComputeShaderConfig& config,
                              uint64_t address) {
    const auto resources = fixtures::dispatch_resources();
    return recompile_compute(code.data(), code.size(), &resources, config,
                             {RecompileDiagnosticStage::Compute, address});
}
void check(const std::vector<uint32_t>& code, const ComputeShaderConfig& config, uint64_t address) {
    const auto module = compile(code, config, address);
    EXPECT_FALSE(module.empty()) << last_terminal_reject_reason(address);
}
void reject(const std::vector<uint32_t>& code, const ComputeShaderConfig& config, uint64_t address,
            uint32_t pc, const std::string& reason) {
    TerminalRejectCapture capture;
    EXPECT_TRUE(compile(code, config, address).empty());
    const auto reasons = capture.take();
    bool exact_consumer = false;
    std::string diagnostic;
    for (const auto& [tag, payload] : reasons) {
        diagnostic.append(tag).append(" ").append(payload).append("\n");
        if (payload.find("pc=" + std::to_string(pc) + " ") != std::string::npos &&
            payload.find(reason) != std::string::npos)
            exact_consumer = true;
    }
    EXPECT_TRUE(exact_consumer) << diagnostic;
}
uint32_t largest_dispatch(const std::vector<uint32_t>& module) {
    uint32_t count = 0;
    for (size_t word = 5; word < module.size();) {
        const auto length = module[word] >> 16;
        if (length == 0 || word + length > module.size()) return 0;
        if ((module[word] & 0xffffu) == 251u && length >= 3) // OpSwitch
            count = std::max(count, (length - 3) / 2);
        word += length;
    }
    return count;
}
} // namespace

TEST(LegacyCoverageFixtures, B32AliasHasBothDecodedVectorInputs) {
    const auto code = fixtures::b32_alias();
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code.data(), code.size(), ins);
    ASSERT_GT(ins.size(), 1u);
    const auto& compare = ins[1];
    ASSERT_EQ(compare.pc, fixtures::b32_compare_word);
    ASSERT_EQ(compare.fmt, Rdna2Format::VOPC);
    EXPECT_EQ(compare.opcode, 0xc4u);
    EXPECT_EQ(compare.src[0].kind, OperandKind::VGPR);
    EXPECT_EQ(compare.src[0].value, 0);
    EXPECT_EQ(compare.src[1].kind, OperandKind::VGPR);
    EXPECT_EQ(compare.src[1].value, 1);
    check(code, fixtures::wave32(false), 0x5125000010ull);
}
TEST(LegacyCoverageFixtures, CndmaskHasRealScalarAndComparisonInputs) {
    check(fixtures::cndmask_scalar(), fixtures::branch_wave64(), 0x5125000020ull);
}
TEST(LegacyCoverageFixtures, LshlOrPreservesItsRestoreAndWholePacketBoundaries) {
    const auto code = fixtures::lshl_or_scalar();
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code.data(), code.size(), ins);
    bool restore = false, first_nop = false, second_nop = false;
    for (const auto& in : ins) {
        if (in.pc == fixtures::lshl_or_tail_word + 12) {
            EXPECT_EQ(in.fmt, Rdna2Format::SOP1);
            EXPECT_EQ(in.opcode, kSop1OpcodeMovB64);
            EXPECT_EQ(in.dst.value, 126);
            EXPECT_EQ(in.src[0].value, 14);
            restore = true;
        }
        if (in.pc == fixtures::lshl_or_tail_word + 13) {
            EXPECT_EQ(in.fmt, Rdna2Format::SOPP);
            EXPECT_EQ(in.opcode, 0u);
            EXPECT_EQ(in.len_dwords, 1u);
            first_nop = true;
        }
        if (in.pc == fixtures::lshl_or_tail_word + 14) {
            EXPECT_EQ(in.fmt, Rdna2Format::SOPP);
            EXPECT_EQ(in.opcode, 0u);
            EXPECT_EQ(in.len_dwords, 1u);
            second_nop = true;
        }
    }
    EXPECT_TRUE(restore && first_nop && second_nop);
    check(code, fixtures::branch_wave64(), 0x5125000030ull);
}
TEST(LegacyCoverageFixtures, TrapBypassHasAnActualInitialScalar) {
    check(fixtures::trap_bypass(), fixtures::wave32(true), 0x5125000040ull);
}
TEST(LegacyCoverageFixtures, NorValuHasTheInputWordItsPrefixConsumes) {
    const auto& code = prosper::test::coverage_inputs::mask_nor;
    TerminalRejectCapture capture;
    const auto module = recompile_valu(code.data(), code.size(), fixtures::nor_input_words, 0);
    const auto reasons = capture.take();
    std::string diagnostic;
    for (const auto& [tag, payload] : reasons)
        diagnostic.append(tag).append(" ").append(payload).append("\n");
    EXPECT_FALSE(module.empty()) << diagnostic;
}

TEST(LegacyCoverageFixtures, MissingB32SecondVectorInputStillRejectsAtTheComparison) {
    auto code = fixtures::b32_alias();
    code[0] = 0xbf800000u;
    reject(code, fixtures::wave32(false), 0x5125000110ull, 1, "fabricated-fresh-mask-source");
}
TEST(LegacyCoverageFixtures, MissingCndmaskComparisonInputStillRejectsAtTheComparison) {
    auto code = fixtures::cndmask_scalar();
    code[0] = 0xbf800000u;
    reject(code, fixtures::branch_wave64(), 0x5125000120ull, 10, "mode=unresolved-operand");
}
TEST(LegacyCoverageFixtures, MissingLshlOrTailInputStillRejectsAtTheComparison) {
    auto code = fixtures::lshl_or_scalar();
    code[0] = 0xbf800000u;
    reject(code, fixtures::branch_wave64(), 0x5125000130ull, fixtures::lshl_or_tail_word + 15,
           "mode=unresolved-operand");
}
TEST(LegacyCoverageFixtures, MissingTrapScalarStillRejectsAtTheBranch) {
    auto code = fixtures::trap_bypass();
    code[0] = 0xbf800000u;
    reject(code, fixtures::wave32(true), 0x5125000140ull, 2, "poisoned-scc");
}
TEST(LegacyCoverageFixtures, ZeroValuInputsStillRejectsNorAtTheComparison) {
    const auto& code = prosper::test::coverage_inputs::mask_nor;
    TerminalRejectCapture capture;
    EXPECT_TRUE(recompile_valu(code.data(), code.size(), 0, 0).empty());
    const auto reasons = capture.take();
    bool comparison_refused = false;
    for (const auto& [tag, payload] : reasons)
        if (tag == "recompile-reject" && payload.find("pc=1 ") != std::string::npos &&
            payload.find("fabricated-fresh-mask-source") != std::string::npos)
            comparison_refused = true;
    EXPECT_TRUE(comparison_refused);
}
TEST(LegacyCoverageFixtures, CndmaskAmbiguousSourceStillRejectsAtTheSamePacket) {
    auto code = fixtures::cndmask_scalar();
    code[fixtures::cndmask_source_word] = 0x00382518u;
    reject(code, fixtures::branch_wave64(), 0x5125000220ull, fixtures::cndmask_source_word - 1,
           "wave64-ambiguous-mask-read");
}
TEST(LegacyCoverageFixtures, LshlOrAmbiguousSourceStillRejectsAtTheSamePacket) {
    auto code = fixtures::lshl_or_scalar();
    code[fixtures::lshl_or_source_word] = 0x01a92009u;
    reject(code, fixtures::branch_wave64(), 0x5125000230ull, fixtures::lshl_or_source_word - 1,
           "wave64-ambiguous-mask-read");
}
TEST(LegacyCoverageFixtures, CndmaskWithoutItsActualBranchInputRejectsAtTheBranch) {
    auto config = fixtures::branch_wave64();
    config.user_sgprs.clear();
    reject(fixtures::cndmask_scalar(), config, 0x5125000240ull, 2, "poisoned-scc");
}
TEST(LegacyCoverageFixtures, LshlOrWithoutItsActualBranchInputRejectsAtTheBranch) {
    auto config = fixtures::branch_wave64();
    config.user_sgprs.clear();
    reject(fixtures::lshl_or_scalar(), config, 0x5125000250ull, 2, "poisoned-scc");
}
TEST(LegacyCoverageFixtures, DynamicBranchInputRetainsBothArmsInsteadOfFoldingTheAmbiguity) {
    for (auto code : {fixtures::cndmask_scalar(), fixtures::lshl_or_scalar()}) {
        std::vector<Rdna2Inst> ins;
        rdna2_walk(code.data(), code.size(), ins);
        ASSERT_GT(ins.size(), 5u);
        const auto& compare = ins[1];
        ASSERT_EQ(compare.pc, 1u);
        EXPECT_EQ(compare.fmt, Rdna2Format::SOPC);
        EXPECT_EQ(compare.opcode, 6u); // s_cmp_eq_u32 s0,0
        EXPECT_EQ(compare.src[0].kind, OperandKind::SGPR);
        EXPECT_EQ(compare.src[0].value, 0);
        EXPECT_EQ(ins[2].pc, 2u);
        EXPECT_EQ(ins[2].opcode, 4u); // s_cbranch_scc0 +3 -> mask arm at pc6
        EXPECT_EQ(ins[2].simm16, 3);
        EXPECT_EQ(ins[5].pc, 5u);
        EXPECT_EQ(ins[5].opcode, 2u); // scalar arm s_branch +1 -> merge at pc7
        EXPECT_EQ(ins[5].simm16, 1);
        auto zero = fixtures::branch_wave64();
        auto one = zero;
        ASSERT_EQ(zero.user_sgprs.size(), 1u);
        one.user_sgprs[0] = 1;
        const auto dynamic_zero = compile(code, zero, 0x5125000300ull);
        const auto dynamic_one = compile(code, one, 0x5125000310ull);
        ASSERT_FALSE(dynamic_zero.empty());
        ASSERT_FALSE(dynamic_one.empty());
        EXPECT_EQ(dynamic_zero, dynamic_one)
            << "push-constant values remain runtime inputs, including values selecting either arm";
        // Matched compiling control: an inline value really folds the branch. Both source forms
        // are genuine, but only the runtime form keeps the original mask/scalar join observable.
        code.insert(code.begin() + 1, 0xbe800380u);   // s_mov_b32 s0,0 before the compare
        const auto folded = compile(code, zero, 0x5125000320ull);
        ASSERT_FALSE(folded.empty());
        EXPECT_GT(largest_dispatch(dynamic_zero), largest_dispatch(folded));
    }
}
