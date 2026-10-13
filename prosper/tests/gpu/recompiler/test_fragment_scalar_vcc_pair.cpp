// Wave64 fragment scalar CSELECTs write physical VCC words. A later live mask must use both
// genuine words and the producing SCC, rather than requiring a stale Bool view or inventing one.
#include "fixtures/fragment_compare_kill.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <gtest/gtest.h>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace prosper::gpu;

std::vector<uint32_t> program() {
    return {
        0xbe840381u,              // s4 = 1
        0xbf0d8004u,              // SCC = s4 != 0
        0x856b80c0u,              // VCC_HI = SCC ? 64 : 0
        0xbf0d8104u,              // SCC = s4 != 1
        0x856a80ffu, 0x00000080u, // VCC_LO = SCC ? 128 : 0
        0x886a6a6bu,              // VCC_LO = VCC_HI | VCC_LO
        0x7e080c6au,              // observe the physical scalar word
        0x02020100u,              // v1 = implicit VCC ? v1 : v0
        0x7e000280u, 0xbf800000u, 0x7e040280u, 0x7e060280u, 0xf800000fu, 0x03020100u, 0xbf810000u,
    };
}

struct Compilation {
    std::vector<uint32_t> module;
    std::string reason;
};

Compilation compile(const std::vector<uint32_t>& code, uint64_t address) {
    const auto inputs = prosper::test::fragment_compare_kill::position_inputs();
    auto module = recompile_fragment(code.data(), code.size(), nullptr, &inputs, UINT32_MAX,
                                     nullptr, false, {RecompileDiagnosticStage::Fragment, address});
    return {std::move(module), last_terminal_reject_reason(address)};
}

void rejected_at(const Compilation& result, uint32_t pc) {
    EXPECT_TRUE(result.module.empty()) << result.reason;
    EXPECT_NE(result.reason.find("pc=" + std::to_string(pc) + " "), std::string::npos)
        << "the missing source must be rejected at its actual consumer: " << result.reason;
}
} // namespace

TEST(FragmentScalarVccPair, GenuinePhysicalWordsRetainTheLiveLaneIdentityContract) {
    const auto result = compile(program(), 0x27610001u);
    ASSERT_FALSE(result.module.empty()) << result.reason;
    EXPECT_EQ(fragment_spirv_required_subgroup_size(result.module), 64u);
    EXPECT_TRUE(fragment_spirv_required_subgroup_reasons(result.module) &
                kFragmentWaveReasonLaneId);
}

TEST(FragmentScalarVccPair, CompletedSelectPublishesItsMaskWithoutALaterLogicalWrite) {
    auto code = program();
    code[6] = 0xbf800000u;
    const auto result = compile(code, 0x27610005u);
    ASSERT_FALSE(result.module.empty()) << result.reason;
    EXPECT_EQ(fragment_spirv_required_subgroup_size(result.module), 64u);
    EXPECT_TRUE(fragment_spirv_required_subgroup_reasons(result.module) &
                kFragmentWaveReasonLaneId);
}

TEST(FragmentScalarVccPair, GenuineArithmeticSccProducerAlsoSuppliesTheSelect) {
    auto code = program();
    code[1] =
        0x80048104u; // s_add_u32 s4,s4,1 produces actual SCC, rather than a compare-only proof.
    const auto result = compile(code, 0x27610006u);
    ASSERT_FALSE(result.module.empty()) << result.reason;
    EXPECT_EQ(fragment_spirv_required_subgroup_size(result.module), 64u);
}

TEST(FragmentScalarVccPair, MissingHighWordCannotCreateTheLivePair) {
    auto code = program();
    code[2] = 0xbf800000u;
    const auto result = compile(code, 0x27610002u);
    rejected_at(result, 4);
}

TEST(FragmentScalarVccPair, MissingScalarConditionCannotCreateTheLivePair) {
    auto code = program();
    code[1] = code[3] = 0xbf800000u;
    const auto result = compile(code, 0x27610003u);
    rejected_at(result, 4);
}

TEST(FragmentScalarVccPair, FabricatedScalarRelayCannotCreateTheLivePair) {
    auto code = program();
    code[4] = 0x856a8014u; // VCC_LO = SCC ? absent s20 : 0
    code[5] = 0xbf800000u;
    const auto result = compile(code, 0x27610004u);
    rejected_at(result, 4);
}

TEST(FragmentScalarVccPair, ACompareOfAnAbsentWordCannotSupplyTheSelect) {
    auto code = program();
    code[3] = 0xbf0d8014u;   // Fresh SCC depends on absent s20 despite the earlier genuine compare.
    const auto result = compile(code, 0x27610007u);
    rejected_at(result, 4);
}

namespace {
std::vector<uint32_t> both_arm_high_word() {
    auto code = program();
    code.insert(code.begin() + 2, {0xbf850002u, 0x856b80c0u, 0xbf820001u});
    // Both scalar successors now independently assign the physical high word before the low select.
    return code;
}
}   // namespace

TEST(FragmentScalarVccPair, BothScalarPredecessorsSupplyTheHighWord) {
    const auto result = compile(both_arm_high_word(), 0x27610008u);
    ASSERT_FALSE(result.module.empty()) << result.reason;
    EXPECT_EQ(fragment_spirv_required_subgroup_size(result.module), 64u);
}

TEST(FragmentScalarVccPair, OneScalarPredecessorCannotSupplyTheHighWord) {
    auto code = both_arm_high_word();
    code[5] = 0xbf800000u;
    const auto result = compile(code, 0x27610009u);
    rejected_at(result, 7);
}

TEST(FragmentScalarVccPair, RefreshingSccCannotAuthenticateThePreviouslyFabricatedHighWord) {
    auto code = program();
    code[1] = 0xbf800000u;
    const auto result = compile(code, 0x2761000au);
    rejected_at(result, 4);
}

TEST(FragmentScalarVccPair, ALiveBoolCannotAuthenticateItsFabricatedScalarLowWord) {
    const std::vector<uint32_t> genuine{
        0x886a8081u,   // VCC_LO = genuine 1 | 0
        0x886b80c0u,   // VCC_HI = 64 | 0; genuine current SCC
        0x856a806au,   // VCC_LO = SCC ? old physical VCC_LO : 0
        0x02020100u, 0x7e040280u, 0x7e0602f2u, 0xf800180fu, 0x03020100u, 0xbf810000u,
    };
    const auto control = compile(genuine, 0x2761000bu);
    ASSERT_FALSE(control.module.empty()) << control.reason;
    auto fabricated = genuine;
    fabricated[0] = 0x886a80fdu;   // initial stored bfalse SCC is not a produced scalar word.
    const auto result = compile(fabricated, 0x2761000cu);
    rejected_at(result, 2);
}

TEST(FragmentScalarVccPair, ObservableIntermediateMasksCannotAuthenticateAFabricatedLowWord) {
    const std::vector<uint32_t> genuine{
        0x886a8081u, 0x02020100u, 0x886b80c0u, 0x02020100u, 0x856a806au, 0x02020100u,
        0x7e040280u, 0x7e0602f2u, 0xf800180fu, 0x03020100u, 0xbf810000u,
    };
    const auto control = compile(genuine, 0x2761000du);
    ASSERT_FALSE(control.module.empty()) << control.reason;
    auto fabricated = genuine;
    fabricated[0] = 0x886a80fdu;
    const auto result = compile(fabricated, 0x2761000eu);
    rejected_at(result, 4);
}

TEST(FragmentScalarVccPair, ScalarOnlySelectMayReadItsProducedScc) {
    const std::vector<uint32_t> code{
        0xbf0d8081u,   // genuine SCC = 1 != 0
        0x856a80fdu,   // scalar-only VCC_LO = SCC ? architectural SCC : 0
        0x7e02026au, 0x7e000280u, 0x7e040280u, 0x7e0602f2u, 0xf800180fu, 0x03020100u, 0xbf810000u,
    };
    const auto result = compile(code, 0x2761000fu);
    ASSERT_FALSE(result.module.empty()) << result.reason;
    EXPECT_EQ(fragment_spirv_required_subgroup_size(result.module), 0u);
}

TEST(FragmentScalarVccPair, ScalarOnlySelectMayReadTheArchitecturalNullWord) {
    const std::vector<uint32_t> code{
        0xbf0d8081u, 0x856a807du,   // genuine SCC; scalar-only source SGPR_NULL
        0x7e02026au, 0x7e000280u, 0x7e040280u, 0x7e0602f2u, 0xf800180fu, 0x03020100u, 0xbf810000u,
    };
    const auto result = compile(code, 0x27610010u);
    ASSERT_FALSE(result.module.empty()) << result.reason;
    EXPECT_EQ(fragment_spirv_required_subgroup_size(result.module), 0u);
}

TEST(FragmentScalarVccPair, ScalarOnlyCompletePairDoesNotRequireLaneIdentity) {
    const std::vector<uint32_t> code{
        0xbf0d8081u, 0x856b80c0u, 0x856a80ffu, 0x00000080u, 0x7e02026au, 0x7e000280u,
        0x7e040280u, 0x7e0602f2u, 0xf800180fu, 0x03020100u, 0xbf810000u,
    };
    const auto result = compile(code, 0x27610011u);
    ASSERT_FALSE(result.module.empty()) << result.reason;
    EXPECT_EQ(fragment_spirv_required_subgroup_size(result.module), 0u);
}

TEST(FragmentScalarVccPair, ScalarOnlyHighWordSourceDoesNotRequireLaneIdentity) {
    const std::vector<uint32_t> code{
        0xbf0d8081u, 0x856b80c0u, 0x856a806bu, 0x7e02026au, 0x7e000280u,
        0x7e040280u, 0x7e0602f2u, 0xf800180fu, 0x03020100u, 0xbf810000u,
    };
    const auto result = compile(code, 0x27610012u);
    ASSERT_FALSE(result.module.empty()) << result.reason;
    EXPECT_EQ(fragment_spirv_required_subgroup_size(result.module), 0u);
}

TEST(FragmentScalarVccPair, LiveSelectMayReadTheGenuineHighWord) {
    auto code = program();
    code[4] = 0x856a806bu;
    code[5] = 0xbf800000u;
    const auto result = compile(code, 0x27610013u);
    ASSERT_FALSE(result.module.empty()) << result.reason;
    EXPECT_EQ(fragment_spirv_required_subgroup_size(result.module), 64u);
}
