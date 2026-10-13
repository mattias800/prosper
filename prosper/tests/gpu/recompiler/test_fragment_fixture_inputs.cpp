// Reduced fragment control-flow fixtures must declare the two genuine varying words their
// comparisons consume. Missing/partial mappings remain refusals at the actual compare, while
// usable inputs preserve the NULL-export and saved-ballot dispatcher contracts.
#include "fixtures/fragment_control_flow_inputs.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <iterator>
#include <array>
#include <string>
#include <vector>

using namespace prosper::gpu;
namespace {
constexpr uint32_t null_tail[] = {
    0x7d840080u, 0x8aea6a7eu, 0xbf840004u, 0xbefe046au, 0xf8001890u, 0x00000000u, 0xbf810000u,
};
// Save and spill both EXEC words, cross the non-lexical VCC branch region, then restore
// the same pair. The two coordinate inputs are consumed at PCs 9, 11 and 15.
constexpr uint32_t ballot_spill[] = {
    0xbeac047eu, 0xd7610014u, 0x0001022du, 0xd7610014u, 0x0001002cu, 0x7e040280u, 0x7e060280u,
    0x7e080280u, 0x7e0a0280u, 0x7c020300u, 0xbf860003u, 0x7c020300u, 0xbf860002u, 0x7e040281u,
    0x7e060281u, 0x7c020300u, 0xbf860002u, 0x7e080281u, 0xbf820001u, 0x7e0a0281u, 0xd7600006u,
    0x00010114u, 0xd7600007u, 0x00010314u, 0x7e000206u, 0x7e020207u, 0xbefe0406u, 0x7e040280u,
    0x7e0602f2u, 0xf800000fu, 0x03020100u, 0xbf810000u,
};

bool has_opcode(const std::vector<uint32_t>& words, uint32_t opcode) {
    for (size_t i = 5; i < words.size();) {
        const uint32_t count = words[i] >> 16;
        if (!count || count > words.size() - i) return false;
        if ((words[i] & 0xffffu) == opcode) return true;
        i += count;
    }
    return false;
}

template <size_t N>
std::vector<uint32_t> compile_without_mapping(const uint32_t (&code)[N], uint64_t address,
                                              const PixelSystemInputMapping* inputs = nullptr) {
    return recompile_fragment(code, N, nullptr, inputs, UINT32_MAX, nullptr, false,
                              {RecompileDiagnosticStage::Fragment, address});
}
} // namespace

TEST(FragmentFixtureInputs, NullExportTailRequiresItsComparedInput) {
    const auto module = prosper::test::compile_fragment_control_flow_fixture(
        null_tail, {RecompileDiagnosticStage::Fragment, 0x50630001u});
    ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(0x50630001u);
    EXPECT_TRUE(has_opcode(module, 252)) << "the NULL export retains inactive-fragment discard";
    EXPECT_TRUE(compile_without_mapping(null_tail, 0x50630002u).empty());
    const auto reason = last_terminal_reject_reason(0x50630002u);
    EXPECT_NE(reason.find("fabricated-fresh-mask-source"), std::string::npos) << reason;
    EXPECT_NE(reason.find("pc=0 words=7d840080"), std::string::npos) << reason;
}

TEST(FragmentFixtureInputs, BothComparedInputsKeepTheSavedBallotDispatcherUsable) {
    const auto module = prosper::test::compile_fragment_control_flow_fixture(
        ballot_spill, {RecompileDiagnosticStage::Fragment, 0x50630003u});
    ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(0x50630003u);
    EXPECT_TRUE(has_opcode(module, 251)) << "the spill crosses the CFG dispatcher";
    EXPECT_TRUE(has_opcode(module, 339)) << "the saved EXEC retains physical ballot words";
    EXPECT_EQ(fragment_spirv_required_subgroup_size(module), 64u);
    EXPECT_TRUE(compile_without_mapping(ballot_spill, 0x50630004u).empty());
    const auto reason = last_terminal_reject_reason(0x50630004u);
    EXPECT_NE(reason.find("cfg-recompile-reject mode=unresolved-operand"), std::string::npos)
        << reason;
    EXPECT_NE(reason.find("pc=9 words=7c020300"), std::string::npos) << reason;
}

TEST(FragmentFixtureInputs, OneNamedInputCannotInitializeTheOtherCoordinate) {
    ASSERT_FALSE(prosper::test::compile_fragment_control_flow_fixture(
                     ballot_spill, {RecompileDiagnosticStage::Fragment, 0x50630005u})
                     .empty());
    PixelSystemInputMapping only_x;
    only_x.ena = only_x.addr = 1u << 8;
    EXPECT_TRUE(compile_without_mapping(ballot_spill, 0x50630006u, &only_x).empty());
    const auto reason = last_terminal_reject_reason(0x50630006u);
    EXPECT_NE(reason.find("cfg-recompile-reject mode=unresolved-operand"), std::string::npos)
        << reason;
    EXPECT_NE(reason.find("pc=9 words=7c020300"), std::string::npos) << reason;
}

TEST(FragmentFixtureInputs, DepthInputSuppliesTheVaryingLoopBound) {
    constexpr uint32_t code[] = {
        0xBE800380u, 0x7E020302u, 0x4A020281u, 0x7E021101u, 0x7D020200u, 0xBF860002u,
        0x81008100u, 0xBF82FFFCu, 0x7E0002F2u, 0xF800180Fu, 0x00000000u, 0xBF810000u,
    };
    const auto module = prosper::test::compile_fragment_control_flow_fixture(
        code, {RecompileDiagnosticStage::Fragment, 0x50630007u}, true);
    ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(0x50630007u);
    EXPECT_EQ(fragment_spirv_required_subgroup_size(module), 64u);
    EXPECT_TRUE(fragment_spirv_required_subgroup_reasons(module) & kFragmentWaveReasonWaveAny);
    EXPECT_TRUE(prosper::test::compile_fragment_control_flow_fixture(
                    code, {RecompileDiagnosticStage::Fragment, 0x50630008u}, false)
                    .empty());
    const auto reason = last_terminal_reject_reason(0x50630008u);
    EXPECT_NE(reason.find("fabricated-fresh-mask-source"), std::string::npos) << reason;
    EXPECT_NE(reason.find("pc=4 words=7d020200"), std::string::npos) << reason;
}

TEST(FragmentFixtureInputs, UnrelatedScalarSelectRequiresItsActualOperands) {
    constexpr std::array code = {
        0x7e000280u, 0xbe800380u, 0x7e020280u, 0x7c2200f0u, 0xbf880002u, 0xbe8503f2u,
        0x7e020205u, 0xbec0037eu, 0xbf060000u, 0x856b807eu, 0x8a406b40u, 0xbefe0940u,
        0x7e000280u, 0x7e040280u, 0x7e0602f2u, 0xf800180fu, 0x03020100u, 0xbf810000u,
    };
    const auto compile = [](const auto& words, uint64_t address) {
        return recompile_fragment(words.data(), words.size(), nullptr, nullptr, UINT32_MAX, nullptr,
                                  true, {RecompileDiagnosticStage::Fragment, address});
    };
    const auto module = compile(code, 0x50630009u);
    ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(0x50630009u);
    EXPECT_EQ(fragment_spirv_required_subgroup_reasons(module), kFragmentWaveReasonWaveAny);
    auto missing_vector = code;
    missing_vector[0] = 0xbf800000u; // S_NOP keeps the consumer PC and branch displacements.
    EXPECT_TRUE(compile(missing_vector, 0x5063000au).empty());
    EXPECT_NE(last_terminal_reject_reason(0x5063000au).find("pc=3 words=7c2200f0"),
              std::string::npos)
        << last_terminal_reject_reason(0x5063000au);
    auto missing_scalar = code;
    missing_scalar[1] = 0xbf800000u;
    EXPECT_TRUE(compile(missing_scalar, 0x5063000bu).empty());
    EXPECT_NE(last_terminal_reject_reason(0x5063000bu).find("pc=10 words=8a406b40"),
              std::string::npos)
        << last_terminal_reject_reason(0x5063000bu);
}
