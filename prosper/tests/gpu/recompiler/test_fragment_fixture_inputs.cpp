// Reduced fragment control-flow fixtures must declare the two genuine varying words their
// comparisons consume. Missing/partial mappings remain refusals at the actual compare, while
// usable inputs preserve the NULL-export and saved-ballot dispatcher contracts.
#include "fixtures/fragment_control_flow_inputs.hpp"
#include "fixtures/native_structural_launch.hpp"
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
    // X alone leaves v1 unloaded, which #5185 makes arbitrary on every route (#5039). Enable
    // POS_FIXED_PT too: hardware loads it into v1, and the backend does not supply it (#5160), so
    // the coordinate the compare reads is hard.
    PixelSystemInputMapping only_x;
    only_x.ena = only_x.addr = (1u << 8) | (1u << 15);
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
    // Without the depth input v2 is a word hardware never loads: an arbitrary bound, which the
    // loop header now keeps (#5193). The refused twin loads POS_FIXED_PT into v2 instead: hardware
    // loads it and the backend does not supply it (#5160), so the bound is hard.
    EXPECT_FALSE(prosper::test::compile_fragment_control_flow_fixture(
                     code, {RecompileDiagnosticStage::Fragment, 0x50630009u}, false)
                     .empty())
        << last_terminal_reject_reason(0x50630009u);
    auto fixed_point = prosper::test::fragment_compare_kill::position_inputs();
    fixed_point.ena = fixed_point.addr = fixed_point.ena | (1u << 15);
    EXPECT_TRUE(recompile_fragment(code, std::size(code), nullptr, &fixed_point, UINT32_MAX,
                                   nullptr, false,
                                   {RecompileDiagnosticStage::Fragment, 0x50630008u})
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

TEST(StructuralFixtureInputs, VccHighScratchCompareRequiresItsActualVectorInput) {
    constexpr std::array code = {
        0xbeeb03ffu, 0x3f400000u, // VCC_HI scalar data before the implicit comparison.
        0x7e160280u,              // The original reduced compare consumes v11.
        0x7d841680u, 0x7e4e026bu, 0xbf810000u,
    };
    ComputeShaderConfig wave32;
    wave32.wave_size = 32;
    const auto wave64 = prosper::test::native_structural::wave64_config(wave32);
    for (const auto& config : {wave32, wave64}) {
        const uint64_t address = 0x50630200u + config.wave_size;
        const auto compile = [&](const auto& words) {
            return recompile_compute(words.data(), words.size(), nullptr, config,
                                     {RecompileDiagnosticStage::Compute, address});
        };
        const auto module = compile(code);
        ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(address);
        if (config.wave_size == 64)
            EXPECT_TRUE(has_opcode(module, 339)) << "the Wave64 comparison reconstructs HIGH, "
                                                    "rather than retaining stale scalar data";
        auto missing = code;
        missing[2] =
            0xbf800000u; // Preserve the original comparison PC while omitting its producer.
        // Without a producer v11 is a never-written VGPR compute does not load: #5174 admits it
        // as an arbitrary value, and #5185 admits a relay of another such word too.
        EXPECT_FALSE(compile(missing).empty()) << last_terminal_reject_reason(address);
        missing[2] = 0x7e16031fu;   // v_mov_b32 v11, v31
        EXPECT_FALSE(compile(missing).empty()) << last_terminal_reject_reason(address);
        // A HARD producer still reaches the guard at the compare: the old LDS word an atomic
        // returns at the never-written address v31 (ds_add_rtn_u32 v11, v31, v0). #5193 makes a
        // plain read there arbitrary; an atomic at an arbitrary address stays hard.
        std::vector<uint32_t> hard(code.begin(), code.end());
        hard[2] = 0xd8800000u;
        hard.insert(hard.begin() + 3, (11u << 24) | 31u);
        EXPECT_TRUE(compile(hard).empty());
        const auto reason = last_terminal_reject_reason(address);
        EXPECT_NE(reason.find("fabricated-fresh-mask-source"), std::string::npos) << reason;
        EXPECT_NE(reason.find("pc=4 words=7d841680"), std::string::npos) << reason;
    }
}

TEST(StructuralFixtureInputs, TwoSourceCarryRequiresTheOriginalComparedWord) {
    constexpr uint32_t body[] = {
        0x7d8a06f9u, 0x06868080u, 0xbf068008u, 0xbf840001u, 0xbe800380u, 0xd70f0016u,
        0x00021f1bu, 0xd5286a17u, 0x00024080u, 0x7e040280u, 0x7c020300u, 0xbf860001u,
        0x7e040281u, 0x7d840100u, 0xbf870001u, 0xbf82fffdu, 0x7e040d02u, 0xbf810000u,
    };
    ComputeShaderConfig config;
    config.wave_size = 32;
    config.local_x = 64;
    config.tgid_x_en = true;
    auto code = prosper::test::declared_two_source_cfg_fixture(body);
    const auto compile = [&](uint64_t address) {
        return recompile_compute(code.data(), code.size(), nullptr, config,
                                 {RecompileDiagnosticStage::Compute, address});
    };
    const auto module = compile(0x50630300u);
    ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(0x50630300u);
    EXPECT_TRUE(has_opcode(module, 251)) << "the original join retains its CFG dispatcher";
    // v3 supplies the original SDWA predicate. An unrelated v27 producer cannot replace it: a v3
    // relayed from s30, which no launch loads, is hard on every route. (A never-written v3 would
    // be arbitrary under #5185 on every route, #5039.)
    code[4] = 0x7e06021eu;   // v_mov_b32 v3, s30
    EXPECT_TRUE(compile(0x50630301u).empty());
    const auto reason = last_terminal_reject_reason(0x50630301u);
    EXPECT_NE(reason.find("fabricated-fresh-mask-source"), std::string::npos) << reason;
    EXPECT_NE(reason.find("pc=9 words=7d8a06f9,06868080"), std::string::npos) << reason;
}

TEST(StructuralFixtureInputs, RuntimeBranchKeepsTheOriginalThreeSourceAmbiguity) {
    constexpr uint32_t body[] = {
        0x7d8a06f9u, 0x06868080u, 0xbf068008u, 0xbf840001u, 0xbe800380u, 0xd5690001u,
        0x00020302u, 0x7e040280u, 0x7c020300u, 0xbf860001u, 0x7e040281u, 0x7d840100u,
        0xbf870001u, 0xbf82fffdu, 0x7e040d02u, 0xbf810000u,
    };
    ComputeShaderConfig config;
    config.wave_size = 32;
    config.local_x = 64;
    config.tgid_x_en = true;
    auto code = prosper::test::declared_two_source_cfg_fixture(body);
    const auto compile = [&](uint64_t address) {
        return recompile_compute(code.data(), code.size(), nullptr, config,
                                 {RecompileDiagnosticStage::Compute, address});
    };
    const auto module = compile(0x50630302u);
    ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(0x50630302u);
    EXPECT_TRUE(has_opcode(module, 251));
    code[prosper::test::two_source_cfg_inputs.size() + 5] = 0xd5430001u;
    EXPECT_TRUE(compile(0x50630303u).empty())
        << "the genuine third source reads the original path-dependent s0 lifetime";
    code[0] = 0xbe880380u; // A constant branch selects only the genuine scalar-data path.
    EXPECT_FALSE(compile(0x50630304u).empty()) << last_terminal_reject_reason(0x50630304u);
}

TEST(StructuralFixtureInputs, PhiReductionDeclaresItsActualPredicate) {
    constexpr uint32_t code[] = {
        0x93eaff03u, 0x00080008u, 0x876bff03u, 0x000000ffu, 0x8f6a8c6au, 0x887c6a6bu,
        0xbf900009u, 0x906a8803u, 0x81ea6a80u, 0x90fe6ac1u, 0xf8000941u, 0x00000000u,
        0x81ea0380u, 0x90fe6ac1u, 0xbe80246au, 0xbf880001u, 0x8afe7e00u, 0xbefe0400u,
        0x34040a81u, 0x36060ac2u, 0x7e000280u, 0x7e0202f2u, 0x36040482u, 0x4a0606c1u,
        0x4a0404c1u, 0x7e060b03u, 0x7e040b02u, 0xf80008cfu, 0x01000302u, 0xbf810000u,
    };
    const auto module = prosper::test::compile_ngg_phi_fixture_with_declared_predicate(code);
    ASSERT_FALSE(module.empty());
    EXPECT_TRUE(has_opcode(module, 245)) << "the original wave-pack/forward-if PHI is exercised";
    for (size_t i = 5; i < module.size();) {
        const uint32_t count = module[i] >> 16;
        ASSERT_GT(count, 0u);
        ASSERT_LE(count, module.size() - i);
        if ((module[i] & 0xffffu) == 245) {
            ASSERT_GE(count, 5u);
            EXPECT_NE(module[i + 2], 0u);
            for (size_t word = i + 3; word < i + count; ++word) EXPECT_NE(module[word], 0u);
        }
        i += count;
    }
    EXPECT_TRUE(recompile_vertex(code, std::size(code), nullptr, nullptr, false, 0,
                                 {RecompileDiagnosticStage::Vertex, 0x50630305u})
                    .empty());
    EXPECT_NE(last_terminal_reject_reason(0x50630305u).find("pc=14 words=be80246a"),
              std::string::npos)
        << last_terminal_reject_reason(0x50630305u);
    EXPECT_TRUE(
        prosper::test::compile_ngg_phi_fixture_with_declared_predicate(std::span(code).first(14))
            .empty());
}
