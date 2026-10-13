// #5041: DATA dependencies of an SCC-preserving instruction cannot poison an earlier
// comparison. A genuine SCC writer replaces old provenance; a missing source stays missing.
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_spilled_mask_halves.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace prosper::gpu;

void transfer(SpilledMaskHalves& state, uint32_t encoding, bool missing) {
    const auto in = rdna2_decode_one(&encoding, 1);
    ASSERT_NE(in.fmt, Rdna2Format::Unknown);
    advance_defined_slots(state, in, {}, {}, {{in.dst.value, 1}}, {}, {.scalar_relay = missing},
                          true, 64);
}

std::vector<uint32_t> relative_read_program(uint32_t compare, bool boundary) {
    std::vector<uint32_t> words{
        0xbe840385u, // s4 = 5
        0xbe850386u, // s5 = 6
        0xbe860387u, // s6 = 7
        0xbefc0381u, // M0 = 1
        compare,
        0xbe802e04u, // s0 = s[4+M0]; SCC unchanged, although later candidates are absent
    };
    if (boundary) words.push_back(0xbf880000u); // EXECZ +0 puts the next PC in another block
    words.insert(words.end(), {0xbf850001u, 0xbe800389u, 0x7e000200u, 0xbf810000u});
    return words;
}

std::vector<uint32_t> compile(const std::vector<uint32_t>& code) {
    return recompile_valu(code.data(), code.size(), 1, 0, nullptr, 0, kDefaultComputePgmRsrc1,
                          true);
}
} // namespace

TEST(ScalarConditionPreservation, RelativeReadKeepsEarlierComparisonAcrossDispatcher) {
    for (const uint32_t compare : {0xbf018005u, 0xbf008005u})
        for (const bool boundary : {false, true}) {
            SCOPED_TRACE(compare);
            SCOPED_TRACE(boundary);
            EXPECT_FALSE(compile(relative_read_program(compare, boundary)).empty());
        }
}

TEST(ScalarConditionPreservation, MissingCompareSourceStillRefusesAcrossTheRelativeRead) {
    for (const bool boundary : {false, true}) {
        SCOPED_TRACE(boundary);
        EXPECT_TRUE(compile(relative_read_program(0xbf018014u, boundary)).empty())
            << "the same comparison now reads absent s20";
    }
}

TEST(ScalarConditionPreservation, PreservingInstructionsDoNotPoisonGenuineScc) {
    // Same DATA taint, different architectural preservers: relative MOV, CSELECT and BFM.
    for (const uint32_t encoding : {0xbe802e04u, 0x85010414u, 0x93010414u}) {
        SCOPED_TRACE(encoding);
        SpilledMaskHalves state;
        transfer(state, encoding, true);
        EXPECT_TRUE(state.relayed_words.contains(1) || state.relayed_words.contains(0))
            << "the destination's missing DATA dependency remains tracked";
        EXPECT_FALSE(state.relayed_words.contains(kRelayedScc));
    }
}

TEST(ScalarConditionPreservation, PreservingInstructionsCannotRepairPoisonedScc) {
    for (const uint32_t encoding : {0xbe802e04u, 0x85010414u, 0x93010414u}) {
        SCOPED_TRACE(encoding);
        SpilledMaskHalves state;
        state.relayed_words.insert(kRelayedScc);
        transfer(state, encoding, false);
        EXPECT_TRUE(state.relayed_words.contains(kRelayedScc));
    }
}

TEST(ScalarConditionPreservation, GenuineWritersReplaceOldSccProvenance) {
    // ADD and AND write a fresh scalar condition from their current genuine sources.
    for (const uint32_t encoding : {0x80018481u, 0x87018481u, 0xbe810704u}) {
        SCOPED_TRACE(encoding);
        SpilledMaskHalves state;
        state.relayed_words.insert(kRelayedScc);
        transfer(state, encoding, false);
        EXPECT_FALSE(state.relayed_words.contains(kRelayedScc));
    }
}

TEST(ScalarConditionPreservation, WritersWithMissingSourcesKeepSccPoisoned) {
    for (const uint32_t encoding : {0x80018481u, 0x87018481u, 0xbe810704u}) {
        SCOPED_TRACE(encoding);
        SpilledMaskHalves state;
        transfer(state, encoding, true);
        EXPECT_TRUE(state.relayed_words.contains(kRelayedScc));
    }
}

TEST(ScalarConditionPreservation, MissingCarryInCannotBecomeAFreshCondition) {
    for (const uint32_t encoding : {0x82018481u, 0x82818481u}) {
        SCOPED_TRACE(encoding); // ADDC and SUBB: explicit constants, implicit incoming SCC
        const auto in = rdna2_decode_one(&encoding, 1);
        for (const bool missing : {false, true}) {
            SCOPED_TRACE(missing);
            SpilledMaskHalves state;
            if (missing) state.relayed_words.insert(kRelayedScc);
            const auto source = spilled_vector_source_marks(state, in, {}, {}, !missing, 64);
            EXPECT_EQ(source.scalar_relay, missing);
            advance_defined_slots(state, in, {}, {}, {{in.dst.value, 1}}, {}, source, !missing, 64);
            EXPECT_EQ(state.relayed_words.contains(kRelayedScc), missing);
        }
    }
}

namespace {
struct SopkCompilation {
    std::vector<uint32_t> module;
    std::string reason;
    uint32_t branch_pc = 0;
};

SopkCompilation compile_sopk(uint32_t instruction, bool old_destination, bool old_scc) {
    std::vector<uint32_t> code;
    code.reserve(16);
    code.push_back(0x7e020280u); // genuine v1 used by the original irreducible tail
    if (old_destination) code.push_back(0xbe8c0385u); // s12 = 5
    code.push_back(old_scc ? 0xbf068080u : 0xbf068014u); // 0 == 0, or absent s20 == 0
    code.push_back(instruction);
    const uint32_t branch_pc = static_cast<uint32_t>(code.size());
    code.insert(code.end(),
                {0xbf850001u, 0x7e060281u, 0x7e040280u, 0x7c020300u, 0xbf860001u, 0x7e040281u,
                 0x7d840100u, 0xbf870001u, 0xbf82fffdu, 0x7e040d02u, 0xbf810000u});
    // Fixture-owned raw one-group Wave64 launch, not authority inferred from host subgroup width.
    ComputeShaderConfig config;
    config.local_x = config.threads_x = config.wave_size = config.native_subgroup_size = 64;
    config.dispatch_launch.known = ComputeDispatchLaunchState::kAll;
    config.dispatch_launch.initiator = 1;
    config.dispatch_launch.num_thread = {64, 1, 1};
    config.dispatch_launch.dimensions = {1, 1, 1};
    static uint64_t next_address = 0x5041001000ull;
    const uint64_t address = next_address++;
    auto module = recompile_compute(code.data(), code.size(), nullptr, config,
                                    {RecompileDiagnosticStage::Compute, address});
    return {std::move(module), last_terminal_reject_reason(address), branch_pc};
}

bool uses_dispatcher(const std::vector<uint32_t>& module) {
    for (std::size_t at = 5; at < module.size();) {
        const uint32_t width = module[at] >> 16;
        if (!width || width > module.size() - at) return false;
        if ((module[at] & 0xffffu) == 251u) return true;
        at += width;
    }
    return false;
}

void poisoned_sopk_branch(const SopkCompilation& result) {
    EXPECT_TRUE(result.module.empty());
    EXPECT_NE(result.reason.find("poisoned-scc"), std::string::npos) << result.reason;
    EXPECT_NE(result.reason.find("pc=" + std::to_string(result.branch_pc) + " "), std::string::npos)
        << "the SCC consumer must be the actual rejecting site: " << result.reason;
}
}   // namespace

TEST(ScalarConditionPreservation, AddkRequiresItsImplicitOldDestinationAtTheSccBranch) {
    // A genuine old s12 supplies ADDK's new SCC even after a poisoned comparison. Without s12,
    // clearing a MAY taint cannot grant a condition: the dispatcher's independent MUST gate rejects.
    const auto genuine = compile_sopk(0xb78c0001u, true, false);
    ASSERT_FALSE(genuine.module.empty()) << genuine.reason;
    ASSERT_TRUE(uses_dispatcher(genuine.module)) << "the real CFG route must own the condition";
    poisoned_sopk_branch(compile_sopk(0xb78c0001u, false, false));
}

TEST(ScalarConditionPreservation, MulkCannotRepairTheEarlierCondition) {
    const auto genuine = compile_sopk(0xb80c0001u, true, true);
    ASSERT_FALSE(genuine.module.empty()) << genuine.reason;
    ASSERT_TRUE(uses_dispatcher(genuine.module)) << "the real CFG route must own the condition";
    const auto unobserved_missing_data = compile_sopk(0xb80c0001u, false, true);
    ASSERT_FALSE(unobserved_missing_data.module.empty()) << unobserved_missing_data.reason;
    ASSERT_TRUE(uses_dispatcher(unobserved_missing_data.module));
    poisoned_sopk_branch(compile_sopk(0xb80c0001u, true, false));
    poisoned_sopk_branch(compile_sopk(0xb80c0001u, false, false));
}
