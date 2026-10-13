// A complete genuine write belongs to every lane of current EXEC. Saving that exact mask before
// the write must not freeze its DATA snapshot forever: restoring the unchanged value reads the
// written word. A narrower mask, overwritten alias or one-predecessor equality cannot grant it.
#include "fixtures/checked_compute_launch.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace prosper::gpu;

struct Result {
    std::vector<uint32_t> module;
    std::string reject;
};

Result compile(const std::vector<uint32_t>& code, uint32_t native_width = 64) {
    ComputeShaderConfig config;
    config.user_sgprs.assign(1, 0);
    config.tgid_x_en = true; // s1 is genuine, runtime workgroup X for scalar branch controls
    config.local_x = 64;
    config.wave_size = 64;
    config.native_subgroup_size = native_width;
    config.force_exchange_dispatcher = true;
    prosper::test::complete_compute_launch(config, 2);
    static uint64_t next = 0x5069000100ull;
    const uint64_t address = next++;
    auto module = recompile_compute(code.data(), code.size(), nullptr, config,
                                    {RecompileDiagnosticStage::Compute, address});
    return {std::move(module), last_terminal_reject_reason(address)};
}

void dispatcher(const Result& result) {
    ASSERT_FALSE(result.module.empty()) << result.reject;
    bool found = false;
    for (std::size_t word = 5; word < result.module.size();) {
        const uint32_t count = result.module[word] >> 16;
        ASSERT_NE(count, 0u);
        ASSERT_LE(word + count, result.module.size());
        found |=
            (result.module[word] & 0xffffu) == 251; // OpSwitch identifies the actual dispatcher
        word += count;
    }
    EXPECT_TRUE(found);
}

void missing(const Result& result, uint32_t pc) {
    EXPECT_TRUE(result.module.empty());
    EXPECT_NE(result.reject.find("mode=unresolved-operand"), std::string::npos) << result.reject;
    EXPECT_NE(result.reject.find("pc=" + std::to_string(pc) + " "), std::string::npos)
        << result.reject;
}

std::vector<uint32_t> unchanged_mask(bool genuine = true) {
    return {
        0x7dea0080u, // A = actual incoming EXEC & (local-ID != 0)
        0xbe84047eu, // s[4:5] = actual A, before the word is written
        genuine ? 0x7e040281u : 0x7e040309u,
        0x7da40080u, // another narrowing cannot remove DATA already genuine throughout A
        0xbefe0404u, // restore the unchanged A
        0x7d280480u,
        0xbf810000u,
    };
}

std::vector<uint32_t> both_paths_equal(bool equal) {
    return {
        0x7dea0080u,
        0xbe84047eu,
        0xbf068001u,
        0xbf840002u, // genuine runtime workgroup X selects the scalar predecessor
        0xbf800000u,
        0xbf820001u,
        equal ? 0xbefe0404u : 0x7da40081u, // keep A or narrow to local-ID 1 on this predecessor
        0x7e040281u, // write under the actual path's CURRENT EXEC
        0xbefe0404u,
        0x7d280480u,
        0xbf810000u,
    };
}
} // namespace

TEST(SavedExecData, UnchangedSavedMaskReceivesTheLaterCompleteWord) {
    dispatcher(compile(unchanged_mask()));
}

TEST(SavedExecData, PortableUnchangedSavedMaskReceivesTheLaterCompleteWord) {
    dispatcher(compile(unchanged_mask(), 0));
}

TEST(SavedExecData, SavedMaskCannotAuthenticateAMissingWriterSource) {
    missing(compile(unchanged_mask(false)), 5);
}

TEST(SavedExecData, PortableSavedMaskCannotAuthenticateAMissingWriterSource) {
    missing(compile(unchanged_mask(false), 0), 5);
}

TEST(SavedExecData, CopiedEqualMasksReceiveOnlyTheirCurrentValueData) {
    auto code = unchanged_mask();
    code.insert(code.begin() + 2, 0xbe860404u);
    code[5] = 0xbefe0406u;
    dispatcher(compile(code));
}

TEST(SavedExecData, ACurrentSubsetCannotRefreshItsWiderSavedParent) {
    auto code = unchanged_mask();
    code.insert(code.begin(), 0xbe86047eu); // retain wider ENTRY before A is selected
    code[5] = 0xbefe0406u;
    missing(compile(code), 6);
}

TEST(SavedExecData, EveryScalarPredecessorRetainsTheSameActualMask) {
    dispatcher(compile(both_paths_equal(true)));
}

TEST(SavedExecData, OneScalarPredecessorCannotBorrowTheOthersEquality) {
    missing(compile(both_paths_equal(false)), 9);
}

TEST(SavedExecData, WiderReplacementCannotUseTheRetiredParentEquality) {
    auto code = unchanged_mask();
    code.insert(code.begin(), 0xbe86047eu);
    code.insert(code.begin() + 3,
                0xbe840406u); // replace s[4:5] with wider ENTRY while A stays active
    missing(compile(code), 7);
}

TEST(SavedExecData, IndependentEqualCopySurvivesTheOriginalPairReplacement) {
    auto code = unchanged_mask();
    code.insert(code.begin(), 0xbe88047eu); // retain ENTRY separately
    code.insert(code.begin() + 3, 0xbe860404u); // retain A independently
    code.insert(code.begin() + 4, 0xbe840408u); // original pair now holds wider ENTRY
    code[7] = 0xbefe0406u;
    dispatcher(compile(code));
}
