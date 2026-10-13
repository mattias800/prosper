// A complete genuine write belongs to every lane of current EXEC. Saving that exact mask before
// the write must not freeze its DATA snapshot forever: restoring the unchanged value reads the
// written word. A narrower mask, overwritten alias or one-predecessor equality cannot grant it.
#include "fixtures/checked_compute_launch.hpp"
#include "gpu/recompiler/rdna2_active_vector_words.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <initializer_list>
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

TEST(SavedExecData, AndOfTheSameActualMaskRetainsEquality) {
    auto code = unchanged_mask();
    code.insert(code.begin() + 2, 0x87fe047eu); // EXEC = actual A & saved A
    dispatcher(compile(code));
}

TEST(SavedExecData, OrOfTheSameActualMaskRetainsEquality) {
    auto code = unchanged_mask();
    code.insert(code.begin() + 2, 0x88fe047eu); // EXEC = actual A | saved A
    dispatcher(compile(code));
}

TEST(SavedExecData, AndWithANarrowerValueCannotRefreshTheWiderOperand) {
    auto code = unchanged_mask();
    code.insert(code.begin() + 2, {0x7da40081u, 0x87fe047eu});
    missing(compile(code), 7);
}

TEST(SavedExecData, ReversedAndOperandsCannotAuthenticateTheWiderMask) {
    auto code = unchanged_mask();
    code.insert(code.begin() + 2, {0x7da40081u, 0x87fe7e04u});
    missing(compile(code), 7);
}

TEST(SavedExecData, AndOfDistinctValuesCannotTreatEitherAsTheResult) {
    const std::vector<uint32_t> code{
        0xbe88047eu, 0x7da40080u, 0xbe84047eu, 0xbefe0408u, 0x7da40081u, 0xbe86047eu, 0xbefe0404u,
        0x87fe7e06u, // B & A is empty, while saved B itself still enables lane 1
        0x7e040281u, 0xbefe0406u, 0x7d280480u, 0xbf810000u,
    };
    missing(compile(code), 10);
}

TEST(SavedExecData, AndOfIdenticalSavedValuesWritesThroughoutThatActualValue) {
    const std::vector<uint32_t> code{
        0xbe88047eu, 0x7da40080u, 0xbe84047eu, 0xbefe0408u, 0x7da40081u, 0xbe86047eu, 0xbefe0404u,
        0x87fe0606u, // B & B enables every lane of saved B, independently of old current A
        0x7e040281u, 0xbefe0406u, 0x7d280480u, 0xbf810000u,
    };
    dispatcher(compile(code));
}

TEST(SavedExecData, SaveexecWithAnEqualOperandCapturesTheSameCurrentValue) {
    auto code = unchanged_mask();
    code.insert(code.begin() + 2, 0xbe862404u);
    code[5] = 0xbefe0406u;
    dispatcher(compile(code));
}

TEST(SavedExecData, SaveexecWithAWiderOperandCannotRefreshThatOperand) {
    auto code = unchanged_mask();
    code.insert(code.begin() + 2, {0x7da40081u, 0xbe862404u});
    missing(compile(code), 7);
}

TEST(SavedExecData, OverlappingDestinationCopiesTheOldSourceBeforeRetirement) {
    auto code = unchanged_mask();
    code.insert(code.begin() + 2, 0xbe850404u); // s[5:6] = old s[4:5], overlapping by one word
    code[5] = 0xbefe0405u;
    dispatcher(compile(code));
}

TEST(SavedExecData, SelfCopyKeepsTheSameActualPairValue) {
    auto code = unchanged_mask();
    code.insert(code.begin() + 2, 0xbe840404u);
    dispatcher(compile(code));
}

namespace {
std::vector<uint32_t> backedge(bool genuine) {
    return {
        0x7dea0080u,
        0xbe84047eu,
        0x7e040281u,
        0xbe860380u,
        0x7d280480u, // header must consume the CURRENT generation, not its preheader writer
        genuine ? 0x7e040281u : 0x7e040309u,
        0xbefe0404u,
        0x80068106u,
        0xbf0a8206u,
        0xbf85fffau,
        0xbf810000u,
    };
}
} // namespace

TEST(SavedExecData, EveryBackedgeRetainsTheGenuineWrittenWord) {
    dispatcher(compile(backedge(true)));
}

TEST(SavedExecData, SavedEqualityCannotReviveAnEarlierDataGeneration) {
    missing(compile(backedge(false)), 4);
}

// Pin the actual MUST boundary directly, so an independent missing-mask emission guard cannot
// conceal a broken equality join or pair lifetime. v0 is the sole genuine hand-supplied input;
// every later word and mask relationship comes from decoded production transfers.
namespace {
void step(ActiveVectorWords& state, std::initializer_list<uint32_t> words,
          const std::vector<std::pair<int, uint32_t>>& scalars = {},
          const std::vector<int>& vectors = {}, int mask = -1, bool complete = false) {
    const std::vector<uint32_t> code(words);
    const auto instruction = rdna2_decode_one(code.data(), code.size());
    std::set<int> relayed;
    for (int word = 0; word < 256; ++word) relayed.insert(word);
    const bool genuine =
        active_vector_sources_defined(state, instruction, relayed, {}, {}, {}, true);
    advance_active_vector_words(state, instruction, scalars, vectors, mask, genuine, complete);
}

ActiveVectorWords saved_mask() {
    ActiveVectorWords state;
    state.enabled = state.current_is_entry = state.current_within_entry = true;
    state.current = state.entry_words = {0};
    step(state, {0x7dea0080u}); // actual current A; CMPX writes EXEC only, preserving VCC
    step(state, {0xbe84047eu}, {{4, 2}}); // unchanged s[4:5] = A before v2 is written
    return state;
}

void narrow(ActiveVectorWords& state) {
    step(state, {0x7da40081u}); // match the actual dispatcher's mask_write=-1
}

void write_and_restore_parent(ActiveVectorWords& state) {
    step(state, {0x7e040281u}, {}, {2}, -1, true);
    step(state, {0xbefe0404u}, {{126, 2}});
}

ActiveVectorWords saved_alias(bool equal) {
    auto state = saved_mask();
    if (equal) step(state, {0xbe860404u}, {{6, 2}});
    narrow(state);
    if (!equal) step(state, {0xbe86047eu}, {{6, 2}});
    return state;
}

void write_through_alias(ActiveVectorWords& state) {
    step(state, {0xbefe0406u}, {{126, 2}});
    write_and_restore_parent(state);
}

ActiveVectorWords replace_high_word() {
    auto state = saved_mask();
    step(state, {0xbe860404u}, {{6, 2}}); // independent copy retains actual A
    step(state, {0xbe8503c1u}, {{5, 1}}); // old pair s[4:5] no longer holds actual A
    step(state, {0xbe880404u}, {{8, 2}}); // a copy cannot reauthenticate that retired value
    step(state, {0x7e040281u}, {}, {2}, -1, true);
    return state;
}
} // namespace

TEST(SavedExecData, CurrentEqualityMeetBothPredecessorsRefreshTheGenuineWord) {
    auto left = saved_mask();
    auto right = saved_mask();
    meet_active_vector_words(left, right);
    write_and_restore_parent(left);
    EXPECT_TRUE(left.current.contains(2));
}

TEST(SavedExecData, CurrentEqualityMeetOnePredecessorCannotAuthenticateTheOther) {
    auto left = saved_mask();
    auto right = saved_mask();
    narrow(right);
    meet_active_vector_words(left, right);
    write_and_restore_parent(left);
    EXPECT_FALSE(left.current.contains(2));
}

TEST(SavedExecData, CurrentEqualityMeetBothNarrowerPredecessorsKeepTheWiderWordMissing) {
    auto left = saved_mask();
    auto right = saved_mask();
    narrow(left);
    narrow(right);
    meet_active_vector_words(left, right);
    write_and_restore_parent(left);
    EXPECT_FALSE(left.current.contains(2));
}

TEST(SavedExecData, AliasEqualityMeetBothPredecessorsRetainTheSameSavedValue) {
    auto left = saved_alias(true);
    auto right = saved_alias(true);
    meet_active_vector_words(left, right);
    write_through_alias(left);
    EXPECT_TRUE(left.current.contains(2));
}

TEST(SavedExecData, AliasEqualityMeetOneDifferentValueCannotRefreshTheWiderParent) {
    auto left = saved_alias(true);
    auto right = saved_alias(false);
    meet_active_vector_words(left, right);
    write_through_alias(left);
    EXPECT_FALSE(left.current.contains(2));
}

TEST(SavedExecData, AliasEqualityMeetBothDifferentValuesRetainTheMissingParentWord) {
    auto left = saved_alias(false);
    auto right = saved_alias(false);
    meet_active_vector_words(left, right);
    write_through_alias(left);
    EXPECT_FALSE(left.current.contains(2));
}

TEST(SavedExecData, HighWordReplacementCannotAuthenticateACopyOfTheRetiredPair) {
    auto state = replace_high_word();
    step(state, {0xbefe0408u}, {{126, 2}});
    EXPECT_FALSE(state.current.contains(2));
}

TEST(SavedExecData, IndependentSavedCopySurvivesAHighWordReplacement) {
    auto state = replace_high_word();
    step(state, {0xbefe0406u}, {{126, 2}});
    EXPECT_TRUE(state.current.contains(2));
}
