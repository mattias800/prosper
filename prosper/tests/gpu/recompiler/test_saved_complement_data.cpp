// Saving exact B=P&~A, narrowing for an inner region, then restoring unchanged B must retain
// the first-half DATA obligation under A. Complete genuine writes under both halves cover P;
// an unrelated restore or an incomplete half must never authenticate P's untouched lanes.
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
// The old value of the read word in the lanes no writer covers: hard on every route. s30 is not
// loaded at launch here (no user SGPR or system value names it), so its value is real but unknown.
// A never-written VGPR would not do: under #5185 it holds an arbitrary value hardware could hold,
// which a fresh predicate may read on any route (#5039). Shifts every pc by one.
std::vector<uint32_t> hard_old(std::vector<uint32_t> code, uint32_t vgpr) {
    code.insert(code.begin(), 0x7e000200u | (vgpr << 17u) | 30u);   // v_mov_b32 vN, s30
    return code;
}
struct Result {
    std::vector<uint32_t> module;
    std::string reject;
};

Result compile(const std::vector<uint32_t>& code, uint32_t native_width = 64) {
    ComputeShaderConfig config;
    config.user_sgprs.assign(1, 0);
    config.local_x = 64;
    config.wave_size = 64;
    config.native_subgroup_size = native_width;
    config.force_exchange_dispatcher = true;
    prosper::test::complete_compute_launch(config, 2);
    static uint64_t next = 0x5075000100ull;
    const auto address = next++;
    auto module = recompile_compute(code.data(), code.size(), nullptr, config,
                                    {RecompileDiagnosticStage::Compute, address});
    return {std::move(module), last_terminal_reject_reason(address)};
}

void dispatcher(const Result& result) {
    ASSERT_FALSE(result.module.empty()) << result.reject;
    bool found = false;
    for (std::size_t word = 5; word < result.module.size();) {
        const auto count = result.module[word] >> 16;
        ASSERT_NE(count, 0u);
        ASSERT_LE(word + count, result.module.size());
        found |= (result.module[word] & 0xffffu) == 251;
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

// A missing half writes v2 from s1 (0x7e040201): only s0 is a user SGPR, so hardware never loads
// s1 and the word is hard on every route. A never-loaded VGPR would be arbitrary under #5185 on
// every route (#5039).
std::vector<uint32_t> saved_complement(bool first = true, bool second = true) {
    return {
        0xbe84047eu, // P is the actual producer EXEC, not a width-derived full-wave certificate
        0x7da40080u, // A=P & (local-ID==0)
        first ? 0x7e040281u : 0x7e040201u,
        0x8afe7e04u, // exact B=P & ~A
        0xbe86047eu, // save actual B before the nested EXEC change
        0x7da40081u, // C=B & (local-ID==1), a strict subset for the supplied 64-lane workgroup
        0xbefe0406u, // restore unchanged B, including its lanes outside C
        second ? 0x7e040282u : 0x7e040201u,
        0xbefe0404u,
        0x7d280480u,
        0xbf810000u,
    };
}
} // namespace

TEST(SavedComplementData, RestoringTheSavedExactComplementRetainsTheFirstHalfData) {
    dispatcher(compile(saved_complement()));
}

TEST(SavedComplementData, MissingFirstHalfCannotBeRepairedByTheSecondHalf) {
    missing(compile(saved_complement(false, true)), 9);
}

TEST(SavedComplementData, PortableWave64RetainsTheSavedExactComplement) {
    dispatcher(compile(saved_complement(), 32));
}

TEST(SavedComplementData, PortableWave64CannotRepairTheMissingFirstHalf) {
    missing(compile(saved_complement(false, true), 32), 9);
}

TEST(SavedComplementData, MissingSecondHalfCannotAuthenticateTheSavedParent) {
    missing(compile(saved_complement(true, false)), 9);
}

TEST(SavedComplementData, InnerSubsetWithoutRestoreCannotCoverTheSavedComplement) {
    auto code = saved_complement();
    code[6] = 0xbf800000u;
    missing(compile(hard_old(code, 2)), 10);
}

TEST(SavedComplementData, AnUnrelatedSavedMaskCannotRestoreComplementMembership) {
    auto code = saved_complement();
    code.insert(code.begin() + 3, 0xbe88047eu); // retain A separately before selecting B
    code[7] = 0xbefe0408u; // restoring A again leaves B's lanes unwritten
    missing(compile(hard_old(code, 2)), 11);
}

TEST(SavedComplementData, CopyingTheSavedComplementRetainsItsEquation) {
    auto code = saved_complement();
    code.insert(code.begin() + 5, 0xbe8a0406u);
    code[7] = 0xbefe040au;
    dispatcher(compile(code));
}

TEST(SavedComplementData, ReplacingTheCopyByAnUnrelatedMaskCannotCoverTheParent) {
    auto code = saved_complement();
    code.insert(code.begin() + 6, 0xbe8a047eu); // copy actual inner C, not the saved exact B
    code[7] = 0xbefe040au;
    missing(compile(hard_old(code, 2)), 11);
}

namespace {
// Direct production transfers isolate the MUST/lifetime proof from independent mask-type,
// relay and scalar validity checks. Only genuine producer v0 is supplied at entry; every
// later mask relationship and word guarantee comes from the decoded instruction sequence.
Rdna2Inst inst(std::initializer_list<uint32_t> words) {
    const std::vector<uint32_t> encoded(words);
    return rdna2_decode_one(encoded.data(), encoded.size());
}

void step(ActiveVectorWords& state, std::initializer_list<uint32_t> words,
          const std::vector<std::pair<int, uint32_t>>& scalar = {},
          const std::vector<int>& vectors = {}, bool genuine = true, bool complete = false) {
    const auto instruction = inst(words);
    advance_active_vector_words(state, instruction, scalar, vectors, -1, genuine, complete);
}

ActiveVectorWords saved_second_half() {
    ActiveVectorWords state;
    state.enabled = state.current_is_entry = state.current_within_entry = true;
    state.current = state.entry_words = {0};
    step(state, {0xbe84047eu}, {{4, 2}});
    step(state, {0x7da40080u});
    step(state, {0x7e040281u}, {}, {2}, true, true);
    step(state, {0x8afe7e04u}, {{126, 2}});
    step(state, {0xbe86047eu}, {{6, 2}});
    step(state, {0x7da40081u});
    return state;
}

void complete_parent(ActiveVectorWords& state, uint32_t carrier = 6, uint32_t parent = 4) {
    step(state, {0xbefe0400u | carrier}, {{126, 2}});
    step(state, {0x7e040282u}, {}, {2}, true, true);
    step(state, {0xbefe0400u | parent}, {{126, 2}});
}

void peer_overwrite(ActiveVectorWords& state, int destination = 2) {
    const auto instruction = inst({0xd7610000u | static_cast<uint32_t>(destination), 0x00010209u});
    EXPECT_EQ(instruction.opcode, 0x361u);
    EXPECT_FALSE(active_vector_sources_defined(state, instruction, {}, {}, {0}, {}, true));
    advance_active_vector_words(state, instruction, {}, {destination}, -1, false, false);
}
} // namespace

TEST(SavedComplementData, SavedContextMeetBothPredecessorsKeepTheActualFirstHalf) {
    auto left = saved_second_half(), right = saved_second_half();
    ASSERT_TRUE(left.complement.saved.at(6).first_words.contains(2));
    ASSERT_EQ(left.complement.parent, -1);
    meet_active_vector_words(left, right);
    complete_parent(left);
    EXPECT_TRUE(left.current.contains(2));
    EXPECT_TRUE(left.masks[4].contains(2));
}

TEST(SavedComplementData, SavedContextMeetOneMissingFirstHalfCannotBorrowTheOtherPath) {
    auto left = saved_second_half(), right = saved_second_half();
    peer_overwrite(right);
    ASSERT_TRUE(left.complement.saved.at(6).first_words.contains(2));
    ASSERT_FALSE(right.complement.saved.at(6).first_words.contains(2));
    meet_active_vector_words(left, right);
    complete_parent(left);
    EXPECT_FALSE(left.current.contains(2));
    EXPECT_FALSE(left.masks[4].contains(2));
}

TEST(SavedComplementData, SavedContextMeetBothMissingFirstHalvesRemainMissing) {
    auto left = saved_second_half(), right = saved_second_half();
    peer_overwrite(left);
    peer_overwrite(right);
    meet_active_vector_words(left, right);
    complete_parent(left);
    EXPECT_FALSE(left.current.contains(2));
}

TEST(SavedComplementData, PeerOverwriteOfAnotherWordRetainsTheStoredFirstHalf) {
    auto state = saved_second_half();
    peer_overwrite(state, 3);
    complete_parent(state);
    EXPECT_TRUE(state.current.contains(2));
}

TEST(SavedComplementData, OnePathWithoutTheCarrierEquationCannotSupplyItAtTheJoin) {
    auto left = saved_second_half(), right = saved_second_half();
    step(right, {0xbe860404u}, {{6, 2}}); // replace B by P; no equation for that current value
    meet_active_vector_words(left, right);
    // Inspect the joined relational fact directly: a later full-P writer would otherwise
    // hide this missing proof by supplying genuinely complete DATA independently.
    EXPECT_FALSE(left.complement.saved.contains(6));
}

TEST(SavedComplementData, BothPathsWithoutTheCarrierEquationRemainUnproved) {
    auto left = saved_second_half(), right = saved_second_half();
    step(left, {0xbe860404u}, {{6, 2}});
    step(right, {0xbe860404u}, {{6, 2}});
    meet_active_vector_words(left, right);
    EXPECT_FALSE(left.complement.saved.contains(6));
}

TEST(SavedComplementData, SelfCopySnapshotsTheActualSavedComplement) {
    auto state = saved_second_half();
    step(state, {0xbe860406u}, {{6, 2}});
    complete_parent(state);
    EXPECT_TRUE(state.current.contains(2));
}

TEST(SavedComplementData, OverlappingCopySnapshotsTheActualSavedComplement) {
    auto state = saved_second_half();
    step(state, {0xbe870406u}, {{7, 2}});
    ASSERT_FALSE(state.complement.saved.contains(6));
    complete_parent(state, 7);
    EXPECT_TRUE(state.current.contains(2));
}

TEST(SavedComplementData, CarrierHighWordReplacementExpiresItsEquation) {
    auto state = saved_second_half();
    step(state, {0xbe8703c1u}, {{7, 1}});
    EXPECT_FALSE(state.complement.saved.contains(6));
    complete_parent(state);
    EXPECT_FALSE(state.current.contains(2));
}

TEST(SavedComplementData, IndependentCarrierCopySurvivesAnOldCarrierHighWordReplacement) {
    auto state = saved_second_half();
    step(state, {0xbe8a0406u}, {{10, 2}});
    step(state, {0xbe8703c1u}, {{7, 1}});
    complete_parent(state, 10);
    EXPECT_TRUE(state.current.contains(2));
}

TEST(SavedComplementData, ParentHighWordReplacementExpiresEveryEquationUsingThatValue) {
    auto state = saved_second_half();
    step(state, {0xbe8503c1u}, {{5, 1}});
    EXPECT_FALSE(state.complement.saved.contains(6));
    complete_parent(state);
    EXPECT_FALSE(state.current.contains(2));
}

TEST(SavedComplementData, ActualParentCopyRetainsTheEquationAfterAnOldParentOverwrite) {
    auto state = saved_second_half();
    step(state, {0xbe880404u}, {{8, 2}});
    step(state, {0xbe8503c1u}, {{5, 1}});
    ASSERT_TRUE(state.complement.saved.at(6).parents.contains(8));
    complete_parent(state, 6, 8);
    EXPECT_TRUE(state.current.contains(2));
}

TEST(SavedComplementData, OverlappingParentCopyUsesThePrewriteParentValue) {
    auto state = saved_second_half();
    step(state, {0xbe8a0406u},
         {{10, 2}}); // independent B carrier before the aliasing parent copy
    step(state, {0xbe850404u}, {{5, 2}});
    ASSERT_FALSE(state.complement.saved.contains(6));
    ASSERT_EQ(state.complement.saved.at(10).parents, (std::set<int>{5}));
    complete_parent(state, 10, 5);
    EXPECT_TRUE(state.current.contains(2));
}

TEST(SavedComplementData, PartialParentCopyCannotPublishANewCompleteParentAlias) {
    auto state = saved_second_half();
    step(state, {0xbe8a0406u}, {{10, 2}});
    step(state, {0xbe850304u}, {{5, 1}});
    EXPECT_FALSE(state.complement.saved.contains(10));
    complete_parent(state, 10, 5);
    EXPECT_FALSE(state.current.contains(2));
}

TEST(SavedComplementData, OneOverwrittenParentCannotBorrowTheOtherPredecessorsGeneration) {
    auto left = saved_second_half(), right = saved_second_half();
    step(right, {0xbe8503c1u}, {{5, 1}});
    meet_active_vector_words(left, right);
    EXPECT_FALSE(left.complement.saved.contains(6));
    complete_parent(left);
    EXPECT_FALSE(left.current.contains(2));
}

TEST(SavedComplementData, BothOverwrittenParentsKeepTheGenerationUnproved) {
    auto left = saved_second_half(), right = saved_second_half();
    step(left, {0xbe8503c1u}, {{5, 1}});
    step(right, {0xbe8503c1u}, {{5, 1}});
    meet_active_vector_words(left, right);
    EXPECT_FALSE(left.complement.saved.contains(6));
    complete_parent(left);
    EXPECT_FALSE(left.current.contains(2));
}

namespace {
ActiveVectorWords parent_alias() {
    auto state = saved_second_half();
    step(state, {0xbe880404u}, {{8, 2}});
    return state;
}
} // namespace

TEST(SavedComplementData, ParentRelationMeetBothPredecessorsRetainBothCurrentAliases) {
    auto left = parent_alias(), right = parent_alias();
    meet_saved_complement_data(left.complement.saved, right.complement.saved);
    EXPECT_EQ(left.complement.saved.at(6).parents, (std::set<int>{4, 8}));
}

TEST(SavedComplementData, ParentRelationMeetDisjointSurvivingAliasesCannotInventAnAllPathParent) {
    auto left = parent_alias(), right = parent_alias();
    step(left, {0xbe8503c1u}, {{5, 1}});
    step(right, {0xbe8903c1u}, {{9, 1}});
    ASSERT_EQ(left.complement.saved.at(6).parents, (std::set<int>{8}));
    ASSERT_EQ(right.complement.saved.at(6).parents, (std::set<int>{4}));
    // Inspect the relational meet itself so a separate live-parent lookup cannot conceal
    // a broken intersection by rejecting the resulting stale references later.
    meet_saved_complement_data(left.complement.saved, right.complement.saved);
    EXPECT_FALSE(left.complement.saved.contains(6));
}

TEST(SavedComplementData, ParentRelationMeetBothPredecessorsRetainTheIndependentAlias) {
    auto left = parent_alias(), right = parent_alias();
    step(left, {0xbe8503c1u}, {{5, 1}});
    step(right, {0xbe8503c1u}, {{5, 1}});
    meet_saved_complement_data(left.complement.saved, right.complement.saved);
    EXPECT_EQ(left.complement.saved.at(6).parents, (std::set<int>{8}));
}

TEST(SavedComplementData, SaveexecSnapshotsOldComplementBeforeNarrowing) {
    auto state = saved_second_half();
    step(state, {0xbefe0406u}, {{126, 2}});
    step(state, {0xbe8a2404u}, {{10, 2}}); // stores old B even though P&B remains B
    step(state, {0x7da40081u});
    complete_parent(state, 10);
    EXPECT_TRUE(state.current.contains(2));
}

TEST(SavedComplementData, SaveexecCannotSnapshotAnUnrelatedInnerSubsetAsComplement) {
    auto state = saved_second_half();
    step(state, {0xbe8a2404u}, {{10, 2}});   // old C is a subset of B, not the complete complement
    EXPECT_FALSE(state.complement.saved.contains(10));
    complete_parent(state, 10);
    EXPECT_FALSE(state.current.contains(2));
}

namespace {
std::vector<uint32_t> backedge(bool genuine) {
    auto code = saved_complement();
    code.resize(9);   // both halves completed and the original P restored before the header
    code.insert(code.end(), {0xbe8a0380u, 0x7d280480u, genuine ? 0x7e040281u : 0x7e040201u,
                             0x800a810au, 0xbf0a820au, 0xbf85fffbu, 0xbf810000u});
    return code;
}
}   // namespace

TEST(SavedComplementData, LoopBackedgeRetainsTheGenuineCurrentGeneration) {
    dispatcher(compile(backedge(true)));
}

TEST(SavedComplementData, LoopBackedgeCannotReviveStoredEarlierFirstHalfData) {
    missing(compile(backedge(false)), 10);
}

TEST(SavedComplementData, Andn1SnapshotsOldComplementBeforeSelectingItsSubset) {
    auto state = saved_second_half();
    step(state, {0xbefe0406u}, {{126, 2}});
    step(state, {0xbe8a3704u}, {{10, 2}});
    ASSERT_TRUE(state.complement.saved.contains(10));
    step(state, {0x7da40081u});
    complete_parent(state, 10);
    EXPECT_TRUE(state.current.contains(2));
}

TEST(SavedComplementData, Andn1CannotSaveAnInnerSubsetAsTheExactComplement) {
    auto state = saved_second_half();
    step(state, {0xbe8a3704u}, {{10, 2}});
    EXPECT_FALSE(state.complement.saved.contains(10));
    complete_parent(state, 10);
    EXPECT_FALSE(state.current.contains(2));
}
