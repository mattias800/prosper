// Complete writes under A and parent&~A cover the saved parent. Neither an absent source in one
// half nor a register-file placeholder may authenticate that whole-parent DATA at a fresh compare.
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

std::vector<uint32_t> partition(bool first = true, bool second = true) {
    return {
        0xbe84047eu, // save actual parent EXEC in s[4:5]
        0x7dea0080u, // A = parent & (genuine local-ID v0 != 0)
        0xbf880001u, // an empty A skips its writer, without creating a missing active lane
        first ? 0x7e040281u : 0x7e040309u,
        0x8afe7e04u, // B = parent & ~A
        0xbf880001u, // an empty B skips its writer
        second ? 0x7e040282u : 0x7e040309u,
        0xbefe0404u, // restore the same parent value
        0x7d280480u, // fresh predicate consumes the entire parent word v2
        0xbf810000u,
    };
}

struct Result {
    std::vector<uint32_t> module;
    std::string reject;
};

Result compile(const std::vector<uint32_t>& code) {
    ComputeShaderConfig config;
    config.user_sgprs.assign(1, 0);
    config.local_x = 64;
    config.wave_size = 64;
    config.native_subgroup_size = 64;
    config.force_exchange_dispatcher = true;
    prosper::test::complete_compute_launch(config);
    static uint64_t next = 0x5066000100ull;
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
        found |= (result.module[word] & 0xffffu) == 251; // OpSwitch
        word += count;
    }
    EXPECT_TRUE(found);
}

void missing(const Result& result, uint32_t pc = 8) {
    EXPECT_TRUE(result.module.empty());
    EXPECT_NE(result.reject.find("mode=unresolved-operand"), std::string::npos) << result.reject;
    EXPECT_NE(result.reject.find("pc=" + std::to_string(pc) + " "), std::string::npos)
        << result.reject;
}
} // namespace

TEST(ComplementaryExecData, BothActiveHalvesDefineTheRestoredParent) {
    dispatcher(compile(partition()));
}

TEST(ComplementaryExecData, MissingFirstHalfCannotDefineTheRestoredParent) {
    missing(compile(partition(false, true)));
}

TEST(ComplementaryExecData, MissingSecondHalfCannotDefineTheRestoredParent) {
    missing(compile(partition(true, false)));
}

TEST(ComplementaryExecData, MissingBothHalvesCannotDefineTheRestoredParent) {
    missing(compile(partition(false, false)));
}

namespace {
std::vector<uint32_t> scalar_first_half(bool alternate_genuine) {
    return {
        0xbe84047eu, 0x7dea0080u, 0xbf060000u,
        0xbf840002u, // independent genuine SCC, not an EXEC-empty condition
        0x7e040281u, 0xbf820001u, alternate_genuine ? 0x7e040281u : 0x7e040309u,
        0x8afe7e04u, 0xbf880001u, 0x7e040282u,
        0xbefe0404u, 0x7d280480u, 0xbf810000u,
    };
}
} // namespace

TEST(ComplementaryExecData, BothScalarPredecessorsCanSupplyTheFirstHalf) {
    dispatcher(compile(scalar_first_half(true)));
}

TEST(ComplementaryExecData, ScalarBranchCannotInventAnEmptyFirstHalf) {
    missing(compile(scalar_first_half(false)), 11);
}

TEST(ComplementaryExecData, Andn2SaveexecCoversTheSameSavedParent) {
    auto code = partition();
    code[4] = 0xbe862704u; // EXEC = saved parent & ~old EXEC; s[6:7] receives old EXEC
    dispatcher(compile(code));
}

TEST(ComplementaryExecData, Andn2SaveexecCannotReplaceAMissingFirstHalf) {
    auto code = partition(false, true);
    code[4] = 0xbe862704u;
    missing(compile(code));
}

TEST(ComplementaryExecData, PartialSecondWriterCannotDefineTheRestoredParent) {
    auto code = partition();
    code[5] = 0xbf880002u; // skip the entire two-dword SDWA writer when B is empty
    code[6] = 0x7e0402f9u;
    code.insert(code.begin() + 7, 0x00851480u); // WORD_0 / UNUSED_PRESERVE
    missing(compile(code), 9);
}

namespace {
std::vector<uint32_t> partition_backedge(bool genuine) {
    auto code = partition();
    code.resize(8); // complete both halves and restore the parent before the loop header
    code.insert(code.end(),
                {
                    0xbe860380u, // genuine scalar counter s6=0
                    0x7d280480u, // header: compare current v2 under the restored parent
                    genuine ? 0x7e040281u : 0x7e040309u,
                    0x80068106u,
                    0xbf0a8206u,
                    0xbf85fffbu, // counter++; counter<2; backedge -> compare
                    0xbf810000u,
                });
    return code;
}
} // namespace

TEST(ComplementaryExecData, BackedgeRetainsGenuineCurrentReplacement) {
    dispatcher(compile(partition_backedge(true)));
}

TEST(ComplementaryExecData, BackedgeCannotReviveAnEarlierPartitionWord) {
    missing(compile(partition_backedge(false)), 9);
}

TEST(ComplementaryExecData, ZeroDisplacementExeczKeepsBothConditionOutcomes) {
    auto code = partition();
    code[2] = 0xbf880000u;
    dispatcher(compile(code));
}

TEST(ComplementaryExecData, ZeroDisplacementExeczCannotInventAnEmptyFirstHalf) {
    auto code = partition(false, true);
    code[2] = 0xbf880000u;
    missing(compile(code));
}

namespace {
std::vector<uint32_t> execnz_partition(bool first, bool second) {
    return {
        0xbe84047eu,
        0x7dea0080u,
        0xbf890001u,
        0xbf820001u,
        first ? 0x7e040281u : 0x7e040309u,
        0x8afe7e04u,
        0xbf890001u,
        0xbf820001u,
        second ? 0x7e040282u : 0x7e040309u,
        0xbefe0404u,
        0x7d280480u,
        0xbf810000u,
    };
}

std::vector<uint32_t> accumulating_partition(bool first, bool second, bool explicit_source = true,
                                             bool alias = false, bool e64 = false) {
    auto code = partition(first, second);
    code.resize(8);
    code.push_back(0x7dea0080u); // narrow again to A, with genuine DATA retained in both halves
    if (e64) {
        code.push_back(0xd51f0002u); // v_mac_f32_e64 v2, v0/v2, v0/v9 (implicit old v2)
        code.push_back((alias ? 0x102u : 0x100u) | (explicit_source ? 0x20000u : 0x21200u));
    } else {
        code.push_back(0x3e040000u | (alias ? 0x102u : 0x100u) | (explicit_source ? 0u : 0x1200u));
    }
    code.insert(code.end(), {0xbefe0404u, 0x7d280480u, 0xbf810000u});
    return code;
}
} // namespace

TEST(ComplementaryExecData, ExecnzFallthroughCoversOnlyItsEmptyCurrentDomain) {
    dispatcher(compile(execnz_partition(true, true)));
}

TEST(ComplementaryExecData, ExecnzFallthroughCannotInventFirstHalfData) {
    missing(compile(execnz_partition(false, true)), 10);
}

TEST(ComplementaryExecData, ExecnzFallthroughCannotInventSecondHalfData) {
    missing(compile(execnz_partition(true, false)), 10);
}

TEST(ComplementaryExecData, MacRetainsGenuinePartitionData) {
    dispatcher(compile(accumulating_partition(true, true)));
}

TEST(ComplementaryExecData, MacCannotInventItsOldAccumulator) {
    missing(compile(accumulating_partition(false, true)), 11);
}

TEST(ComplementaryExecData, MacCannotRepairTheOtherMissingHalf) {
    missing(compile(accumulating_partition(true, false)), 11);
}

TEST(ComplementaryExecData, MacCannotBorrowAMissingExplicitSource) {
    missing(compile(accumulating_partition(true, true, false)), 11);
}

TEST(ComplementaryExecData, AliasingMacReadsTheGenuinePriorWord) {
    dispatcher(compile(accumulating_partition(true, true, true, true)));
}

TEST(ComplementaryExecData, AliasingMacCannotReadAMissingPriorWord) {
    missing(compile(accumulating_partition(false, true, true, true)), 11);
}

TEST(ComplementaryExecData, E64MacRetainsGenuinePartitionData) {
    dispatcher(compile(accumulating_partition(true, true, true, false, true)));
}

TEST(ComplementaryExecData, E64MacCannotInventItsOldAccumulator) {
    missing(compile(accumulating_partition(false, true, true, false, true)), 12);
}

TEST(ComplementaryExecData, E64MacCannotBorrowAMissingExplicitSource) {
    missing(compile(accumulating_partition(true, true, false, false, true)), 12);
}

namespace {
std::vector<uint32_t> scalar_accumulator(bool genuine) {
    auto code = scalar_first_half(genuine);
    code.resize(11);
    code.insert(code.end(), {0x7dea0080u, 0x3e040100u, 0xbefe0404u, 0x7d280480u, 0xbf810000u});
    return code;
}

std::vector<uint32_t> mac_backedge(bool genuine) {
    auto code = partition();
    code.resize(8);
    code.insert(code.end(), {
                                0xbe860380u,
                                0x3e040100u,
                                0x7d280480u,
                                genuine ? 0x7e040281u : 0x7e040309u,
                                0x80068106u,
                                0xbf0a8206u,
                                0xbf85fffau,
                                0xbf810000u,
                            });
    return code;
}

std::vector<uint32_t> overwritten_parent(bool restore_copy) {
    auto code = partition();
    code.insert(code.begin(), 0x7da40080u);   // actual parent initially enables only local-ID zero
    code.insert(code.begin() + 2, 0xbe860404u);   // retain that parent in independent s[6:7]
    code[6] = 0x8afe7e06u;   // use the retained parent to form the exact complement
    code.insert(code.begin() + 9,
                0xbe8503c1u);   // overwrite only s5: old s[4:5] now enables high lanes
    code[10] = restore_copy ? 0xbefe0406u : 0xbefe0404u;
    return code;
}
}   // namespace

TEST(ComplementaryExecData, MacRetainsBothPredecessorsGenuineAccumulator) {
    dispatcher(compile(scalar_accumulator(true)));
}

TEST(ComplementaryExecData, MacCannotBorrowOnePredecessorsAccumulator) {
    missing(compile(scalar_accumulator(false)), 14);
}

TEST(ComplementaryExecData, MacBackedgeRetainsGenuinePriorGeneration) {
    dispatcher(compile(mac_backedge(true)));
}

TEST(ComplementaryExecData, MacBackedgeCannotBorrowAnEarlierGeneration) {
    missing(compile(mac_backedge(false)), 10);
}

TEST(ComplementaryExecData, SavedParentCopySurvivesAnUnrelatedHighWordOverwrite) {
    dispatcher(compile(overwritten_parent(true)));
}

TEST(ComplementaryExecData, SavedParentHighWordOverwriteCannotReviveCoverage) {
    missing(compile(overwritten_parent(false)), 11);
}

TEST(ComplementaryExecData, RepeatingTheFirstMaskCannotCoverTheParent) {
    auto code = partition();
    code[4] = 0x87fe7e04u;   // parent & old EXEC repeats A instead of selecting parent & ~A
    missing(compile(code));
}

TEST(ComplementaryExecData, SelectedLaneWriterCannotReplaceTheMissingComplement) {
    auto code = partition();
    code[5] = 0xbf880002u;
    code[6] = 0xd7610002u;
    code.insert(code.begin() + 7, 0x00010200u);   // write lane 1, while B's missing lane is 0
    missing(compile(code), 9);
}

namespace {
std::vector<uint32_t> peer_overwrite_arm(bool poison) {
    return {
        0xbe84047eu,
        0x7dea0080u,
        0xbf880001u,
        0x7e040281u,
        0x8afe7e04u,
        0xbf060000u,
        0xbf840003u,
        poison ? 0xd7610002u : 0xbf800000u,
        poison ? 0x00010209u
               : 0xbf800000u,   // absent s9 overwrites active A's lane 1, ignoring EXEC B
        0xbf820001u,
        0xbf800000u,
        0xbf880001u,
        0x7e040282u,
        0xbefe0404u,
        0x7d280480u,
        0xbf810000u,
    };
}
}   // namespace

TEST(ComplementaryExecData, BothPredecessorsRetainTheActualFirstHalfWord) {
    dispatcher(compile(peer_overwrite_arm(false)));
}

TEST(ComplementaryExecData, OnePeerOverwriteCannotBorrowTheOtherPredecessorsFirstHalf) {
    missing(compile(peer_overwrite_arm(true)), 14);
}

TEST(ComplementaryExecData, NewParentValueCannotReuseAnEarlierComplement) {
    auto code = partition();
    code.insert(code.begin(),
                {0xbe86047eu, 0x7da40080u});   // preserve ENTRY, then narrow parent to lane 0
    code.insert(code.begin() + 7,
                0xbe840406u);   // while old B is active, replace parent by wider ENTRY
    missing(compile(code), 11);
}

// Exercise the actual DATA MUST boundary without an independent spill/relay emission refusal
// hiding a broken first-half meet. The hand-established v0 entry is the genuine producer input;
// subsequent ownership comes from decoded production transfers, not invented word definitions.
namespace {
Rdna2Inst inst(std::initializer_list<uint32_t> words) {
    const std::vector<uint32_t> encoded(words);
    return rdna2_decode_one(encoded.data(), encoded.size());
}
ActiveVectorWords first_half() {
    // The one supplied word is genuine under producer ENTRY EXEC.
    ActiveVectorWords state;
    state.enabled = state.current_is_entry = state.current_within_entry = true;
    state.current = state.entry_words = {0};
    advance_active_vector_words(state, inst({0xbe84047eu}), {{4, 2}}, {}, 4, true, false);
    advance_active_vector_words(state, inst({0x7dea0080u}), {}, {}, -1, true, false);
    advance_active_vector_words(state, inst({0x7e040281u}), {}, {2}, -1, true, true);
    advance_active_vector_words(state, inst({0x8afe7e04u}), {{126, 2}}, {}, -1, true, false);
    return state;
}
void selected_peer_overwrite(ActiveVectorWords& state) {
    // This exact encoded instruction writes lane 1 irrespective of current B EXEC.
    // The absent s9 input cannot preserve the genuine first-half word in active A.
    const auto peer = inst({0xd7610002u, 0x00010209u});
    EXPECT_EQ(peer.fmt, Rdna2Format::VOP3);
    EXPECT_EQ(peer.opcode, 0x361u);
    EXPECT_EQ(peer.dst.value, 2);
    EXPECT_FALSE(active_vector_sources_defined(state, peer, {2}, {}, {0}, {}, true));
    advance_active_vector_words(state, peer, {}, {2}, -1, false, false);
}
void finish_second_half(ActiveVectorWords& state) {
    advance_active_vector_words(state, inst({0x7e040282u}), {}, {2}, -1, true, true);
    advance_active_vector_words(state, inst({0xbefe0404u}), {{126, 2}}, {}, -1, true, false);
}
}   // namespace
TEST(ComplementaryExecData, FirstHalfMeetBothPredecessorsRetainAndPublishTheGenuineWord) {
    auto left = first_half();
    auto right = first_half();
    ASSERT_EQ(left.complement.parent, 4);
    ASSERT_TRUE(left.complement.first_words.contains(2));
    ASSERT_FALSE(left.current.contains(2));   // B has no complete writer yet.
    meet_active_vector_words(left, right);
    finish_second_half(left);
    EXPECT_TRUE(left.current.contains(2));
    EXPECT_TRUE(left.masks[4].contains(2));
}
TEST(ComplementaryExecData, FirstHalfMeetOneLostWordCannotBorrowTheOtherPredecessor) {
    auto left = first_half();
    auto right = first_half();
    selected_peer_overwrite(right);
    ASSERT_TRUE(left.complement.first_words.contains(2));
    ASSERT_FALSE(right.complement.first_words.contains(2));
    ASSERT_EQ(left.complement.parent, right.complement.parent);
    ASSERT_EQ(left.complement.parents, right.complement.parents);
    ASSERT_EQ(left.complement.current_parents, right.complement.current_parents);
    meet_active_vector_words(left, right);
    finish_second_half(left);
    EXPECT_FALSE(left.current.contains(2));
    EXPECT_FALSE(left.masks[4].contains(2));
}
TEST(ComplementaryExecData, FirstHalfMeetBothLostWordsRemainUncovered) {
    auto left = first_half();
    auto right = first_half();
    selected_peer_overwrite(left);
    selected_peer_overwrite(right);
    meet_active_vector_words(left, right);
    finish_second_half(left);
    EXPECT_FALSE(left.current.contains(2));
    EXPECT_FALSE(left.masks[4].contains(2));
}
