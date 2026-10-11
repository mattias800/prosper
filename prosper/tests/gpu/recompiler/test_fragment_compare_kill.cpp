// #4895: a non-CMPX explicit SGPR compare replaces its low word, including inactive
// lanes. Ignoring that write refuses dead B32 mask domains at a structured join;
// treating its sibling as replaced can instead hide a surviving physical-word read.
#include "fixtures/fragment_compare_kill.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <gtest/gtest.h>
#include <cstdint>
#include <string>
#include <vector>

namespace prosper::gpu {
namespace {
namespace fixture = prosper::test::fragment_compare_kill;

std::vector<Rdna2Inst> decode(const std::vector<uint32_t>& code) {
    std::vector<Rdna2Inst> result;
    EXPECT_EQ(rdna2_walk(code.data(), code.size(), result), code.size());
    return result;
}

bool dead(const std::vector<uint32_t>& code, int reg, ScalarMergeBlocker* blocker = nullptr) {
    return sgpr_dead_at_merge(decode(code), 0, reg, ScalarMergeProof::AnyRead, blocker);
}

std::vector<uint32_t> compile(const std::vector<uint32_t>& code, uint64_t address) {
    return recompile_fragment(code.data(), code.size(), nullptr, nullptr, UINT32_MAX, nullptr, true,
                              {RecompileDiagnosticStage::Fragment, address});
}

void join_refuses(const std::vector<uint32_t>& code, uint64_t address) {
    EXPECT_TRUE(compile(code, address).empty());
    EXPECT_NE(last_terminal_reject_reason(address).find("live b32 mask domain differs"),
              std::string::npos)
        << "an unrelated refusal does not test the join proof";
}
} // namespace

TEST(FragmentCompareKill, SyntheticCompareHasExplicitLowDestination) {
    const auto instructions = decode(fixture::program());
    ASSERT_FALSE(instructions.empty());
    EXPECT_EQ(instructions[0].fmt, Rdna2Format::VOPC);
    EXPECT_EQ(instructions[0].dst.kind, OperandKind::SGPR);
    EXPECT_EQ(instructions[0].dst.value, 20);
    EXPECT_FALSE(vopc_is_cmpx(instructions[0].opcode));
    EXPECT_FALSE(instructions[0].has_modifier);
}

TEST(FragmentCompareKill, FreshLowWordMakesOldLowWordDead) {
    EXPECT_TRUE(dead(
        {fixture::compare(), fixture::compare_sd(20), fixture::mov(30, 20), fixture::end}, 20));
}

TEST(FragmentCompareKill, SiblingRemainsLiveAfterWave32Compare) {
    EXPECT_FALSE(dead(
        {fixture::compare(), fixture::compare_sd(20), fixture::mov(30, 21), fixture::end}, 21));
}

TEST(FragmentCompareKill, LastOrdinarySgprLowWordCanBeReplaced) {
    EXPECT_TRUE(dead(
        {fixture::compare(), fixture::compare_sd(105), fixture::mov(30, 105), fixture::end}, 105));
}

TEST(FragmentCompareKill, CompareReadsOldScalarSourceBeforeReplacingIt) {
    ScalarMergeBlocker blocker;
    EXPECT_FALSE(dead(
        {fixture::compare(), fixture::compare_sd(20, 20, true), fixture::mov(30, 20), fixture::end},
        20, &blocker));
    EXPECT_EQ(blocker.pc, 0u);
    EXPECT_STREQ(blocker.kind, "source-dword");
}

TEST(FragmentCompareKill, EarlierPhysicalReadKeepsOldWordLive) {
    EXPECT_FALSE(dead(
        {fixture::mov(30, 20), fixture::compare(), fixture::compare_sd(20), fixture::end}, 20));
}

TEST(FragmentCompareKill, CmpxDoesNotReplaceEncodedScalarDestination) {
    EXPECT_FALSE(dead(
        {fixture::compare(0xd2), fixture::compare_sd(20), fixture::mov(30, 20), fixture::end}, 20));
}

TEST(FragmentCompareKill, EveryForwardSuccessorMustReplaceBeforeRead) {
    EXPECT_TRUE(
        dead({0xbf880003u, fixture::compare(), fixture::compare_sd(20), 0xbf820002u,
              fixture::compare(), fixture::compare_sd(20), fixture::mov(30, 20), fixture::end},
             20));
    EXPECT_FALSE(dead({0xbf880002u, fixture::compare(), fixture::compare_sd(20),
                       fixture::mov(30, 20), fixture::end},
                      20));
}

TEST(FragmentCompareKill, UnrelatedDestinationDoesNotRetireOldMask) {
    EXPECT_FALSE(dead(
        {fixture::compare(), fixture::compare_sd(22), fixture::mov(30, 20), fixture::end}, 20));
}

TEST(FragmentCompareKill, StructuredWave32JoinCompilesAfterReplacement) {
    EXPECT_FALSE(compile(fixture::program(), 0x489501u).empty());
}

TEST(FragmentCompareKill, UnrelatedVectorBetweenJoinAndReplacementCompiles) {
    EXPECT_FALSE(compile(fixture::program(true), 0x489502u).empty());
}

TEST(FragmentCompareKill, StructuredJoinRefusesAReadBeforeReplacement) {
    auto code = fixture::program();
    code.insert(code.begin() + 7, fixture::mov(126, 20));
    join_refuses(code, 0x489503u);
}

TEST(FragmentCompareKill, StructuredJoinRefusesCmpxAsReplacement) {
    auto code = fixture::program();
    code[7] = fixture::compare(0xd2);
    join_refuses(code, 0x489504u);
}

TEST(FragmentCompareKill, StructuredJoinRefusesAPathBypassingReplacement) {
    auto code = fixture::program();
    code.insert(code.begin() + 7, 0xbf880002u); // second EXECZ skips the new compare
    join_refuses(code, 0x489505u);
}
} // namespace prosper::gpu
