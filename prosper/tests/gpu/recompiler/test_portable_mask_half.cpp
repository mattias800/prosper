// #4926: a physical HIGH replacement must not leave the previous complete saved-mask Bool alive.
// Execute actual portable production modules; missing original words remain refusals, not zeros.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/portable_mask_half.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

namespace {
using namespace prosper::gpu;
namespace f = prosper::test::portable_mask_half;
constexpr uint64_t all = UINT64_MAX;
void expect_values(const std::vector<uint32_t>& code, uint64_t mask, bool dispatcher = false,
                   uint32_t workgroup = 0) {
    const auto module = f::compile(code, dispatcher);
    ASSERT_FALSE(module.empty());
    bpermute_oracle::Interpreter oracle(module);
    const auto actual =
        oracle.run_buffers(64, {{3, std::vector<uint32_t>(64, 0xdeadbeef)}}, 3, workgroup);
    ASSERT_TRUE(oracle.error.empty()) << oracle.error;
    ASSERT_EQ(actual.size(), 64u);
    for (uint32_t lane = 0; lane < 64; ++lane)
        EXPECT_EQ(actual[lane], ((mask >> lane) & 1u) ? 9u : 0u) << "lane " << lane;
    for (size_t offset = 5; offset < module.size();) {
        const auto width = module[offset] >> 16;
        ASSERT_NE(width, 0u);
        EXPECT_NE(module[offset] & 0xffffu, 339u) << "no ballot may manufacture saved words";
        offset += width;
    }
}
struct Emitter {
    SpirvCompute b;
    RegState rs;
    Emitter() {
        b.begin(4, nullptr, 64, 1, 1, 64, 0, true, true);
        rs.exec = b.btrue();
        rs.vcc = b.bfalse();
        rs.sreg_bool[20] = b.btrue();
        rs.sreg[20] = b.uconst(UINT32_MAX);
        rs.sreg[21] = b.uconst(0);
        rs.absent_sgpr_reads_fabricated_zero = true;
    }
    bool run(bool dispatcher = false) {
        const std::vector<uint32_t> code{f::mov(126, 20, true), f::end};
        std::vector<Rdna2Inst> decoded;
        rdna2_walk(code.data(), code.size(), decoded);
        const auto exp = [](RegState&, const Rdna2Inst&) { return false; };
        if (dispatcher)
            return emit_cfg_state_machine(b, rs, decoded, {}, nullptr, true, false, exp,
                                          code.data(), code.size());
        return emit_body(b, rs, decoded, {}, nullptr, true, false, exp, code.data(), code.size());
    }
};
} // namespace

TEST(PortableMaskHalf, UnchangedWholeBoolRestoreKeepsAllLanes) {
    expect_values(f::program(false), all);
}
TEST(PortableMaskHalf, UnchangedNumericRestoreKeepsAllLanes) {
    expect_values(f::program(true), all);
}
TEST(PortableMaskHalf, HighNumericReplacementChangesOnlyItsPhysicalHalf) {
    for (uint32_t word : {0u, 0x80000001u})
        expect_values(f::program(true, 1, word), (uint64_t(word) << 32) | UINT32_MAX);
}
TEST(PortableMaskHalf, LowNumericReplacementKeepsTheOriginalHighWord) {
    for (uint32_t word : {0u, 0x80000001u})
        expect_values(f::program(true, 0, word), 0xffffffff00000000ull | word);
}
TEST(PortableMaskHalf, HighReplacementCannotUseAnAbsentOriginalLowWord) {
    EXPECT_TRUE(f::compile(f::program(false, 1)).empty());
}
TEST(PortableMaskHalf, LowReplacementCannotUseAnAbsentOriginalHighWord) {
    EXPECT_TRUE(f::compile(f::program(false, 0)).empty());
}
TEST(PortableMaskHalf, CompleteZeroPairIsAGenuineEmptyMask) {
    auto code = f::program(true, 0);
    code.insert(code.begin() + 4, f::mov(21, 128));
    expect_values(code, 0);
}
TEST(PortableMaskHalf, CopiedNumericGenerationReceivesItsOwnHighReplacement) {
    expect_values(f::program(true, 1, 0, false, true), UINT32_MAX);
}
TEST(PortableMaskHalf, CopiedBoolDoesNotAuthenticateAMissingPhysicalHalf) {
    EXPECT_TRUE(f::compile(f::program(false, 1, 0, false, true)).empty());
}
TEST(PortableMaskHalf, EarlierWholeBoolCopySurvivesAnotherPairsReplacement) {
    auto code = f::program(false, -1, 0, false, true);
    code.insert(code.begin() + 3, {f::mov(21, 255), 0u});
    expect_values(code, all);
}
TEST(PortableMaskHalf, CurrentNumericWordsPrecedeAnOlderCompetingBool) {
    Emitter e;
    const auto old = e.rs.exec;
    ASSERT_TRUE(e.run());
    EXPECT_NE(e.rs.exec, old);
}
TEST(PortableMaskHalf, FabricatedCurrentWordCannotFallBackToAnOlderBool) {
    for (int marked : {20, 21}) {
        Emitter e;
        e.rs.sreg_merge_placeholder.insert(marked);
        EXPECT_FALSE(e.run()) << "word " << marked;
    }
}
TEST(PortableMaskHalf, EntryInputCannotResurrectAnOverwrittenMissingWord) {
    for (int missing : {20, 21}) {
        Emitter e;
        e.rs.sreg_input[missing] = e.rs.sreg.at(missing);
        e.rs.sreg.erase(missing);
        e.rs.sreg_written.insert(missing);
        EXPECT_FALSE(e.run()) << "word " << missing;
    }
}
TEST(PortableMaskHalf, OneArmCannotProvideAMissingPhysicalWordAtTheJoin) {
    EXPECT_TRUE(f::compile(f::one_arm(false)).empty());
}
TEST(PortableMaskHalf, BothReachingGenerationsHaveGenuineNumericWords) {
    expect_values(f::one_arm(true), UINT32_MAX, false, 0);
    expect_values(f::one_arm(true), all, false, 1);
}
TEST(PortableMaskHalf, CountedPostludeUsesRetainedNumericWordsWithoutABallot) {
    expect_values(f::program(true, 1, 0, true), UINT32_MAX);
}
TEST(PortableMaskHalf, CountedPostludeDoesNotRecoverAnAbsentPhysicalHalf) {
    EXPECT_TRUE(f::compile(f::program(false, 1, 0, true)).empty());
}
TEST(PortableMaskHalf, EarlyLaneExitsCannotChangeTheRetainedNumericGeneration) {
    expect_values(f::early_lane_exits(true), UINT32_MAX);
}
TEST(PortableMaskHalf, EarlyLaneExitsCannotAuthorizeALateMissingWord) {
    EXPECT_TRUE(f::compile(f::early_lane_exits(false)).empty());
}
TEST(PortableMaskHalf, UnchangedWholeBoolSurvivesEarlyLaneExits) {
    expect_values(f::early_lane_exits(false, false), all);
}
TEST(PortableMaskHalf, DispatcherKeepsTheSameCurrentPairContract) {
    expect_values(f::program(true, 1), UINT32_MAX, true);
    expect_values(f::program(false), all, true);
    EXPECT_TRUE(f::compile(f::program(false, 1), true).empty());
}

TEST(PortableMaskHalf, ATrackedB32PredicateIsNotACompleteSavedPair) {
    Emitter e;
    e.rs.sreg.clear();
    e.rs.sreg_bool_b32.insert(20);
    EXPECT_FALSE(e.run());
}

TEST(PortableMaskHalf, Wave32AndNggRetainTheirExistingAliasPolicy) {
    for (int domain = 0; domain < 3; ++domain) {
        Emitter e;
        if (domain == 0) e.b.wave_size = 32;
        if (domain == 1) e.b.ngg_workgroup_shell = true;
        if (domain == 2) e.b.ngg_one_lane = true;
        const std::vector<uint32_t> code{f::mov(21, 128)};
        std::vector<Rdna2Inst> ins;
        rdna2_walk(code.data(), code.size(), ins);
        const auto saved = snapshot_saved_b64_masks(e.b, e.rs, ins.front());
        EXPECT_FALSE(saved.portable_numeric_replacement);
        e.rs.sreg[21] = e.b.uconst(0);
        expire_saved_b64_mask(e.rs, saved, 21);
        EXPECT_TRUE(e.rs.sreg_bool.contains(20));
    }
}
TEST(PortableMaskHalf, MaskSlotReloadKeepsTheEstablishedPortableRepresentation) {
    Emitter e;
    const std::vector<uint32_t> code{0xd7600015u, 0x00010f12u};
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code.data(), code.size(), ins);
    const auto saved = snapshot_saved_b64_masks(e.b, e.rs, ins.front());
    ASSERT_TRUE(saved.portable_numeric_replacement);
    expire_saved_b64_mask(e.rs, saved, 21, 21, true);
    EXPECT_TRUE(e.rs.sreg_bool.contains(20));
    // The SAME inventory's genuine DATA reload does end the old complete mask.
    expire_saved_b64_mask(e.rs, saved, 21, -1, false);
    EXPECT_FALSE(e.rs.sreg_bool.contains(20));
}
TEST(PortableMaskHalf, NumericHalfReplacementExpiresTheAliasForOtherMaskConsumers) {
    for (int replaced : {0, 1}) {
        auto code = f::program(true, replaced, 0x80000001u);
        code[code.size() - 6] = 0x87fe14c1u; // S_AND_B64 EXEC, -1, s[20:21]
        expect_values(code, replaced ? 0x80000001ffffffffull : 0xffffffff80000001ull);
    }
}
TEST(PortableMaskHalf, OtherMaskConsumersCannotRestoreAnAbsentUntouchedHalf) {
    auto code = f::program(false, 1);
    code[code.size() - 6] = 0x87fe14c1u;
    EXPECT_TRUE(f::compile(code).empty());
}
