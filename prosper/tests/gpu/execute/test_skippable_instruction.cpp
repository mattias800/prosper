// test_skippable_instruction -- program_may_skip_by_scalar_branch (src/gpu/execute/skippable_instruction).
//
// build_stage_table binds a null image for a T# whose words are not a descriptor only when scalar
// branch decisions can finish the program without executing the consuming instruction (#4796). Both
// directions matter: a false "skippable" turns a refused, visibly broken program into a silently
// wrong one (the #4801 review's alpha-kill case), and a false "must execute" leaves admission at the
// mercy of stale descriptor-ring bytes. Every program here is real gfx1030 machine code, decoded by
// the production walker rather than hand-built instruction records.
#include "gpu/execute/skippable_instruction.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

using namespace prosper::gpu;

namespace {

constexpr uint32_t kNop = 0xBF800000u;   // s_nop 0
constexpr uint32_t kEnd = 0xBF810000u;   // s_endpgm
constexpr uint32_t kCmpS0Zero = 0xBF068000u;   // s_cmp_eq_u32 s0, 0
constexpr uint32_t kSample0 = 0xF0800F08u;   // image_sample v[8:11], v[0:1], s[12:19], s[20:23]
constexpr uint32_t kSample1 = 0x00A30000u;
constexpr uint32_t kSetpc = 0xBE802000u;   // s_setpc_b64 s[0:1]

constexpr uint32_t sopp(uint32_t opcode, int16_t simm) {
    return 0xBF800000u | (opcode << 16) | static_cast<uint16_t>(simm);
}
constexpr uint32_t kBranch = 0x02, kScc0 = 0x04, kScc1 = 0x05, kVccz = 0x06, kExecz = 0x08;
constexpr uint32_t kCmpxGt = 0x7C280300u;   // v_cmpx_gt_f32 v0, v1 (an alpha kill)

std::vector<Rdna2Inst> decode(const std::vector<uint32_t>& words) {
    std::vector<Rdna2Inst> out;
    rdna2_walk(words.data(), words.size(), out);
    return out;
}

TEST(SkippableInstruction, StraightLineMustExecute) {
    const auto program = decode({kSample0, kSample1, kEnd});
    EXPECT_FALSE(program_may_skip_by_scalar_branch(program, 0));
}

TEST(SkippableInstruction, ForwardBranchOverTheUseSkipsIt) {
    // pc0 cmp; pc1 s_cbranch_scc1 -> pc4; pc2 sample; pc4 endpgm
    const auto program = decode({kCmpS0Zero, sopp(kScc1, 2), kSample0, kSample1, kEnd});
    EXPECT_TRUE(program_may_skip_by_scalar_branch(program, 2));
    EXPECT_FALSE(program_may_skip_by_scalar_branch(program, 1)) << "the branch itself always runs";
}

TEST(SkippableInstruction, TheMergeOfADiamondMustExecute) {
    // pc0 cmp; pc1 scc1 -> pc4; pc2 nop; pc3 s_branch -> pc4; pc4 sample; pc6 endpgm
    const auto program =
        decode({kCmpS0Zero, sopp(kScc1, 2), kNop, sopp(kBranch, 0), kSample0, kSample1, kEnd});
    EXPECT_FALSE(program_may_skip_by_scalar_branch(program, 4)) << "both arms reach the sample";
    EXPECT_TRUE(program_may_skip_by_scalar_branch(program, 2)) << "one arm skips the nop";
}

TEST(SkippableInstruction, ALoopBodyCanRunZeroTimes) {
    // Kena's shape (#4775): pc1 cmp; pc2 s_cbranch_scc0 -> pc6 (exit); pc3 sample; pc5 s_branch -> pc1
    const auto program =
        decode({kNop, kCmpS0Zero, sopp(kScc0, 3), kSample0, kSample1, sopp(kBranch, -5), kEnd});
    EXPECT_TRUE(program_may_skip_by_scalar_branch(program, 3));
    EXPECT_FALSE(program_may_skip_by_scalar_branch(program, 1))
        << "the loop test runs at least once";
}

TEST(SkippableInstruction, CodeBehindAnUnconditionalBranchIsSkipped) {
    // pc0 s_branch -> pc3; pc1 sample (dead); pc3 endpgm
    const auto program = decode({sopp(kBranch, 2), kSample0, kSample1, kEnd});
    EXPECT_TRUE(program_may_skip_by_scalar_branch(program, 1));
}

TEST(SkippableInstruction, AnIncompleteGraphAnswersFalse) {
    // Indirect control flow: no scan over direct displacements can bound the graph.
    EXPECT_FALSE(program_may_skip_by_scalar_branch(
        decode({kCmpS0Zero, sopp(kScc1, 3), kSample0, kSample1, kSetpc, kEnd}), 2));
    // A branch into code the walk never decoded.
    EXPECT_FALSE(program_may_skip_by_scalar_branch(
        decode({kCmpS0Zero, sopp(kScc1, 40), kSample0, kSample1, kEnd}), 2));
    // A use pc that is the middle of an instruction, or past the program.
    const auto program = decode({kCmpS0Zero, sopp(kScc1, 2), kSample0, kSample1, kEnd});
    EXPECT_FALSE(program_may_skip_by_scalar_branch(program, 3));
    EXPECT_FALSE(program_may_skip_by_scalar_branch(program, 99));
    EXPECT_FALSE(program_may_skip_by_scalar_branch({}, 0));
}

TEST(SkippableInstruction, AMaskBranchDoesNotMakeTheUseSkippable) {
    // #4801 review: an alpha kill (v_cmpx, s_cbranch_execz to the end) lets a WAVE skip the sample,
    // but every visible pixel executes it. Same shape with the scalar branch is skippable.
    const auto alpha_kill = decode({kCmpxGt, sopp(kExecz, 2), kSample0, kSample1, kEnd});
    EXPECT_FALSE(program_may_skip_by_scalar_branch(alpha_kill, 2));
    const auto scalar = decode({kCmpS0Zero, sopp(kScc1, 2), kSample0, kSample1, kEnd});
    EXPECT_TRUE(program_may_skip_by_scalar_branch(scalar, 2));
    // A divergent if (vccz) likewise.
    const auto divergent = decode({kCmpS0Zero, sopp(kVccz, 2), kSample0, kSample1, kEnd});
    EXPECT_FALSE(program_may_skip_by_scalar_branch(divergent, 2));
}

TEST(SkippableInstruction, AnSccBranchOnTheLaneMaskIsAMaskBranch) {
    // #4801 re-review: LLVM's early-terminate shape. s_andn2_b64 exec, exec, vcc sets SCC from the
    // lane mask, and s_cbranch_scc0 to the end is an alpha kill, not a draw-uniform decision.
    constexpr uint32_t kAndn2ExecVcc = 0x8AFE6A7Eu;   // s_andn2_b64 exec, exec, vcc
    EXPECT_FALSE(program_may_skip_by_scalar_branch(
        decode({kAndn2ExecVcc, sopp(kScc0, 2), kSample0, kSample1, kEnd}), 2));
    // A compare that reads EXEC itself is a mask test too.
    constexpr uint32_t kCmpExecZero = 0xBF12807Eu;   // s_cmp_eq_u64 exec, 0
    EXPECT_FALSE(program_may_skip_by_scalar_branch(
        decode({kCmpExecZero, sopp(kScc1, 2), kSample0, kSample1, kEnd}), 2));
    // The uniform control: the same branch fed by a compare of SGPR data is a skip.
    EXPECT_TRUE(program_may_skip_by_scalar_branch(
        decode({kCmpS0Zero, sopp(kScc0, 2), kSample0, kSample1, kEnd}), 2));
    // A compare separated from the branch only by instructions that cannot write SCC still counts.
    EXPECT_TRUE(program_may_skip_by_scalar_branch(
        decode({kCmpS0Zero, kNop, sopp(kScc0, 2), kSample0, kSample1, kEnd}), 3));
    // An intervening scalar ALU op is a possible SCC writer of unknown meaning: not uniform.
    EXPECT_FALSE(program_may_skip_by_scalar_branch(
        decode({kCmpS0Zero, kAndn2ExecVcc, sopp(kScc0, 2), kSample0, kSample1, kEnd}), 3));
}

TEST(SkippableInstruction, AnSccDefinitionFromAnotherBlockIsUnknown) {
    // pc0 andn2 exec (SCC from the mask); pc1 scc1 -> pc3; pc2 cmp s0; pc3 scc0 -> pc6;
    // pc4 sample; pc6 endpgm. The compare right before pc3 is uniform, but pc3 is also reached by
    // the jump from pc1, carrying the mask-derived SCC, so its definition is unknown.
    constexpr uint32_t kAndn2ExecVcc = 0x8AFE6A7Eu;
    const auto program = decode(
        {kAndn2ExecVcc, sopp(kScc1, 1), kCmpS0Zero, sopp(kScc0, 2), kSample0, kSample1, kEnd});
    EXPECT_FALSE(program_may_skip_by_scalar_branch(program, 4));
}

TEST(SkippableInstruction, AScalarSkipStillWinsPastAMaskBranch) {
    // pc0 cmp; pc1 scc1 -> pc6 (end); pc2 execz -> pc5; pc3 sample; pc5 nop; pc6 endpgm. The mask
    // branch alone cannot avoid the sample, but the scalar decision before it can.
    const auto program =
        decode({kCmpS0Zero, sopp(kScc1, 4), sopp(kExecz, 2), kSample0, kSample1, kNop, kEnd});
    EXPECT_TRUE(program_may_skip_by_scalar_branch(program, 3));
}

TEST(SkippableInstruction, AQueryAppendsTheDiscardTail) {
    // pc0 cmp; pc1 scc1 -> pc5 (a tail AFTER the first s_endpgm); pc2 sample; pc4 endpgm;
    // pc5 endpgm. rdna2_walk stops at pc4, so only the appended tail lets the branch resolve.
    const std::vector<uint32_t> words = {kCmpS0Zero, sopp(kScc1, 3), kSample0,
                                         kSample1,   kEnd,           kEnd};
    EXPECT_FALSE(program_may_skip_by_scalar_branch(decode(words), 2))
        << "PREMISE: without the tail the branch names undecoded code";
    const std::vector<uint32_t> walked(words.begin(), words.begin() + 5);
    SkippableInstructionQuery query(nullptr, walked, words.data(), words.size());
    EXPECT_TRUE(query.may_skip(2));
    SkippableInstructionQuery no_live(nullptr, walked, nullptr, 0);
    EXPECT_FALSE(no_live.may_skip(2)) << "the live stream is what supplies the tail";
}

TEST(SkippableInstruction, CachedAnswersFollowTheOwner) {
    const std::vector<uint32_t> skip = {kCmpS0Zero, sopp(kScc1, 2), kSample0, kSample1, kEnd};
    const std::vector<uint32_t> always = {kSample0, kSample1, kNop, kNop, kEnd};
    auto owner = std::make_shared<int>(1);
    EXPECT_TRUE(SkippableInstructionQuery(owner, skip, nullptr, 0).may_skip(2));
    EXPECT_TRUE(SkippableInstructionQuery(owner, always, nullptr, 0).may_skip(2))
        << "a live owner's cached answer is served: the owner names the program";
    // An expired owner's answers are never served to a new program, even at a reused address
    // (make_shared of the same size commonly reuses it; the weak reference catches it either way).
    owner.reset();
    auto fresh = std::make_shared<int>(2);
    EXPECT_FALSE(SkippableInstructionQuery(fresh, always, nullptr, 0).may_skip(2));
}

TEST(SkippableInstruction, AStreamThatNeverEndsAnswersFalse) {
    // The walk stops at the buffer end with no s_endpgm: falling off it is not a program end.
    EXPECT_FALSE(program_may_skip_by_scalar_branch(
        decode({kCmpS0Zero, sopp(kScc1, 2), kSample0, kSample1, kNop}), 2));
}

}   // namespace
