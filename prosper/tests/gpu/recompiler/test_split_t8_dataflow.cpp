// test_split_t8_dataflow -- the split-T# proof follows the CFG, not index ranges.
//
// THE DEFECT. The proof used to scan lexical intervals: the instructions between a load and the
// consumer, and between a back edge's target and its branch. Each of the reviewed sequences below
// reaches the consumer along a path those intervals do not contain, with a descriptor word that no
// longer holds the bits the CPU fold published. The proof is now a must-dataflow over the CFG:
// every path into the consumer has to deliver the same load word to each descriptor register.
//
// Every refused arm is paired with a control that differs in one register and must be admitted, so
// a refusal cannot come from the fold failing to see the shape at all.
//
// WHAT EACH PAIR KILLS:
//   CopyReplayedAfterItsSourceChanged     a loop before the consumer re-runs a copy (#4203 review)
//   LoopPathOutsideTheLexicalBody         a loop exits its [target, branch] range (#4203 review)
//   ReplayedCopyThroughAForwardBranch     #4218's lexical loop range missed the copy (#4218 review)
//   VectorReadlaneWritesAnSgpr            v_readfirstlane/v_readlane SGPR writes (#4203 review)
//   MimgWritersAreFailClosed              an unlisted image writer keeps the proof (#4208 review)
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

using namespace prosper::gpu;

namespace {

alignas(16) uint32_t g_table[16];

std::vector<SrtUse> uses_for(const std::vector<uint32_t>& code) {
    for (uint32_t i = 0; i < 16; ++i) g_table[i] = 0xD1000000u + i;
    const uint64_t base = reinterpret_cast<uint64_t>(g_table);
    const uint32_t seed[2] = { static_cast<uint32_t>(base), static_cast<uint32_t>(base >> 32u) };
    std::vector<SrtUse> result;
    resolve_dynamic_fetch(code.data(), code.size(), seed, 2, 0, &result);
    return result;
}

bool has_use(const std::vector<uint32_t>& code, uint32_t pc) {
    const auto uses = uses_for(code);
    return std::any_of(uses.begin(), uses.end(),
                       [pc](const SrtUse& u) { return u.kind == 0 && u.use_pc == pc; });
}

// Encodings (llvm-mc -mcpu=gfx1030).
constexpr uint32_t kLoadX8 = 0xF40C0100u, kLoadX8Off = 0xFA000000u;   // s_load_dwordx8 s[4:11], s[0:1], 0
constexpr uint32_t kLoadX4 = 0xF4080600u, kLoadX4Off = 0xFA000020u;   // s_load_dwordx4 s[24:27], s[0:1], 0x20
constexpr uint32_t kImageLoad0 = 0xF0000308u, kImageLoad1 = 0x00050409u; // image_load ... s[20:27]
constexpr uint32_t kEnd = 0xBF810000u;
constexpr uint32_t mov(uint32_t dst, uint32_t src) { return 0xBE800300u | (dst << 16) | src; }
constexpr uint32_t mov_one(uint32_t dst) { return 0xBE800381u | (dst << 16); }  // s_mov_b32 sD, 1
constexpr uint32_t sopp(uint32_t op, int at, int target) {
    return 0xBF800000u | (op << 16) | (static_cast<uint32_t>(target - (at + 1)) & 0xFFFFu);
}
constexpr uint32_t kBranch = 0x02, kScc0 = 0x04, kScc1 = 0x05, kVccnz = 0x07;

// s_mov_b32 s20..s23, s8..s11: words 4-7 of the x8 load become the low half of the T# s[20:27].
void copies(std::vector<uint32_t>& c) {
    for (uint32_t k = 0; k < 4; ++k) c.push_back(mov(20 + k, 8 + k));
}

// pc 0-3 loads; pc 4 L: copies; pc 8 <clobber>; pc 9 s_cbranch_scc1 L; pc 10 the consumer.
std::vector<uint32_t> copy_replay(uint32_t clobber) {
    std::vector<uint32_t> c{kLoadX8, kLoadX8Off, kLoadX4, kLoadX4Off};
    copies(c);
    c.push_back(mov_one(clobber));
    c.push_back(sopp(kScc1, 9, 4));
    c.insert(c.end(), {kImageLoad0, kImageLoad1, kEnd});
    return c;
}

// pc 0-7 loads and copies; pc 8 the consumer; pc 10 s_cbranch_scc0 X; pc 11 B: s_cbranch_scc1 use;
// pc 12 s_branch END; pc 13 X: <clobber>; pc 14 s_branch B; pc 15 END. The path 10 -> 13 -> 14 ->
// 11 -> 8 leaves the lexical loop [8, 11] and re-enters the consumer.
std::vector<uint32_t> escaping_loop(uint32_t clobber) {
    std::vector<uint32_t> c{kLoadX8, kLoadX8Off, kLoadX4, kLoadX4Off};
    copies(c);
    c.insert(c.end(), {kImageLoad0, kImageLoad1});
    c.push_back(sopp(kScc0, 10, 13));
    c.push_back(sopp(kScc1, 11, 8));
    c.push_back(sopp(kBranch, 12, 15));
    c.push_back(mov_one(clobber));
    c.push_back(sopp(kBranch, 14, 11));
    c.push_back(kEnd);
    return c;
}

// The #4218 review sequence: pc 4 C: copies; pc 8 s_cbranch_scc1 U; pc 9 J: s_branch C; pc 10 U:
// the consumer; pc 12 <clobber>; pc 13 s_cbranch_vccnz J. The path 13 -> 9 -> 4 re-runs the copies.
std::vector<uint32_t> forward_branch_replay(uint32_t clobber) {
    std::vector<uint32_t> c{kLoadX8, kLoadX8Off, kLoadX4, kLoadX4Off};
    copies(c);
    c.push_back(sopp(kScc1, 8, 10));
    c.push_back(sopp(kBranch, 9, 4));
    c.insert(c.end(), {kImageLoad0, kImageLoad1});
    c.push_back(mov_one(clobber));
    c.push_back(sopp(kVccnz, 13, 9));
    c.push_back(kEnd);
    return c;
}

// pc 0-7 loads and copies; pc 8 the consumer; pc 10.. <body>; then a branch back to the consumer.
std::vector<uint32_t> loop_after_use(std::initializer_list<uint32_t> body) {
    std::vector<uint32_t> c{kLoadX8, kLoadX8Off, kLoadX4, kLoadX4Off};
    copies(c);
    c.insert(c.end(), {kImageLoad0, kImageLoad1});
    c.insert(c.end(), body.begin(), body.end());
    const int at = static_cast<int>(c.size());
    c.push_back(sopp(kScc1, at, 8));
    c.push_back(kEnd);
    return c;
}

// pc 0-7 loads and copies; pc 8-9 an image op on an unrelated descriptor s[40:47]; pc 10 consumer.
std::vector<uint32_t> image_op_before_use(uint32_t opcode) {
    std::vector<uint32_t> c{kLoadX8, kLoadX8Off, kLoadX4, kLoadX4Off};
    copies(c);
    c.push_back(0xF0000000u | ((opcode & 0x7Fu) << 18) | (0xFu << 8) | ((opcode >> 7) & 1u));
    c.push_back(0x000A0400u);
    c.insert(c.end(), {kImageLoad0, kImageLoad1, kEnd});
    return c;
}

}  // namespace

TEST(SplitT8Dataflow, CopyReplayedAfterItsSourceChanged) {
    EXPECT_FALSE(has_use(copy_replay(8), 10u))
        << "the second pass copies s8 = 1 into s20, so the published T# is stale";
    EXPECT_TRUE(has_use(copy_replay(30), 10u))
        << "control: re-running the copies over unchanged sources yields the same T#";
}

TEST(SplitT8Dataflow, LoopPathOutsideTheLexicalBody) {
    EXPECT_FALSE(has_use(escaping_loop(21), 8u))
        << "s21 is rewritten on a path that leaves [target, branch] and re-enters the consumer";
    EXPECT_TRUE(has_use(escaping_loop(30), 8u))
        << "control: the same CFG writing an unrelated register keeps the proof";
}

TEST(SplitT8Dataflow, ReplayedCopyThroughAForwardBranch) {
    EXPECT_FALSE(has_use(forward_branch_replay(8), 10u))
        << "13 -> 9 -> 4 copies s8 = 1 into s20 and reaches the consumer again";
    EXPECT_TRUE(has_use(forward_branch_replay(30), 10u))
        << "control: the replayed copies carry the original load words";
}

TEST(SplitT8Dataflow, VectorReadlaneWritesAnSgpr) {
    // v_readfirstlane_b32 sN, v0 is VOP1 0x02 with N in the VDST field.
    EXPECT_FALSE(has_use(loop_after_use({0x7E2A0500u}), 8u))
        << "v_readfirstlane_b32 s21, v0 rewrites a descriptor word inside the loop";
    EXPECT_TRUE(has_use(loop_after_use({0x7E3C0500u}), 8u))
        << "control: v_readfirstlane_b32 s30, v0 leaves the descriptor alone";
    // v_readlane_b32 sN, v0, 0 is VOP3 0x360.
    EXPECT_FALSE(has_use(loop_after_use({0xD7600015u, 0x00010100u}), 8u))
        << "v_readlane_b32 s21, v0, 0 rewrites a descriptor word inside the loop";
    EXPECT_TRUE(has_use(loop_after_use({0xD760001Eu, 0x00010100u}), 8u))
        << "control: v_readlane_b32 s30, v0, 0 leaves the descriptor alone";
}

TEST(SplitT8Dataflow, MimgWritersAreFailClosed) {
    // Every writer the old list missed: store_pck, store_mip_pck, cmpswap, inc, dec and the
    // float atomics, plus an unassigned slot inside the sample range.
    for (uint32_t op : {0x0Au, 0x0Bu, 0x10u, 0x1Bu, 0x1Cu, 0x1Du, 0x1Eu, 0x1Fu, 0x42u})
        EXPECT_FALSE(has_use(image_op_before_use(op), 10u)) << "image opcode 0x" << std::hex << op;
    for (uint32_t op : {0x00u, 0x0Eu, 0x20u, 0x47u, 0x60u, 0x6Fu})
        EXPECT_TRUE(has_use(image_op_before_use(op), 10u))
            << "control: image opcode 0x" << std::hex << op << " only reads";
}
