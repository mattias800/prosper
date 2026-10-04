// test_rdna2_closed_tail — adversarial pins for rdna2_append_closed_tail_blocks.
//
// Compilers place a divergent early-out (discard, kill) AFTER the body's s_endpgm and branch to
// it from the body, so a branch target can name code the linear walk never decoded. The helper
// decodes every block reachable through such a target and appends it — but ONLY when the extra
// blocks are CLOSED. A wrong accept executes bytes that were never meant to run (padding,
// literal pools, the next shader) as instructions.
//
// Each refusal arm below mutates exactly one field of the accepted packet, and the accepted arm
// carries a same-shape control, so a failure names the exact admission rule that moved (per the
// recompiler folder's reject-contract guidance: a reject without a control is not evidence).
// Pure decode; no device, no tables. Branch arithmetic follows the decoder itself
// (target = pc + len_dwords + simm16 over SOPP 0xBF80 + (op << 16) + simm16).
#include "gpu/recompiler/rdna2_decode.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using namespace prosper::gpu;

namespace {

constexpr uint32_t kEndpgm = 0xBF810000u;  // SOPP s_endpgm (op 0x01)
constexpr uint32_t kNop = 0xBF800000u;     // SOPP s_nop (op 0x00)

// SOPP branch word: op 0x02 = s_branch, 0x04 = s_cbranch_scc0. simm16 is signed.
uint32_t sopp(uint32_t op, int simm) {
    return 0xBF800000u | (op << 16) | (static_cast<uint32_t>(simm) & 0xFFFFu);
}

// Walk code[0, n) linearly; the walk stops at the first s_endpgm, which is the `full` prefix
// the append helper requires (it refuses when the last walked instruction is not an end).
std::vector<Rdna2Inst> walk_body(const uint32_t* code, size_t n) {
    std::vector<Rdna2Inst> full;
    rdna2_walk(code, n, full);
    return full;
}

}  // namespace

TEST(Rdna2ClosedTail, AcceptsADivergentEarlyOutPastEndpgm) {
    // Body: mov s0,s1; cbranch_scc0 -> pc4; end. Tail at pc4: mov s1,s2; end. Padding nop at pc3
    // is never decoded — only branch targets past the end are.
    const uint32_t code[] = {
        0xBE800301u,  // pc0: s_mov_b32 s0, s1
        sopp(0x04u, 2),  // pc1: s_cbranch_scc0 -> pc4 (1 + 1 + 2)
        kEndpgm,        // pc2: s_endpgm (main_end = 3)
        kNop,           // pc3: padding between the end and the tail
        0xBE810302u,  // pc4: s_mov_b32 s1, s2
        kEndpgm,        // pc5: s_endpgm closes the tail run
    };
    std::vector<Rdna2Inst> full = walk_body(code, std::size(code));
    ASSERT_EQ(full.size(), 3u);
    ASSERT_TRUE(full.back().is_end);
    EXPECT_TRUE(rdna2_append_closed_tail_blocks(code, std::size(code), full));
    ASSERT_EQ(full.size(), 5u);
    EXPECT_EQ(full[3].pc, 4u);
    EXPECT_EQ(full[4].pc, 5u);
    EXPECT_TRUE(full[4].is_end);
    EXPECT_EQ(full[3].opcode, 0x03u);
    EXPECT_EQ(full[3].dst.value, 1);
}

TEST(Rdna2ClosedTail, BranchTargetPastTheBufferRefuses) {
    // Same body with the branch aimed at pc9 in a 6-dword buffer: nothing is appended.
    const uint32_t code[] = {
        0xBE800301u, sopp(0x04u, 7), kEndpgm, kNop, 0xBE810302u, kEndpgm,
        //             target = 1 + 1 + 7 = 9 >= 6 dwords --^
    };
    std::vector<Rdna2Inst> full = walk_body(code, std::size(code));
    ASSERT_EQ(full.size(), 3u);
    EXPECT_FALSE(rdna2_append_closed_tail_blocks(code, std::size(code), full));
    EXPECT_EQ(full.size(), 3u);
}

TEST(Rdna2ClosedTail, NegativeBranchTargetRefuses) {
    // Target 1 + 1 - 4 = -2 names no dword at all.
    const uint32_t code[] = {0xBE800301u, sopp(0x04u, -4), kEndpgm};
    std::vector<Rdna2Inst> full = walk_body(code, std::size(code));
    ASSERT_EQ(full.size(), 3u);
    EXPECT_FALSE(rdna2_append_closed_tail_blocks(code, std::size(code), full));
    EXPECT_EQ(full.size(), 3u);
}

TEST(Rdna2ClosedTail, BranchBackIntoTheBodyRefuses) {
    // The tail's own s_branch targets pc0: the tail could run before the body or re-enter it.
    const uint32_t code[] = {
        0xBE800301u, sopp(0x04u, 2), kEndpgm, kNop,
        sopp(0x02u, -5),  // pc4: s_branch -> pc0 (4 + 1 - 5), inside the body
        kEndpgm,
    };
    std::vector<Rdna2Inst> full = walk_body(code, std::size(code));
    ASSERT_EQ(full.size(), 3u);
    EXPECT_FALSE(rdna2_append_closed_tail_blocks(code, std::size(code), full));
    EXPECT_EQ(full.size(), 3u);
}

TEST(Rdna2ClosedTail, UnknownEncodingInTailRefuses) {
    // The tail target (pc4) decodes as an unknown prefix (top6 0x3F): garbage bytes stay undecoded.
    // pc1 target = 1 + 1 + 2 = 4.
    const uint32_t code[] = {
        0xBE800301u, sopp(0x04u, 2), kEndpgm, kNop,
        0xFFFFFFFFu,  // pc4: not any RDNA2 encoding
        kEndpgm,
    };
    std::vector<Rdna2Inst> full = walk_body(code, std::size(code));
    ASSERT_EQ(full.size(), 3u);
    EXPECT_FALSE(rdna2_append_closed_tail_blocks(code, std::size(code), full));
    EXPECT_EQ(full.size(), 3u);
}

TEST(Rdna2ClosedTail, TargetLandingMidInstructionRefuses) {
    // pc4..5 are one 2-dword VOP1 (v_mov_b32 v2, literal) whose second word is an s_nop. A
    // second branch aims at pc5, reinterpreting the literal as an instruction start: the sorted
    // tail would hold pc4 (len 2) immediately followed by pc5, so the fall-through edge is wrong.
    // pc0 target = 0 + 1 + 3 = 4; pc1 target = 1 + 1 + 3 = 5.
    const uint32_t fixed[] = {
        sopp(0x04u, 3), sopp(0x04u, 3), kEndpgm, kNop, 0x7E0402FFu, kNop, kEndpgm,
    };
    std::vector<Rdna2Inst> full = walk_body(fixed, std::size(fixed));
    ASSERT_EQ(full.size(), 3u);
    EXPECT_FALSE(rdna2_append_closed_tail_blocks(fixed, std::size(fixed), full));
    EXPECT_EQ(full.size(), 3u);
}

TEST(Rdna2ClosedTail, SameShapeWithWholeInstructionTargetAccepts) {
    // Positive control for the mid-instruction arm: aiming the second branch at pc6 (the real
    // end) instead of pc5 admits the tail, proving the refusal above is about the overlap.
    // pc0 target = 0 + 1 + 3 = 4; pc1 target = 1 + 1 + 4 = 6.
    const uint32_t code[] = {
        sopp(0x04u, 3), sopp(0x04u, 4), kEndpgm, kNop, 0x7E0402FFu, kNop, kEndpgm,
    };
    std::vector<Rdna2Inst> full = walk_body(code, std::size(code));
    ASSERT_EQ(full.size(), 3u);
    EXPECT_TRUE(rdna2_append_closed_tail_blocks(code, std::size(code), full));
    // Tail runs {pc4 (len 2)} and {pc6 (end)} join at pc6: 4 + 2 == 6, so the edge is exact.
    ASSERT_EQ(full.size(), 5u);
    EXPECT_EQ(full[3].pc, 4u);
    EXPECT_EQ(full[4].pc, 6u);
}

TEST(Rdna2ClosedTail, TargetExactlyAtMainEndIsDecoded) {
    // The boundary: a target naming the first dword past the end is queued (>= main_end) and, if
    // it closes immediately, appended.
    const uint32_t code[] = {0xBE800301u, sopp(0x04u, 1), kEndpgm, kEndpgm};
    std::vector<Rdna2Inst> full = walk_body(code, std::size(code));
    ASSERT_EQ(full.size(), 3u);
    EXPECT_TRUE(rdna2_append_closed_tail_blocks(code, std::size(code), full));
    ASSERT_EQ(full.size(), 4u);
    EXPECT_EQ(full[3].pc, 3u);
}

TEST(Rdna2ClosedTail, DegenerateInputsRefuseOrAcceptVacuously) {
    std::vector<Rdna2Inst> empty;
    const uint32_t one[] = {kEndpgm};
    EXPECT_FALSE(rdna2_append_closed_tail_blocks(nullptr, 0, empty));
    EXPECT_FALSE(rdna2_append_closed_tail_blocks(one, 0, empty));
    // A body with no past-end branches appends nothing and reports success.
    const uint32_t plain[] = {0xBE800301u, kEndpgm};
    std::vector<Rdna2Inst> full = walk_body(plain, std::size(plain));
    ASSERT_EQ(full.size(), 2u);
    EXPECT_TRUE(rdna2_append_closed_tail_blocks(plain, std::size(plain), full));
    EXPECT_EQ(full.size(), 2u);
    // A walk that never reaches s_endpgm is not a body at all.
    const uint32_t noend[] = {0xBE800301u};
    std::vector<Rdna2Inst> open = walk_body(noend, std::size(noend));
    ASSERT_EQ(open.size(), 1u);
    EXPECT_FALSE(rdna2_append_closed_tail_blocks(noend, std::size(noend), open));
}
