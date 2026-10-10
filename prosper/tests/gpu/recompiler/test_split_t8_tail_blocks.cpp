// test_split_t8_tail_blocks -- a divergent early-out placed after the first s_endpgm must not
// refuse the split-T# proof.
//
// THE DEFECT. The proof that a T# assembled from adjacent scalar loads still holds at its consumer
// walks the whole program, but `rdna2_walk` stops at the first `s_endpgm`. Compilers put a discard
// path (`s_mov_b32 exec_lo, 0; exp null; s_endpgm`) after that end and branch to it from the body, so
// the branch target had no decoded instruction and the proof returned "unresolved". Assassin's Creed
// Black Flag Resynced's fragment program `0x407eecf000` has exactly this shape (its `image_load` T# is
// s[4:11], loaded by an x8 and an x4 scalar load) and the draw was dropped for it.
//
// WHAT EACH TEST KILLS:
//   TailBlockAfterFirstEndKeepsProof     the tail is refused again (the defect)
//   TailBranchingBackIntoBodyIsRefused   a tail that can re-enter the body is admitted
//   UndecodableTailIsRefused             an unknown encoding in the tail is treated as code
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "split_t8_fold_harness.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

using namespace prosper::gpu;

namespace {

// Entry user data is s0..s11; the table pointer is s[10:11].
std::vector<SrtUse> uses_for(const std::vector<uint32_t>& code) {
    return test::split_t8_uses_for(code.data(), code.size(), 10, 12u);
}

bool has_use(const std::vector<SrtUse>& uses, uint32_t pc) {
    return test::split_t8_has_image_use(uses, pc);
}

// The title's body followed by `tail`.
std::vector<uint32_t> program(std::vector<uint32_t> tail) {
    std::vector<uint32_t> code = test::split_t8_tail_block_body();
    code.insert(code.end(), tail.begin(), tail.end());
    return code;
}

const std::vector<uint32_t> kDiscardTail = {
    0xBEFE0480u, 0xF8001890u, 0x00000000u, 0xBF810000u,   // s_mov_b32 exec_lo, 0; exp null; s_endpgm
};

}  // namespace

TEST(SplitT8TailBlocks, TailBlockAfterFirstEndKeepsProof) {
    EXPECT_TRUE(has_use(uses_for(program(kDiscardTail)), 10u))
        << "the T# is loaded on every path to the image_load; the discard block only ends the wave";
}

TEST(SplitT8TailBlocks, TailBranchingBackIntoBodyIsRefused) {
    // The discard block jumps back to pc 3: it could run, then re-enter the body.
    const auto code = program({ 0xBEFE0480u, 0xBF82FFECu /* s_branch pc3 */, 0xBF810000u });
    EXPECT_FALSE(has_use(uses_for(code), 10u));
}

TEST(SplitT8TailBlocks, UndecodableTailIsRefused) {
    EXPECT_FALSE(has_use(uses_for(program({ 0xFFFFFFFFu, 0xFFFFFFFFu })), 10u));
}
