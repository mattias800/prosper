// test_split_t8_loop_defs -- a loop around a split-T# consumer has to protect the descriptor WORDS,
// not the registers they were copied from.
//
// THE DEFECT. #4203 admits a loop around the consumer only if its body leaves the descriptor words,
// the registers they were copied from, and the pointer they were loaded through alone. The sources
// need no protection of their own: a copy captures bits, so a loop that cannot re-run the load or
// the copy is unaffected by later writes to its sources, and a loop that can re-run either writes
// the descriptor words again and is refused by the word check. Protecting the sources refused a
// loop that merely reused a dead source register. Assassin's Creed Black Flag Resynced's fragment
// program `0x4005a2e500` does exactly that: it copies s12..s15 into the T#, then reloads s12
// inside its loop.
//
// WHAT EACH TEST KILLS:
//   LoopAfterTheCopiesMayReuseSourceRegisters    the sources are protected again
//   LoopThatReRunsTheCopiesIsRefused             a loop that re-runs the copies is admitted
//   LoopStillProtectsTheDescriptorWords          the descriptor words stop being protected
#include "gpu/execute/gpu_execute.hpp"

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

bool has_use(const std::vector<SrtUse>& uses, uint32_t pc) {
    return std::any_of(uses.begin(), uses.end(),
                       [pc](const SrtUse& u) { return u.kind == 0 && u.use_pc == pc; });
}

// pc 0-1 s_load_dwordx8 s[4:11], s[0:1], 0      pc 2-3 s_load_dwordx4 s[24:27], s[0:1], 0x20
// pc 4-7 s_mov_b32 s20..s23, s8..s11            (the T# is s[20:27])
// pc 8-9 image_load ... s[20:27]                (the consumer, use pc 8)
std::vector<uint32_t> prefix() {
    return {
        0xF40C0100u, 0xFA000000u,
        0xF4080600u, 0xFA000020u,
        0xBE940308u, 0xBE950309u, 0xBE96030Au, 0xBE97030Bu,
        0xF0000308u, 0x00050409u,
    };
}

std::vector<uint32_t> with(std::vector<uint32_t> code, std::initializer_list<uint32_t> tail) {
    code.insert(code.end(), tail.begin(), tail.end());
    return code;
}

}  // namespace

TEST(SplitT8LoopDefs, LoopAfterTheCopiesMayReuseSourceRegisters) {
    // pc 10 s_mov_b32 s9, 1 (s9 was copied into s21 long ago); pc 11 loop back to the consumer.
    const auto uses = uses_for(with(prefix(), { 0xBE890381u, 0xBF85FFFCu, 0xBF810000u }));
    EXPECT_TRUE(has_use(uses, 8u))
        << "a loop that cannot re-run the copies may reuse a source register";
}

TEST(SplitT8LoopDefs, LoopThatReRunsTheCopiesIsRefused) {
    // The branch returns to pc 4, so the copies run again and rewrite the descriptor words.
    const auto uses = uses_for(with(prefix(), { 0xBE890381u, 0xBF85FFF8u, 0xBF810000u }));
    EXPECT_FALSE(has_use(uses, 8u))
        << "a loop that re-runs the copies rewrites the descriptor words and must be refused";
}

TEST(SplitT8LoopDefs, LoopStillProtectsTheDescriptorWords) {
    // pc 10 s_mov_b32 s21, 1 rewrites a word of the T# itself.
    const auto uses = uses_for(with(prefix(), { 0xBE950381u, 0xBF85FFFCu, 0xBF810000u }));
    EXPECT_FALSE(has_use(uses, 8u)) << "a loop must never rewrite a descriptor word";
}
