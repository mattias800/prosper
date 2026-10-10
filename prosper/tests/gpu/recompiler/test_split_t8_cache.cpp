// test_split_t8_cache -- the split-T# proof's code-only half is cached; its address half is not.
//
// THE DEFECT. mapped_split_t8_reaches_use() re-parsed and re-analysed the whole program on every
// scalar fold -- once per dispatch, every frame -- so it was capped at 2048 dwords as a cost guard.
// Assassin's Creed Black Flag Resynced's compute programs 0x407f243000 (3720 dwords) and
// 0x407f27b700 (2912) pair two T#s in one s_load_dwordx16 in branchy code, so the proof refused
// them for their size alone and their image ops were never published. The CFG analysis now runs
// once per program version and consumer (SplitT8ProofCache, owned by the decoded program) and the
// cap is a memory guard.
//
// WHAT EACH TEST KILLS:
//   ProgramPastTheOldCapIsProved        the 2048-dword cap comes back
//   CachedAndUncachedAgree              the cache changes an answer
//   CacheHitStillChecksTheWordRead      a cache hit skips the per-call address check
//   ProgramPastTheMemoryGuardIsRefused  the memory guard is dropped
//   ChangedTailAtTheSameAddressIsReanalysed  a hit trusts bytes past the first s_endpgm it never
//                                       re-read (#4712 review B1: the decode cache validates only
//                                       the body, so same body + new tail reached the old answer)
//   ProducerThroughAnOutOfRangePointerIsRefused  documents the bound; the later per-call check also
//                                       refuses s106+, so only a sanitizer sees the code-only half's
//                                       out-of-range read if its bound is dropped (#4712 review B2)
//   SaveexecOverTheDescriptorIsRefusedByTheProof  a saveexec destination is not treated as written
//   ReservedSop1EncodingIsAnUnknownWrite  a reserved SOP1 opcode >= 0x20 is charged only its SDST
//   CacheKeyIncludesTheBaseAndProducers  the cache key drops tbase or the producer pcs
//   FoldAnalysesPastTheOldCap            the call site clamps the program to 2048 dwords again
//   SameLengthEditIsReanalysed           a cache hit is checked by length only
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/execute/split_t8_proof.hpp"
#include "split_t8_fold_harness.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

using namespace prosper::gpu;

namespace {

alignas(16) std::array<uint32_t, 16> g_table{};

// pc 0-1  s_load_dwordx8 s[4:11], s[2:3], 0     pc 2-3  s_load_dwordx4 s[12:15], s[2:3], 0x20
// pc 4    s_nop                                 pc 5-6  image_store ... s[8:15]
// then `padding` s_nops, then s_endpgm. The T# in s[8:15] is words 4..7 of the first load and
// words 0..3 of the second.
std::vector<uint32_t> program(size_t padding) {
    std::vector<uint32_t> code = {
        0xF40C0101u, 0xFA000000u, 0xF4080301u, 0xFA000020u, 0xBF800000u, 0xF0200108u, 0x00020009u,
    };
    code.insert(code.end(), padding, 0xBF800000u);
    code.push_back(0xBF810000u);
    return code;
}

struct Inputs {
    std::array<uint32_t, 4> user{};
    std::array<uint32_t, 8> source_pc{0, 0, 0, 0, 2, 2, 2, 2};
    std::array<uint64_t, 8> source_addr{};
};

Inputs inputs() {
    Inputs in;
    const auto base = reinterpret_cast<uint64_t>(g_table.data());
    in.user = {0u, 0u, static_cast<uint32_t>(base), static_cast<uint32_t>(base >> 32u)};
    for (uint32_t lane = 0; lane < 4; ++lane) {
        in.source_addr[lane] = base + 16u + uint64_t{4} * lane;   // first load, words 4..7
        in.source_addr[lane + 4u] = base + 0x20u + uint64_t{4} * lane;   // second load, words 0..3
    }
    return in;
}

bool proves(const std::vector<uint32_t>& code, const Inputs& in, SplitT8ProofCache* cache) {
    return mapped_split_t8_reaches_use(code.data(), code.size(), 5u, 8, in.source_pc,
                                       in.source_addr, in.user.data(), 4u, 0u, {}, cache);
}

}   // namespace

TEST(SplitT8Cache, ProgramPastTheOldCapIsProved) {
    const auto code = program(3000);
    ASSERT_GT(code.size(), 2048u);
    const auto cache = make_split_t8_proof_cache();
    EXPECT_TRUE(proves(code, inputs(), cache.get()))
        << "a 3008-dword program is analysed once, not refused for its size";
}

TEST(SplitT8Cache, CachedAndUncachedAgree) {
    const auto code = program(8);
    const auto cache = make_split_t8_proof_cache();
    EXPECT_TRUE(proves(code, inputs(), nullptr));
    EXPECT_TRUE(proves(code, inputs(), cache.get())) << "first call fills the cache";
    EXPECT_TRUE(proves(code, inputs(), cache.get())) << "second call is a cache hit";
}

TEST(SplitT8Cache, CacheHitStillChecksTheWordRead) {
    const auto code = program(8);
    const auto cache = make_split_t8_proof_cache();
    Inputs in = inputs();
    ASSERT_TRUE(proves(code, in, cache.get()));
    in.source_addr[0] += 4u;   // the fold claims word 5 where the use holds word 4
    EXPECT_FALSE(proves(code, in, cache.get()))
        << "the cached structure must still be checked against the words the fold read";
    EXPECT_TRUE(proves(code, inputs(), cache.get())) << "and the right words still prove";
}

TEST(SplitT8Cache, ProgramPastTheMemoryGuardIsRefused) {
    const auto code = program(kSplitT8MaxDwords);
    ASSERT_GT(code.size(), kSplitT8MaxDwords);
    EXPECT_FALSE(proves(code, inputs(), nullptr));
}

namespace {

// One fixed buffer, so both programs live at the same address and reach the same decoded program.
alignas(256) std::array<uint32_t, 64> g_code{};

bool image_load_published(const std::array<uint32_t, 4>& tail) {
    const auto& body = test::split_t8_tail_block_body();
    const auto end = std::copy(body.begin(), body.end(), g_code.begin());
    std::copy(tail.begin(), tail.end(), end);
    return test::split_t8_has_image_use(
        test::split_t8_uses_for(g_code.data(), body.size() + tail.size(), 10, 12u), 10u);
}

}   // namespace

TEST(SplitT8Cache, ChangedTailAtTheSameAddressIsReanalysed) {
    // s_mov_b32 exec_lo, 0; exp null; s_endpgm -- a closed discard block.
    constexpr std::array<uint32_t, 4> kClosed = {0xBEFE0480u, 0xF8001890u, 0x00000000u,
                                                 0xBF810000u};
    // s_mov_b32 exec_lo, 0; s_branch pc3; s_nop; s_endpgm -- it can re-enter the body. The same
    // length as kClosed, so a re-check that compares only lengths reuses the stale proof.
    constexpr std::array<uint32_t, 4> kReenters = {0xBEFE0480u, 0xBF82FFECu, 0xBF800000u,
                                                   0xBF810000u};
    ASSERT_TRUE(image_load_published(kClosed)) << "the closed tail proves the T#";
    EXPECT_FALSE(image_load_published(kReenters))
        << "same address, same body, new tail: the cached proof must not be reused";
    EXPECT_TRUE(image_load_published(kClosed)) << "and the closed tail proves again";
}

TEST(SplitT8Cache, ProducerThroughAnOutOfRangePointerIsRefused) {
    // The first producer reads its pointer from s[106:107] (VCC), outside the tracked SGPRs.
    auto code = program(8);
    code[0] = 0xF40C0135u;   // s_load_dwordx8 s[4:11], s[106:107], 0
    const auto cache = make_split_t8_proof_cache();
    EXPECT_FALSE(proves(code, inputs(), cache.get()));
    EXPECT_FALSE(proves(code, inputs(), nullptr));
}

TEST(SplitT8Cache, SaveexecOverTheDescriptorIsRefusedByTheProof) {
    // pc 4 becomes s_and_saveexec_b64 s[8:9], vcc: it overwrites the first two T# words. Asked
    // directly, so the fold's own bookkeeping cannot refuse it first.
    auto code = program(8);
    code[4] = 0xBE88246Au;
    EXPECT_FALSE(proves(code, inputs(), nullptr));
    const auto cache = make_split_t8_proof_cache();
    EXPECT_FALSE(proves(code, inputs(), cache.get()));
}

TEST(SplitT8Cache, ReservedSop1EncodingIsAnUnknownWrite) {
    // pc 4 becomes SOP1 opcode 0x23 (reserved on RDNA2) naming s[20:21], away from the T# in
    // s[8:15]. It has no defined effect, so the proof may not assume it writes only s[20:21].
    auto code = program(8);
    ASSERT_TRUE(proves(code, inputs(), nullptr)) << "the s_nop control proves";
    code[4] = 0xBE94236Au;
    EXPECT_FALSE(proves(code, inputs(), nullptr));
    // A defined opcode in the same range that names s[20:21] (s_and_saveexec_b64) still proves.
    code[4] = 0xBE94246Au;
    EXPECT_TRUE(proves(code, inputs(), nullptr));
}

TEST(SplitT8Cache, CacheKeyIncludesTheBaseAndProducers) {
    const auto code = program(8);
    const Inputs in = inputs();
    const auto cache = make_split_t8_proof_cache();
    ASSERT_TRUE(proves(code, in, cache.get()));
    // The same words claimed for s[9:16]: s16 is never loaded, so this use does not prove.
    EXPECT_FALSE(mapped_split_t8_reaches_use(code.data(), code.size(), 5u, 9, in.source_pc,
                                             in.source_addr, in.user.data(), 4u, 0u, {},
                                             cache.get()))
        << "a different T# base is a different question";
    // Lane 4 claimed from the first load at the second load's address. The cached entry's producers
    // accept that address, so only a key that includes the producer pcs refuses it.
    Inputs wrong = in;
    wrong.source_pc[4] = 0u;
    EXPECT_FALSE(proves(code, wrong, cache.get()))
        << "different producers are a different question";
    EXPECT_TRUE(proves(code, in, cache.get()));
}

TEST(SplitT8Cache, FoldAnalysesPastTheOldCap) {
    // Through the real scalar fold, which clamps the span it hands the proof.
    const auto code = program(3000);
    ASSERT_GT(code.size(), 2048u);
    EXPECT_TRUE(test::split_t8_has_image_use(test::split_t8_uses_for(code), 5u))
        << "a 3008-dword program's split T# is published";
}

TEST(SplitT8Cache, SameLengthEditIsReanalysed) {
    // One buffer, one cache: the second program has the same length and an EXEC save over s[8:9]
    // where the first had an s_nop, so only a content check sees that the entry no longer applies.
    alignas(16) static std::array<uint32_t, 16> code{};
    const auto original = program(8);
    ASSERT_LE(original.size(), 16u);
    std::copy(original.begin(), original.end(), code.begin());
    const Inputs in = inputs();
    const auto cache = make_split_t8_proof_cache();
    auto ask = [&] {
        return mapped_split_t8_reaches_use(code.data(), original.size(), 5u, 8, in.source_pc,
                                           in.source_addr, in.user.data(), 4u, 0u, {}, cache.get());
    };
    ASSERT_TRUE(ask());
    code[4] = 0xBE88246Au;   // s_and_saveexec_b64 s[8:9], vcc
    EXPECT_FALSE(ask()) << "same length, different bytes: the cached proof must not be reused";
}
