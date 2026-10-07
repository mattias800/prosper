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
//   ProducerThroughAnOutOfRangePointerIsRefused  the pointer-register bound leaves the code-only half
//                                       (#4712 review B2: the dataflow then reads past its array)
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/execute/split_t8_proof.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

using namespace prosper::gpu;

namespace {

alignas(16) uint32_t g_table[16];

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
    const auto base = reinterpret_cast<uint64_t>(g_table);
    in.user = {0u, 0u, static_cast<uint32_t>(base), static_cast<uint32_t>(base >> 32u)};
    for (uint32_t lane = 0; lane < 4; ++lane) {
        in.source_addr[lane] = base + 16u + 4u * lane;   // first load, words 4..7
        in.source_addr[lane + 4u] = base + 0x20u + 4u * lane;   // second load, words 0..3
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

// The tail-block program of test_split_t8_tail_blocks: pc3 x8 + pc5 x4 load the T# in s[4:11] through
// the entry pointer s[10:11]; pc10 image_load; pc16 branches to the block after the first s_endpgm.
constexpr uint32_t kBody[] = {
    0xBFA00001u, 0x7E000F02u, 0x7E020F03u, 0xF40C0005u, 0xFA000000u, 0xF4080205u, 0xFA000020u,
    0xBF8CC07Fu, 0xF4201A80u, 0xFA000000u, 0xF0000108u, 0x00010000u, 0xBF8C0070u, 0x3600006Au,
    0x7D840080u, 0x8AEA6A7Eu, 0xBF840004u, 0xBEFE046Au, 0xF8001890u, 0x00000000u, 0xBF810000u,
};
constexpr size_t kBodyDwords = sizeof(kBody) / sizeof(kBody[0]);

// One fixed buffer, so both programs live at the same address and reach the same decoded program.
alignas(256) uint32_t g_code[64];

bool image_load_published(const uint32_t* tail, size_t tail_dwords) {
    std::copy(kBody, kBody + kBodyDwords, g_code);
    std::copy(tail, tail + tail_dwords, g_code + kBodyDwords);
    for (uint32_t i = 0; i < 16; ++i) g_table[i] = 0xD1000000u + i;
    const auto base = reinterpret_cast<uint64_t>(g_table);
    uint32_t seed[12] = {};
    seed[10] = static_cast<uint32_t>(base);
    seed[11] = static_cast<uint32_t>(base >> 32u);
    std::vector<SrtUse> uses;
    resolve_dynamic_fetch(g_code, kBodyDwords + tail_dwords, seed, 12, 0, &uses);
    return std::any_of(uses.begin(), uses.end(),
                       [](const SrtUse& u) { return u.kind == 0 && u.use_pc == 10u; });
}

}   // namespace

TEST(SplitT8Cache, ChangedTailAtTheSameAddressIsReanalysed) {
    // s_mov_b32 exec_lo, 0; exp null; s_endpgm -- a closed discard block.
    constexpr uint32_t kClosed[] = {0xBEFE0480u, 0xF8001890u, 0x00000000u, 0xBF810000u};
    // s_mov_b32 exec_lo, 0; s_branch pc3; s_endpgm; s_nop -- it can re-enter the body.
    constexpr uint32_t kReenters[] = {0xBEFE0480u, 0xBF82FFECu, 0xBF810000u, 0xBF800000u};
    ASSERT_TRUE(image_load_published(kClosed, 4)) << "the closed tail proves the T#";
    EXPECT_FALSE(image_load_published(kReenters, 4))
        << "same address, same body, new tail: the cached proof must not be reused";
    EXPECT_TRUE(image_load_published(kClosed, 4)) << "and the closed tail proves again";
}

TEST(SplitT8Cache, ProducerThroughAnOutOfRangePointerIsRefused) {
    // The first producer reads its pointer from s[106:107] (VCC), outside the tracked SGPRs.
    auto code = program(8);
    code[0] = 0xF40C0135u;   // s_load_dwordx8 s[4:11], s[106:107], 0
    const auto cache = make_split_t8_proof_cache();
    EXPECT_FALSE(proves(code, inputs(), cache.get()));
    EXPECT_FALSE(proves(code, inputs(), nullptr));
}
