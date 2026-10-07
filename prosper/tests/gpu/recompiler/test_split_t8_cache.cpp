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
#include "gpu/execute/split_t8_proof.hpp"

#include <gtest/gtest.h>

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
