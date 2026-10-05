// test_self_segment_key (#339) — the SELF data-segment map key is the SCE self-segment id in flag
// bits [31:20], masked to 12 bits. Without the mask, a data segment whose flags carry any bit >= 32
// keys the map by a garbage index, so every phdr lookup misses and the segment silently falls back to
// `elf_base + p_offset` (mapping garbage bytes). This pins the mask math (matches Kyty Elf.cpp).
#include "../src/self/module.hpp"

#include <gtest/gtest.h>

#include <cstdint>

using namespace prosper;

TEST(SelfSegmentKey, RealMessengerFlagsKeyTheId) {
    // Real Messenger-style flags (data flag 0x800 + id in [31:20], no upper bits): key == the id.
    EXPECT_EQ(self_segment_key(0x002804), 0u) << "flags 0x002804 -> id 0";
    EXPECT_EQ(self_segment_key(0x102804), 1u) << "flags 0x102804 -> id 1";
    EXPECT_EQ(self_segment_key(0x302804), 3u) << "flags 0x302804 -> id 3";
    EXPECT_EQ(self_segment_key(0xb02804), 0xbu) << "flags 0xb02804 -> id 0xb";
}

TEST(SelfSegmentKey, IdFieldIsTwelveBits) {
    // The id field is 12 bits [31:20]; its max is 0xFFF.
    EXPECT_EQ(self_segment_key(0xFFF00000u), 0xFFFu) << "max 12-bit id 0xFFF";
}

TEST(SelfSegmentKey, HighFlagBitsDoNotLeakIntoTheKey) {
    // The bug: a flag bit >= 32 must NOT leak into the key. Bit 40 set alongside id 1 -> still id 1
    // (unmasked this would be 0x100001, a garbage index that misses every phdr lookup).
    constexpr uint64_t kBit40AndId1 = 0x10000102804ull;
    // The arm's premise, stated as an assertion of its own: the UNMASKED key really is garbage, so a
    // mask regression below cannot be explained away as an input that would have been fine anyway.
    EXPECT_EQ(kBit40AndId1 >> 20, 0x100001ull)
        << "sanity: the UNMASKED key would be garbage 0x100001";
    EXPECT_EQ(self_segment_key(kBit40AndId1), 1u)
        << "bit 40 set + id 1 -> masked to id 1 (not 0x100001)";
    EXPECT_EQ(self_segment_key(0xFFFFFFFF00000000ull | 0x702804), 7u)
        << "all upper bits set + id 7 -> id 7";
}