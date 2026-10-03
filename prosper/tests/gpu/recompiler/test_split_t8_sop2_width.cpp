// test_split_t8_sop2_width -- a 32-bit scalar ALU result clobbers one SGPR, not its neighbour.
//
// THE DEFECT. The split-T# proof asks whether any instruction before a descriptor load may write the
// pointer register the load goes through. For SOP2 it assumed every result was an SGPR PAIR, so a
// plain `s_add_i32 s3, ...` was taken to clobber s4 as well. Assassin's Creed Black Flag Resynced's
// fragment program `0x407edf1400` adds into s3 and then loads its T# through s[4:5], so the proof was
// refused for a write that never touched the pointer.
//
// WHAT EACH TEST KILLS:
//   Sop2DestWidths                       a 32-bit opcode reports two dwords, or a 64-bit one reports one
//   AddIntoRegisterBelowThePointerKeepsProof   the pair assumption comes back
//   PairWriteOverlappingThePointerIsRefused    a 64-bit write stops being treated as a pair
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

using namespace prosper::gpu;

namespace {

alignas(16) uint32_t g_table[16];

// The entry user data is s0..s3; the table pointer is s[2:3].
std::vector<SrtUse> uses_for(const std::vector<uint32_t>& code) {
    for (uint32_t i = 0; i < 16; ++i) g_table[i] = 0xD1000000u + i;
    const uint64_t base = reinterpret_cast<uint64_t>(g_table);
    const uint32_t seed[4] = { 0u, 0u, static_cast<uint32_t>(base),
                               static_cast<uint32_t>(base >> 32u) };
    std::vector<SrtUse> result;
    resolve_dynamic_fetch(code.data(), code.size(), seed, 4, 0, &result);
    return result;
}

bool has_use(const std::vector<SrtUse>& uses, uint32_t pc) {
    return std::any_of(uses.begin(), uses.end(),
                       [pc](const SrtUse& u) { return u.kind == 0 && u.use_pc == pc; });
}

// pc 0      <scalar write>
// pc 1-2    s_load_dwordx8 s[4:11], s[2:3], 0      pc 3-4  s_load_dwordx4 s[12:15], s[2:3], 0x20
// pc 5-6    image_store ... s[8:15]  (the consumer)
std::vector<uint32_t> program(uint32_t scalar_write) {
    return {
        scalar_write,
        0xF40C0101u, 0xFA000000u,
        0xF4080301u, 0xFA000020u,
        0xF0200108u, 0x00020009u,
        0xBF810000u,
    };
}

}  // namespace

TEST(SplitT8Sop2Width, Sop2DestWidths) {
    EXPECT_EQ(rdna2_sop2_dest_dwords(0x02u), 1u) << "s_add_i32 writes one SGPR";
    EXPECT_EQ(rdna2_sop2_dest_dwords(0x0au), 1u) << "s_cselect_b32 writes one SGPR";
    EXPECT_EQ(rdna2_sop2_dest_dwords(0x0eu), 1u) << "s_and_b32 writes one SGPR";
    EXPECT_EQ(rdna2_sop2_dest_dwords(0x1eu), 1u) << "s_lshl_b32 writes one SGPR";
    EXPECT_EQ(rdna2_sop2_dest_dwords(0x0bu), 2u) << "s_cselect_b64 writes a pair";
    EXPECT_EQ(rdna2_sop2_dest_dwords(0x0fu), 2u) << "s_and_b64 writes a pair";
    EXPECT_EQ(rdna2_sop2_dest_dwords(0x1fu), 2u) << "s_lshl_b64 writes a pair";
    EXPECT_EQ(rdna2_sop2_dest_dwords(0x7fu), 2u) << "an unknown opcode fails closed to a pair";
}

TEST(SplitT8Sop2Width, AddIntoRegisterBelowThePointerKeepsProof) {
    // s_add_i32 s1, 1, 1: a 32-bit write one register below the pointer pair s[2:3].
    const auto uses = uses_for(program(0x81018181u));
    EXPECT_TRUE(has_use(uses, 5u))
        << "a 32-bit scalar add into s1 does not touch the pointer in s[2:3]";
}

TEST(SplitT8Sop2Width, PairWriteOverlappingThePointerIsRefused) {
    // s_and_b64 into s[1:2]: a pair write that DOES overlap the pointer.
    const auto uses = uses_for(program(0x87818181u));
    EXPECT_FALSE(has_use(uses, 5u)) << "a 64-bit write that overlaps the pointer must be refused";
}
