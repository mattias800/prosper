// test_split_t8_saveexec -- an EXEC save names its destination; it is not a relative SGPR write.
//
// THE DEFECT. The split-T# proof's may_write() treated every SOP1 opcode at or above 0x20 as able to
// write ANY scalar register, a rule meant for the M0-relative moves (s_movreld_*, s_movrelsd_2_b32).
// That range also holds every s_*_saveexec form, which writes only its named destination and EXEC,
// and which every structured `if` in a compute shader starts with. So the first branch after a
// descriptor load erased the load's provenance and the image op behind it was never published.
// Assassin's Creed Black Flag Resynced lost four compute programs this way (0x407f1fb700,
// 0x407f208200, 0x407f1ae000, 0x407f26b000): each loses its T# at an s_and_saveexec_b64.
// rdna2_escapes_decoded_effects() is the one shared rule for "effects the operands do not name"
// (#4529); the proof now uses it.
//
// WHAT EACH TEST KILLS:
//   SaveexecBetweenLoadAndUseKeepsProof   the opcode-range rule comes back
//   SaveexecOverTheDescriptorIsRefused    the saveexec destination stops being treated as written
//   RelativeMoveIsStillRefused            the relative moves stop being treated as unknown writes
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
    const auto base = reinterpret_cast<uint64_t>(g_table);
    const uint32_t seed[4] = {0u, 0u, static_cast<uint32_t>(base),
                              static_cast<uint32_t>(base >> 32u)};
    std::vector<SrtUse> result;
    resolve_dynamic_fetch(code.data(), code.size(), seed, 4, 0, &result);
    return result;
}

bool has_use(const std::vector<SrtUse>& uses, uint32_t pc) {
    return std::any_of(uses.begin(), uses.end(),
                       [pc](const SrtUse& u) { return u.kind == 0 && u.use_pc == pc; });
}

// pc 0-1    s_load_dwordx8 s[4:11], s[2:3], 0      pc 2-3  s_load_dwordx4 s[12:15], s[2:3], 0x20
// pc 4      <scalar write between the loads and the consumer>
// pc 5-6    image_store ... s[8:15]  (the consumer; its T# spans both loads)
std::vector<uint32_t> program(uint32_t scalar_write) {
    return {
        0xF40C0101u, 0xFA000000u,
        0xF4080301u, 0xFA000020u,
        scalar_write,
        0xF0200108u, 0x00020009u,
        0xBF810000u,
    };
}

}  // namespace

TEST(SplitT8Saveexec, SaveexecBetweenLoadAndUseKeepsProof) {
    // s_and_saveexec_b64 s[20:21], vcc: writes s20, s21 and EXEC, nothing the T# lives in.
    const auto uses = uses_for(program(0xBE94246Au));
    EXPECT_TRUE(has_use(uses, 5u))
        << "an EXEC save into s[20:21] does not touch the descriptor in s[8:15]";
}

TEST(SplitT8Saveexec, SaveexecOverTheDescriptorIsRefused) {
    // s_and_saveexec_b64 s[8:9], vcc: the saved mask overwrites the first two T# words.
    const auto uses = uses_for(program(0xBE88246Au));
    EXPECT_FALSE(has_use(uses, 5u)) << "an EXEC save into s[8:9] clobbers the descriptor";
}

TEST(SplitT8Saveexec, RelativeMoveIsStillRefused) {
    // s_movreld_b32 s0, s1 writes SGPR[0 + M0], which the encoding does not name.
    const auto uses = uses_for(program(0xBE803001u));
    EXPECT_FALSE(has_use(uses, 5u)) << "an M0-relative write may land in the descriptor";
}
