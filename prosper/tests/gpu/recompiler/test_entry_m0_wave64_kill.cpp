// test_entry_m0_wave64_kill — a register that saved entry M0 and was then reused for data is data
// again after a CFG-dispatcher block edge, when the Wave64 MUST analysis proves the data write on
// every path.
//
// The dispatcher re-arms the entry-M0 token (#3133) at each block entry from a MAY dataflow with no
// KILL (`entry_m0_in`, rdna2_emit_cfg.cpp), so before this rule a register that ever held the token
// rejected at its first read in any later block, even after the program had overwritten it with
// ordinary data. Kena's compute program `0x5008dc0000` saves M0 into s14 at pc85, reuses s14 as a
// mask half and then as a loop counter (`s_mov_b32 s14, 0` at pc490) and was refused at the
// counter's first read in the next block (pc491, `s_cmp_lt_u32 s14, s79`).
//
// The Wave64 MUST scalar-word analysis is the KILL: a word in it at a block entry had a
// value-publishing scalar write after the last save on every path, so the token cannot be live.
//
// WHAT EACH ARM KILLS:
//   DataWriteOnEveryPathIsData      the missing KILL (the pre-fix behaviour rejects it)
//   TokenAloneStillRejects          a KILL that drops the token unconditionally (the read would
//                                   silently become the save's zero placeholder)
//   DataWriteOnOnePathStillRejects  a MAY-style KILL that fires when only one path wrote data
//
// Every kernel appends kTail (kernel 17d of test_rdna2_to_spirv), an irreducible loop that forces
// the CFG dispatcher, and puts the read at a join (below), in a later case than the save. Execution needs
// a device that can REQUIRE a 64-lane compute subgroup (RADV yes, CI's lavapipe no), so it is gated;
// the compile-level assertions run everywhere.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <gtest/gtest.h>
#include "fixtures/compute_runner.h"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <vector>

using namespace prosper::gpu;

namespace {

constexpr uint32_t kLanes = 64;
constexpr uint32_t kOpSwitch = 251;

// Every kernel reads s2 at the HEADER of a two-trip loop, the shape Kena's counter loop has: a
// backward-branch target starts its own dispatcher case, entered from the prologue and from the
// back edge. (A forward diamond does not do: the dispatcher structures it inside one case, so the
// token never crosses a case edge there.)
//   pc0 s_mov_b32 s2, m0          save entry M0: s2 holds the token
//   pc1 <s_mov_b32 s2, 41 | s_nop>
//   pc2 s_mov_b32 s3, 0
//   pc3 v_mov_b32 v3, s2          L: the read
//   pc4 <s_nop | s_mov_b32 s2, 41>
//   pc5 s_add_u32 s3, s3, 1
//   pc6 s_cmp_lt_u32 s3, 2
//   pc7 s_cbranch_scc1 L
//   s_mov_b64 exec, -1 | buffer_store_dword v3 -> out[x]
constexpr uint32_t kWrite = 0xBE8203A9u;   // s_mov_b32 s2, 41
constexpr uint32_t kNop = 0xBF800000u;     // s_nop 0
#define LOOP_KERNEL(before, inside)                                                              \
    {0xBE82037Cu, before, 0xBE830380u, 0x7E060202u, inside, 0x80838103u, 0xBF0A8203u,           \
     0xBF85FFFBu, 0xBEFE04C1u, 0xE0702000u, 0x80020300u}
// Written before the loop: data on both edges into the header.
const uint32_t kDataWriteOnEveryPath[] = LOOP_KERNEL(kWrite, kNop);
// Never written: the token on both edges.
const uint32_t kTokenAlone[] = LOOP_KERNEL(kNop, kNop);
// Written only inside the loop, after the read: data on the back edge, the token on the entry edge.
const uint32_t kDataWriteOnOnePath[] = LOOP_KERNEL(kNop, kWrite);
#undef LOOP_KERNEL
const uint32_t kTail[] = {
    0x7e040280u, 0x7c020300u, 0xbf860001u, 0x7e040281u, 0x7d840100u,
    0xbf870001u, 0xbf82fffdu, 0x7e040d02u, 0xbf810000u,
};

bool has_opcode(const std::vector<uint32_t>& spv, uint32_t opcode) {
    for (size_t i = 5; i < spv.size();) {
        const uint32_t words = spv[i] >> 16u;
        if (words == 0 || i + words > spv.size()) return false;
        if ((spv[i] & 0xffffu) == opcode) return true;
        i += words;
    }
    return false;
}

// out[] is binding 3 (run_compute's cbuf1), addressed through the direct V# at s[8:11].
ShaderResourceTable output_table() {
    ShaderResourceTable table;
    ShaderResource out{};
    out.cls = ResourceClass::ConstantBuffer;
    out.format = DataFormat::Uint32;
    out.num_components = 1;
    out.binding = 3;
    out.stride = 4;
    out.sgpr_base = 8;
    table.resources.push_back(out);
    return table;
}

template <size_t N>
std::vector<uint32_t> compile(const uint32_t (&prefix)[N]) {
    std::vector<uint32_t> code(std::begin(prefix), std::end(prefix));
    code.insert(code.end(), std::begin(kTail), std::end(kTail));
    ComputeShaderConfig config;
    config.local_x = 64;
    config.wave_size = 64;
    config.native_subgroup_size = 64;
    const ShaderResourceTable table = output_table();
    return recompile_compute(code.data(), code.size(), &table, config);
}

}   // namespace

TEST(EntryM0Wave64Kill, DataWriteOnEveryPathIsData) {
    const std::vector<uint32_t> spv = compile(kDataWriteOnEveryPath);
    ASSERT_FALSE(spv.empty())
        << "s2 was overwritten with data after the M0 save on the only path to the read";
    EXPECT_TRUE(has_opcode(spv, kOpSwitch)) << "it lowered through the CFG dispatcher";
    if (!prosper::test::default_compute_required_subgroup_supported(64u, kLanes))
        GTEST_SKIP() << "the device cannot require a 64-lane compute subgroup";
    std::vector<uint32_t> out;
    prosper::test::run_compute(spv, std::vector<float>(kLanes, 0.0f), kLanes, kLanes, {},
                               std::vector<uint32_t>(kLanes, 0xdeadbeefu), &out, kLanes, nullptr,
                               nullptr, nullptr, 64u);
    ASSERT_EQ(out.size(), kLanes);
    for (uint32_t lane = 0; lane < kLanes; ++lane)
        EXPECT_EQ(out[lane], 41u) << "lane " << lane << ": the reloaded data word, not a zero";
}

TEST(EntryM0Wave64Kill, TokenAloneStillRejects) {
    EXPECT_TRUE(compile(kTokenAlone).empty())
        << "a read that only the save reaches must keep rejecting, not read a zero placeholder";
}

TEST(EntryM0Wave64Kill, DataWriteOnOnePathStillRejects) {
    EXPECT_TRUE(compile(kDataWriteOnOnePath).empty())
        << "the data write is skipped on one path, so the token may still be live at the read";
}
