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
//   FragmentDispatcherAppliesTheSameKill  the three shapes above in a Wave64 pixel program
//                                   (compile level; the compute arm executes the value)
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
#include <string>
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
constexpr uint32_t kNop = 0xBF800000u;   // s_nop 0
#define LOOP_KERNEL(before, inside)                                                                \
    {0xBE82037Cu, before,      0xBE830380u, 0x7E060202u, inside,     0x80038103u,                  \
     0xBF0A8203u, 0xBF85FFFBu, 0xBEFE04C1u, 0xE0702000u, 0x80020300u}
// Written before the loop: data on both edges into the header.
const uint32_t kDataWriteOnEveryPath[] = LOOP_KERNEL(kWrite, kNop);
// Never written: the token on both edges.
const uint32_t kTokenAlone[] = LOOP_KERNEL(kNop, kNop);
// Written only inside the loop, after the read: data on the back edge, the token on the entry edge.
const uint32_t kDataWriteOnOnePath[] = LOOP_KERNEL(kNop, kWrite);
// The same three shapes as a Wave64 PIXEL program, whose dispatcher runs the same KILL: the loop,
// the irreducible tail, then v1 = float(v3) / 255 exported as green.
#define FRAGMENT_LOOP(before, inside)                                                              \
    {0xBE82037Cu, before,      0xBE830380u, 0x7E060202u, inside,      0x80038103u,                 \
     0xBF0A8203u, 0xBF85FFFBu, 0x7e040280u, 0x7c020300u, 0xbf860001u, 0x7e040281u,                 \
     0x7d840100u, 0xbf870001u, 0xbf82fffdu, 0x7e020d03u, 0x100202ffu, 0x3b808081u,                 \
     0x7e000280u, 0x7e040280u, 0x7e0602f2u, 0xf800180fu, 0x03020100u, 0xbf810000u}
const uint32_t kFragmentDataWriteOnEveryPath[] = FRAGMENT_LOOP(kWrite, kNop);
const uint32_t kFragmentTokenAlone[] = FRAGMENT_LOOP(kNop, kNop);
const uint32_t kFragmentDataWriteOnOnePath[] = FRAGMENT_LOOP(kNop, kWrite);
#undef FRAGMENT_LOOP
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
std::vector<uint32_t> compile(const uint32_t (&prefix)[N], bool local_y = true,
                              uint64_t program = 0x49900001ull) {
    std::vector<uint32_t> code(std::begin(prefix), std::end(prefix));
    code.insert(code.end(), std::begin(kTail), std::end(kTail));
    // Without local Y, v1 is never loaded and so arbitrary under #5185 on every route; give it a
    // hard value from s30, which no launch loads either (#5039). Shifts every pc by one.
    if (!local_y) code.insert(code.begin(), 0x7e02021eu);   // v_mov_b32 v1, s30
    ComputeShaderConfig config;
    config.local_x = 64;
    config.wave_size = 64;
    config.native_subgroup_size = 64;
    // The irreducible tail compares v0/v1. Supply both real local IDs before testing M0.
    config.tidig_comp_cnt = local_y ? 1 : 0;
    const ShaderResourceTable table = output_table();
    return recompile_compute(code.data(), code.size(), &table, config,
                             {RecompileDiagnosticStage::Compute, program});
}

template <size_t N>
std::vector<uint32_t> compile_fragment(const uint32_t (&code)[N], bool position = true,
                                       uint64_t program = 0x49900002ull) {
    // PS position X/Y are genuine varying inputs for the same tail's v0/v1 compare.
    PixelSystemInputMapping inputs{};
    inputs.addr = inputs.ena = (1u << 8) | (1u << 9);
    return recompile_fragment(code, N, nullptr, position ? &inputs : nullptr, UINT32_MAX, nullptr,
                              false, {RecompileDiagnosticStage::Fragment, program});
}

}   // namespace

TEST(EntryM0Wave64Kill, DataWriteOnEveryPathCompiles) {
    const std::vector<uint32_t> spv = compile(kDataWriteOnEveryPath);
    ASSERT_FALSE(spv.empty())
        << "s2 was overwritten with data after the M0 save on the only path to the read";
    EXPECT_TRUE(has_opcode(spv, kOpSwitch)) << "it lowered through the CFG dispatcher";
}

TEST(EntryM0Wave64Kill, DataWriteOnEveryPathIsData) {
    const std::vector<uint32_t> spv = compile(kDataWriteOnEveryPath);
    ASSERT_FALSE(spv.empty());
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
    constexpr uint64_t program = 0x49900005ull;
    EXPECT_TRUE(compile(kTokenAlone, true, program).empty())
        << "a read that only the save reaches must keep rejecting, not read a zero placeholder";
    EXPECT_NE(
        last_terminal_reject_reason(program).find("mode=unresolved-operand pc=3 words=7e060202"),
        std::string::npos);
}

TEST(EntryM0Wave64Kill, DataWriteOnOnePathStillRejects) {
    constexpr uint64_t program = 0x49900006ull;
    EXPECT_TRUE(compile(kDataWriteOnOnePath, true, program).empty())
        << "the data write is skipped on one path, so the token may still be live at the read";
    EXPECT_NE(
        last_terminal_reject_reason(program).find("mode=unresolved-operand pc=3 words=7e060202"),
        std::string::npos);
}

TEST(EntryM0Wave64Kill, FragmentDispatcherAppliesTheSameKill) {
    const std::vector<uint32_t> data = compile_fragment(kFragmentDataWriteOnEveryPath);
    ASSERT_FALSE(data.empty())
        << "a Wave64 pixel program's dispatcher kills the token the same way";
    EXPECT_TRUE(has_opcode(data, kOpSwitch)) << "it lowered through the CFG dispatcher";
    constexpr uint64_t token_program = 0x49900007ull;
    EXPECT_TRUE(compile_fragment(kFragmentTokenAlone, true, token_program).empty())
        << "the token alone still rejects in a pixel program";
    EXPECT_NE(last_terminal_reject_reason(token_program)
                  .find("mode=unresolved-operand pc=3 words=7e060202"),
              std::string::npos);
    constexpr uint64_t one_path_program = 0x49900008ull;
    EXPECT_TRUE(compile_fragment(kFragmentDataWriteOnOnePath, true, one_path_program).empty())
        << "a data write on one path only still rejects in a pixel program";
    EXPECT_NE(last_terminal_reject_reason(one_path_program)
                  .find("mode=unresolved-operand pc=3 words=7e060202"),
              std::string::npos);
}

TEST(EntryM0Wave64Kill, MissingLocalYRefusesAtTheTailCompare) {
    constexpr uint64_t program = 0x49900003ull;
    EXPECT_TRUE(compile(kDataWriteOnEveryPath, false, program).empty());
    EXPECT_NE(
        last_terminal_reject_reason(program).find("mode=unresolved-operand pc=13 words=7c020300"),
        std::string::npos);
}

TEST(EntryM0Wave64Kill, MissingPositionInputsRefuseAtTheTailCompare) {
    constexpr uint64_t program = 0x49900004ull;
    EXPECT_TRUE(compile_fragment(kFragmentDataWriteOnEveryPath, false, program).empty());
    EXPECT_NE(
        last_terminal_reject_reason(program).find("mode=unresolved-operand pc=9 words=7c020300"),
        std::string::npos);
}
