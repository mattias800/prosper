// #4847: a backward branch no longer costs a raw-wide SMEM load its stable-source proof, provided
// every back edge in the program lands AFTER the load.
//
// The Pathless's last refused NGG program (hash 5006269a9073860b, ~133k dropped vertex draws)
// reads `s_load_dwordx4 s[76:79], s[28:29], vcc_lo` at pc 37 and runs a loop at pc 982 in its GS
// main. The proof behind `rdna2_proven_raw_*_wide_data_loads` used to return nothing for the whole
// program at the first backward branch, because it is written as "forward-only control permits a
// textual prefix scan". That premise only has to hold for the code that can run BEFORE the load:
// when every back edge targets a pc above the load, every edge into a pc at or below the load is a
// forward edge, so any path that reaches the load is strictly increasing in pc. The load then
// runs at most once, after exactly a subset of its textual prefix, and the only way past it
// without running it is a forward branch from that prefix -- which the bypass proof already walks,
// loops included.
//
// Arms, each changing one property of the same program:
//   * back edge after the load: admitted. Mutation that reddens it: restore the whole-program
//     `return proven` on a backward branch.
//   * back edge to the load itself, or above it: refused, both with a base-pointer write after
//     the load (the register-offset proof ignores writes after the load because the load runs
//     once) and with a guest store after the load (the immediate proof checks only the textual
//     prefix for aliasing writes). Mutations that redden them: admitting a load AT the loop floor
//     (`>=` relaxed to `>`), or dropping the floor check altogether (a blanket admit).
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

using namespace prosper::gpu;

namespace {

constexpr uint32_t kLoadPc = 1;
constexpr uint32_t kLoopPc = 4;   // first instruction after the load and its numeric reader

constexpr uint32_t kSOffsetS2 = 0x04000000u;     // SMEM word 1: SOFFSET s2, immediate 0
constexpr uint32_t kSOffsetNull = 0xfa000000u;   // SMEM word 1: SOFFSET null, immediate 0
constexpr uint32_t kBaseWrite = 0x801c901cu;     // s_add_u32 s28, s28, 16
// global_store_dword v[0:1], v2, off
constexpr uint32_t kStore0 = 0xdc708000u;
constexpr uint32_t kStore1 = 0x007d0200u;

//  0  s_mov_b32 s2, 16
//  1  s_load_dwordx4 s[16:19], s[28:29], <soffset>       <- the load
//  3  v_mov_b32 v0, s16                                  numeric reader of a loaded word
//  4  v_mov_b32 v1, s17                                  <- kLoopPc
//  .  (in_loop)
//  .  s_add_u32 s0, s0, 1
//  .  s_cmp_lt_u32 s0, 4
//  .  s_cbranch_scc1 <back_edge_target>
//  .  s_endpgm
std::vector<Rdna2Inst> program(uint32_t soffset_word, uint32_t back_edge_target,
                               const std::vector<uint32_t>& in_loop = {}) {
    std::vector<uint32_t> code{0xbe820390u, 0xf408040eu, soffset_word, 0x7e000210u, 0x7e020211u};
    code.insert(code.end(), in_loop.begin(), in_loop.end());
    code.insert(code.end(), {0x80008100u, 0xbf0a8400u});
    const uint32_t branch_pc = static_cast<uint32_t>(code.size());
    code.push_back(0xbf850000u | ((back_edge_target - (branch_pc + 1u)) & 0xffffu));
    code.push_back(0xbf810000u);
    std::vector<Rdna2Inst> instructions;
    EXPECT_EQ(rdna2_walk(code.data(), code.size(), instructions), code.size());
    return instructions;
}

const Rdna2Inst& at(const std::vector<Rdna2Inst>& instructions, uint32_t pc) {
    for (const Rdna2Inst& in : instructions)
        if (in.pc == pc) return in;
    ADD_FAILURE() << "no instruction at pc " << pc;
    return instructions.front();
}

const std::vector<uint32_t> kLoad{kLoadPc};

// The decode the arms rely on: the load, its register offset, the back edge and the store.
TEST(RawWideLoopAfterLoad, DecodePreconditions) {
    const auto ins = program(kSOffsetS2, kLoopPc, {kStore0, kStore1});
    const Rdna2Inst& load = at(ins, kLoadPc);
    EXPECT_EQ(load.fmt, Rdna2Format::SMEM);
    EXPECT_EQ(load.opcode, 0x2u);
    EXPECT_EQ(load.dst.value, 16);
    EXPECT_EQ(load.src[0].value, 28);
    EXPECT_EQ(load.src[1].kind, OperandKind::SGPR);
    EXPECT_EQ(load.src[1].value, 2);
    EXPECT_TRUE(rdna2_may_write_guest_memory(at(ins, kLoopPc + 1)));
    const Rdna2Inst& branch = ins[ins.size() - 2];
    EXPECT_EQ(branch.fmt, Rdna2Format::SOPP);
    EXPECT_EQ(static_cast<int64_t>(branch.pc) + branch.len_dwords + branch.simm16,
              static_cast<int64_t>(kLoopPc));
    // The load is numeric data with or without a loop, so only the CFG proof decides below.
    EXPECT_EQ(rdna2_raw_wide_data_loads(ins), kLoad);
}

TEST(RawWideLoopAfterLoad, BackEdgeAfterTheLoadIsAdmitted) {
    EXPECT_EQ(rdna2_proven_raw_immediate_wide_data_loads(program(kSOffsetNull, kLoopPc)), kLoad);
    EXPECT_EQ(rdna2_proven_raw_register_wide_entry_loads(program(kSOffsetS2, kLoopPc)), kLoad);
    EXPECT_EQ(rdna2_proven_raw_register_wide_data_loads(program(kSOffsetS2, kLoopPc)), kLoad);
    // The loop may rewrite the base pointer and store to memory: the load can never run again.
    EXPECT_EQ(rdna2_proven_raw_register_wide_data_loads(program(kSOffsetS2, kLoopPc, {kBaseWrite})),
              kLoad);
    EXPECT_EQ(rdna2_proven_raw_immediate_wide_data_loads(
                  program(kSOffsetNull, kLoopPc, {kStore0, kStore1})),
              kLoad);
}

TEST(RawWideLoopAfterLoad, BackEdgeToOrAboveTheLoadIsRefused) {
    for (const uint32_t target : {kLoadPc, 0u}) {
        SCOPED_TRACE(target);
        // The second visit reads a different address: the base moved after the first.
        EXPECT_TRUE(
            rdna2_proven_raw_register_wide_data_loads(program(kSOffsetS2, target, {kBaseWrite}))
                .empty());
        // The store follows the load textually but precedes its second visit.
        EXPECT_TRUE(rdna2_proven_raw_immediate_wide_data_loads(
                        program(kSOffsetNull, target, {kStore0, kStore1}))
                        .empty());
        // Refused for the loop alone, whatever the body does.
        EXPECT_TRUE(
            rdna2_proven_raw_immediate_wide_data_loads(program(kSOffsetNull, target)).empty());
        EXPECT_TRUE(rdna2_proven_raw_register_wide_data_loads(program(kSOffsetS2, target)).empty());
    }
}

}   // namespace
