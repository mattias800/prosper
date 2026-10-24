// #5497: a current-EXEC overwrite must not read the previous iteration's inactive lanes.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/checked_compute_launch.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_scalar_exec_scopes.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

namespace {
using namespace prosper::gpu;

std::vector<uint32_t> loop_program(bool project = false) {
    std::vector<uint32_t> program{
        0x7e040300u, // v2 = lane, output index
        0x7e080300u, // v4 = lane, loop counter
        0x7e020280u, // v1 = 0, output count
        0xbe950380u, // s21 = 0, the physical mask's high word
        0xbe84047eu, // save EXEC in s[4:5]
        0x7e000281u, // header: v0 = 1 in the current EXEC domain
        0x7e280500u, // s20 = readfirstlane(v0)
        0x4a060281u, // v3 = v1 + 1
        0x7d840014u, // vcc = s20 == v0
        0x02020701u, // v1 = VCC ? v3 : v1
        0x7e00023cu, // dead back-edge generation: v0 = never-written s60
        0xd4c1006au, 0x00018104u, // vcc = v4 < 64
        0x87fe6a7eu, // EXEC &= vcc
        0xbf880002u, // execz -> exit
        0x4a080881u, // v4 += 1
        0xbf82fff4u, // branch to header, pc 5
        0xbefe0404u, // exit: restore EXEC
        0xe0702000u, 0x80020102u, // output[v2] = v1
        0xbf810000u,
    };
    if (project) {
        program[3] = 0xbe920381u; // s18 = 1
        program[6] = 0x7ed60500u; // s107 = readfirstlane(v0)
        program[8] = 0x936a6b12u; // s106 = s18 * s107
        program.insert(program.begin() + 9, 0x9aea6b6au); // s106 = mul_hi(s106, s107)
        program.insert(program.begin() + 10, 0x7d8a006au); // fresh vcc = s106 != v0
        program[18] = 0xbf82fff2u; // adjusted back edge to header pc 5
    }
    return program;
}

ComputeShaderConfig config() {
    ComputeShaderConfig c;
    c.local_x = c.wave_size = c.native_subgroup_size = 64;
    c.user_sgprs.resize(16, 0);
    prosper::test::complete_compute_launch(c);
    return c;
}

ShaderResourceTable resources() {
    ShaderResource out;
    out.cls = ResourceClass::ConstantBuffer;
    out.format = DataFormat::Uint32;
    out.num_components = 1;
    out.binding = 3;
    out.stride = 4;
    out.sgpr_base = 8;
    ShaderResourceTable table;
    table.resources.push_back(out);
    return table;
}

bool has_op(const std::vector<uint32_t>& module, uint32_t opcode) {
    for (size_t at = 5; at < module.size();) {
        const uint32_t width = module[at] >> 16u;
        if (!width || width > module.size() - at) return false;
        if ((module[at] & 65535u) == opcode) return true;
        at += width;
    }
    return false;
}
void values(bool project) {
    const auto program = loop_program(project);
    const auto c = config();
    const auto table = resources();
    const uint64_t address = 0x5497200100ull + static_cast<uint64_t>(project);
    const auto module = recompile_compute(program.data(), program.size(), &table, c,
                                          {RecompileDiagnosticStage::Compute, address});
    ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(address);
    ASSERT_TRUE(has_op(module, 246));
    ASSERT_FALSE(has_op(module, 251));
    bpermute_oracle::Interpreter vm(module);
    vm.native_subgroup_width = 64;
    vm.push_constants = c.user_sgprs;
    const auto out = vm.run_buffers(64, {{3, std::vector<uint32_t>(64, 0xdeadbeef)}}, 3);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_EQ(out.size(), 64u);
    for (uint32_t lane = 0; lane < 64; ++lane) EXPECT_EQ(out[lane], 65u - lane) << "lane " << lane;
}

struct ScopedScalar {
    SpirvCompute builder;
    RegState registers;
    ScalarSourceMarks source;
    ScopedScalar() {
        builder.is_compute = true;
        builder.wave_size = 64;
        registers.exec = 7;
        registers.exec_narrowed = true;
        registers.vector_wave_size = 64;
        registers.sreg[20] = 101;
        source.placeholder = source.scoped_complete_word = true;
        source.outside_exec = registers.exec;
        source.blanket_roots = {blanket_root(5, kVectorBlanketBase)};
        registers.sreg_merge_placeholder.insert(20);
        registers.sreg_loop_blanket[20] = source.blanket_roots;
        const uint32_t code = 0x7e280500u; // READFIRSTLANE s20, v0
        record_scalar_exec_scope(registers, rdna2_decode_one(&code, 1), 20, false, source, true);
        registers.vreg[0] = 201;
        auto& missing = registers.vreg_source_marks[0];
        missing.placeholder = missing.hard = true; // the actual back edge is missing
    }
    bool compare() {
        registers.vreg[1] = 202; // a defined current operand; v0 below is the actual back edge
        const uint32_t code = 0x7d840214u; // vcc = s20 == v1
        return fresh_mask_sources_defined(builder, registers, rdna2_decode_one(&code, 1));
    }
    bool back_edge_holds() {
        LoopBlanketCheck check(registers, std::set<int>{}, 5);
        check.carried[kVectorBlanketBase];   // a genuine preheader v0, as in the execution witness
        LoopCheckMarks exits;
        return check.holds(builder, registers, exits);
    }
};
}   // namespace

TEST(NativeLoopScalarScope, CurrentOverwriteFeedsScalarCompare) {
    values(false);
}
TEST(NativeLoopScalarScope, CurrentOverwriteFeedsVccNumericScratch) {
    values(true);
}

TEST(NativeLoopScalarScope, ObservedMissingBackEdgeStillRefuses) {
    auto program = loop_program();
    program[5] = 0xbf800000u;   // no current overwrite; the back-edge generation is now read
    const auto c = config();
    const auto table = resources();
    const uint64_t address = 0x5497200200ull;
    EXPECT_TRUE(recompile_compute(program.data(), program.size(), &table, c,
                                  {RecompileDiagnosticStage::Compute, address})
                    .empty());
    EXPECT_NE(last_terminal_reject_reason(address).find("loop-carried v0 is fabricated"),
              std::string::npos)
        << last_terminal_reject_reason(address);
    EXPECT_NE(last_terminal_reject_reason(address).find("header pc=5"), std::string::npos);
}

TEST(NativeLoopScalarScope, CompleteCurrentCohortDoesNotReadTheMissingBackEdge) {
    ScopedScalar fixture;
    EXPECT_TRUE(fixture.compare());
    EXPECT_TRUE(fixture.builder.loop_blanket_roots_read.empty());
    EXPECT_TRUE(fixture.back_edge_holds());
}

TEST(NativeLoopScalarScope, ExpandedExecStillNeedsTheBackEdge) {
    ScopedScalar fixture;
    fixture.registers.exec = 8;   // no subset proof from this wider reader to the producer
    EXPECT_TRUE(fixture.compare());
    EXPECT_FALSE(fixture.builder.loop_blanket_roots_read.empty());
    EXPECT_FALSE(fixture.back_edge_holds());
}

TEST(NativeLoopScalarScope, ANewValueCannotBorrowThePriorGenerationsScope) {
    ScopedScalar fixture;
    fixture.registers.sreg[20] = 102;
    EXPECT_TRUE(fixture.compare());
    EXPECT_FALSE(fixture.back_edge_holds());
}

TEST(NativeLoopScalarScope, AnEqualValueJoinNeedsTheOtherEdgesProof) {
    ScopedScalar fixture;
    const auto roots = fixture.registers.sreg_loop_blanket.at(20);
    // The other edge has the same stored value but only the global loop assumption.
    join_merge_placeholder(fixture.registers, 20, true, false, &roots);
    EXPECT_TRUE(fixture.compare());
    EXPECT_FALSE(fixture.back_edge_holds());
}

TEST(NativeLoopScalarScope, ASelectedPeerIsNotProvedByCurrentExec) {
    ScopedScalar fixture;
    const uint32_t code[] = {0xd7600014u, 256u | (191u << 9u)};   // READLANE s20, v0, 63
    record_scalar_exec_scope(fixture.registers, rdna2_decode_one(code, 2), 20, false,
                             fixture.source);
    EXPECT_TRUE(fixture.compare());
    EXPECT_FALSE(fixture.back_edge_holds());
}

TEST(NativeLoopScalarScope, CurrentCohortDoesNotAuthenticateWholeWaveMaskBits) {
    ScopedScalar fixture;
    EXPECT_TRUE(scalar_words_projectable(fixture.builder, fixture.registers, 20, 1));
    EXPECT_FALSE(fixture.back_edge_holds());
}

TEST(NativeLoopScalarScope, NumericVccPreflightDoesNotObserveLaneBits) {
    ScopedScalar fixture;
    const uint32_t code = 0x936a8114u;   // s_mul_i32 s106, s20, 1
    const auto instruction = rdna2_decode_one(&code, 1);
    bool ok = true;
    {
        VccMaskViewDrop result(fixture.builder, fixture.registers);
        EXPECT_FALSE(
            refuse_mask_write(fixture.builder, fixture.registers, instruction, result, ok));
        EXPECT_TRUE(ok);
        EXPECT_TRUE(fixture.builder.loop_blanket_roots_read.empty());
        // The scalar DATA route retires the old Bool and writes only the physical dword.
        fixture.registers.vcc = 0;
    }
    EXPECT_TRUE(fixture.builder.loop_blanket_roots_read.empty());
    EXPECT_TRUE(fixture.back_edge_holds());
}

TEST(NativeLoopScalarScope, ARealMaskReadKeepsItsOwnObligation) {
    ScopedScalar fixture;
    const uint32_t code = 0x936a8114u;
    bool ok = true;
    {
        VccMaskViewDrop result(fixture.builder, fixture.registers);
        EXPECT_FALSE(refuse_mask_write(fixture.builder, fixture.registers,
                                       rdna2_decode_one(&code, 1), result, ok));
        EXPECT_TRUE(scalar_words_projectable(fixture.builder, fixture.registers, 20, 1));
        fixture.registers.vcc = 0;
    }
    EXPECT_FALSE(fixture.back_edge_holds());
}

TEST(NativeLoopScalarScope, APublishedMaskCommitsThePreflightObligation) {
    ScopedScalar fixture;
    const uint32_t code = 0x936a8114u;
    bool ok = true;
    {
        VccMaskViewDrop result(fixture.builder, fixture.registers);
        EXPECT_FALSE(refuse_mask_write(fixture.builder, fixture.registers,
                                       rdna2_decode_one(&code, 1), result, ok));
        fixture.registers.vcc = 301;   // a route that publishes a mask must retain its roots
    }
    EXPECT_FALSE(fixture.back_edge_holds());
}

TEST(NativeLoopScalarScope, LaneVaryingReadfirstlaneDoesNotBorrowTheUniformScope) {
    auto program = loop_program();
    program[5] = 0x7e000300u;   // header v0 = lane: THIS lane is not a broadcast
    const auto c = config();
    const auto table = resources();
    const uint64_t address = 0x5497200300ull;
    EXPECT_TRUE(recompile_compute(program.data(), program.size(), &table, c,
                                  {RecompileDiagnosticStage::Compute, address})
                    .empty());
    EXPECT_NE(last_terminal_reject_reason(address).find("loop-carried v0 is fabricated"),
              std::string::npos)
        << last_terminal_reject_reason(address);
}

TEST(NativeLoopScalarScope, AReadWithoutAnAgreementProofCannotBorrowThePriorScope) {
    ScopedScalar fixture;
    const uint32_t code = 0x7e280501u;   // same destination, no agreement proof for this producer
    record_scalar_exec_scope(fixture.registers, rdna2_decode_one(&code, 1), 20, false,
                             fixture.source);
    EXPECT_TRUE(fixture.compare());
    EXPECT_FALSE(fixture.back_edge_holds());
}

TEST(NativeLoopScalarScope, NumericScratchCannotRemoveAnAlreadyObservedRoot) {
    ScopedScalar fixture;
    fixture.builder.loop_blanket_roots_read = fixture.source.blanket_roots;
    const uint32_t code = 0x936a8114u;
    bool ok = true;
    {
        VccMaskViewDrop result(fixture.builder, fixture.registers);
        EXPECT_FALSE(refuse_mask_write(fixture.builder, fixture.registers,
                                       rdna2_decode_one(&code, 1), result, ok));
        fixture.registers.vcc = 0;
    }
    EXPECT_FALSE(fixture.back_edge_holds());
}

TEST(NativeLoopScalarScope, AgreementProofNeedsTheOriginalPacketAndReadonlyLaunchWord) {
    ScopedScalar fixture;
    auto& b = fixture.builder;
    b.native_subgroup_size = 64;
    b.saved_mask_original_words = loop_program();
    auto read = rdna2_decode_one(b.saved_mask_original_words.data() + 6, 1);
    read.pc = 6;
    EXPECT_TRUE(scalar_scope_first_lane_agrees(b, fixture.registers, read));
    auto changed = read;
    changed.src[0].value = 1;
    EXPECT_FALSE(scalar_scope_first_lane_agrees(b, fixture.registers, changed));
    b.saved_mask_original_words[5] = 0x7e00020au;   // v0 = launch s10
    fixture.registers.sreg[10] = 202;
    EXPECT_TRUE(scalar_scope_first_lane_agrees(b, fixture.registers, read));
    b.saved_mask_original_words[3] = 0xbe8a0381u;   // s10 is now a shader-written word
    EXPECT_FALSE(scalar_scope_first_lane_agrees(b, fixture.registers, read));
}
