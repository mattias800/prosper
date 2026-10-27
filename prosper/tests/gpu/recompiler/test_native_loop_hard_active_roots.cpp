// #5497: inactive old-word obligations must not contaminate an active complete overwrite.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/checked_compute_launch.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_cfg_scalar_sources.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <set>
#include <vector>

namespace {
using namespace prosper::gpu;
std::vector<uint32_t> program() {
    return {
        0x7e040300u, // v2 = lane, output index
        0x7e080300u, // v4 = lane, iteration counter
        0x7e020280u, // v1 = 0, output count
        0x7e0e0281u, // v7 = 1, genuine carried seed
        0xbe84047eu, // save EXEC
        0x7e000281u, // header pc5: v0 = 1 in every active lane
        0x4a100f00u, // v8 = v0 + v7; absent old v8 survives outside EXEC
        0x7e000308u, // v0 = v8; soft old v0 survives outside EXEC
        0x4a060281u, // v3 = output count + 1
        0x7d840100u, // vcc = v0 == v0, observes only active lanes
        0x02020701u, // increment output in active lanes
        0x7e000210u, // missing back-edge v0, dead before the next active read
        0x7e0e0281u, // v7 = 1 on the back edge
        0xd4c1006au, 0x00018104u, // vcc = counter < 64
        0x87fe6a7eu, // EXEC &= vcc
        0xbf880002u, // execz -> restore
        0x4a080881u, // counter += 1
        0xbf82fff2u, // branch to pc5
        0xbefe0404u, // restore EXEC
        0xe0702000u, 0x80020102u, // output[lane] = count
        0xbf810000u,
    };
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
} // namespace

TEST(NativeLoopHardActiveRoots, CompleteRewriteDoesNotReadInactiveOldGeneration) {
    const auto code = program();
    const auto c = config();
    const auto table = resources();
    const uint64_t address = 0x5497300100ull;
    const auto module = recompile_compute(code.data(), code.size(), &table, c,
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
    for (uint32_t lane = 0; lane < 64; ++lane) EXPECT_EQ(out[lane], 65u - lane) << lane;
}

TEST(NativeLoopHardActiveRoots, AReadBeforeTheHeaderRewriteStillNeedsTheBackEdge) {
    auto code = program();
    code[5] = 0xbf800000u;
    const auto c = config();
    const auto table = resources();
    const uint64_t address = 0x5497300200ull;
    EXPECT_TRUE(recompile_compute(code.data(), code.size(), &table, c,
                                  {RecompileDiagnosticStage::Compute, address})
                    .empty());
    EXPECT_NE(last_terminal_reject_reason(address).find("loop-carried v0 is fabricated"),
              std::string::npos)
        << last_terminal_reject_reason(address);
}

TEST(NativeLoopHardActiveRoots, AMissingActiveOperandStillRefuses) {
    auto code = program();
    code[3] = 0x7e0e0210u; // v7 = unknown possible scratch-offset s16
    const auto c = config();
    const auto table = resources();
    const uint64_t address = 0x5497300300ull;
    EXPECT_TRUE(recompile_compute(code.data(), code.size(), &table, c,
                                  {RecompileDiagnosticStage::Compute, address})
                    .empty());
    EXPECT_NE(last_terminal_reject_reason(address).find("fabricated-fresh-mask-source"),
              std::string::npos)
        << last_terminal_reject_reason(address);
}

namespace {
struct HardActiveWord {
    SpirvCompute b;
    RegState rs;
    const uint64_t old_root = blanket_root(5, kVectorBlanketBase);
    const uint64_t source_root = blanket_root(5, kVectorBlanketBase + 7);
    explicit HardActiveWord(bool nested = false) {
        b.is_compute = true;
        b.wave_size = b.native_subgroup_size = 64;
        rs.exec = nested ? 8 : 7;
        rs.exec_narrowed = true;
        rs.vector_full_exec = false;
        rs.vector_wave_size = 64;
        if (nested) rs.exec_subset_of[8] = 7;
        rs.vreg[0] = 201;
        rs.vreg[8] = 202;
        auto& old = rs.vreg_source_marks[0];
        old.placeholder = old.scoped_complete_word = true;
        old.outside_exec = 7;
        old.blanket_roots = {old_root};
        ScalarSourceMarks source;
        source.placeholder = source.hard = true;
        source.outside_exec = rs.exec;
        source.blanket_roots = {source_root};
        rs.vreg_source_marks[8] = source;
        const uint32_t copy = 0x7e000308u; // v0 = v8
        record_vector_word_marks(rs, rdna2_decode_one(&copy, 1), source);
    }
    bool compare() {
        const uint32_t cmp = 0x7d840100u; // vcc = v0 == v0
        return fresh_mask_sources_defined(b, rs, rdna2_decode_one(&cmp, 1));
    }
    ScalarSourceMarks& marks() { return rs.vreg_source_marks.at(0); }
};
} // namespace

TEST(NativeLoopHardActiveRoots, ActiveReadChargesOnlyTheNewSources) {
    HardActiveWord f;
    EXPECT_TRUE(f.marks().hard);
    EXPECT_EQ(f.marks().blanket_roots, (std::set<uint64_t>{f.old_root, f.source_root}));
    EXPECT_TRUE(f.marks().scoped_complete_word);
    EXPECT_EQ(f.marks().scoped_blanket_roots, (std::set<uint64_t>{f.source_root}));
    EXPECT_TRUE(f.compare());
    EXPECT_EQ(f.b.loop_blanket_roots_read, (std::set<uint64_t>{f.source_root}));
}

TEST(NativeLoopHardActiveRoots, AReaderOutsideTheOwnerStillChargesTheOldGeneration) {
    HardActiveWord f;
    f.rs.exec = 9; // no subset proof to the writer
    EXPECT_TRUE(f.compare()); // arbitrary permission remains conditional on the global roots
    EXPECT_TRUE(f.b.loop_blanket_roots_read.contains(f.old_root));
    EXPECT_TRUE(f.b.loop_blanket_roots_read.contains(f.source_root));
}

TEST(NativeLoopHardActiveRoots, ARepeatedHardWriteKeepsTheInactiveObligation) {
    HardActiveWord f;
    const auto source = f.marks();
    const uint32_t copy = 0x7e000300u; // v0 = v0, still predicated by EXEC
    record_vector_word_marks(f.rs, rdna2_decode_one(&copy, 1), source);
    EXPECT_TRUE(f.compare());
    EXPECT_FALSE(f.b.loop_blanket_roots_read.contains(f.old_root));
    f.b.loop_blanket_roots_read.clear();
    f.rs.exec = 9;
    EXPECT_TRUE(f.compare());
    EXPECT_TRUE(f.b.loop_blanket_roots_read.contains(f.old_root));
}

TEST(NativeLoopHardActiveRoots, ANestedWriterCannotAuthenticateTheWiderOwner) {
    HardActiveWord f(true);
    EXPECT_TRUE(f.compare());
    EXPECT_FALSE(f.b.loop_blanket_roots_read.contains(f.old_root));
    f.b.loop_blanket_roots_read.clear();
    f.rs.exec = 7;
    EXPECT_TRUE(f.compare());
    EXPECT_TRUE(f.b.loop_blanket_roots_read.contains(f.old_root));
}

TEST(NativeLoopHardActiveRoots, AnInvalidInactiveRootRetiresOnlyTheGlobalView) {
    HardActiveWord f;
    retire_vector_word_roots(f.marks(), {f.old_root});
    EXPECT_TRUE(f.compare());
    EXPECT_FALSE(f.b.loop_blanket_roots_read.contains(f.old_root));
    f.rs.exec = 9;
    EXPECT_FALSE(f.compare());
}

TEST(NativeLoopHardActiveRoots, AnInvalidActiveRootCannotRetainTheOwner) {
    HardActiveWord f;
    retire_vector_word_roots(f.marks(), {f.source_root});
    EXPECT_FALSE(f.compare());
}

TEST(NativeLoopHardActiveRoots, AMissingOtherEdgeCannotBorrowTheWriter) {
    HardActiveWord f;
    VectorWordMarks other;
    other[0].placeholder = other[0].hard = true;
    join_vector_word_marks(f.rs.vreg_source_marks, other, &f.rs);
    EXPECT_FALSE(f.compare());
}

TEST(NativeLoopHardActiveRoots, ADefinedAddressReadsOnlyItsCurrentExecRoots) {
    HardActiveWord f;
    Rdna2Inst load;
    load.fmt = Rdna2Format::MUBUF;
    load.src[0].kind = OperandKind::VGPR;
    load.src[0].value = 0;
    load.src[2].kind = OperandKind::InlineInt;
    load.literal = 0x1000; // one current address VGPR
    const auto before = snapshot_saved_b64_masks(f.b, f.rs, load);
    EXPECT_TRUE(before.source_marks.memory);
    EXPECT_TRUE(f.b.loop_blanket_roots_read.contains(f.source_root));
    EXPECT_FALSE(f.b.loop_blanket_roots_read.contains(f.old_root));
}

TEST(NativeLoopHardActiveRoots, AWiderRetainedOwnerAlsoNeedsItsPriorActiveSources) {
    HardActiveWord f;
    f.rs.exec = 8;
    f.rs.exec_subset_of[8] = 7;
    ScalarSourceMarks source;
    source.placeholder = source.hard = true;
    source.outside_exec = 8;
    const uint64_t next_root = blanket_root(5, kVectorBlanketBase + 9);
    source.blanket_roots = {next_root};
    const uint32_t copy = 0x7e000308u;
    record_vector_word_marks(f.rs, rdna2_decode_one(&copy, 1), source);
    EXPECT_EQ(f.marks().outside_exec, 7u);
    f.rs.exec = 7;
    EXPECT_TRUE(f.compare());
    EXPECT_TRUE(f.b.loop_blanket_roots_read.contains(next_root));
    EXPECT_TRUE(f.b.loop_blanket_roots_read.contains(f.source_root));
    EXPECT_FALSE(f.b.loop_blanket_roots_read.contains(f.old_root));
}

TEST(NativeLoopHardActiveRoots, ACompletedPartitionMeetsBothActiveProofs) {
    HardActiveWord f;
    f.marks().outside_exec = 9;
    f.rs.exec = 8;
    f.rs.exec_subset_of[8] = f.rs.exec_subset_of[9] = 7;
    f.rs.exec_complement_in[8] = {7, 9};
    ScalarSourceMarks source;
    source.placeholder = source.hard = true;
    source.outside_exec = 8;
    const uint64_t next_root = blanket_root(5, kVectorBlanketBase + 10);
    source.blanket_roots = {next_root};
    const uint32_t copy = 0x7e000308u;
    record_vector_word_marks(f.rs, rdna2_decode_one(&copy, 1), source);
    EXPECT_EQ(f.marks().outside_exec, 7u);
    f.rs.exec = 7;
    EXPECT_TRUE(f.compare());
    EXPECT_TRUE(f.b.loop_blanket_roots_read.contains(next_root));
    EXPECT_TRUE(f.b.loop_blanket_roots_read.contains(f.source_root));
    EXPECT_FALSE(f.b.loop_blanket_roots_read.contains(f.old_root));
}

namespace {
std::vector<uint32_t> buffer_program() {
    auto code = program();
    code[6] = 0xe0302000u; // buffer_load_dword v0, v0, s[12:15], 0 idxen
    code[7] = 0x80030000u;
    code[9] = 0x7d840087u; // vcc = 7 == v0; the loaded value is observed
    return code;
}
ShaderResourceTable buffer_resources() {
    auto table = resources();
    ShaderResource input = table.resources.front();
    input.binding = 4;
    input.sgpr_base = 12;
    input.size = 8;
    table.resources.push_back(input);
    return table;
}
} // namespace

TEST(NativeLoopHardActiveRoots, CompleteBufferReadDoesNotReadInactiveOldGeneration) {
    const auto code = buffer_program();
    const auto c = config();
    const auto table = buffer_resources();
    const uint64_t address = 0x5497300400ull;
    const auto module = recompile_compute(code.data(), code.size(), &table, c,
                                          {RecompileDiagnosticStage::Compute, address});
    ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(address);
    ASSERT_TRUE(has_op(module, 246));
    ASSERT_FALSE(has_op(module, 251));
    bpermute_oracle::Interpreter vm(module);
    vm.native_subgroup_width = 64;
    vm.push_constants = c.user_sgprs;
    const auto out =
        vm.run_buffers(64, {{4, {0, 7}}, {3, std::vector<uint32_t>(64, 0xdeadbeef)}}, 3);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_EQ(out.size(), 64u);
    for (uint32_t lane = 0; lane < 64; ++lane) EXPECT_EQ(out[lane], 65u - lane) << lane;
}

TEST(NativeLoopHardActiveRoots, ABufferReadBeforeTheAddressRewriteStillNeedsTheBackEdge) {
    auto code = buffer_program();
    code[5] = 0xbf800000u;
    const auto c = config();
    const auto table = buffer_resources();
    const uint64_t address = 0x5497300500ull;
    EXPECT_TRUE(recompile_compute(code.data(), code.size(), &table, c,
                                  {RecompileDiagnosticStage::Compute, address})
                    .empty());
    EXPECT_NE(last_terminal_reject_reason(address).find("loop-carried v0 is fabricated"),
              std::string::npos)
        << last_terminal_reject_reason(address);
}

TEST(NativeLoopHardActiveRoots, ABufferReadCannotInventAMissingResource) {
    const auto code = buffer_program();
    const auto c = config();
    const auto table = resources(); // output present, input descriptor absent
    const uint64_t address = 0x5497300600ull;
    EXPECT_TRUE(recompile_compute(code.data(), code.size(), &table, c,
                                  {RecompileDiagnosticStage::Compute, address})
                    .empty());
}
