// #5497: a fresh predicate reads each operand in current EXEC. An unrelated
// source's unconfined mark must not resurrect another operand's inactive root.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/checked_compute_launch.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"
#include "gpu/recompiler/rdna2_predicate_operand_roots.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
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
        0x7e00023cu, // missing back-edge v0, dead before the next active read
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
    c.tgid_x_en = true; // genuine uniform s16, dynamic to the compiler
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
std::vector<uint32_t> cohort_program() {
    auto code = program();
    code[9] = 0x7d841300u; // compare v0 with a value loaded in the pre-loop cohort
    // The real Kena shape is SCC-tested: the data predicate is outside the
    // strict EXEC-loop continuation slice, while its operands still cross the EXEC PHI.
    code.erase(code.begin() + 13, code.begin() + 19);
    code.insert(code.begin() + 13, {
                                       0x80298129u, // s41 += 1
                                       0xbf0a2a29u, // s_cmp_lt_u32 s41, s42
                                       0xbf85fff2u, // back to the first complete v0 rewrite
                                   });
    code.insert(code.begin() + 9,
                {
                    0xbf068110u, // SCC = (genuine workgroup id s16 == 1)
                    0xbf850001u, // an inner scalar if selects the general SCC-loop route
                    0x7e140281u, // v10 = 1, scratch on this arm
                });
    code.insert(code.begin() + 5,
                {
                    0x7e0c0283u, // v6 = 3
                    0x7e540506u, // s42 = readfirstlane(v6), uniform bound
                    0xbea90380u, // s41 = 0
                    0x7da6009fu, // EXEC = (31 <= lane), before the loop's EXEC PHI
                    0xe0302000u,
                    0x80030902u, // v9 = input[lane], s[12:15]
                    0xbf8c3f70u,
                });
    return code;
}
ShaderResourceTable cohort_resources() {
    auto table = resources();
    auto input = table.resources.front();
    input.binding = 4;
    input.sgpr_base = 12;
    input.size = 64 * 4;
    table.resources.push_back(input);
    return table;
}
}   // namespace
TEST(FreshMaskOperandRoots, ASeparateOperandDoesNotReadTheInactiveOldGeneration) {
    const auto code = cohort_program();
    const auto c = config();
    const auto table = cohort_resources();
    constexpr uint64_t address = 0x5497040100ull;
    const auto module = recompile_compute(code.data(), code.size(), &table, c,
                                          {RecompileDiagnosticStage::Compute, address});
    ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(address);
    ASSERT_TRUE(has_op(module, 246));
    ASSERT_FALSE(has_op(module, 251));
    bpermute_oracle::Interpreter vm(module);
    vm.native_subgroup_width = 64;
    vm.push_constants = c.user_sgprs;
    const auto out = vm.run_buffers(
        64, {{4, std::vector<uint32_t>(64, 2)}, {3, std::vector<uint32_t>(64, 0xdeadbeefu)}}, 3);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_EQ(out.size(), 64u);
    for (uint32_t lane = 0; lane < 64; ++lane) EXPECT_EQ(out[lane], lane >= 31 ? 3u : 0u) << lane;
}
TEST(FreshMaskOperandRoots, AReadBeforeTheHeaderRewriteStillNeedsTheBackEdge) {
    auto code = cohort_program();
    code[12] = 0xbf800000u; // no complete header rewrite
    const auto c = config();
    const auto table = cohort_resources();
    constexpr uint64_t address = 0x5497040200ull;
    EXPECT_TRUE(recompile_compute(code.data(), code.size(), &table, c,
                                  {RecompileDiagnosticStage::Compute, address})
                    .empty());
    EXPECT_NE(last_terminal_reject_reason(address).find("loop-carried v0 is fabricated"),
              std::string::npos)
        << last_terminal_reject_reason(address);
}
TEST(FreshMaskOperandRoots, ADescriptorCannotBeInventedForTheOtherOperand) {
    const auto code = cohort_program();
    const auto c = config();
    const auto table = resources();
    EXPECT_TRUE(recompile_compute(code.data(), code.size(), &table, c,
                                  {RecompileDiagnosticStage::Compute, 0x5497040300ull})
                    .empty());
    EXPECT_NE(last_terminal_reject_reason(0x5497040300ull).find("pc=9"), std::string::npos)
        << last_terminal_reject_reason(0x5497040300ull);
}

TEST(FreshMaskOperandRoots, ARequiredActiveOperandStillNeedsItsBackEdge) {
    auto code = cohort_program();
    const auto write = std::find(code.rbegin(), code.rend(), 0x7e0e0281u);
    ASSERT_NE(write, code.rend());
    *write = 0x7e0e023du; // v7 = absent s61 on the back edge; v0's current value needs v7
    const auto c = config();
    const auto table = cohort_resources();
    constexpr uint64_t address = 0x5497040400ull;
    EXPECT_TRUE(recompile_compute(code.data(), code.size(), &table, c,
                                  {RecompileDiagnosticStage::Compute, address})
                    .empty());
    EXPECT_NE(last_terminal_reject_reason(address).find("loop-carried v7 is fabricated"),
              std::string::npos)
        << last_terminal_reject_reason(address);
}

namespace {
struct DistinctOwners {
    SpirvCompute builder;
    RegState registers;
    Rdna2Inst compare;
    uint64_t old = blanket_root(5, 256);
    uint64_t active = blanket_root(5, 257);
    DistinctOwners() {
        builder.is_compute = true;
        builder.wave_size = 64;
        registers.vector_wave_size = 64;
        registers.exec_narrowed = true;
        registers.exec = 7;
        registers.vreg[0] = 100;
        registers.vreg[1] = 101;
        auto& current = registers.vreg_source_marks[0];
        current.placeholder = current.hard = current.arbitrary = current.scoped_complete_word =
            true;
        current.outside_exec = 7;
        current.blanket_roots = {old};
        auto& other = registers.vreg_source_marks[1];
        other.placeholder = other.hard = other.arbitrary = true;
        other.blanket_roots = {active};
        const uint32_t word = 0x7d840300u; // vcc = v0 == v1
        compare = rdna2_decode_one(&word, 1);
    }
};
const std::set<uint64_t>& test_roots(const RegState&, const ScalarSourceMarks& marks,
                                     uint32_t exec) {
    return marks.scoped_complete_word && marks.outside_exec == exec ? marks.scoped_blanket_roots
                                                                    : marks.blanket_roots;
}
} // namespace

TEST(FreshMaskOperandRoots, ExistingObservationsAreNeverRolledBack) {
    DistinctOwners fixture;
    fixture.builder.loop_blanket_roots_read.insert(fixture.old);
    EXPECT_TRUE(fresh_mask_sources_defined(fixture.builder, fixture.registers, fixture.compare));
    EXPECT_TRUE(fixture.builder.loop_blanket_roots_read.contains(fixture.old));
    EXPECT_TRUE(fixture.builder.loop_blanket_roots_read.contains(fixture.active));
}

TEST(FreshMaskOperandRoots, DistinctOwnersRetainEveryActiveDependency) {
    DistinctOwners fixture;
    EXPECT_TRUE(fresh_mask_sources_defined(fixture.builder, fixture.registers, fixture.compare));
    EXPECT_FALSE(fixture.builder.loop_blanket_roots_read.contains(fixture.old));
    EXPECT_TRUE(fixture.builder.loop_blanket_roots_read.contains(fixture.active));
    fixture.registers.vreg_source_marks[0].scoped_blanket_roots.insert(fixture.old);
    fixture.builder.loop_blanket_roots_read.clear();
    EXPECT_TRUE(fresh_mask_sources_defined(fixture.builder, fixture.registers, fixture.compare));
    EXPECT_TRUE(fixture.builder.loop_blanket_roots_read.contains(fixture.old));
}

TEST(FreshMaskOperandRoots, UnsupportedFormsKeepTheLegacyObservation) {
    DistinctOwners fixture;
    fixture.compare.has_dpp = true;
    EXPECT_FALSE(predicate_operand_roots(fixture.registers, fixture.compare, &test_roots));
    fixture.compare.has_dpp = false;
    fixture.compare.has_modifier = true;
    EXPECT_FALSE(predicate_operand_roots(fixture.registers, fixture.compare, &test_roots));
    fixture.compare.has_modifier = false;
    fixture.compare.src[0].kind = OperandKind::SGPR;
    EXPECT_FALSE(predicate_operand_roots(fixture.registers, fixture.compare, &test_roots));
}
