#include "gpu/recompiler/rdna2_native_launch.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_loop_exec_data.hpp"
// #5497: a fresh predicate reads each operand in current EXEC. An unrelated
// source's unconfined mark must not resurrect another operand's inactive root.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/checked_compute_launch.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"
#include "gpu/recompiler/rdna2_predicate_operand_roots.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <gtest/gtest.h>
#include <algorithm>
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
    code.insert(code.begin() + 8, 0x7e12023cu); // old v9 = absent s60 before the partial load
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
TEST(NativeLoopExecData, ASeparateOperandDoesNotReadTheInactiveOldGeneration) {
    const auto code = cohort_program();
    const auto c = config();
    const auto table = cohort_resources();
    constexpr uint64_t address = 0x5497080100ull;
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
TEST(NativeLoopExecData, AReadBeforeTheHeaderRewriteStillNeedsTheBackEdge) {
    auto code = cohort_program();
    code[13] = 0xbf800000u; // no complete header rewrite
    const auto c = config();
    const auto table = cohort_resources();
    constexpr uint64_t address = 0x5497080200ull;
    EXPECT_TRUE(recompile_compute(code.data(), code.size(), &table, c,
                                  {RecompileDiagnosticStage::Compute, address})
                    .empty());
}
TEST(NativeLoopExecData, ADescriptorCannotBeInventedForTheOtherOperand) {
    const auto code = cohort_program();
    const auto c = config();
    const auto table = resources();
    EXPECT_TRUE(recompile_compute(code.data(), code.size(), &table, c,
                                  {RecompileDiagnosticStage::Compute, 0x5497080300ull})
                    .empty());
    EXPECT_NE(last_terminal_reject_reason(0x5497080300ull).find("pc=10"), std::string::npos)
        << last_terminal_reject_reason(0x5497080300ull);
}

TEST(NativeLoopExecData, ARequiredActiveOperandStillNeedsItsBackEdge) {
    auto code = cohort_program();
    const auto write = std::find(code.rbegin(), code.rend(), 0x7e0e0281u);
    ASSERT_NE(write, code.rend());
    *write = 0x7e0e023du; // v7 = absent s61 on the back edge; v0's current value needs v7
    const auto c = config();
    const auto table = cohort_resources();
    constexpr uint64_t address = 0x5497080400ull;
    EXPECT_TRUE(recompile_compute(code.data(), code.size(), &table, c,
                                  {RecompileDiagnosticStage::Compute, address})
                    .empty());
}

namespace {
std::vector<uint32_t> narrowing_cohort_program() {
    auto code = cohort_program();
    const auto cohort = std::find(code.begin(), code.end(), 0x7da6009fu);
    code.insert(cohort, 0x36180481u); // v12 = lane & 1, genuine in every lane
    const auto compare = std::find(code.begin(), code.end(), 0x7d841300u);
    code.insert(compare, 0x7da41881u); // EXEC &= (1 == v12), retaining odd lanes of C
    const auto back = std::find(code.begin(), code.end(), 0xbf85fff2u);
    *back = 0xbf85fff1u; // one new instruction inside the loop
    return code;
}
}   // namespace
TEST(NativeLoopExecData, ARealCmpxBackedgeKeepsOnlyItsOriginalEntryCohort) {
    const auto code = narrowing_cohort_program();
    const auto c = config();
    const auto table = cohort_resources();
    constexpr uint64_t address = 0x5497080500ull;
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
    for (uint32_t lane = 0; lane < 64; ++lane)
        EXPECT_EQ(out[lane], lane >= 31 && (lane & 1) ? 3u : 0u) << lane;
}
TEST(NativeLoopExecData, AnUnavailableCohortLoadCannotBorrowTheExecProof) {
    auto code = cohort_program();
    code[10] = 0x7e12023cu;
    code[11] = 0xbf800000u;
    const auto c = config();
    const auto table = cohort_resources();
    constexpr uint64_t address = 0x5497080600ull;
    EXPECT_TRUE(recompile_compute(code.data(), code.size(), &table, c,
                                  {RecompileDiagnosticStage::Compute, address})
                    .empty());
    EXPECT_NE(last_terminal_reject_reason(address).find("fabricated-fresh-mask-source pc=20"),
              std::string::npos);
}

TEST(NativeLoopExecData, AnAlreadyAcceptedGlobalControlKeepsItsValues) {
    auto code = cohort_program();
    const auto poison = std::find(code.begin(), code.end(), 0x7e12023cu);
    ASSERT_NE(poison, code.end());
    *poison = 0xbf800000u;
    const auto c = config();
    const auto table = cohort_resources();
    constexpr uint64_t address = 0x5497080700ull;
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

namespace {
struct BoundFrame {
    SpirvCompute b;
    RegState rs;
    std::vector<Rdna2Inst> ins;
    DivLoop loop;
    explicit BoundFrame(const std::vector<uint32_t>& words) {
        const auto launch = config();
        b.native_subgroup_size = 64;
        b.begin(4, nullptr, 64, 1, 1, 64, 0, true, true);
        prosper::test::bind_checked_compute_launch(b, launch);
        b.saved_mask_original_words = words;
        b.saved_mask_original_compute_launch = b.guest_compute_launch;
        rdna2_walk(words.data(), words.size(), ins);
        rs.exec = b.ucmp(Op_ULessThan, b.localid_comp[0], b.uconst(33));
        rs.exec_narrowed = true;
        rs.vector_wave_size = 64;
        rs.sreg_bool[4] = b.btrue();
        rs.sreg_bool[20] = rs.exec;
        rs.sreg_bool[24] = b.btrue();
        rs.unavailable_saved_mask_words = {20, 21};
        rs.sreg_merge_placeholder = {20, 21};
        loop.header_pc = 0;
        loop.backedge_pc = static_cast<uint32_t>(words.size() - 2);
        loop.exit_pc = static_cast<uint32_t>(words.size() - 1);
        loop.condition = DivLoop::Condition::Scc;
        loop.bottom_tested = true;
    }
    bool bound() const { return native_loop_data_exec_bound(b, ins, loop, rs); }
};
std::vector<uint32_t> both_arms() {
    return {0xbf068110u, 0xbf850002u, 0xbe98047eu, 0xbf820001u,
            0xbe980414u, 0xbefe0418u, 0xbf82fff9u, 0xbf810000u};
}
}   // namespace
TEST(NativeLoopExecData, BothRealSavedMaskArmsProveDataWithoutGrantingARoute) {
    BoundFrame f(both_arms());
    ASSERT_TRUE(f.bound());
    LoopExecSubset proof(f.b, f.ins, f.loop, f.rs);
    EXPECT_FALSE(proof.admitted());
    EXPECT_TRUE(proof.has_data_exec_bound());
    const auto unavailable = f.rs.unavailable_saved_mask_words;
    const auto placeholders = f.rs.sreg_merge_placeholder;
    struct Phi {
        int reg;
        int kind;
        uint32_t value;
        std::size_t patch;
    };
    std::vector<Phi> phis;
    const auto entry = f.rs.exec;
    open_narrowing_loop_exec_phi(f.b, f.rs, f.b.cur_block, phis, proof);
    EXPECT_EQ(f.rs.exec_subset_of.at(f.rs.exec), entry);
    EXPECT_FALSE(proof.admitted());
    EXPECT_EQ(f.rs.unavailable_saved_mask_words, unavailable);
    EXPECT_EQ(f.rs.sreg_merge_placeholder, placeholders);
    EXPECT_TRUE(f.rs.sreg.empty());
}
TEST(NativeLoopExecData, AOneArmSaveCannotHideTheWiderIncomingMask) {
    BoundFrame f({0xbf068110u, 0xbf850001u, 0xbe98047eu, 0xbefe0418u, 0xbf82fffbu, 0xbf810000u});
    EXPECT_FALSE(f.bound());
    LoopExecSubset proof(f.b, f.ins, f.loop, f.rs);
    EXPECT_FALSE(proof.admitted());
    EXPECT_FALSE(proof.has_data_exec_bound());
}
TEST(NativeLoopExecData, APhysicalHighWordReplacementEndsThePairBound) {
    BoundFrame f({0xbe98047eu, 0xbe9903c1u, 0xbefe0418u, 0xbf82fffcu, 0xbf810000u});
    EXPECT_FALSE(f.bound());
}
TEST(NativeLoopExecData, TheStaticCertificateAddsNoDynamicAuditRefusal) {
    BoundFrame f(both_arms());
    LoopExecSubset proof(f.b, f.ins, f.loop, f.rs);
    ASSERT_TRUE(proof.has_data_exec_bound());
    ASSERT_FALSE(proof.admitted());
    proof.open(f.rs);
    f.rs.exec = f.b.id(); // legacy ancestry may not map an unchanged saved-mask PHI
    EXPECT_TRUE(proof.holds(f.b, f.rs));
}
TEST(NativeLoopExecData, ContextOriginalWordsAndB32ViewsCannotSupplyTheCertificate) {
    for (unsigned mutation = 0; mutation < 12; ++mutation) {
        BoundFrame f(both_arms());
        switch (mutation) {
            case 0: f.b.native_subgroup_size = 0; break;
            case 1: f.b.wave_size = 32; break;
            case 2: f.b.is_fragment = true; break;
            case 3: f.b.guest_stage = GuestShaderStage::Standalone; break;
            case 4: f.b.physical_domain = PhysicalExecutionDomain::Invocation; break;
            case 5: f.b.saved_mask_original_compute_launch.wave_size = 32; break;
            case 6: f.b.saved_mask_original_words[2] ^= 1u; break;
            case 7: f.rs.sreg_bool_b32.insert(20); break;
            case 8: f.rs.exec_narrowed = false; break;
            case 9: f.rs.vector_full_exec = true; break;
            case 10: f.loop.backedge_pc = 0; break;
            case 11: f.b.saved_mask_original_words.resize(65537, 0); break;
        }
        EXPECT_FALSE(f.bound()) << mutation;
    }
}

TEST(NativeLoopExecData, ALoopCarriedPairMustMeetItsChangedBackedgeGeneration) {
    BoundFrame f({0xbefe0418u, 0xbe980404u, 0xbf82fffdu, 0xbf810000u});
    f.rs.sreg_bool[24] = f.rs.exec;
    EXPECT_FALSE(f.bound());
}
