// Wave64 fragment scalar CSELECTs write physical VCC words. A later live mask must use both
// genuine words and the producing SCC, rather than requiring a stale Bool view or inventing one.
#include "fixtures/fragment_compare_kill.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace prosper::gpu;

std::vector<uint32_t> program() {
    return {
        0xbe840381u,              // s4 = 1
        0xbf0d8004u,              // SCC = s4 != 0
        0x856b80c0u,              // VCC_HI = SCC ? 64 : 0
        0xbf0d8104u,              // SCC = s4 != 1
        0x856a80ffu, 0x00000080u, // VCC_LO = SCC ? 128 : 0
        0x886a6a6bu,              // VCC_LO = VCC_HI | VCC_LO
        0x7e080c6au,              // observe the physical scalar word
        0x02020100u,              // v1 = implicit VCC ? v1 : v0
        0x7e000280u, 0xbf800000u, 0x7e040280u, 0x7e060280u, 0xf800000fu, 0x03020100u, 0xbf810000u,
    };
}

struct Compilation {
    std::vector<uint32_t> module;
    std::string reason;
};

Compilation compile(const std::vector<uint32_t>& code, uint64_t address) {
    const auto inputs = prosper::test::fragment_compare_kill::position_inputs();
    auto module = recompile_fragment(code.data(), code.size(), nullptr, &inputs, UINT32_MAX,
                                     nullptr, false, {RecompileDiagnosticStage::Fragment, address});
    return {std::move(module), last_terminal_reject_reason(address)};
}

void rejected_at(const Compilation& result, uint32_t pc) {
    EXPECT_TRUE(result.module.empty()) << result.reason;
    EXPECT_NE(result.reason.find("pc=" + std::to_string(pc) + " "), std::string::npos)
        << "the missing source must be rejected at its actual consumer: " << result.reason;
}
} // namespace

TEST(FragmentScalarVccPair, GenuinePhysicalWordsRetainTheLiveLaneIdentityContract) {
    const auto result = compile(program(), 0x27610001u);
    ASSERT_FALSE(result.module.empty()) << result.reason;
    EXPECT_EQ(fragment_spirv_required_subgroup_size(result.module), 64u);
    EXPECT_TRUE(fragment_spirv_required_subgroup_reasons(result.module) &
                kFragmentWaveReasonLaneId);
}

TEST(FragmentScalarVccPair, CompletedSelectPublishesItsMaskWithoutALaterLogicalWrite) {
    auto code = program();
    code[6] = 0xbf800000u;
    const auto result = compile(code, 0x27610005u);
    ASSERT_FALSE(result.module.empty()) << result.reason;
    EXPECT_EQ(fragment_spirv_required_subgroup_size(result.module), 64u);
    EXPECT_TRUE(fragment_spirv_required_subgroup_reasons(result.module) &
                kFragmentWaveReasonLaneId);
}

TEST(FragmentScalarVccPair, GenuineArithmeticSccProducerAlsoSuppliesTheSelect) {
    auto code = program();
    code[1] =
        0x80048104u; // s_add_u32 s4,s4,1 produces actual SCC, rather than a compare-only proof.
    const auto result = compile(code, 0x27610006u);
    ASSERT_FALSE(result.module.empty()) << result.reason;
    EXPECT_EQ(fragment_spirv_required_subgroup_size(result.module), 64u);
}

TEST(FragmentScalarVccPair, MissingHighWordCannotCreateTheLivePair) {
    auto code = program();
    code[2] = 0xbf800000u;
    const auto result = compile(code, 0x27610002u);
    rejected_at(result, 4);
}

TEST(FragmentScalarVccPair, MissingScalarConditionCannotCreateTheLivePair) {
    auto code = program();
    code[1] = code[3] = 0xbf800000u;
    const auto result = compile(code, 0x27610003u);
    rejected_at(result, 4);
}

TEST(FragmentScalarVccPair, FabricatedScalarRelayCannotCreateTheLivePair) {
    auto code = program();
    code[4] = 0x856a8014u; // VCC_LO = SCC ? absent s20 : 0
    code[5] = 0xbf800000u;
    const auto result = compile(code, 0x27610004u);
    rejected_at(result, 4);
}

TEST(FragmentScalarVccPair, ACompareOfAnAbsentWordCannotSupplyTheSelect) {
    auto code = program();
    code[3] = 0xbf0d8014u;   // Fresh SCC depends on absent s20 despite the earlier genuine compare.
    const auto result = compile(code, 0x27610007u);
    rejected_at(result, 4);
}

namespace {
std::vector<uint32_t> both_arm_high_word() {
    auto code = program();
    code.insert(code.begin() + 2, {0xbf850002u, 0x856b80c0u, 0xbf820001u});
    // Both scalar successors now independently assign the physical high word before the low select.
    return code;
}
}   // namespace

TEST(FragmentScalarVccPair, BothScalarPredecessorsSupplyTheHighWord) {
    const auto result = compile(both_arm_high_word(), 0x27610008u);
    ASSERT_FALSE(result.module.empty()) << result.reason;
    EXPECT_EQ(fragment_spirv_required_subgroup_size(result.module), 64u);
}

TEST(FragmentScalarVccPair, OneScalarPredecessorCannotSupplyTheHighWord) {
    auto code = both_arm_high_word();
    code[5] = 0xbf800000u;
    const auto result = compile(code, 0x27610009u);
    rejected_at(result, 7);
}

TEST(FragmentScalarVccPair, RefreshingSccCannotAuthenticateThePreviouslyFabricatedHighWord) {
    auto code = program();
    code[1] = 0xbf800000u;
    const auto result = compile(code, 0x2761000au);
    rejected_at(result, 4);
}

TEST(FragmentScalarVccPair, ALiveBoolCannotAuthenticateItsFabricatedScalarLowWord) {
    const std::vector<uint32_t> genuine{
        0x886a8081u,   // VCC_LO = genuine 1 | 0
        0x886b80c0u,   // VCC_HI = 64 | 0; genuine current SCC
        0x856a806au,   // VCC_LO = SCC ? old physical VCC_LO : 0
        0x02020100u, 0x7e040280u, 0x7e0602f2u, 0xf800180fu, 0x03020100u, 0xbf810000u,
    };
    const auto control = compile(genuine, 0x2761000bu);
    ASSERT_FALSE(control.module.empty()) << control.reason;
    auto fabricated = genuine;
    fabricated[0] = 0x886a80fdu;   // initial stored bfalse SCC is not a produced scalar word.
    const auto result = compile(fabricated, 0x2761000cu);
    rejected_at(result, 2);
}

TEST(FragmentScalarVccPair, ObservableIntermediateMasksCannotAuthenticateAFabricatedLowWord) {
    const std::vector<uint32_t> genuine{
        0x886a8081u, 0x02020100u, 0x886b80c0u, 0x02020100u, 0x856a806au, 0x02020100u,
        0x7e040280u, 0x7e0602f2u, 0xf800180fu, 0x03020100u, 0xbf810000u,
    };
    const auto control = compile(genuine, 0x2761000du);
    ASSERT_FALSE(control.module.empty()) << control.reason;
    auto fabricated = genuine;
    fabricated[0] = 0x886a80fdu;
    const auto result = compile(fabricated, 0x2761000eu);
    rejected_at(result, 4);
}

TEST(FragmentScalarVccPair, ScalarOnlySelectMayReadItsProducedScc) {
    const std::vector<uint32_t> code{
        0xbf0d8081u,   // genuine SCC = 1 != 0
        0x856a80fdu,   // scalar-only VCC_LO = SCC ? architectural SCC : 0
        0x7e02026au, 0x7e000280u, 0x7e040280u, 0x7e0602f2u, 0xf800180fu, 0x03020100u, 0xbf810000u,
    };
    const auto result = compile(code, 0x2761000fu);
    ASSERT_FALSE(result.module.empty()) << result.reason;
    EXPECT_EQ(fragment_spirv_required_subgroup_size(result.module), 0u);
}

TEST(FragmentScalarVccPair, ScalarOnlySelectMayReadTheArchitecturalNullWord) {
    const std::vector<uint32_t> code{
        0xbf0d8081u, 0x856a807du,   // genuine SCC; scalar-only source SGPR_NULL
        0x7e02026au, 0x7e000280u, 0x7e040280u, 0x7e0602f2u, 0xf800180fu, 0x03020100u, 0xbf810000u,
    };
    const auto result = compile(code, 0x27610010u);
    ASSERT_FALSE(result.module.empty()) << result.reason;
    EXPECT_EQ(fragment_spirv_required_subgroup_size(result.module), 0u);
}

TEST(FragmentScalarVccPair, ScalarOnlyCompletePairDoesNotRequireLaneIdentity) {
    const std::vector<uint32_t> code{
        0xbf0d8081u, 0x856b80c0u, 0x856a80ffu, 0x00000080u, 0x7e02026au, 0x7e000280u,
        0x7e040280u, 0x7e0602f2u, 0xf800180fu, 0x03020100u, 0xbf810000u,
    };
    const auto result = compile(code, 0x27610011u);
    ASSERT_FALSE(result.module.empty()) << result.reason;
    EXPECT_EQ(fragment_spirv_required_subgroup_size(result.module), 0u);
}

TEST(FragmentScalarVccPair, ScalarOnlyHighWordSourceDoesNotRequireLaneIdentity) {
    const std::vector<uint32_t> code{
        0xbf0d8081u, 0x856b80c0u, 0x856a806bu, 0x7e02026au, 0x7e000280u,
        0x7e040280u, 0x7e0602f2u, 0xf800180fu, 0x03020100u, 0xbf810000u,
    };
    const auto result = compile(code, 0x27610012u);
    ASSERT_FALSE(result.module.empty()) << result.reason;
    EXPECT_EQ(fragment_spirv_required_subgroup_size(result.module), 0u);
}

TEST(FragmentScalarVccPair, LiveSelectMayReadTheGenuineHighWord) {
    auto code = program();
    code[4] = 0x856a806bu;
    code[5] = 0xbf800000u;
    const auto result = compile(code, 0x27610013u);
    ASSERT_FALSE(result.module.empty()) << result.reason;
    EXPECT_EQ(fragment_spirv_required_subgroup_size(result.module), 64u);
}

namespace {
std::vector<uint32_t> dispatcher_select(bool produced_scc) {
    auto code = program();
    code[1] = 0xbf800000u;
    code[2] = 0xbeeb03c0u;   // Genuine VCC_HI move preserves the incoming SCC.
    code[3] = 0xbf800000u;
    // The established irreducible tail requires the dispatcher; all compares have real inputs.
    code.insert(code.begin(),
                {produced_scc ? 0xbf0d8081u : 0xbf800000u, 0x7e060281u, 0x7e040280u, 0x7c020300u,
                 0xbf860001u, 0x7e040281u, 0x7d840100u, 0xbf870001u, 0xbf82fffdu, 0x7e040d02u});
    return code;
}

bool has_opcode(const std::vector<uint32_t>& module, uint32_t opcode) {
    for (std::size_t at = 5; at < module.size();) {
        const uint32_t width = module[at] >> 16;
        if (!width || width > module.size() - at) return false;
        if ((module[at] & 0xffffu) == opcode) return true;
        at += width;
    }
    return false;
}
}   // namespace

TEST(FragmentScalarVccPair, DispatcherCannotAuthenticateTheEntrySccPlaceholder) {
    const auto genuine = compile(dispatcher_select(true), 0x27610014u);
    ASSERT_FALSE(genuine.module.empty()) << genuine.reason;
    ASSERT_TRUE(has_opcode(genuine.module, 251u)) << "OpSwitch must own the actual CFG route";
    const auto missing = compile(dispatcher_select(false), 0x27610015u);
    rejected_at(missing, 14);
}

TEST(FragmentScalarVccPair, CountedLoopHeaderSccCannotBecomeAProducedExitValue) {
    std::vector<uint32_t> code{
        0xbe840380u, 0xbe850382u,   // s4=0, s5=2, preserving the absent entry SCC
        0xbf0a0504u, 0xbf840002u,   // header compare; exit to pc6
        0x80048104u, 0xbf82fffcu,   // increment; back to pc2
        0xbf0d8081u,   // matched control refreshes SCC after the loop
        0xbeeb03c0u, 0x856a80ffu, 0x00000080u, 0x02020100u, 0x7e000280u,
        0x7e040280u, 0x7e060280u, 0xf800000fu, 0x03020100u, 0xbf810000u,
    };
    const auto refreshed = compile(code, 0x27610016u);
    ASSERT_FALSE(refreshed.module.empty()) << refreshed.reason;
    ASSERT_TRUE(has_opcode(refreshed.module, 246u)) << "OpLoopMerge must own the loop";
    ASSERT_FALSE(has_opcode(refreshed.module, 251u)) << "this must exercise the structured emitter";
    code[6] = 0xbf800000u;
    const auto carried = compile(code, 0x27610017u);
    rejected_at(carried, 8);
}

TEST(FragmentScalarVccPair, DivergentLoopCheckSccCannotBorrowTheBodyDefinition) {
    std::vector<uint32_t> code{
        0xbe840380u, 0x7e040282u, 0x7e060281u, 0x7d020404u, 0xbf860002u,   // VCC header compare; exit to pc7, preserving absent SCC
        0x80048104u, 0xbf82fffcu,   // body produces SCC; back to pc3
        0xbf0d8081u,   // matched control refreshes SCC after the loop
        0xbeeb03c0u, 0x856a80ffu, 0x00000080u, 0x02020100u, 0x7e000280u, 0x7e040280u,
        0x7e060280u, 0xf800000fu, 0x03020100u, 0xbf810000u,
    };
    const auto refreshed = compile(code, 0x27610018u);
    ASSERT_FALSE(refreshed.module.empty()) << refreshed.reason;
    ASSERT_TRUE(has_opcode(refreshed.module, 246u)) << "OpLoopMerge must own the loop";
    ASSERT_FALSE(has_opcode(refreshed.module, 251u)) << "this must exercise the structured emitter";
    code[7] = 0xbf800000u;
    const auto carried = compile(code, 0x27610019u);
    rejected_at(carried, 9);
}

// #5168: VCC as plain scalar scratch around a uniform branch (Kena ps 0x500b3d0000, pc1531..1629).
// The taken edge skips the arm that rewrites both VCC words with scalar CSELECTs; after the merge
// VCC_LO is rewritten and read only as one scalar dword by a VOP3 float op. Every select is then
// scalar-only, so no Bool mask exists on either edge and the merge has nothing to reconcile. Until
// V_MAX3_F32's sources were charged one dword each (#5150), that read was counted as a VCC pair:
// the scalar-only proof failed, the select took the live-mask path, and the merge refused for a
// mask that is dead there.
namespace {
std::vector<uint32_t> uniform_branch_vcc_scratch() {
    return {
        0xbe840382u,   // s_mov_b32 s4, 2
        0xbeea03f2u,   // s_mov_b32 vcc_lo, 1.0     (scalar data on both edges)
        0xbeeb03f0u,   // s_mov_b32 vcc_hi, 0.5
        0xbf068004u,   // s_cmp_eq_u32 s4, 0
        0xbf840009u,   // pc4: s_cbranch_scc0 pc14   (uniform skip of the arm)
        0xbf068304u,   // s_cmp_eq_u32 s4, 3
        0xbeea03ffu, 0x3f83e577u,   // s_mov_b32 vcc_lo, lit
        0x856b8081u,   // s_cselect_b32 vcc_hi, 1, 0
        0xbf068204u,   // s_cmp_eq_u32 s4, 2
        0x856aff6au, 0x3f80312du,   // s_cselect_b32 vcc_lo, vcc_lo, lit
        0xbf07806bu,   // s_cmp_lg_u32 vcc_hi, 0
        0x8508f26au,   // s_cselect_b32 s8, vcc_lo, 1.0
        0xbeea03ffu, 0x38800006u,   // pc14 (merge): s_mov_b32 vcc_lo, lit
        0xd5540001u, 0x0400d4ffu, 0x3c9374bcu,   // v_max3_f32 v1, lit, vcc_lo, v0 (one dword)
        0x7e040208u,   // v_mov_b32 v2, s8
        0x7e0602f2u,   // v_mov_b32 v3, 1.0
        0xf800180fu, 0x03020100u,   // exp mrt0 v0, v1, v2, v3 done vm
        0xbf810000u,   // s_endpgm
    };
}

// A live (lane-mask) select inside the arm, followed by an implicit VCC reader at the merge.
// `redefine_at_merge` puts a VOPC that replaces the whole pair first, which makes the arm's mask
// dead at the merge; without it the merge reader observes the arm's mask on one edge and only
// scalar data on the other, which the structured join must still refuse.
std::vector<uint32_t> live_select_then_merge(bool redefine_at_merge) {
    std::vector<uint32_t> code{
        0x7e0202f0u,   // v_mov_b32 v1, 0.5
        0x7e0002f2u,   // v_mov_b32 v0, 1.0
        0xbe840382u,   // s_mov_b32 s4, 2
        0xbeea03f2u,   // s_mov_b32 vcc_lo, 1.0     (scalar data, no Bool mask)
        0xbeeb03f0u,   // s_mov_b32 vcc_hi, 0.5
        0xbf068004u,   // s_cmp_eq_u32 s4, 0
        0xbf840006u,   // pc6: s_cbranch_scc0 pc13
        0xbf068304u,   // s_cmp_eq_u32 s4, 3
        0x856b8081u,   // s_cselect_b32 vcc_hi, 1, 0
        0xbf068204u,   // s_cmp_eq_u32 s4, 2
        0x856a80ffu, 0x00000080u,   // s_cselect_b32 vcc_lo, 0x80, 0
        0x02020300u,   // v_cndmask_b32 v1, v0, v1, vcc   (the arm's lane-mask read)
        0x7c080300u,   // pc13 (merge): v_cmp_gt_f32 vcc, v0, v1  (whole pair)
        0x7e0602f2u,   // v_mov_b32 v3, 1.0
        0x02040680u,   // v_cndmask_b32 v2, 0, v3, vcc
        0xf800180fu, 0x03020100u,   // exp mrt0 v0, v1, v2, v3 done vm
        0xbf810000u,   // s_endpgm
    };
    if (!redefine_at_merge) code.erase(code.begin() + 13);
    return code;
}
}   // namespace

TEST(FragmentScalarVccPair, UniformBranchScalarScratchNeedsNoMaskAtTheMerge) {
    const auto result = compile(uniform_branch_vcc_scratch(), 0x51680001u);
    ASSERT_FALSE(result.module.empty()) << result.reason;
    // Scalar-only: the selects keep generic lowering and need no guest lane identity.
    EXPECT_EQ(fragment_spirv_required_subgroup_size(result.module), 0u);
}

TEST(FragmentScalarVccPair, ALiveArmMaskDeadAtTheMergeNeedsNoSkipEdgeMask) {
    const auto result = compile(live_select_then_merge(true), 0x51680002u);
    ASSERT_FALSE(result.module.empty()) << result.reason;
    // The in-arm read makes the select a live mask (lane identity), so this exercises the
    // structured join's liveness rather than the scalar-only route above.
    EXPECT_EQ(fragment_spirv_required_subgroup_size(result.module), 64u);
}

TEST(FragmentScalarVccPair, AnArmMaskLiveAfterTheMergeStillRequiresEveryEdge) {
    const auto result = compile(live_select_then_merge(false), 0x51680003u);
    EXPECT_TRUE(result.module.empty());
    EXPECT_NE(result.reason.find("structured-saved-mask-domain-missing-edge"), std::string::npos)
        << result.reason;
    EXPECT_NE(result.reason.find("pc=6 "), std::string::npos) << result.reason;
}
