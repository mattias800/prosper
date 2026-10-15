// #5039: the #5185 ARBITRARY property must survive the CFG dispatcher's storage reload.
//
// The structured walk keeps ScalarSourceMarks::arbitrary on a VGPR word that hardware never loads
// (rdna2_launch_vgprs) and on lane-wise or cross-lane functions of such words and defined ones, so
// a fresh predicate may read them (test_unloaded_launch_vgpr.cpp). The dispatcher instead reloads
// every guest register from Function storage at each case edge and re-marks the words its MAY
// relay analysis (SpilledMaskHalves::relayed_vectors) cannot prove defined. It used to re-mark them
// hard and NOT arbitrary, so a compute dispatcher refused any predicate over a never-loaded VGPR,
// e.g. kernel 17d's `v_cmp_lt_f32 vcc, v0, v1` at TIDIG_COMP_CNT = 0. The relay analysis now
// keeps the property the same way the walk does: a relayed vector stays arbitrary only while every
// relayed input it reads is an arbitrary VECTOR word. A scalar, SCC, memory or M0-relative input
// makes it hard (nonarbitrary_vectors). Each accepted arm has a refused twin differing in one
// input, plus a control that supplies the twin's missing scalar.
#include "fixtures/native_execution_launch.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <vector>

namespace {
using namespace prosper::gpu;
using Words = std::vector<uint32_t>;

// Kernel 17d (test_rdna2_to_spirv.cpp): an irreducible loop that forces the whole shader through
// the CFG dispatcher. Its pc1 compare reads v1, which TIDIG_COMP_CNT = 0 never loads.
constexpr uint32_t kIrreducibleTail[] = {
    0x7e040280u,   // v_mov_b32 v2, 0
    0x7c020300u,   // v_cmp_lt_f32 vcc, v0, v1
    0xbf860001u,   // s_cbranch_vccz +1
    0x7e040281u,   // v_mov_b32 v2, 1
    0x7d840100u,   // v_cmp_eq_u32 vcc, v0, v0
    0xbf870001u,   // s_cbranch_vccnz +1
    0xbf82fffdu,   // s_branch -3
    0x7e040d02u,   // v_cvt_f32_u32 v2, v2
    0xbf810000u,   // s_endpgm
};
constexpr uint32_t kEdge = 0xbf880000u;   // s_cbranch_execz +0: a dispatcher case edge
constexpr uint32_t kAddV3OneV1 = 0x4a060281u;   // v_add_nc_u32 v3, 1, v1
constexpr uint32_t kAddV3S0V1 = 0x4a060200u;   // v_add_nc_u32 v3, s0, v1
constexpr uint32_t kAddV4S0V1 = 0x4a080200u;   // v_add_nc_u32 v4, s0, v1
constexpr uint32_t kAddV3OneV4 = 0x4a060881u;   // v_add_nc_u32 v3, 1, v4
constexpr uint32_t kCompareV3 = 0x7d840700u;   // v_cmp_eq_u32 vcc, v0, v3

Words program(std::initializer_list<uint32_t> prefix) {
    Words code(prefix);
    code.insert(code.end(), std::begin(kIrreducibleTail), std::end(kIrreducibleTail));
    return code;
}

ComputeShaderConfig launch(uint32_t user_sgprs = 0) {
    ComputeShaderConfig config = prosper::test::native_execution_launch();
    config.tidig_comp_cnt = 0;   // hardware loads v0 only
    config.user_sgprs.assign(user_sgprs, 0u);
    return config;
}

bool compiles(const Words& code, const ComputeShaderConfig& config) {
    return !recompile_compute(code.data(), code.size(), nullptr, config).empty();
}

}   // namespace

TEST(DispatcherArbitraryRelay, NeverLoadedVgprStaysArbitraryAcrossTheReload) {
    EXPECT_TRUE(compiles(program({}), launch()));
}

TEST(DispatcherArbitraryRelay, LoadedVgprIsDefinedControl) {
    // Positive control for the domain: with TIDIG_COMP_CNT = 1 hardware loads v1 itself.
    ComputeShaderConfig config = launch();
    config.tidig_comp_cnt = 1;
    EXPECT_TRUE(compiles(program({}), config));
}

TEST(DispatcherArbitraryRelay, VectorRelayAcrossAnEdgeKeepsArbitrary) {
    // v3 = 1 + v1 is a lane-wise function of an arbitrary word; the compare reads it only after
    // a case edge, where the dispatcher's relay analysis alone supplies its marks.
    EXPECT_TRUE(compiles(program({kAddV3OneV1, kEdge, kCompareV3}), launch()));
}

TEST(DispatcherArbitraryRelay, ScalarRelayAcrossAnEdgeIsHard) {
    // Same shape; s0 is not a user SGPR, so hardware never loads it and v3 is hard after the edge.
    EXPECT_FALSE(compiles(program({kAddV3S0V1, kEdge, kCompareV3}), launch()));
    // Control: the same bytes with s0 loaded as a user SGPR compile.
    EXPECT_TRUE(compiles(program({kAddV3S0V1, kEdge, kCompareV3}), launch(1)));
}

TEST(DispatcherArbitraryRelay, HardnessPropagatesThroughASecondVectorRelay) {
    // v4 is hard (unloaded s0); v3 = 1 + v4 crosses a second edge and must stay hard.
    EXPECT_FALSE(compiles(program({kAddV4S0V1, kEdge, kAddV3OneV4, kEdge, kCompareV3}), launch()));
    EXPECT_TRUE(compiles(program({kAddV4S0V1, kEdge, kAddV3OneV4, kEdge, kCompareV3}), launch(1)));
}
