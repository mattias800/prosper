// The bounded DPP OR ladder of Kena's culling NGG VS programs (#3135 P7), executed in the NGG
// workgroup shell.
//
// Kena's culling primitive shaders (8642d737, db6b0a64, 399ec700, ffbabd0c) OR their per-lane
// primitive masks across the wave with an in-place `v_or_b32_dpp vN, vN, vN row_shr:{1,2,4,8}
// bound_ctrl:1` ladder, then cross rows with v_permlanex16. The shell's dispatcher had the bounded
// ADD form only (Kena's merged packer), so all four refused at the first OR. BOUND_CTRL=1 reads zero
// for a source before the 16-lane row, and OR with zero keeps the lane's own SRC1.
//
// The fixture gives lane L the bit 1 << (L & 15), runs the ladder and exports the result through
// the PRIM export probe: an inclusive prefix OR inside each row, (2 << (L & 15)) - 1. Both the
// portable dispatcher (workgroup scratch) and the exact native Wave64 one run it. Assembled with
// llvm-mc -mcpu=gfx1030.
#include "fixtures/compute_runner.h"
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <gtest/gtest.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <ios>
#include <vector>

namespace {

constexpr uint32_t kLanes = 64;

std::vector<uint32_t> or_ladder() {
    return {
        0x7E000F00u,   // v_cvt_u32_f32 v0, v0 (input: lane * 4)
        0x2C000082u,   // v_lshrrev_b32 v0, 2, v0: the lane
        0x3602008Fu,   // v_and_b32 v1, 15, v0: its row lane
        0x7E040281u,   // v_mov_b32 v2, 1
        0x34140501u,   // v_lshlrev_b32 v10, v1, v2
        0xBF8A0000u,   // s_barrier: the phased dispatcher, as in Kena's programs
        0x381414FAu, 0xFF09110Au,   // v_or_b32_dpp v10, v10, v10 row_shr:1 bound_ctrl:1
        0x381414FAu, 0xFF09120Au,   // ... row_shr:2
        0x381414FAu, 0xFF09140Au,   // ... row_shr:4
        0x381414FAu, 0xFF09180Au,   // ... row_shr:8
        0xF8000941u, 0x0000000Au,   // exp prim v10 done
        0xBF810000u,   // s_endpgm
    };
}

std::vector<float> lane_inputs() {
    std::vector<float> input(kLanes);
    for (uint32_t lane = 0; lane < kLanes; ++lane) input[lane] = static_cast<float>(lane * 4u);
    return input;
}

uint32_t exported(const std::vector<float>& out, uint32_t lane) {
    return std::bit_cast<uint32_t>(
        out[static_cast<size_t>(lane) * prosper::gpu::kNggExportProbeWords]);
}

::testing::AssertionResult inclusive_row_or(const std::vector<float>& out) {
    if (out.size() != size_t{kLanes} * prosper::gpu::kNggExportProbeWords)
        return ::testing::AssertionFailure() << "dispatch returned " << out.size() << " floats";
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
        const uint32_t expected = (2u << (lane & 15u)) - 1u;
        if (exported(out, lane) != expected)
            return ::testing::AssertionFailure()
                   << "lane " << lane << " = 0x" << std::hex << exported(out, lane)
                   << ", expected 0x" << expected;
    }
    return ::testing::AssertionSuccess();
}

}   // namespace

TEST(NggDppOrLadder, PortableShellComputesTheInclusiveRowOr) {
    const std::vector<uint32_t> code = or_ladder();
    const auto module = prosper::gpu::recompile_ngg_exports_for_test(code.data(), code.size(), 1);
    ASSERT_FALSE(module.empty()) << "the bounded OR ladder must compile in the NGG shell";
    const auto out = prosper::test::run_compute(module, lane_inputs(), kLanes,
                                                kLanes * prosper::gpu::kNggExportProbeWords);
    if (out.empty()) GTEST_SKIP() << "no Vulkan compute device";
    EXPECT_TRUE(inclusive_row_or(out));
}

// The exact native Wave64 shell, launched as the live merged shell is: four guest waves in one
// 256-invocation workgroup, ten launch words per lane (v0 first).
TEST(NggDppOrLadder, NativeWave64ShellComputesTheInclusiveRowOr) {
    const std::vector<uint32_t> code = or_ladder();
    const auto module = prosper::gpu::recompile_ngg_exports_for_test(
        code.data(), code.size(), 10, 0, nullptr, 4, 0, {}, /*packed_gs_offsets*/ true,
        /*full_four_wave_launch*/ true, /*native_wave64*/ true);
    ASSERT_FALSE(module.empty()) << "the bounded OR ladder must compile for an exact Wave64";
    constexpr uint32_t kThreads = 256;
    if (!prosper::test::default_compute_required_subgroup_supported(64u, kThreads))
        GTEST_SKIP() << "no exact 64-lane compute subgroup";
    std::vector<float> input(static_cast<size_t>(kThreads) * 10u, 0.0f);
    for (uint32_t t = 0; t < kThreads; ++t)
        input[static_cast<size_t>(t) * 10u] = static_cast<float>((t & 63u) * 4u);
    const auto out = prosper::test::run_compute(
        module, input, kThreads, kThreads * prosper::gpu::kNggExportProbeWords, {}, {}, nullptr,
        kThreads, nullptr, nullptr, nullptr, 64u);
    ASSERT_EQ(out.size(), kThreads * prosper::gpu::kNggExportProbeWords)
        << "native dispatch failed";
    for (uint32_t t = 0; t < kThreads; ++t)
        EXPECT_EQ(exported(out, t), (2u << (t & 15u)) - 1u) << "thread " << t;
}

// The admitted family is exactly the bounded form: the same ladder with BOUND_CTRL=0 (whose
// row-edge lanes would keep their old VDST rather than OR with zero) stays refused here.
TEST(NggDppOrLadder, OnlyTheBoundedFormIsAdmitted) {
    std::vector<uint32_t> code = or_ladder();
    for (size_t k : {7u, 9u, 11u, 13u}) code[k] &= ~(1u << 19);   // clear BOUND_CTRL
    EXPECT_TRUE(prosper::gpu::recompile_ngg_exports_for_test(code.data(), code.size(), 1).empty());
}
