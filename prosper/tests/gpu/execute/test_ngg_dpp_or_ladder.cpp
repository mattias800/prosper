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
// The full-mask in-place BC0 form is also supported (#4896): keeping old VDST at a row edge
// agrees with own|0. #4953 replaces the obsolete BC0 refusal assertion with actual value checks.
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

std::vector<uint32_t> unbounded_or_ladder() {
    auto code = or_ladder();
    for (size_t k : {7u, 9u, 11u, 13u}) code[k] &= ~(1u << 19);
    return code;
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

// Every lane holds the SAME bit (1). OR keeps 1 in every lane; an integer ADD of the same ladder
// would give each lane the count of row lanes at or below it, (L & 15) + 1. The disjoint-bit
// kernel above cannot tell the two apart, since OR and ADD agree on disjoint bits (#4750 review).
TEST(NggDppOrLadder, PortableShellOrsRatherThanAdds) {
    std::vector<uint32_t> code = or_ladder();
    code[4] = 0x7E140302u;   // v_mov_b32 v10, v2 (= 1) instead of the per-lane shift
    const auto module = prosper::gpu::recompile_ngg_exports_for_test(code.data(), code.size(), 1);
    ASSERT_FALSE(module.empty());
    const auto out = prosper::test::run_compute(module, lane_inputs(), kLanes,
                                                kLanes * prosper::gpu::kNggExportProbeWords);
    if (out.empty()) GTEST_SKIP() << "no Vulkan compute device";
    ASSERT_EQ(out.size(), size_t{kLanes} * prosper::gpu::kNggExportProbeWords);
    for (uint32_t lane = 0; lane < kLanes; ++lane)
        EXPECT_EQ(exported(out, lane), 1u) << "lane " << lane;
}

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

    // The same bit in every lane: OR keeps 1, an ADD would count the row lanes (#4750 review).
    std::vector<uint32_t> same = or_ladder();
    same[4] = 0x7E140302u;   // v_mov_b32 v10, v2 (= 1)
    const auto same_module = prosper::gpu::recompile_ngg_exports_for_test(
        same.data(), same.size(), 10, 0, nullptr, 4, 0, {}, true, true, true);
    ASSERT_FALSE(same_module.empty());
    const auto ones = prosper::test::run_compute(
        same_module, input, kThreads, kThreads * prosper::gpu::kNggExportProbeWords, {}, {},
        nullptr, kThreads, nullptr, nullptr, nullptr, 64u);
    ASSERT_EQ(ones.size(), kThreads * prosper::gpu::kNggExportProbeWords);
    for (uint32_t t = 0; t < kThreads; ++t) EXPECT_EQ(exported(ones, t), 1u) << "thread " << t;
}

TEST(NggDppOrLadder, UnboundedPortableShellComputesTheInclusiveRowOr) {
    const auto code = unbounded_or_ladder();
    const auto module = prosper::gpu::recompile_ngg_exports_for_test(code.data(), code.size(), 1);
    ASSERT_FALSE(module.empty()) << "the supported in-place BC0 OR ladder must compile";
    const auto out = prosper::test::run_compute(module, lane_inputs(), kLanes,
                                                kLanes * prosper::gpu::kNggExportProbeWords);
    if (out.empty()) GTEST_SKIP() << "no Vulkan compute device";
    EXPECT_TRUE(inclusive_row_or(out));
}

TEST(NggDppOrLadder, UnboundedPortableShellOrsRatherThanAdds) {
    auto code = unbounded_or_ladder();
    code[4] = 0x7E140302u;   // v_mov_b32 v10, v2 (= 1)
    const auto module = prosper::gpu::recompile_ngg_exports_for_test(code.data(), code.size(), 1);
    ASSERT_FALSE(module.empty());
    const auto out = prosper::test::run_compute(module, lane_inputs(), kLanes,
                                                kLanes * prosper::gpu::kNggExportProbeWords);
    if (out.empty()) GTEST_SKIP() << "no Vulkan compute device";
    ASSERT_EQ(out.size(), size_t{kLanes} * prosper::gpu::kNggExportProbeWords);
    for (uint32_t lane = 0; lane < kLanes; ++lane)
        EXPECT_EQ(exported(out, lane), 1u) << "lane " << lane;
}

TEST(NggDppOrLadder, UnboundedUnsupportedShapesRemainRefused) {
    for (uint32_t control = 0; control < 4; ++control) {
        auto code = unbounded_or_ladder();
        if (control == 0)
            code[6] = (code[6] & ~(0xffu << 9)) | (2u << 9);   // SRC1 is initialized v2
        else if (control == 1)
            code[6] = (code[6] & ~(0xffu << 17)) | (2u << 17);   // VDST differs from SRC0
        else if (control == 2)
            code[7] &= ~(1u << 28);   // row mask is not complete
        else
            code[7] &= ~(1u << 24);   // bank mask is not complete
        EXPECT_TRUE(
            prosper::gpu::recompile_ngg_exports_for_test(code.data(), code.size(), 1).empty())
            << "unsupported BC0 shape " << control;
    }
}

TEST(NggDppOrLadder, UnboundedPortableRowsAlsoWorkWithoutAGuestBarrier) {
    for (bool repeated : {false, true}) {
        auto code = unbounded_or_ladder();
        if (repeated) code[4] = 0x7E140302u;   // every lane holds the same bit
        code.erase(code.begin() + 5);   // remove only the guest s_barrier
        const auto module =
            prosper::gpu::recompile_ngg_exports_for_test(code.data(), code.size(), 1);
        ASSERT_FALSE(module.empty());
        const auto out = prosper::test::run_compute(module, lane_inputs(), kLanes,
                                                    kLanes * prosper::gpu::kNggExportProbeWords);
        if (out.empty()) GTEST_SKIP() << "no Vulkan compute device";
        if (!repeated) {
            EXPECT_TRUE(inclusive_row_or(out));
        } else {
            ASSERT_EQ(out.size(), size_t{kLanes} * prosper::gpu::kNggExportProbeWords);
            for (uint32_t lane = 0; lane < kLanes; ++lane)
                EXPECT_EQ(exported(out, lane), 1u) << "lane " << lane;
        }
    }
}

TEST(NggDppOrLadder, UnboundedNativeRowsKeepBothValueControlsAndBarrierForms) {
    constexpr uint32_t kThreads = 256;
    if (!prosper::test::default_compute_required_subgroup_supported(64u, kThreads))
        GTEST_SKIP() << "no exact 64-lane compute subgroup";
    std::vector<float> input(static_cast<size_t>(kThreads) * 10u, 0.0f);
    for (uint32_t t = 0; t < kThreads; ++t)
        input[static_cast<size_t>(t) * 10u] = static_cast<float>((t & 63u) * 4u);
    for (bool barrier : {false, true}) {
        for (bool repeated : {false, true}) {
            auto code = unbounded_or_ladder();
            if (repeated) code[4] = 0x7E140302u;
            if (!barrier) code.erase(code.begin() + 5);
            const auto module = prosper::gpu::recompile_ngg_exports_for_test(
                code.data(), code.size(), 10, 0, nullptr, 4, 0, {}, true, true, true);
            ASSERT_FALSE(module.empty());
            const auto out = prosper::test::run_compute(
                module, input, kThreads, kThreads * prosper::gpu::kNggExportProbeWords, {}, {},
                nullptr, kThreads, nullptr, nullptr, nullptr, 64u);
            ASSERT_EQ(out.size(), kThreads * prosper::gpu::kNggExportProbeWords);
            for (uint32_t t = 0; t < kThreads; ++t)
                EXPECT_EQ(exported(out, t), repeated ? 1u : (2u << (t & 15u)) - 1u)
                    << "thread " << t << ", barrier " << barrier << ", repeated " << repeated;
        }
    }
}

TEST(NggDppOrLadder, UnboundedPortableInactiveSourcesPreserveOldDestinations) {
    for (bool barrier : {false, true}) {
        auto code = unbounded_or_ladder();
        if (!barrier) code.erase(code.begin() + 5);
        code.insert(code.begin() + 5, 0x7da40501u);   // v_cmpx_eq_u32 v1,v2: only row lane 1
        code.insert(code.end() - 3, 0xbefe04c1u);   // restore full EXEC before raw export
        const auto module =
            prosper::gpu::recompile_ngg_exports_for_test(code.data(), code.size(), 1);
        ASSERT_FALSE(module.empty());
        const auto out = prosper::test::run_compute(module, lane_inputs(), kLanes,
                                                    kLanes * prosper::gpu::kNggExportProbeWords);
        if (out.empty()) GTEST_SKIP() << "no Vulkan compute device";
        ASSERT_EQ(out.size(), size_t{kLanes} * prosper::gpu::kNggExportProbeWords);
        // The active destination's shifted sources are inactive or outside the row. Every
        // other destination is inactive itself, so all lanes retain their original one-hot bit.
        for (uint32_t lane = 0; lane < kLanes; ++lane)
            EXPECT_EQ(exported(out, lane), 1u << (lane & 15u)) << "lane " << lane;
    }
}

namespace {
void check_full_source_overwrite_results(bool native) {
    constexpr uint32_t threads = 256;
    if (native && !prosper::test::default_compute_required_subgroup_supported(64u, threads))
        GTEST_SKIP() << "no exact 64-lane compute subgroup";
    const uint32_t lanes = native ? threads : kLanes;
    const uint32_t words_per_lane = native ? 10u : 1u;
    std::vector<float> input(static_cast<size_t>(lanes) * words_per_lane, 0.0f);
    for (uint32_t lane = 0; lane < lanes; ++lane)
        input[static_cast<size_t>(lane) * words_per_lane] = static_cast<float>((lane & 63u) * 4u);
    for (uint32_t opcode : {0x25u, 0x14u, 0x1cu}) {
        for (bool barrier : {false, true}) {
            for (bool repeated : {false, true}) {
                auto code = unbounded_or_ladder();
                if (repeated) code[4] = 0x7e140302u;
                for (size_t at : {6u, 8u, 10u, 12u})
                    code[at] = (code[at] & 0x81ffffffu) | (opcode << 25u);
                code.insert(code.begin() + 5, {0x7e06030au, 0xd761000au, 0x00010081u, 0x7e140303u});
                // Save the genuine per-lane value in v3, spill v10, then fully replace v10.
                if (!barrier) code.erase(code.begin() + 9);
                const auto module = prosper::gpu::recompile_ngg_exports_for_test(
                    code.data(), code.size(), words_per_lane, 0, nullptr, native ? 4u : 0u, 0, {},
                    native, native, native);
                ASSERT_FALSE(module.empty()) << opcode << '/' << barrier << '/' << repeated;
                const auto out = prosper::test::run_compute(
                    module, input, lanes, lanes * prosper::gpu::kNggExportProbeWords, {}, {},
                    nullptr, lanes, nullptr, nullptr, nullptr, native ? 64u : 0u);
                if (out.empty()) GTEST_SKIP() << "no Vulkan compute device";
                ASSERT_EQ(out.size(),
                          static_cast<size_t>(lanes) * prosper::gpu::kNggExportProbeWords);
                for (uint32_t lane = 0; lane < lanes; ++lane) {
                    const uint32_t expected =
                        repeated
                            ? (opcode == 0x25u ? (lane & 15u) + 1u : 1u)
                            : (opcode == 0x14u ? 1u << (lane & 15u) : (2u << (lane & 15u)) - 1u);
                    EXPECT_EQ(exported(out, lane), expected)
                        << opcode << '/' << barrier << '/' << repeated << "/lane " << lane;
                }
            }
        }
    }
}
}   // namespace

TEST(NggDppOrLadder, PortableFullSourceOverwritesExportActualRowResults) {
    check_full_source_overwrite_results(false);
}

TEST(NggDppOrLadder, NativeFullSourceOverwritesExportActualRowResults) {
    check_full_source_overwrite_results(true);
}

namespace {
void check_complete_integer_source_overwrites(bool native) {
    constexpr uint32_t threads = 256;
    if (native && !prosper::test::default_compute_required_subgroup_supported(64u, threads))
        GTEST_SKIP() << "no exact 64-lane compute subgroup";
    std::vector<float> input(static_cast<size_t>(threads) * 10u, 0.0f);
    for (uint32_t lane = 0; lane < threads; ++lane)
        input[static_cast<size_t>(lane) * 10u] = std::bit_cast<float>(5u);
    for (uint32_t producer : {0x0bu, 0x11u, 0x12u, 0x13u, 0x14u, 0x16u, 0x18u, 0x1au, 0x1bu, 0x1cu,
                              0x1du, 0x1eu, 0x25u, 0x26u, 0x27u}) {
        uint32_t expected = 5u;
        switch (producer) {
            case 0x0bu: expected = 25u; break;
            case 0x16u:
            case 0x18u:
            case 0x1du:
            case 0x26u:
            case 0x27u: expected = 0u; break;
            case 0x1au: expected = 160u; break;
            case 0x1eu: expected = 0xffffffffu; break;
            case 0x25u: expected = 10u; break;
        }
        for (bool barrier : {false, true}) {
            std::vector<uint32_t> code{0x7e020300u, 0xd7610001u, 0x00010081u};
            if (barrier) code.push_back(0xbf8a0000u);
            code.insert(code.end(),
                        {(producer << 25u) | 0x00020100u, 0x380202fau, 0xff011101u, 0x380202fau,
                         0xff011801u, 0xf8000941u, 0x00000001u, 0xbf810000u});
            // Genuine full-word integer producers replace a historical scalar spill. Every lane
            // starts with raw word 5, so the actual exported OR row result is the producer value.
            const auto module = prosper::gpu::recompile_ngg_exports_for_test(
                code.data(), code.size(), 10, 0, nullptr, 4, 0, {}, true, true, native);
            ASSERT_FALSE(module.empty()) << producer << '/' << barrier;
            const auto out = prosper::test::run_compute(
                module, input, threads, threads * prosper::gpu::kNggExportProbeWords, {}, {},
                nullptr, threads, nullptr, nullptr, nullptr, native ? 64u : 0u);
            if (out.empty()) GTEST_SKIP() << "no Vulkan compute device";
            ASSERT_EQ(out.size(), size_t{threads} * prosper::gpu::kNggExportProbeWords);
            for (uint32_t lane = 0; lane < threads; ++lane)
                EXPECT_EQ(exported(out, lane), expected)
                    << producer << '/' << barrier << "/lane " << lane;
        }
    }
}
}   // namespace

TEST(NggDppOrLadder, PortableCompleteIntegerOverwritesExportActualRowResults) {
    check_complete_integer_source_overwrites(false);
}

TEST(NggDppOrLadder, NativeCompleteIntegerOverwritesExportActualRowResults) {
    check_complete_integer_source_overwrites(true);
}

namespace {
void check_other_complete_word_source_overwrites(bool native) {
    constexpr uint32_t threads = 256;
    if (native && !prosper::test::default_compute_required_subgroup_supported(64u, threads))
        GTEST_SKIP() << "no exact 64-lane compute subgroup";
    const std::vector<std::vector<uint32_t>> producers{
        {0x7e026f00u},
        {0x7e027100u},
        {0x06020100u},
        {0xd5030001u, 0x00020100u},
        {0xd5690001u, 0x00020100u},
        {0xd5780001u, 0x02020100u},
        {0xd7110001u, 0x00020100u},
        {0xd8d480e4u, 0x01000000u},
        {0xd7610003u, 0x00010081u, 0x7e060300u, 0xd8d480e4u, 0x01000003u},
        {0xd7610000u, 0x00010081u, 0xd8340000u, 0x00000605u, 0xd8b40000u, 0x01000605u},
    };
    const std::vector<uint32_t> expected{~5u, 0xa0000000u, 0u, 0u, 25u,
                                         0u,  0x00050005u, 5u, 5u, 0u};
    for (size_t shape = 0; shape < producers.size(); ++shape) {
        std::vector<float> input(static_cast<size_t>(threads) * 10u, 0.0f);
        if (shape != 2u && shape != 3u) {
            for (uint32_t lane = 0; lane < threads; ++lane)
                input[static_cast<size_t>(lane) * 10u] = std::bit_cast<float>(5u);
        }
        if (shape == 9u)
            for (uint32_t lane = 0; lane < threads; ++lane)
                input[static_cast<size_t>(lane) * 10u + 5u] = std::bit_cast<float>(lane * 4u);
        for (bool barrier : {false, true}) {
            std::vector<uint32_t> code{0x7e020300u, 0xd7610001u, 0x00010081u};
            if (barrier) code.push_back(0xbf8a0000u);
            code.insert(code.end(), producers[shape].begin(), producers[shape].end());
            code.insert(code.end(), {0x380202fau, 0xff011101u, 0x380202fau, 0xff011801u,
                                     0xf8000941u, 0x00000001u, 0xbf810000u});
            const auto module = prosper::gpu::recompile_ngg_exports_for_test(
                code.data(), code.size(), 10, 0, nullptr, 4, 0, {}, true, true, native);
            ASSERT_FALSE(module.empty()) << shape << '/' << barrier;
            const auto out = prosper::test::run_compute(
                module, input, threads, threads * prosper::gpu::kNggExportProbeWords, {}, {},
                nullptr, threads, nullptr, nullptr, nullptr, native ? 64u : 0u);
            if (out.empty()) GTEST_SKIP() << "no Vulkan compute device";
            ASSERT_EQ(out.size(), size_t{threads} * prosper::gpu::kNggExportProbeWords);
            for (uint32_t lane = 0; lane < threads; ++lane)
                EXPECT_EQ(exported(out, lane), expected[shape])
                    << shape << '/' << barrier << "/lane " << lane;
        }
    }
}
}   // namespace

TEST(NggDppOrLadder, PortableUnaryFloatAndExtendedOverwritesExportActualRowResults) {
    check_other_complete_word_source_overwrites(false);
}

TEST(NggDppOrLadder, NativeUnaryFloatAndExtendedOverwritesExportActualRowResults) {
    check_other_complete_word_source_overwrites(true);
}

TEST(NggDppOrLadder, PortablePhaseRelayReplacementsExportActualRows) {
    constexpr uint32_t threads = 256;
    std::vector<float> input(static_cast<size_t>(threads) * 10u, 0.0f);
    std::vector<uint32_t> source(threads);
    for (uint32_t lane = 0; lane < threads; ++lane) {
        source[lane] = (1u << 24u) | (1u << ((lane & 15u) + 8u));
        input[static_cast<size_t>(lane) * 10u] = std::bit_cast<float>(source[lane]);
    }
    for (bool replace_source : {false, true}) {
        std::vector<uint32_t> code{0xd7610003u, 0x00010081u};
        if (replace_source) code.push_back(0x7e060300u);
        code.insert(code.end(), {0xd8d480e4u, 0x04000003u, 0xbf8a0000u});
        if (!replace_source) code.push_back(0x7e080300u);
        code.insert(code.end(), {0x380808fau, 0xff011104u, 0x380808fau, 0xff011804u, 0xf8000941u,
                                 4u, 0xbf810000u});
        const auto module = prosper::gpu::recompile_ngg_exports_for_test(
            code.data(), code.size(), 10, 0, nullptr, 4, 0, {}, true, true, false);
        ASSERT_FALSE(module.empty()) << replace_source;
        const auto out = prosper::test::run_compute(module, input, threads,
                                                    threads * prosper::gpu::kNggExportProbeWords,
                                                    {}, {}, nullptr, threads);
        if (out.empty()) GTEST_SKIP() << "no Vulkan compute device";
        ASSERT_EQ(out.size(), size_t{threads} * prosper::gpu::kNggExportProbeWords);
        auto expected = source;
        for (uint32_t shift : {1u, 8u}) {
            const auto previous = expected;
            for (uint32_t lane = 0; lane < threads; ++lane)
                if ((lane & 15u) >= shift) expected[lane] |= previous[lane - shift];
        }
        for (uint32_t lane = 0; lane < threads; ++lane)
            EXPECT_EQ(exported(out, lane), expected[lane]) << replace_source << "/lane " << lane;
    }
}
