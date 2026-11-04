// Mutation: keep the legacy lane-local READFIRST instead of the current-EXEC broadcast.
// A varying genuine Input must publish one value even to currently inactive guest lanes.
#include "fixtures/native_fragment_lane_values.hpp"
#include "fixtures/native_fragment_readfirst.hpp"
#include "gpu/recompiler/rdna2_arbitrary_placeholder.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <gtest/gtest.h>
#include <array>
#include <bit>
#include <cstdint>
#include <string>
#include <vector>

namespace {
namespace fixture = prosper::test::native_fragment_readfirst;
namespace input = prosper::test::native_pixel_parameter;
void witness(bool first, bool high, uint32_t wanted_width = 0) {
    for (bool wave32 : {true, false}) {
        const uint32_t width = wave32 ? 32 : 64;
        if (wanted_width && wanted_width != width) continue;
        std::vector<float> source(width, 0.5f);
        source[width == 32 ? 17 : 37] = 7.5f;
        for (float threshold : {0.0f, 2.5f, 7.5f, 9.5f}) {
            SCOPED_TRACE(width);
            SCOPED_TRACE(threshold);
            const auto result = input::compile(fixture::program(first, high, threshold),
                                               fixture::mapping(), wave32);
            ASSERT_FALSE(result.module.empty()) << result.reason;
            EXPECT_GT(prosper::gpu::arbitrary_placeholder_reads(result.module).total(), 0u);
            // This is an independently authored expectation, not the emitter's ballot/index.
            const float chosen = first && (threshold == 2.5f || threshold == 7.5f) ? 7.5f : 0.5f;
            const auto bits = std::bit_cast<uint32_t>(chosen / 8.0f);
            const auto actual = prosper::test::native_fragment_lane_values(result.module, source);
            ASSERT_EQ(actual.size(), width);
            for (const auto& pixel : actual)
                EXPECT_EQ(pixel, (std::array<uint32_t, 4>{bits, bits, bits, 0x3f800000u}));
        }
    }
}
}   // namespace

TEST(NativeFragmentReadfirst, FixedLaneControlIgnoresGuestExecForAnOrdinaryScalar) {
    witness(false, false);
}

TEST(NativeFragmentReadfirst, FixedLaneControlIgnoresGuestExecForVccHigh) {
    witness(false, true);
}

TEST(NativeFragmentReadfirst, CurrentExecChoosesOneValueInNative32) {
    witness(true, false, 32);
}
TEST(NativeFragmentReadfirst, CurrentExecChoosesOneValueInNative64) {
    witness(true, false, 64);
}
TEST(NativeFragmentReadfirst, CurrentExecPublishesVccHighInNative32) {
    witness(true, true, 32);
}
TEST(NativeFragmentReadfirst, CurrentExecPublishesVccHighInNative64) {
    witness(true, true, 64);
}

TEST(NativeFragmentReadfirst, BallotHarnessObservesTheIndependentlyChosenHighLane) {
    using namespace prosper::gpu;
    SpirvCompute b;
    b.wave_size = 64;
    b.begin_fragment(nullptr, 1);
    const auto selected = b.ucmp(Op_IEqual, b.subgroup_local_id(), b.uconst(37));
    const auto low = b.fragment_wave_ballot_half(selected, 0);
    const auto high = b.fragment_wave_ballot_half(selected, 1);
    b.export_color(0, low, high, b.uconst(37), b.uconst(0x3f800000u));
    const auto actual =
        prosper::test::native_fragment_lane_values(b.finish(), std::vector<float>(64, 0.5f));
    for (const auto& pixel : actual)
        EXPECT_EQ(pixel, (std::array<uint32_t, 4>{0, 32, 37, 0x3f800000u}));
}

TEST(NativeFragmentReadfirst, ARealMissingPerspectiveInputStaysHard) {
    auto code = fixture::program();
    // Full EXEC needs no threshold reader: the missing launch word is observed first by
    // READFIRST itself, not by the earlier mask compare. Keep the independent retry marker.
    code.erase(code.begin() + 5, code.begin() + 8);
    for (bool wave32 : {true, false}) {
        const auto result = input::compile(code, fixture::mapping(true), wave32);
        EXPECT_TRUE(result.module.empty());
        EXPECT_NE(result.reason.find("origin=launch-input word=v0"), std::string::npos)
            << result.reason;
        EXPECT_NE(result.reason.find("pc=5"), std::string::npos) << result.reason;
    }
}

TEST(NativeFragmentReadfirst, IntegerScanHarnessDefinesZeroAndBothBitBoundaries) {
    using namespace prosper::gpu;
    SpirvCompute b;
    b.begin_fragment(nullptr, 1);
    const auto empty = b.ibin(Op_IAdd, b.find_ilsb(b.uconst(0)), b.uconst(1));
    b.export_color(0, empty, b.find_ilsb(b.uconst(0x80000000u)), b.find_ilsb(b.uconst(64)),
                   b.uconst(0x3f800000u));
    const auto actual =
        prosper::test::native_fragment_lane_values(b.finish(), std::vector<float>(32, 0.5f));
    for (const auto& pixel : actual)
        EXPECT_EQ(pixel, (std::array<uint32_t, 4>{0, 31, 6, 0x3f800000u}));
}
