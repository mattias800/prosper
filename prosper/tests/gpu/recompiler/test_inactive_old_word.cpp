// Mutations: charge inactive preservation as an active read, or forget its missing outside lanes.
#include "fixtures/inactive_old_word.hpp"
#include "fixtures/inactive_compute_word.hpp"
#include "fixtures/inactive_old_word_values.hpp"
#include "fixtures/native_pixel_parameter.hpp"
#include "gpu/recompiler/rdna2_arbitrary_placeholder.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include <gtest/gtest.h>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace fixture = prosper::test::inactive_old_word;
namespace input = prosper::test::native_pixel_parameter;
using Writer = fixture::Writer;
void witness(Writer writer, uint32_t width, bool provided = false, bool expanded = false) {
    std::vector<float> x(width, 0.5f), red(width);
    x[width == 32 ? 17 : 37] = 7.5f;
    for (size_t lane = 0; lane < width; ++lane) red[lane] = 0.375f + 0.125f * (lane % 3);
    for (float threshold : {0.0f, 2.5f, 9.5f}) {
        SCOPED_TRACE(threshold);
        const auto result = input::compile(fixture::program(writer, provided, expanded, threshold),
                                           fixture::mapping(), width == 32);
        ASSERT_FALSE(result.module.empty()) << result.reason;
        EXPECT_GT(prosper::gpu::arbitrary_placeholder_reads(result.module).total(), 0u);
        const auto values = fixture::values(result.module, x, red);
        ASSERT_EQ(values.size(), width);
        for (size_t lane = 0; lane < width; ++lane) {
            uint32_t wanted = std::bit_cast<uint32_t>(expanded ? 0.875f : 0.25f);
            if (x[lane] >= threshold) {
                if (writer == Writer::Interpolate) wanted = std::bit_cast<uint32_t>(red[lane]);
                if (writer == Writer::Multiply) wanted = std::bit_cast<uint32_t>(x[lane] * 0.125f);
                if (writer == Writer::Accumulate)
                    wanted = std::bit_cast<uint32_t>(0.875f + x[lane] * 0.125f);
                if (writer == Writer::Half)
                    wanted = 0x3f600000u | (std::bit_cast<uint32_t>(x[lane]) & 0xffffu);
            }
            EXPECT_EQ(values[lane], (std::array<uint32_t, 4>{wanted, wanted, wanted, 0x3f800000u}));
        }
    }
}
} // namespace

TEST(InactiveOldWord, InterpolationOverwritesAnUnsuppliedDestinationInNative32) {
    witness(Writer::Interpolate, 32);
}
TEST(InactiveOldWord, InterpolationOverwritesAnUnsuppliedDestinationInNative64) {
    witness(Writer::Interpolate, 64);
}
TEST(InactiveOldWord, MultiplyOverwritesAnUnsuppliedDestinationInNative32) {
    witness(Writer::Multiply, 32);
}
TEST(InactiveOldWord, MultiplyOverwritesAnUnsuppliedDestinationInNative64) {
    witness(Writer::Multiply, 64);
}

TEST(InactiveOldWord, GenuineOldWordsKeepTheirValueAcrossEitherReaderScope) {
    for (Writer writer : {Writer::Interpolate, Writer::Multiply, Writer::Accumulate, Writer::Half})
        for (uint32_t width : {32u, 64u})
            for (bool expanded : {false, true}) witness(writer, width, true, expanded);
}

TEST(InactiveOldWord, AnExpandedReaderStillOwesTheUnwrittenLaunchLanes) {
    for (Writer writer : {Writer::Interpolate, Writer::Multiply})
        for (bool wave32 : {false, true}) {
            const auto result =
                input::compile(fixture::program(writer, false, true), fixture::mapping(), wave32);
            EXPECT_TRUE(result.module.empty());
            EXPECT_NE(result.reason.find("hard-placeholder-read"), std::string::npos)
                << result.reason;
        }
}

TEST(InactiveOldWord, ActiveAccumulatorsAndPartialWordsStillNeedTheirOldInput) {
    for (Writer writer : {Writer::Accumulate, Writer::Half})
        for (bool wave32 : {false, true}) {
            const auto result =
                input::compile(fixture::program(writer), fixture::mapping(), wave32);
            EXPECT_TRUE(result.module.empty());
            EXPECT_NE(result.reason.find("origin=launch-input word=v0"), std::string::npos)
                << result.reason;
            EXPECT_NE(result.reason.find("pc=" + std::to_string(fixture::writer_pc(writer))),
                      std::string::npos)
                << result.reason;
        }
}

TEST(InactiveOldWord, ARealMissingSourceIsNotMadeAnArbitraryValue) {
    for (bool wave32 : {false, true}) {
        const auto result =
            input::compile(fixture::program(Writer::Missing, true), fixture::mapping(), wave32);
        EXPECT_TRUE(result.module.empty());
        EXPECT_NE(result.reason.find("origin=launch-input word=v1"), std::string::npos)
            << result.reason;
    }
}

TEST(InactiveOldWord, TheInterpolantMustBeAnActualSuppliedInput) {
    for (bool wave32 : {false, true}) {
        const auto result =
            input::compile(fixture::program(Writer::Interpolate, true), fixture::mapping(), wave32);
        ASSERT_FALSE(result.module.empty()) << result.reason;
        const uint32_t width = wave32 ? 32 : 64;
        EXPECT_THROW(fixture::values(result.module, std::vector<float>(width, 7.5f),
                                     std::vector<float>(width, 0.625f), false),
                     std::runtime_error);
    }
}

TEST(InactiveOldWord, AnOriginalStrictSuccessStillNeedsNoArbitraryAdmission) {
    for (Writer writer : {Writer::Interpolate, Writer::Multiply})
        for (bool wave32 : {false, true}) {
            auto code = fixture::program(writer);
            // Remove only the LLVM-assembled unloaded marker and its fixed-lane comparison.
            code.erase(code.begin() + 6, code.begin() + 10);
            const auto result = input::compile(code, fixture::mapping(), wave32);
            ASSERT_FALSE(result.module.empty()) << result.reason;
            EXPECT_EQ(prosper::gpu::arbitrary_placeholder_reads(result.module).total(), 0u);
        }
}

TEST(InactiveOldWord, AComputeLoopOverwritesItsOldWordBeforeTheFutureMemoryRead) {
    namespace compute = prosper::test::inactive_compute_word;
    const auto code = compute::program();
    const auto resources = compute::resources();
    const auto launch = compute::launch();
    const auto module = prosper::gpu::recompile_compute(
        code.data(), code.size(), &resources, launch,
        {prosper::gpu::RecompileDiagnosticStage::Compute, 0x5523c000u});
    ASSERT_FALSE(module.empty()) << prosper::gpu::last_terminal_reject_reason(0x5523c000u);
    EXPECT_GT(prosper::gpu::arbitrary_placeholder_reads(module).total(), 0u);
    for (uint32_t seed : {7u, 13u}) {
        bpermute_oracle::Interpreter vm(module);
        vm.native_subgroup_width = 64;
        vm.supplied_push_constants = launch.user_sgprs;
        const auto actual = vm.run_buffers(
            64, {{2, compute::input(seed)}, {3, std::vector<uint32_t>(64, 0xdeadbeefu)}}, 3);
        ASSERT_TRUE(vm.error.empty()) << vm.error;
        EXPECT_EQ(actual, compute::expected(seed));
    }
}
