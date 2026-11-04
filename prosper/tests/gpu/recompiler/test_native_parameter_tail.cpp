// Mutation: authenticate only the first-END prefix, omitting the already retained trailer.
#include "fixtures/native_parameter_tail.hpp"
#include "fixtures/native_pixel_parameter.hpp"
#include "fixtures/native_fragment_spirv_oracle.hpp"
#include "gpu/recompiler/rdna2_arbitrary_placeholder.hpp"
#include "gpu/recompiler/rdna2_native_parameter_base.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <gtest/gtest.h>
#include <array>
#include <bit>
#include <cstdint>
#include <vector>

namespace {
namespace tail = prosper::test::native_parameter_tail;
namespace input = prosper::test::native_pixel_parameter;
}   // namespace

TEST(NativeParameterTail, TheClosedTrailerKeepsRealInputAndTheReusedScalarLifetime) {
    for (bool wave32 : {false, true})
        for (bool smooth : {false, true}) {
            const auto result =
                input::compile(tail::program(smooth), input::mapping(30, smooth), wave32);
            ASSERT_FALSE(result.module.empty()) << result.reason;
            EXPECT_GT(prosper::gpu::arbitrary_placeholder_reads(result.module).total(), 0u);
            for (float red : {0.625f, 0.875f}) {
                const auto r = std::bit_cast<uint32_t>(red);
                const auto actual = prosper::test::native_fragment_words(
                    result.module, {{0, {r, 0, 0, 0x3f800000u}}},
                    {{15, {0x3f000000u, 0x3f000000u, 0, 0x3f800000u}}}, wave32 ? 32 : 64);
                EXPECT_EQ(actual, (std::array<uint32_t, 4>{r, 0x3f000000u, r, 0x3f800000u}));
            }
        }
}

TEST(NativeParameterTail, UnknownAndWrongLaunchPrefixesStillRefuse) {
    auto unknown = input::mapping();
    unknown.parameter_launch = {};
    for (bool wave32 : {false, true}) {
        EXPECT_TRUE(input::compile(tail::program(), unknown, wave32).module.empty());
        EXPECT_TRUE(input::compile(tail::program(), input::mapping(31), wave32).module.empty());
    }
}

TEST(NativeParameterTail, ARealNumericalSccReadStillRefusesAtItsOwnInstruction) {
    auto code = tail::program();
    code[14] = 0x85148180u;   // LLVM gfx1030: s_cselect_b32 s20, 0, 1; no later s20 sink
    for (bool wave32 : {false, true}) {
        const auto result = input::compile(code, input::mapping(), wave32);
        EXPECT_TRUE(result.module.empty());
        EXPECT_NE(result.reason.find("pc=14"), std::string::npos) << result.reason;
        EXPECT_NE(result.reason.find("origin=unknown-launch"), std::string::npos) << result.reason;
    }
}

TEST(NativeParameterTail, TruncationAndANumericalTailObserverCannotBorrowTheService) {
    auto truncated = tail::program();
    truncated.pop_back();
    auto numerical = tail::program();
    numerical[numerical.size() - 4] = 0x7e00027cu; // replace EXEC=0 with v0=M0
    for (const auto& code : {truncated, numerical})
        for (bool wave32 : {false, true})
            EXPECT_TRUE(input::compile(code, input::mapping(), wave32).module.empty());
}

TEST(NativeParameterTail, ASuccessfulOriginalRouteKeepsItsExactModuleBytes) {
    auto code = tail::program();
    code[3] = 0x7e0e02f0u;   // a real 0.5 replaces the unloaded source before the later comparison
    code[7] = 0x7e1002f2u;   // a real 1.0 replaces the native Input's legacy provenance
    code[8] = 0xbf800000u;   // keep the old two-word CNDMASK's extent
    code[6] = 0x7d840f07u;   // use the existing strict-route control without a READLANE relay
    code.erase(code.begin() + 4, code.begin() + 6);
    auto unknown = input::mapping();
    unknown.parameter_launch = {};
    for (bool wave32 : {false, true}) {
        const auto original = input::compile(code, unknown, wave32);
        const auto known = input::compile(code, input::mapping(), wave32);
        ASSERT_FALSE(original.module.empty()) << original.reason;
        ASSERT_FALSE(known.module.empty()) << known.reason;
        EXPECT_EQ(known.module, original.module);
    }
}

TEST(NativeParameterTail, OnlyClosedOriginalSourceCanAuthenticateTheTrailer) {
    using prosper::gpu::native_parameter_base_plan;
    const auto launch = input::mapping().parameter_launch;
    const auto code = tail::program();
    std::vector<prosper::gpu::Rdna2Inst> prefix;
    prosper::gpu::rdna2_walk(code.data(), code.size(), prefix);
    EXPECT_FALSE(native_parameter_base_plan(launch, prefix));   // the old incomplete inventory
    EXPECT_TRUE(native_parameter_base_plan(launch, code.data(), code.size()));
    EXPECT_FALSE(native_parameter_base_plan(launch, nullptr, code.size()));
    EXPECT_FALSE(native_parameter_base_plan(launch, code.data(), code.size() - 1));
    auto numerical = code;
    numerical[numerical.size() - 4] = 0x7e00027cu;   // a real numerical M0 observer in the trailer
    EXPECT_FALSE(native_parameter_base_plan(launch, numerical.data(), numerical.size()));
    auto reentry = code;
    const auto tail_pc = static_cast<uint32_t>(reentry.size() - 4);
    reentry[tail_pc] = 0xbf820000u | static_cast<uint16_t>(-static_cast<int32_t>(tail_pc + 1));
    EXPECT_FALSE(native_parameter_base_plan(launch, reentry.data(), reentry.size()));
}
