// test_fragment_float_flags — what a launch register's FLOAT_MODE and float flags actually decode to.
//
// Register absence and an OBSERVED ZERO are independent producing inputs, and conflating them is the
// defect this pins: a shader compiled without the guest's float-mode authority reads as the same
// input as one compiled with an explicit all-clear, so the cache (see test_fragment_float_flags_key)
// reuses a module the guest never asked for. The complete 32-bit word is therefore carried as evidence
// alongside the derived policy fields, so a future register bit cannot be silently dropped by a
// decoder that only knows today's fields.
//
// The expected values are named locals rather than braced temporaries: a comma inside a brace
// initialiser is an argument separator to the preprocessor, so `EXPECT_EQ(x, T{a, b})` does not
// compile.
#include "gpu/pm4/pm4_registers.hpp"
#include "gpu/state/render_state.hpp"

#include <gtest/gtest.h>

#include <cstdint>

using namespace prosper::gpu;
namespace P = prosper::agc::Pm4;

namespace {
constexpr FragmentFloatFlags kUnknownFlags{};
constexpr FragmentLaunchRsrc1 kUnavailableWord{};
constexpr FragmentFloatFlags kKnownFlagsClear{true, false, false};
constexpr FragmentFloatMode kUnknownMode{};
constexpr FragmentFloatMode kKnownModeZero{true, 0};
constexpr FragmentLaunchRsrc1 kRawEvidenceZero{true, 0};
constexpr FragmentLaunchRsrc1 kUnavailablePayload{false, 1};
constexpr FragmentFloatFlags kUnknownIeee{false, true, false};
constexpr FragmentFloatFlags kUnknownDx10{false, false, true};
constexpr FragmentFloatFlags kBothUnknown{false, true, true};
constexpr FragmentFloatFlags kIndependentlyKnown{true, true, true};
}   // namespace

TEST(FragmentFloatFlags, AnAbsentLaunchStateIsUnknownAndCanonical) {
    // Register absence and an observed zero are independent producing inputs.
    const auto absent = extract_render_state(GpuState{});
    EXPECT_EQ(absent.ps_float_flags, kUnknownFlags) << "absent launch flags are unknown";
    EXPECT_TRUE(absent.ps_float_flags.canonical()) << "absent launch flags are canonical";
    EXPECT_FALSE(absent.ps_float_mode.available) << "absent launch FLOAT_MODE is unknown";
    EXPECT_EQ(absent.ps_launch_rsrc1, kUnavailableWord) << "absent full word remains unavailable";
}

TEST(FragmentFloatFlags, EveryLaunchRegisterExtractionDecodesItsThreeAuthoritiesIndependently) {
    for (unsigned mode = 0; mode < 256; ++mode) {
        for (unsigned flags = 0; flags < 4; ++flags) {
            GpuState launch;
            // Literal published positions: FLOAT_MODE [19:12], DX10_CLAMP 21,
            // DEBUG_MODE 22 and IEEE_MODE 23. Expected fields never use PM4_FIELD.
            launch.sh[P::SPI_SHADER_PGM_RSRC1_PS] =
                (mode << 12) | ((flags & 1u) << 21) | ((flags & 2u) << 22) | (1u << 22) | 0x3fu;
            const auto decoded = extract_render_state(launch);
            const FragmentFloatFlags expected{true, (flags & 2u) != 0, (flags & 1u) != 0};
            const FragmentFloatMode expected_mode{true, static_cast<uint8_t>(mode)};
            const FragmentLaunchRsrc1 expected_word{true, launch.sh.at(P::SPI_SHADER_PGM_RSRC1_PS)};
            SCOPED_TRACE(::testing::Message() << "FLOAT_MODE=" << mode << " flags=" << flags);
            EXPECT_EQ(decoded.ps_float_flags, expected) << "both launch flags survive extraction";
            EXPECT_TRUE(decoded.ps_float_flags.canonical()) << "observed flags are canonical";
            EXPECT_EQ(decoded.ps_float_mode, expected_mode)
                << "launch flags do not contaminate FLOAT_MODE";
            EXPECT_EQ(decoded.ps_launch_rsrc1, expected_word)
                << "complete actual word survives independently of derived policy fields";
        }
    }
}

TEST(FragmentFloatFlags, AnObservedZeroIsKnownFlagsClearAndAvailableEvidence) {
    GpuState observed_zero;
    observed_zero.sh[P::SPI_SHADER_PGM_RSRC1_PS] = 0;
    const auto zero = extract_render_state(observed_zero);
    EXPECT_EQ(zero.ps_float_flags, kKnownFlagsClear) << "observed zero is known flags-clear";
    EXPECT_EQ(zero.ps_float_mode, kKnownModeZero) << "observed zero has known FLOAT_MODE";
    EXPECT_EQ(zero.ps_launch_rsrc1, kRawEvidenceZero) << "observed zero is available raw evidence";
}

TEST(FragmentFloatFlags, NonPolicyAndFuturePolicyBitsSurviveAsEvidence) {
    for (uint32_t word : {0u, 1u << 29, UINT32_MAX}) {
        SCOPED_TRACE(::testing::Message() << "word=0x" << std::hex << word);
        GpuState launch;
        launch.sh[P::SPI_SHADER_PGM_RSRC1_PS] = word;
        const FragmentLaunchRsrc1 expected{true, word};
        EXPECT_EQ(extract_render_state(launch).ps_launch_rsrc1, expected)
            << "non-policy and future-policy bits survive byte-exactly as evidence";
    }
    EXPECT_FALSE(kUnavailablePayload.canonical()) << "unavailable raw payload rejects";
}

TEST(FragmentFloatFlags, FlagAuthorityIsIndependentOfModeAuthority) {
    // Direct compiler/capture callers may know the flags without knowing FLOAT_MODE.
    EXPECT_TRUE(kIndependentlyKnown.canonical() && !kUnknownMode.available)
        << "flag authority is independent of mode authority";
    EXPECT_FALSE(kUnknownIeee.canonical()) << "unknown IEEE payload rejects";
    EXPECT_FALSE(kUnknownDx10.canonical()) << "unknown DX10 payload rejects";
    EXPECT_FALSE(kBothUnknown.canonical()) << "both unknown payloads reject";
}