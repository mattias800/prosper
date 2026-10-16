// AMD70648 6.2/6.4 hand answers: output scale is precision-aware, flushes denormals and -0,
// and must round before deciding whether an underflow candidate is still denormal.
#include "gpu/recompiler/ir/ops/float_output_modifier.hpp"
#include <gtest/gtest.h>
#include <array>
#include <cstdint>
using namespace prosper::gpu::ir;
namespace {
struct Answer {
    uint32_t input, scale, rounding, output;
};
void check(FloatResultWidth width, const Answer& x) {
    const auto got =
        eval_float_output_modifier(x.input, {uint8_t(x.scale), uint8_t(x.rounding), width});
    ASSERT_TRUE(got.has_value());
    EXPECT_EQ(got.value_or(0), x.output)
        << std::hex << x.input << " scale " << x.scale << " rounding " << x.rounding;
}
} // namespace
TEST(IrCoreFloatOutputModifier, Binary32HandWorkedNormalUnderflowAndZero) {
    for (const auto& x :
         std::array{Answer{0x40880000, 1, 0, 0x41080000}, // 4.25 * 2 = 8.5
                    Answer{0x40880000, 2, 0, 0x41880000}, // 4.25 * 4 = 17
                    Answer{0x40880000, 3, 0, 0x40080000}, // 4.25 / 2 = 2.125
                    Answer{0x00800000, 3, 0, 0},          // minimum normal / 2 flushes
                    Answer{0x80800000, 3, 0, 0},
                    Answer{0x80000000, 1, 0, 0},          // -0 -> +0 only when applied
                    Answer{0x80000000, 0, 0, 0x80000000},
                    Answer{0x00400000, 1, 0, 0x00800000}, // subnormal * 2 reaches normal
                    Answer{0x00ffffff, 3, 0, 0x00800000}, // tie rounds up to EVEN min normal
                    Answer{0x00ffffff, 3, 3, 0}, Answer{0x00ffffff, 3, 1, 0x00800000},
                    Answer{0x80ffffff, 3, 2, 0x80800000}, Answer{0x80ffffff, 3, 1, 0}})
        check(FloatResultWidth::F32, x);
}
TEST(IrCoreFloatOutputModifier, Binary16UsesItsOwnNormalBoundaryAndRounding) {
    // Inputs/results here are the EXACT F32 representation of a half result before packing.
    for (const auto& x : std::array{
             Answer{0x40880000, 1, 0, 0x41080000},
             Answer{0x38800000, 3, 0, 0}, // 2^-14 / 2 is a half subnormal, not F32 subnormal
             Answer{0xb8800000, 3, 0, 0}, Answer{0x80000000, 2, 0, 0},
             Answer{0x80000000, 0, 0, 0x80000000}, Answer{0x38000000, 1, 0, 0x38800000},
             Answer{0x38ffe000, 3, 0, 0x38800000}, // halfway below min half normal
             Answer{0x38ffe000, 3, 3, 0}, Answer{0x38ffe000, 3, 1, 0x38800000},
             Answer{0xb8ffe000, 3, 2, 0xb8800000}, Answer{0xb8ffe000, 3, 1, 0}})
        check(FloatResultWidth::F16, x);
}
TEST(IrCoreFloatOutputModifier, OverflowUsesTheSignAndTheResultPrecisionsRoundMode) {
    for (auto width : {FloatResultWidth::F32, FloatResultWidth::F16}) {
        const uint32_t maximum = width == FloatResultWidth::F32 ? 0x7f7fffff : 0x477fe000;
        for (uint32_t sign : {0u, 0x80000000u})
            for (uint32_t round = 0; round < 4; ++round) {
                const bool infinite = round == 0 || (round == 1 && !sign) || (round == 2 && sign);
                check(width, {maximum | sign, 1, round, sign | (infinite ? 0x7f800000 : maximum)});
            }
    }
}
TEST(IrCoreFloatOutputModifier, DisabledModifierDoesNotRoundFlushOrQuietItsInput) {
    for (auto width : {FloatResultWidth::F16, FloatResultWidth::F32})
        for (uint32_t bits : {0u, 0x80000000u, 1u, 0x807fffffu, 0x7f800001u, 0xffffffffu})
            check(width, {bits, 0, 0, bits});
    EXPECT_FALSE(eval_float_output_modifier(std::nullopt, {1, 0, FloatResultWidth::F32}));
}
TEST(IrCoreFloatOutputModifier, TrueInfinityAndNanPayloadsAreKeptInBothPrecisions) {
    for (uint8_t round = 0; round < 4; ++round)
        for (uint32_t sign : {0u, 0x80000000u})
            for (auto width : {FloatResultWidth::F16, FloatResultWidth::F32}) {
                const uint32_t payload = width == FloatResultWidth::F16 ? 0x2000 : 1;
                check(width, {sign | 0x7f800000, 1, round, sign | 0x7f800000});
                check(width, {sign | 0x7f800000 | payload, 1, round, sign | 0x7fc00000 | payload});
            }
}
