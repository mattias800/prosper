// Pure loaded-word calculations must keep component positions and the second word's bytes.
// Access/admission/predicated publication are deliberately exercised by the integration witness.
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/ir/lower/buffer_format_values.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstdint>
#include <cstddef>
#include <functional>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::gpu::ir;

namespace {
using Words = std::array<uint32_t, 4>;
using Emit = std::function<std::vector<uint32_t>(SpirvCompute&, const Words&)>;

std::vector<uint32_t> run(const std::vector<Words>& lanes, uint32_t outputs, const Emit& emit,
                          bool half_profile = false) {
    const auto n = static_cast<uint32_t>(lanes.size());
    SpirvCompute b;
    b.begin(4, nullptr, n, 1, 1, n, 0, true, true);
    Words ids{};
    for (uint32_t i = 0; i < 4; ++i) ids[i] = b.load_input(i);
    const auto result = emit(b, ids);
    EXPECT_EQ(result.size(), outputs);
    for (uint32_t k = 0; k < result.size(); ++k) b.store_output_word(result[k], outputs, k);
    std::vector<uint32_t> data;
    for (const auto& lane : lanes) data.insert(data.end(), lane.begin(), lane.end());
    bpermute_oracle::Interpreter vm(b.finish());
    vm.half_rte_preserve_denorm = half_profile;
    auto out = vm.run_buffers(n, {{0, data}, {1, std::vector<uint32_t>(size_t{n} * outputs)}}, 1);
    EXPECT_TRUE(vm.error.empty()) << vm.error;
    return out;
}

uint32_t extend(uint32_t raw, uint32_t bits) {
    const uint32_t sign = 1u << (bits - 1);
    return (raw ^ sign) - sign;
}

TEST(IrLowerBufferFormatValues, RawShiftsJoinsAndExtensionsRetainBothWords) {
    for (uint32_t wave : {32u, 64u}) {
        std::vector<Words> lanes;
        lanes.reserve(64);
        for (uint32_t lane = 0; lane < wave; ++lane)
            lanes.push_back(
                {lane & 3u, 0x91827364u ^ (lane * 0x10001u), 0xfedcba98u ^ (lane * 0x1010101u), 0});
        const auto out = run(lanes, 6, [](SpirvCompute& b, const Words& x) {
            const auto first = lower_buffer_raw_shift(b, x[0], x[1]);
            const auto raw = lower_buffer_raw_join(b, first.joined, x[2], first.shift);
            return std::vector<uint32_t>{
                first.shift,
                first.joined,
                raw,
                lower_buffer_half_join(b, first.joined, x[2], first.shift),
                lower_buffer_dynamic_join(b, first.joined, x[2], first.shift),
                lower_buffer_raw_field(b, raw, 16, true)};
        });
        ASSERT_EQ(out.size(), wave * 6);
        for (uint32_t lane = 0; lane < wave; ++lane) {
            const auto& x = lanes[lane];
            const uint32_t shift = x[0] * 8;
            const uint32_t joined = static_cast<uint32_t>(((uint64_t{x[2]} << 32) | x[1]) >> shift);
            EXPECT_EQ(out[size_t{lane} * 6], shift);
            EXPECT_EQ(out[size_t{lane} * 6 + 1], x[1] >> shift);
            for (uint32_t k = 2; k < 5; ++k) EXPECT_EQ(out[size_t{lane} * 6 + k], joined);
            EXPECT_EQ(out[size_t{lane} * 6 + 5], extend(joined & 0xffffu, 16));
        }
    }
}

TEST(IrLowerBufferFormatValues, ComponentLocationsWrapWithoutDroppingByteOffsets) {
    for (uint32_t bytes : {1u, 2u})
        for (uint32_t component = 0; component < 4; ++component) {
            std::vector<Words> lanes;
            lanes.reserve(64);
            for (uint32_t lane = 0; lane < 64; ++lane)
                lanes.push_back(
                    {lane < 4 ? 0xfffffffcu + lane : lane * 33u + 3u, 0xfffffffeu + lane, 0, 0});
            const auto out = run(lanes, 6, [=](SpirvCompute& b, const Words& x) {
                const auto dyn = lower_buffer_dynamic_location(b, x[0], component, bytes);
                const auto half = lower_buffer_half_location(b, x[0], component);
                const auto stat = lower_buffer_static_location(b, x[1], component, bytes);
                return std::vector<uint32_t>{dyn.index,  dyn.shift,  half.index,
                                             half.shift, stat.index, b.uconst(stat.bit_offset)};
            });
            ASSERT_EQ(out.size(), 64u * 6);
            for (uint32_t lane = 0; lane < 64; ++lane) {
                const uint32_t dynamic_address = lanes[lane][0] + component * bytes;
                const uint32_t half_address = lanes[lane][0] + component * 2;
                EXPECT_EQ(out[size_t{lane} * 6], dynamic_address / 4);
                EXPECT_EQ(out[size_t{lane} * 6 + 1], (dynamic_address % 4) * 8);
                EXPECT_EQ(out[size_t{lane} * 6 + 2], half_address / 4);
                EXPECT_EQ(out[size_t{lane} * 6 + 3], (half_address % 4) * 8);
                EXPECT_EQ(out[size_t{lane} * 6 + 4], lanes[lane][1] + component * bytes / 4);
                EXPECT_EQ(out[size_t{lane} * 6 + 5], (component * bytes % 4) * 8);
            }
        }
}

TEST(IrLowerBufferFormatValues, IntegerPackedFieldsUseTheArchitecturalWidths) {
    for (uint32_t component = 0; component < 4; ++component) {
        std::vector<Words> lanes;
        lanes.reserve(64);
        for (uint32_t lane = 0; lane < 64; ++lane)
            lanes.push_back({0x81c37e55u ^ (lane * 0x1020304u), 0, 0, 0});
        const auto out = run(lanes, 2, [=](SpirvCompute& b, const Words& x) {
            return std::vector<uint32_t>{
                lower_buffer_packed_field(b, x[0], component, false, true, false, false),
                lower_buffer_packed_field(b, x[0], component, false, false, true, false)};
        });
        ASSERT_EQ(out.size(), 128u);
        const uint32_t width = component == 3 ? 2 : 10;
        for (uint32_t lane = 0; lane < 64; ++lane) {
            const uint32_t raw = (lanes[lane][0] >> (component * 10)) & ((1u << width) - 1);
            EXPECT_EQ(out[size_t{lane} * 2], raw);
            EXPECT_EQ(out[size_t{lane} * 2 + 1], extend(raw, width));
        }
    }
}

TEST(IrLowerBufferFormatValues, HalfAndUnsignedFloatFieldsHaveHandWorkedFiniteValues) {
    // binary16 {+0,+1,-1,+0.5}; unsigned F11's +1 is exponent15/mantissa0 = 0x3c0.
    constexpr uint32_t halves[] = {0, 0x3c00, 0xbc00, 0x3800};
    constexpr uint32_t float_bits[] = {0, 0x3f800000, 0xbf800000, 0x3f000000};
    std::vector<Words> lanes;
    lanes.reserve(64);
    for (uint32_t lane = 0; lane < 64; ++lane)
        lanes.push_back({halves[lane % 4] | (halves[(lane + 1) % 4] << 16),
                         0x3c0u | (0x380u << 11) | (0x1e0u << 22), 0, 0});
    const auto out = run(
        lanes, 5,
        [](SpirvCompute& b, const Words& x) {
            return std::vector<uint32_t>{
                lower_buffer_half_field(b, x[0]),
                lower_buffer_static_field(b, x[0], 16, 2, true, false, false, false, 0),
                lower_buffer_packed_field(b, x[1], 0, true, false, false, false),
                lower_buffer_packed_field(b, x[1], 1, true, false, false, false),
                lower_buffer_packed_field(b, x[1], 2, true, false, false, false)};
        },
        true);
    ASSERT_EQ(out.size(), 320u);
    for (uint32_t lane = 0; lane < 64; ++lane) {
        EXPECT_EQ(out[size_t{lane} * 5], float_bits[lane % 4]);
        EXPECT_EQ(out[size_t{lane} * 5 + 1], float_bits[(lane + 1) % 4]);
        EXPECT_EQ(out[size_t{lane} * 5 + 2], 0x3f800000u);
        EXPECT_EQ(out[size_t{lane} * 5 + 3], 0x3f000000u);
        EXPECT_EQ(out[size_t{lane} * 5 + 4], 0x3f800000u);
    }
}

TEST(IrLowerBufferFormatValues, HalfUnpackRequiresTheExplicitInterpreterProfile) {
    SpirvCompute b;
    b.begin(1, nullptr, 32, 1, 1, 32, 0, true, true);
    b.store_output_word(lower_buffer_half_field(b, b.load_input(0)), 1, 0);
    bpermute_oracle::Interpreter vm(b.finish());
    const auto out = vm.run_buffers(
        32, {{0, std::vector<uint32_t>(32, 0x3c00)}, {1, std::vector<uint32_t>(32)}}, 1);
    EXPECT_TRUE(out.empty());
    EXPECT_EQ(vm.error, "unsupported live ExtInst");
}

TEST(IrLowerBufferFormatValues, NormalizedAndIntegerFieldsRetainEndpointsAndSign) {
    for (uint32_t bytes : {1u, 2u}) {
        const uint32_t width = bytes * 8, mask = (1u << width) - 1;
        const uint32_t sign = 1u << (width - 1);
        const float norm = static_cast<float>(sign - 1);
        std::vector<Words> lanes;
        lanes.reserve(64);
        const uint32_t edges[] = {0, 1, sign - 1, sign, mask};
        for (uint32_t lane = 0; lane < 64; ++lane) lanes.push_back({edges[lane % 5], 0, 0, 0});
        const auto out = run(lanes, 4, [=](SpirvCompute& b, const Words& x) {
            return std::vector<uint32_t>{
                lower_buffer_dynamic_field(b, x[0], bytes, true, true, norm, false),
                lower_buffer_dynamic_field(b, x[0], bytes, false, false, 0, false),
                lower_buffer_static_field(b, x[0], 0, bytes, false, false, true, false, 0),
                lower_buffer_static_field(b, x[0], 0, bytes, false, false, false, false,
                                          static_cast<float>(mask))};
        });
        ASSERT_EQ(out.size(), 256u);
        for (uint32_t lane = 0; lane < 64; ++lane) {
            const uint32_t raw = lanes[lane][0];
            // The most-negative signed endpoint clamps to -1; -1 maps to -1/(2^(n-1)-1).
            const auto signed_value = std::bit_cast<int32_t>(extend(raw, width));
            const float snorm = signed_value < -static_cast<int32_t>(sign - 1)
                                    ? -1.0f
                                    : static_cast<float>(signed_value) / norm;
            const float unorm = static_cast<float>(raw) / static_cast<float>(mask);
            EXPECT_EQ(out[size_t{lane} * 4], std::bit_cast<uint32_t>(snorm));
            EXPECT_EQ(out[size_t{lane} * 4 + 1], raw);
            EXPECT_EQ(out[size_t{lane} * 4 + 2], extend(raw, width));
            EXPECT_EQ(out[size_t{lane} * 4 + 3], std::bit_cast<uint32_t>(unorm));
        }
    }
}
}   // namespace
