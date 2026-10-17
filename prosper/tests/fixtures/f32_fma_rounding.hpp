// #5225: the four existing fused F32 encodings. Runtime GLSL Fma has MED portability;
// the CPU expectations select finite normal-or-zero RTE, without proving other guest MODEs.
#pragma once
#include "fixtures/f32_mad_rounding.hpp"
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
namespace prosper::test::fma_rounding {
using mad_rounding::Addend;
using mad_rounding::buffer;
using mad_rounding::Encoding;
using mad_rounding::execute;
using mad_rounding::inputs;
using mad_rounding::LaneInputs;
using mad_rounding::program;
using mad_rounding::Shape;
using mad_rounding::supports;
using prosper::gpu::Rdna2Format;
inline constexpr std::array<Encoding, 4> kEncodings = {{
    {"fmac_e32", Rdna2Format::VOP2, 0x2b, Addend::Destination, false},
    {"fmamk", Rdna2Format::VOP2, 0x2c, Addend::Source2, true},
    {"fmaak", Rdna2Format::VOP2, 0x2d, Addend::Literal, false},
    {"fma_e64", Rdna2Format::VOP3, 0x14b, Addend::Source2, false},
}};
inline uint32_t fused_expected(const Encoding& e, Shape s, const LaneInputs& in, size_t lane,
                               uint32_t rsrc1) {
    if (!in.active[lane]) return in.old_d[lane];
    const auto bits = mad_rounding::sources(e, s, in, lane);
    float result = std::fma(std::bit_cast<float>(bits[0]), std::bit_cast<float>(bits[1]),
                            std::bit_cast<float>(bits[2]));
    if (s.omod && !(rsrc1 & ((0x20u << 12) | (1u << 23))))
        result *= s.omod == 1 ? 2.0f : s.omod == 2 ? 4.0f : 0.5f;
    if (s.clamp) result = std::fmin(std::fmax(result, 0.0f), 1.0f);
    return std::bit_cast<uint32_t>(result);
}
} // namespace prosper::test::fma_rounding
