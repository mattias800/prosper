// Test-owned FLR encodings and finite bit reference; no host floating-point arithmetic.
#pragma once
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <iterator>
#include <vector>
namespace prosper::test::ir::cvt_flr_e64 {
struct Form {
    uint32_t source = 258, literal = 0, scalar = 0;
    bool initialize = false;
};
inline Form form(uint32_t index) {
    Form f;
    switch (index) {
        case 1: f.source = 257; break; // Old VDST is the source.
        case 2:
            f.source = 50;
            f.scalar = 0xc0780000;
            f.initialize = true;
            break; // -3.875.
        case 3:
            f.source = 255;
            f.literal = 0x3fe00000;
            break; // 1.75.
        case 4:
            f.source = 255;
            f.literal = 0x807fffff;
            break;
        case 5: f.source = 241; break; // Inline -0.5.
        case 6: f.source = 129; break; // Integer word1 is a positive F32 subnormal.
        case 7: f.source = 128; break;
        default: break;
    }
    return f;
}
inline std::vector<uint32_t> words(Form f = {}, uint32_t encoding = 0) {
    std::vector<uint32_t> out;
    if (f.initialize) out.insert(out.end(), {0xbeb203ff, f.scalar});
    if (encoding == 2)
        out.push_back(vop1(0x0d, kVdst, f.source));
    else
        out.insert(out.end(), {(encoding ? 0xd18d0001u : 0xd58d0001u), f.source});
    if (f.source == 255) out.push_back(f.literal);
    return out;
}
inline uint32_t source(Form f, const LaneInputs& in, size_t lane) {
    if (f.source == 258) return in.s0[lane];
    if (f.source == 257) return in.old_d[lane];
    if (f.source == 255) return f.literal;
    if (f.source == 241) return 0xbf000000;
    if (f.source == 129) return 1;
    if (f.source == 128) return 0;
    return f.scalar;
}
// AMD70648: downward conversion. Interpret the finite dyadic value as integer
// magnitude plus discarded bits. Exceptional saturation is deliberately outside this oracle.
inline std::optional<uint32_t> finite_floor(uint32_t bits) {
    const uint32_t magnitude = bits & 0x7fffffffu;
    const bool negative = (bits >> 31u) != 0;
    if (!magnitude) return 0;
    const uint32_t exponent = magnitude >> 23u;
    if (exponent < 127) return negative ? UINT32_MAX : 0;
    if (negative && magnitude == 0x4f000000u) return 0x80000000u;
    if (exponent > 157) return std::nullopt;
    const uint32_t e = exponent - 127;
    const uint32_t mantissa = (magnitude & 0x7fffffu) | 0x800000u;
    const uint32_t integer = e >= 23 ? mantissa << (e - 23) : mantissa >> (23 - e);
    const bool fraction = e < 23 && (mantissa & ((1u << (23 - e)) - 1u)) != 0;
    return negative ? 0u - integer - uint32_t(fraction) : integer;
}
inline LaneInputs inputs(uint32_t seed) {
    auto in = lane_inputs(seed);
    constexpr uint32_t edges[] = {0,          0x80000000, 1,          0x80000001, 0x007fffff,
                                  0x807fffff, 0x00800000, 0x80800000, 0x3f7fffff, 0xbf7fffff,
                                  0x3f800000, 0xbf800000, 0x3f800001, 0xbf800001, 0x3fc00000,
                                  0xbfc00000, 0x40100000, 0xc0100000, 0x4affffff, 0xcaffffff,
                                  0x4b800000, 0xcb800000, 0x4effffff, 0xcf000000};
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        seed = seed * 1664525u + 1013904223u;
        in.s0[lane] = lane < std::size(edges)
                          ? edges[lane]
                          : (seed & 0x807fffffu) | (((seed >> 23u) % 158u) << 23u);
        in.old_d[lane] = edges[(lane * 7 + seed) % std::size(edges)];
    }
    return in;
}
inline LaneRun execute(const std::vector<uint32_t>& module, const LaneInputs& in, bool flush) {
    std::vector<uint32_t> buffer;
    buffer.reserve(4 * in.lanes());
    for (const auto* array : {&in.s0, &in.s1, &in.old_d})
        buffer.insert(buffer.end(), array->begin(), array->end());
    for (bool active : in.active) buffer.push_back(active ? 1u : 0u);
    bpermute_oracle::Interpreter vm(module);
    vm.floor_flush_denorm_input = flush;
    auto out = vm.run_buffers(static_cast<uint32_t>(in.lanes()), {{3, buffer}}, 3);
    if (!vm.error.empty()) return {{}, vm.error};
    out.resize(in.lanes());
    return {out, {}};
}
} // namespace prosper::test::ir::cvt_flr_e64
