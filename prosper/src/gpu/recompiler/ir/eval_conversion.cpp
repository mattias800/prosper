// AMD 70648 12.8: saturate out-of-range conversions, truncate and map NaN to zero.
// Half unpack is exact for finite values/infinity; its unspecified NaN payload stays unknown.
// Integer-to-float uses nominal RTE, as the existing CPU test oracle does. No guest fold widens.
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include <optional>
#include "gpu/recompiler/ir/eval_conversion.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
namespace prosper::gpu::ir {
namespace {
Known unpack(uint32_t a, uint32_t component) {
    if (component > 1) return std::nullopt;
    const uint32_t h = (a >> (component * 16)) & 0xffffu;
    const uint32_t sign = (h & 0x8000u) << 16;
    const uint32_t exponent = (h >> 10) & 31u, fraction = h & 1023u;
    if (exponent == 31) return fraction ? std::nullopt : Known(sign | 0x7f800000u);
    if (!exponent && !fraction) return sign;
    const float magnitude = exponent ? std::ldexp(float(1024 + fraction), int(exponent) - 25)
                                     : std::ldexp(float(fraction), -24);
    return sign | std::bit_cast<uint32_t>(magnitude);
}
}   // namespace
Known evaluate_conversion(Opcode op, uint32_t imm, uint32_t a, uint32_t b, uint32_t c) {
    const float x = std::bit_cast<float>(a);
    switch (op) {
        case Opcode::I32ToF32:
            return std::bit_cast<uint32_t>(static_cast<float>(std::bit_cast<int32_t>(a)));
        case Opcode::U32ToF32: return std::bit_cast<uint32_t>(static_cast<float>(a));
        case Opcode::F32ToU32Sat:
            if (std::isnan(x) || x <= 0) return 0;
            if (x >= 0x1p32f) return 0xffffffffu;
            return static_cast<uint32_t>(x);
        case Opcode::F32ToI32Sat:
            if (std::isnan(x)) return 0;
            if (x <= -0x1p31f) return 0x80000000u;
            if (x >= 0x1p31f) return 0x7fffffffu;
            return static_cast<uint32_t>(static_cast<int32_t>(x));
        case Opcode::BfeS: {
            const uint32_t off = std::min(b, 32u), count = std::min(c, 32u - off);
            if (!count) return 0;
            const uint32_t mask = count == 32 ? 0xffffffffu : (1u << count) - 1u;
            const uint32_t field = (a >> off) & mask;
            return field & (1u << (count - 1)) ? field | ~mask : field;
        }
        case Opcode::UnpackHalf: return unpack(a, imm);
        case Opcode::F32Ge: return uint32_t(x >= std::bit_cast<float>(b));
        default: return std::nullopt;
    }
}
}   // namespace prosper::gpu::ir
