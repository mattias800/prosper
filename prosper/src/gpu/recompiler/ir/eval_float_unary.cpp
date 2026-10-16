// Nominal binary32 reference for AMD 70648 12.8 / GLSL.std.450. Unknown inputs and
// unspecified domains remain unknown. This does not establish a driver's approximation bits
// or its implicit FP mode, and does not widen the guest constant folder.
#include "gpu/recompiler/ir/eval.hpp"
#include <optional>
#include "gpu/recompiler/ir/eval_float_unary.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include <bit>
#include <cmath>
#include <cstdint>
namespace prosper::gpu::ir {
namespace {
float round_even(float x) {
    const float lower = std::floor(x), fraction = x - lower;
    const float rounded = fraction < 0.5f                  ? lower
                          : fraction > 0.5f                ? lower + 1.0f
                          : std::fmod(lower, 2.0f) == 0.0f ? lower
                                                           : lower + 1.0f;
    return rounded == 0.0f ? std::copysign(0.0f, x) : rounded;
}
}   // namespace
Known evaluate_float_unary(Opcode op, uint32_t word) {
    const float x = std::bit_cast<float>(word);
    if (!std::isfinite(x)) return std::nullopt;
    float result;
    switch (op) {
        case Opcode::F32Fract: result = x - std::floor(x); break;
        case Opcode::F32Trunc: result = std::trunc(x); break;
        case Opcode::F32Ceil: result = std::ceil(x); break;
        case Opcode::F32RoundEven: result = round_even(x); break;
        case Opcode::F32Floor: result = std::floor(x); break;
        case Opcode::F32Exp2: result = std::exp2(x); break;
        case Opcode::F32Log2:
            if (x <= 0.0f) return std::nullopt;
            result = std::log2(x);
            break;
        case Opcode::F32Rcp:
            if (x == 0.0f) return std::nullopt;
            result = 1.0f / x;
            break;
        case Opcode::F32InverseSqrt:
            if (x <= 0.0f) return std::nullopt;
            result = 1.0f / std::sqrt(x);
            break;
        case Opcode::F32Sqrt:
            if (x < 0.0f) return std::nullopt;
            result = std::sqrt(x);
            break;
        case Opcode::F32Sin: result = std::sin(x); break;
        case Opcode::F32Cos: result = std::cos(x); break;
        default: return std::nullopt;
    }
    if (!std::isfinite(result)) return std::nullopt;
    return std::bit_cast<uint32_t>(result);
}
} // namespace prosper::gpu::ir
