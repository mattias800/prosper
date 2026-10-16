// AMD70648 sections 6.2 and 6.4: an enabled output scale rounds in the result precision,
// flushes denormals, and canonicalizes zero. Launch-state selection belongs in lower/.
#pragma once
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include <cstdint>
namespace prosper::gpu {
struct SpirvCompute;
}
namespace prosper::gpu::ir {
enum class FloatResultWidth { F16, F32, F64 };
struct FloatOutputModifier {
    uint8_t scale = 0;
    uint8_t rounding = 0;
    FloatResultWidth width = FloatResultWidth::F32;
};
Value build_float_output_modifier(Builder& ir, Value bits, FloatOutputModifier modifier);
Known eval_float_output_modifier(Known bits, FloatOutputModifier modifier);
uint32_t emit_float_output_modifier(SpirvCompute& b, uint32_t bits, FloatOutputModifier modifier);
} // namespace prosper::gpu::ir
