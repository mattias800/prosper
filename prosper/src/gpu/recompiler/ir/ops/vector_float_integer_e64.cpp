// AMD 70648 12.8: truncate, saturate out-of-range values, and convert NaN to zero.
#include "gpu/recompiler/ir/ops/vector_float_integer_e64.hpp"
namespace prosper::gpu::ir {
namespace {
constexpr ConversionDef kDefs[] = {
    {Rdna2Format::VOP3, 0x187, build_f32_to_u32_sat},
    {Rdna2Format::VOP3, 0x188, build_f32_to_i32_sat},
};
} // namespace
std::span<const ConversionDef> vector_float_integer_e64_defs() {
    return kDefs;
}
const ConversionDef* find_vector_float_integer_e64_op(Rdna2Format fmt, uint32_t opcode) {
    for (const auto& def : kDefs)
        if (def.fmt == fmt && def.opcode == opcode) return &def;
    return nullptr;
}
} // namespace prosper::gpu::ir
