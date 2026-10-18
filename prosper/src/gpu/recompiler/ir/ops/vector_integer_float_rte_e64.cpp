// AMD70648 6.4/12.8: FP_ROUND[1:0] selects the integer-to-F32 result's rounding.
// The lowering supplies a known-RTE launch and a device-gated RoundingModeRTE 32 module.
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/ir/ops/vector_conversion.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
#include "gpu/recompiler/ir/ops/vector_integer_float_rte_e64.hpp"

namespace prosper::gpu::ir {
namespace {
Value signed_value(Builder& ir, const ConversionIn& x) {
    const Value converted = ir.i32_to_f32(x.s0);
    // Vulkan's implicit NSZ may change the conversion's zero sign. Select raw integer +0
    // after the conversion/bitcast; every nonzero integer has a finite nonzero F32 result.
    return ir.select(ir.ieq(x.s0, ir.c32(0)), ir.c32(0), converted);
}
Value unsigned_value(Builder& ir, const ConversionIn& x) {
    const Value converted = ir.u32_to_f32(x.s0);
    return ir.select(ir.ieq(x.s0, ir.c32(0)), ir.c32(0), converted);
}
constexpr ConversionDef defs[] = {
    {Rdna2Format::VOP3, 0x185, signed_value},
    {Rdna2Format::VOP3, 0x186, unsigned_value},
};
} // namespace
std::span<const ConversionDef> vector_integer_float_rte_e64_defs() {
    return defs;
}
const ConversionDef* find_vector_integer_float_rte_e64_op(Rdna2Format fmt, uint32_t opcode) {
    for (const auto& def : defs)
        if (def.fmt == fmt && def.opcode == opcode) return &def;
    return nullptr;
}
} // namespace prosper::gpu::ir
