// AMD70648 12.8: F16-to-F32 widening is exact. The lowerer admits normal constants only.
#include "gpu/recompiler/ir/ops/vector_half_float_exact_e64.hpp"
#include "gpu/recompiler/ir/ops/vector_conversion.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
namespace prosper::gpu::ir {
const ConversionDef* find_vector_half_float_exact_e64_op(Rdna2Format fmt, uint32_t opcode) {
    if (fmt != Rdna2Format::VOP3 || opcode != 0x18b) return nullptr;
    static const ConversionDef alias = [] {
        const auto* old = find_vector_conversion_op(Rdna2Format::VOP1, 0x0b);
        return ConversionDef{Rdna2Format::VOP3, 0x18b, old ? old->build : nullptr};
    }();
    return alias.build ? &alias : nullptr;
}
} // namespace prosper::gpu::ir
