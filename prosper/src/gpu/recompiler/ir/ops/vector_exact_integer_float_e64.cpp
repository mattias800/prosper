// AMD70648 12.8/12.8.1: the e64 integer-to-F32 values are the existing VOP1 conversions.
#include "gpu/recompiler/ir/ops/vector_exact_integer_float_e64.hpp"
#include "gpu/recompiler/ir/ops/vector_conversion.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <array>
#include <cstdint>
#include <span>

namespace prosper::gpu::ir {
std::span<const ConversionDef> vector_exact_integer_float_e64_defs() {
    static const std::array defs{
        ConversionDef{Rdna2Format::VOP3, 0x185,
                      find_vector_conversion_op(Rdna2Format::VOP1, 0x05)->build},
        ConversionDef{Rdna2Format::VOP3, 0x186,
                      find_vector_conversion_op(Rdna2Format::VOP1, 0x06)->build},
    };
    return defs;
}
const ConversionDef* find_vector_exact_integer_float_e64_op(Rdna2Format fmt, uint32_t opcode) {
    for (const auto& def : vector_exact_integer_float_e64_defs())
        if (def.fmt == fmt && def.opcode == opcode) return &def;
    return nullptr;
}
} // namespace prosper::gpu::ir
