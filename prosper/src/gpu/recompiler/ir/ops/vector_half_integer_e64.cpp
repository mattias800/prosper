// AMD70648 12.8: truncating half-to-integer aliases reuse the exact existing e32 graphs.
#include "gpu/recompiler/ir/ops/vector_half_integer_e64.hpp"
#include "gpu/recompiler/ir/ops/vector_conversion.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <span>
#include <cstdint>
namespace prosper::gpu::ir {
namespace {
constexpr ConversionDef kDefs[] = {
    {Rdna2Format::VOP3, 0x1d2, build_half_to_u16_sat},
    {Rdna2Format::VOP3, 0x1d3, build_half_to_i16_sat},
};
}
std::span<const ConversionDef> vector_half_integer_e64_defs() {
    return kDefs;
}
const ConversionDef* find_vector_half_integer_e64_op(Rdna2Format fmt, uint32_t opcode) {
    for (const auto& def : kDefs)
        if (def.fmt == fmt && def.opcode == opcode) return &def;
    return nullptr;
}
} // namespace prosper::gpu::ir
