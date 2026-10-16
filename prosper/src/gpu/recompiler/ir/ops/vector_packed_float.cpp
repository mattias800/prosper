// AMD70648 section12.10: packed ADD/MUL/MIN/MAX. Root retains half expansion,
// rounding, selectors and modifiers. No fused operation or new encoding is admitted.
#include "gpu/recompiler/ir/ops/vector_packed_float.hpp"
#include "gpu/recompiler/ir/ops/vector_binary_float.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
namespace {
constexpr TernaryIntegerDef kDefs[] = {{Rdna2Format::VOP3P, 0x0f, build_float_add},
                                       {Rdna2Format::VOP3P, 0x10, build_float_mul},
                                       {Rdna2Format::VOP3P, 0x11, build_float_nmin},
                                       {Rdna2Format::VOP3P, 0x12, build_float_nmax}};
}
std::span<const TernaryIntegerDef> vector_packed_float_defs() {
    return kDefs;
}
const TernaryIntegerDef* find_vector_packed_float_op(Rdna2Format fmt, uint32_t opcode) {
    for (const auto& def : kDefs)
        if (def.fmt == fmt && def.opcode == opcode) return &def;
    return nullptr;
}
}   // namespace prosper::gpu::ir
