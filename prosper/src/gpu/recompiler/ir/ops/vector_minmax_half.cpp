// AMD 70648 12.12 ops849/852/855: half expansion is exact. MIN3/MAX3 nest
// their one-NaN operations; MED3 explicitly selects MIN3 when any source is NaN.
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/ir/ops/vector_minmax_half.hpp"
#include "gpu/recompiler/ir/ops/vector_minmax_float.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
namespace {
Value min3(Builder& ir, const TernaryInputs& x) {
    return build_float_min3(ir, x);
}
Value max3(Builder& ir, const TernaryInputs& x) {
    return build_float_max3(ir, x);
}
Value med3(Builder& ir, const TernaryInputs& x) {
    return build_float_med3(ir, x);
}
constexpr TernaryIntegerDef kDefs[] = {{Rdna2Format::VOP3, 0x351, min3},
                                       {Rdna2Format::VOP3, 0x354, max3},
                                       {Rdna2Format::VOP3, 0x357, med3}};
}   // namespace
std::span<const TernaryIntegerDef> vector_minmax_half_defs() {
    return kDefs;
}
const TernaryIntegerDef* find_vector_minmax_half_op(Rdna2Format fmt, uint32_t opcode) {
    for (const auto& def : kDefs)
        if (def.fmt == fmt && def.opcode == opcode) return &def;
    return nullptr;
}
}   // namespace prosper::gpu::ir
