// AMD70648 12.7 op7: zero times anything is zero (DX9). Preserve the
// original two sequential source reads, then zero, comparison nesting and final select.
#include "gpu/recompiler/ir/ops/vector_legacy_mul.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
namespace {
Value legacy_mul(Builder& ir, const TernaryInputs& x) {
    const Value s0 = x.s0(), s1 = x.s1(), zero = ir.c32(0);
    const Value any_zero = ir.boolor_(ir.f32eq(s0, zero), ir.f32eq(s1, zero));
    return ir.select(any_zero, zero, ir.f32mul(s0, s1));
}
constexpr TernaryIntegerDef kDefs[] = {{Rdna2Format::VOP3, 0x107, legacy_mul}};
}   // namespace
std::span<const TernaryIntegerDef> vector_legacy_mul_defs() {
    return kDefs;
}
const TernaryIntegerDef* find_vector_legacy_mul_op(Rdna2Format fmt, uint32_t opcode) {
    for (const auto& def : kDefs)
        if (def.fmt == fmt && def.opcode == opcode) return &def;
    return nullptr;
}
}   // namespace prosper::gpu::ir
