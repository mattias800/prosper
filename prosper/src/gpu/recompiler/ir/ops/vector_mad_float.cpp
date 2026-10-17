// Non-fused F32 MAD rounds its product before addition, without contraction (#5225).
// The product stage must precede the lazy third source read, just as in the legacy emitter.
#include "gpu/recompiler/ir/ops/vector_mad_float.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <array>
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
Value build_nonfused_mad_float(Builder& ir, const TernaryInputs& x) {
    const Value product = ir.f32mul_no_contraction(x.s0(), x.s1());
    return ir.f32add_no_contraction(product, x.s2());
}
namespace {
constexpr MadFloatDef kDefs[] = {{Rdna2Format::VOP3, 0x141, build_nonfused_mad_float}};
}   // namespace
uint32_t emit_nonfused_mad_values(SpirvCompute& b, uint32_t a, uint32_t c, uint32_t addend) {
    const std::array<uint32_t, 3> sources = {a, c, addend};
    return emit_mad_float_op(b, kDefs[0], [&](uint32_t slot) { return sources[slot]; });
}
std::span<const MadFloatDef> vector_mad_float_defs() {
    return kDefs;
}
const MadFloatDef* find_vector_mad_float_op(Rdna2Format fmt, uint32_t opcode) {
    for (const auto& def : kDefs)
        if (def.fmt == fmt && def.opcode == opcode) return &def;
    return nullptr;
}
Known eval_mad_float_op(const MadFloatDef& def, const std::array<Known, 3>& sources) {
    return eval_ternary_integer_op(def, sources);
}
uint32_t emit_mad_float_op(SpirvCompute& b, const MadFloatDef& def,
                           const TernarySourceReader& read) {
    return emit_ternary_integer_op(b, def, read);
}
}   // namespace prosper::gpu::ir
