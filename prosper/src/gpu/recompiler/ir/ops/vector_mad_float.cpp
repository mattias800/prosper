// Non-fused MAD rounds its product before addition. Fused forms remain separate (#5225).
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
namespace {
constexpr MadFloatDef kDefs[] = {
    {Rdna2Format::VOP3, 0x141, [](Builder& ir, const TernaryInputs& x) {
         const Value product = ir.f32mul(x.s0(), x.s1());
         return ir.f32add(product, x.s2());
     }}};
}   // namespace
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
