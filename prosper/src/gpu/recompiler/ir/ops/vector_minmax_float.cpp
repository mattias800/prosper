// AMD 70648 12.12, operations 337/340/343. MED3's any-NaN fallback is explicitly MIN3.
#include "gpu/recompiler/ir/ops/vector_minmax_float.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <array>
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
namespace {
using In = TernaryInputs;
Value min3(Builder& ir, const In& x) {
    return ir.f32nmin(ir.f32nmin(x.s0(), x.s1()), x.s2());
}
Value max3(Builder& ir, const In& x) {
    return ir.f32nmax(ir.f32nmax(x.s0(), x.s1()), x.s2());
}
Value med3(Builder& ir, const In& x) {
    const Value s0 = x.s0(), s1 = x.s1(), s2 = x.s2();
    const Value mn = ir.f32nmin(s0, s1), mx = ir.f32nmax(s0, s1);
    const Value med = ir.f32nmax(mn, ir.f32nmin(mx, s2));
    const Value min3 = ir.f32nmin(mn, s2);
    const Value nan_any =
        ir.boolor_(ir.f32unordne(s0, s0), ir.boolor_(ir.f32unordne(s1, s1), ir.f32unordne(s2, s2)));
    return ir.select(nan_any, min3, med);
}
constexpr MinmaxFloatDef kDefs[] = {{Rdna2Format::VOP3, 0x151, min3},
                                    {Rdna2Format::VOP3, 0x154, max3},
                                    {Rdna2Format::VOP3, 0x157, med3}};
}   // namespace
std::span<const MinmaxFloatDef> vector_minmax_float_defs() {
    return kDefs;
}
const MinmaxFloatDef* find_vector_minmax_float_op(Rdna2Format fmt, uint32_t opcode) {
    for (const auto& def : kDefs)
        if (def.fmt == fmt && def.opcode == opcode) return &def;
    return nullptr;
}
Known eval_minmax_float_op(const MinmaxFloatDef& def, const std::array<Known, 3>& sources) {
    return eval_ternary_integer_op(def, sources);
}
uint32_t emit_minmax_float_op(SpirvCompute& b, const MinmaxFloatDef& def,
                              const TernarySourceReader& read) {
    return emit_ternary_integer_op(b, def, read);
}
}   // namespace prosper::gpu::ir
