// AMD 70648 12.12: integer triple extrema and median. Keep the old source-read positions.
#include "gpu/recompiler/ir/ops/vector_minmax.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
namespace {
using F = Rdna2Format;
using In = TernaryInputs;
Value median(Builder& ir, const In& x, bool is_signed) {
    const Value s0 = x.s0(), s1 = x.s1(), s2 = x.s2();
    const Value mn = is_signed ? ir.smin(s0, s1) : ir.umin(s0, s1);
    const Value mx = is_signed ? ir.smax(s0, s1) : ir.umax(s0, s1);
    return is_signed ? ir.smax(mn, ir.smin(mx, s2)) : ir.umax(mn, ir.umin(mx, s2));
}
constexpr TernaryIntegerDef kDefs[] = {
    {F::VOP3, 0x152,
     [](Builder& ir, const In& x) { return ir.smin(ir.smin(x.s0(), x.s1()), x.s2()); }},
    {F::VOP3, 0x153,
     [](Builder& ir, const In& x) { return ir.umin(ir.umin(x.s0(), x.s1()), x.s2()); }},
    {F::VOP3, 0x155,
     [](Builder& ir, const In& x) { return ir.smax(ir.smax(x.s0(), x.s1()), x.s2()); }},
    {F::VOP3, 0x156,
     [](Builder& ir, const In& x) { return ir.umax(ir.umax(x.s0(), x.s1()), x.s2()); }},
    {F::VOP3, 0x158, [](Builder& ir, const In& x) { return median(ir, x, true); }},
    {F::VOP3, 0x159, [](Builder& ir, const In& x) { return median(ir, x, false); }},
};
}   // namespace
std::span<const TernaryIntegerDef> vector_minmax_defs() {
    return kDefs;
}
const TernaryIntegerDef* find_vector_minmax_op(Rdna2Format fmt, uint32_t opcode) {
    for (const auto& def : kDefs)
        if (def.fmt == fmt && def.opcode == opcode) return &def;
    return nullptr;
}
}   // namespace prosper::gpu::ir
